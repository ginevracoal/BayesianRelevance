"""
Deterministic Neural Network model.
Last layer is separated from the others.
"""

import os
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as nnf
import torch.optim as torchopt
import torch.nn.functional as F

from utils.data import *
from utils.savedir import *
from utils.seeding import *
from utils.model_settings import baseNN_settings

from lrp.linear import Linear 
from lrp.maxpool import MaxPool2d 
from lrp.conv import Conv2d 
from lrp.sequential import Sequential 

DEBUG = False


class baseNN(nn.Module):

    def __init__(self, input_shape, output_size, dataset_name, hidden_size, activation, 
                       architecture, epochs, lr):
        super(baseNN, self).__init__()

        if math.log(hidden_size, 2).is_integer() is False or hidden_size<16:
            raise ValueError("\nhidden size should be a power of 2 greater than 16.")

        self.dataset_name = dataset_name
        self.architecture = architecture
        self.hidden_size = hidden_size 
        self.activation = activation
        self.lr, self.epochs = lr, epochs
        self.loss_func = nn.CrossEntropyLoss()
        self.set_model(architecture, activation, input_shape, output_size, hidden_size)
        self.name = str(dataset_name)+"_baseNN_hid="+str(hidden_size)+\
                    "_arch="+str(self.architecture)+"_act="+str(self.activation)+\
                    "_ep="+str(self.epochs)+"_lr="+str(self.lr)

        print("\nbaseNN total number of weights =", sum(p.numel() for p in self.parameters()))
        self.n_layers = len(list(self.model.children()))
        learnable_params = self.model.state_dict()
        self.n_learnable_layers = int(len(learnable_params)/2)

    def set_model(self, architecture, activation, input_shape, output_size, hidden_size):

        input_size = input_shape[0]*input_shape[1]*input_shape[2]
        in_channels = input_shape[0]

        if activation == "relu":
            activ = nn.ReLU
        elif activation == "leaky":
            activ = nn.LeakyReLU
        elif activation == "sigm":
            activ = nn.Sigmoid
        elif activation == "tanh":
            activ = nn.Tanh
        else: 
            raise AssertionError("\nWrong activation name.")

        if architecture == "fc":

            self.model = nn.Sequential(
                nn.Flatten(), 
                Linear(input_size, hidden_size),
                activ(),
                Linear(hidden_size, output_size))

            self.learnable_layers_idxs = [1, 3]

        elif architecture == "fc2":
            self.model = nn.Sequential(
                nn.Flatten(),
                Linear(input_size, hidden_size),
                activ(),
                Linear(hidden_size, hidden_size),
                activ(),
                Linear(hidden_size, output_size)
                )

            self.learnable_layers_idxs = [1, 3, 5]

        elif architecture == "fc4":
            self.model = nn.Sequential(
                nn.Flatten(),
                Linear(input_size, hidden_size),
                activ(),
                Linear(hidden_size, hidden_size),
                activ(),
                Linear(hidden_size, hidden_size),
                activ(),
                Linear(hidden_size, hidden_size),
                activ(),
                Linear(hidden_size, output_size))

            self.learnable_layers_idxs = [1, 3, 5, 7, 9]

        elif architecture == "conv":

            if self.dataset_name in ["mnist","fashion_mnist"]:

                self.model = nn.Sequential(
                    Conv2d(in_channels, 32, kernel_size=5),
                    activ(),
                    MaxPool2d(kernel_size=2),
                    Conv2d(32, hidden_size, kernel_size=5),
                    activ(),
                    MaxPool2d(kernel_size=2, stride=1),
                    nn.Flatten(),
                    Linear(int(hidden_size/(4*4))*input_size, output_size))

                self.learnable_layers_idxs = [0, 3, 7]

            elif self.dataset_name in ["cifar"]:

                self.model = nn.Sequential(

                    # Conv Layer block 1
                    Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    activ(),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                    activ(),
                    MaxPool2d(kernel_size=2, stride=2),

                    # Conv Layer block 2
                    Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    activ(),
                    Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                    activ(),
                    MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(p=0.05),

                    # Conv Layer block 3
                    Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    activ(),
                    Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    activ(),
                    MaxPool2d(kernel_size=2, stride=2),

                    # Linear
                    nn.Dropout(p=0.1),
                    nn.Flatten(),
                    Linear(4096, hidden_size),
                    # Linear(8192, hidden_size),
                    # Linear(16384, hidden_size),
                    activ(),
                    Linear(hidden_size, 512),
                    activ(),
                    nn.Dropout(p=0.1),
                    Linear(512, output_size))

                # self.learnable_layers_idxs = [0, 3, 6, 9, 13, 16, 21, 23, 26]

            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features + 1

    def train(self, train_loader, savedir, device):
        print("\n == baseNN training ==")
        self.to(device)

        optimizer = torchopt.Adam(params=self.parameters(), lr=self.lr)

        start = time.time()
        for epoch in range(self.epochs):
            total_loss = 0.0
            correct_predictions = 0.0

            for x_batch, y_batch in train_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).argmax(-1)
                outputs = self.forward(x_batch)
                
                optimizer.zero_grad()
                loss = self.loss_func(outputs, y_batch)
                loss.backward()
                optimizer.step()

                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == y_batch).sum()
                total_loss += loss.data.item() / len(train_loader.dataset)
            
            accuracy = 100 * correct_predictions / len(train_loader.dataset)
            print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", 
                  end="\t")

        execution_time(start=start, end=time.time())
        self.save(savedir)

    def _get_learnable_layer_idx(self, layer_idx):

        if abs(layer_idx)>self.n_layers:
            raise ValueError(f"Max number of available layers is {self.n_layers}")

        if layer_idx<0:
            layer_idx = self.learnable_layers_idxs[layer_idx]
        else:
            layer_idx = self.learnable_layers_idxs[layer_idx]

        return layer_idx

    def _set_correct_layer_idx(self, layer_idx):

        """
        -1 = n_learnable_layers-1 = last learnable layer idx
        0 = -n_learnable_layers = firsy learnable layer idx  
        """
        # if layer_idx is not None:
        if abs(layer_idx)>self.n_layers:
            raise ValueError(f"Max number of available layers is {self.n_layers}")

        if layer_idx==-1:
            layer_idx=None
        else:
            if layer_idx<0:
                layer_idx+=self.n_layers+1
            else:
                layer_idx+=1

        return layer_idx

    def forward(self, inputs, layer_idx=-1, softmax=False, *args, **kwargs):

        layer_idx = self._set_correct_layer_idx(layer_idx)
        # print(self.model(inputs).shape)

        # preds = nn.Sequential(*list(self.model.children())[:layer_idx])(inputs)
        model = Sequential(*list(self.model.children())[:layer_idx])
        preds = model.forward(inputs, *args, **kwargs)

        if softmax:
            preds = nnf.softmax(preds, dim=-1)

        return preds

    def get_logits(self, *args, **kwargs):
        return self.forward(layer_idx=-1, *args, **kwargs)

    def save(self, savedir):

        filename=self.name+"_weights.pt"
        os.makedirs(savedir, exist_ok=True)

        self.to("cpu")
        torch.save(self.state_dict(), os.path.join(savedir, filename))

        if DEBUG:
            print("\nCheck saved weights:")
            print("\nstate_dict()['l2.0.weight'] =", self.state_dict()["l2.0.weight"][0,0,:3])
            print("\nstate_dict()['out.weight'] =",self.state_dict()["out.weight"][0,:3])

    def load(self, device, savedir):

        filename=self.name+"_weights.pt"

        self.load_state_dict(torch.load(os.path.join(savedir, filename)))
        self.to(device)

        if DEBUG:
            print("\nCheck loaded weights:")    
            print("\nstate_dict()['l2.0.weight'] =", self.state_dict()["l2.0.weight"][0,0,:3])
            print("\nstate_dict()['out.weight'] =",self.state_dict()["out.weight"][0,:3])

    def evaluate(self, test_loader, device, *args, **kwargs):
        self.to(device)

        with torch.no_grad():
            correct_predictions = 0.0

            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).argmax(-1)
                outputs = self(x_batch)
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == y_batch).sum()

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("\nAccuracy: %.2f%%" % (accuracy))
            return accuracy