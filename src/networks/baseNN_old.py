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

DEBUG = False

baseNN_settings = {"model_0":{"dataset":"mnist", "hidden_size":512, "activation":"leaky",
                            "architecture":"conv", "epochs":5, "lr":0.001},
                   "model_1":{"dataset":"fashion_mnist", "hidden_size":1024, "activation":"leaky",
                            "architecture":"fc2", "epochs":15, "lr":0.001},
                   # "model_2":{"dataset":"cifar", "hidden_size":512, "activation":"leaky",
                   #          "architecture":"conv", "epochs":20, "lr":0.01}
                            }


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
        print("\nTotal number of weights =", sum(p.numel() for p in self.parameters()))

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
                nn.Linear(input_size, hidden_size),
                activ())
            self.out = nn.Linear(hidden_size, output_size)

        elif architecture == "fc2":
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, hidden_size),
                activ(),
                nn.Linear(hidden_size, hidden_size),
                activ())
            self.out = nn.Linear(hidden_size, output_size)

        elif architecture == "fc4":
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, hidden_size),
                activ(),
                nn.Linear(hidden_size, hidden_size),
                activ(),
                nn.Linear(hidden_size, hidden_size),
                activ(),
                nn.Linear(hidden_size, hidden_size),
                activ())
            self.out = nn.Linear(hidden_size, output_size)

        elif architecture == "conv":

            if self.dataset_name in ["mnist","fashion_mnist"]:

                self.model = nn.Sequential(
                    nn.Conv2d(in_channels, 16, kernel_size=5),
                    activ(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(16, hidden_size, kernel_size=5),
                    activ(),
                    nn.MaxPool2d(kernel_size=2, stride=1),
                    nn.Flatten())
                self.out = nn.Linear(int(hidden_size/(4*4))*input_size, output_size)

            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()

    def train(self, train_loader, savedir, device):
        print("\n == baseNN training ==")
        random.seed(0)
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

    def forward(self, inputs, layer_idx=-1, *args, **kwargs):

        if layer_idx==-1:
            x = self.model(inputs)
            x = self.out(x)
            return x

        else:
            x = nn.Sequential(*list(self.model.children())[:layer_idx+1])(inputs)
            return x

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
        random.seed(0)

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