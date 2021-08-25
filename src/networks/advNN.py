"""
Deterministic Neural Network model with adversarial training.
"""

import os
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
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

from attacks.run_attacks import attack
from networks.baseNN import baseNN

DEBUG = False


class advNN(baseNN):

    def __init__(self, input_shape, output_size, dataset_name, hidden_size, activation, 
                       architecture, epochs, lr, attack_method):
        super(advNN, self).__init__(input_shape, output_size, dataset_name, hidden_size, activation, 
                       architecture, epochs, lr)

        if math.log(hidden_size, 2).is_integer() is False or hidden_size<16:
            raise ValueError("\nhidden size should be a power of 2 greater than 16.")

        self.dataset_name = dataset_name
        self.architecture = architecture
        self.hidden_size = hidden_size 
        self.activation = activation
        self.attack_method = attack_method
        self.lr, self.epochs = lr, epochs
        self.loss_func = nn.CrossEntropyLoss()
        self.set_model(architecture, activation, input_shape, output_size, hidden_size)
        self.name = str(dataset_name)+"_advNN_hid="+str(hidden_size)+\
                    "_arch="+str(self.architecture)+"_act="+str(self.activation)+\
                    "_ep="+str(self.epochs)+"_lr="+str(self.lr)+"_atk="+str(attack_method)

        print("\nadvNN total number of weights =", sum(p.numel() for p in self.parameters()))
        self.n_layers = len(list(self.model.children()))
        learnable_params = self.model.state_dict()
        self.n_learnable_layers = int(len(learnable_params)/2)

    def train(self, train_loader, savedir, device, hyperparams={}):
        print("\n == advNN training ==")
        self.to(device)

        optimizer = torchopt.Adam(params=self.parameters(), lr=self.lr)

        start = time.time()
        for epoch in tqdm(range(self.epochs)):
            total_loss = 0.0
            correct_predictions = 0.0

            for x_batch, y_batch in train_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                self.model.eval()
                x_attack = attack(net=self, x_test=x_batch, y_test=y_batch, device=device, hyperparams=hyperparams,
                                    method=self.attack_method, verbose=False)
                outputs = self.forward(x_attack)
                
                self.model.train()
                optimizer.zero_grad()
                y_batch = y_batch.argmax(-1)
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
        self.model.eval()
        print(self.state_dict().keys())
        self.save(savedir)
