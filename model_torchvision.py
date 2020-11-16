from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from argparse import ArgumentParser
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf
import torchvision
import torchvision.models as models
from torchvision import datasets, models, transforms

import pyro
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
import pyro.optim as pyroopt
from pyro.infer.mcmc import MCMC, HMC, NUTS
from pyro.distributions import OneHotCategorical, Normal, Categorical, Uniform
from pyro.nn import PyroModule
from utils_torchvision import load_data 
softplus = torch.nn.Softplus()

DEBUG=False

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg11_bn = models.vgg11_bn(pretrained=True)

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="resnet")
parser.add_argument("--dataset_name", type=str, default="animals10")
parser.add_argument("--train", type=eval, default="True")
parser.add_argument("--nn_epochs", type=int, default=10)
parser.add_argument("--bnn_epochs", type=int, default=60)
args = parser.parse_args()

dataloaders_dict, batch_size, num_classes = load_data(dataset_name=args.dataset_name)

class torchvisionNN():

    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs=25, 
                    is_inception=False):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0
        self.name = "finetuned_"+str(args.model)

        if model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


class torchvisionBNN(PyroModule):

    def __init__(self, torchvisionNN):
        super(torchvisionBNN, self).__init__()
        self.inference = "svi"
        self.torchvisionNN = torchvisionNN

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):

        model_ft, input_size = self.torchvisionNN.initialize_model(model_name, num_classes, 
                                                                feature_extract, use_pretrained)

        self.model_name = model_name
        self.basenet = model_ft
        self.name = "finetuned_"+str(args.model)+"_svi"
        return model_ft, input_size

    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs=25, 
                    is_inception=False):

        return self.torchvisionNN.train_model(model, dataloaders, criterion, optimizer, 
                                                num_epochs, is_inception)

    def train_svi(self, model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        random.seed(0)
        pyro.set_rng_seed(0)

        since = time.time()
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optimizer, loss=elbo)

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):

            loss=0.0

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        
                        loss += svi.step(x_data=inputs, y_data=labels) #NEW
                        outputs = self.forward(inputs)

                        _, preds = torch.max(outputs, 1)

                        if DEBUG:
                            print("\n", pyro.get_param_store()["outw_mu"])    

                    running_loss += loss * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        print("\nlearned variational params:\n")
        print(pyro.get_param_store().get_all_param_names())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history
   

    def _last_layer(self, net):

        if self.model_name == "resnet":
            w, b = net.fc.weight, net.fc.bias
            w_name, b_name = 'fc.weight', 'fc.bias'

        elif self.model_name == "alexnet":
            w, b = net.classifier[6].weight, net.classifier[6].bias
            w_name, b_name = 'classifier[6].weight', 'classifier[6].bias'

        elif self.model_name == "vgg":
            w, b = net.classifier[6].weight, net.classifier[6].bias
            w_name, b_name = 'classifier[6].weight', 'classifier[6].bias'

        return w, b, w_name, b_name

    def model(self, x_data, y_data):

        net = self.basenet
        w, b, w_name, b_name = self._last_layer(net)

        for weights_name in ["outw_mu","outw_sigma","outb_mu","outb_sigma"]:
            pyro.get_param_store()[weights_name].requires_grad=True

        outw_prior = Normal(loc=torch.zeros_like(w), scale=torch.ones_like(w))
        outb_prior = Normal(loc=torch.zeros_like(b), scale=torch.ones_like(b))
        
        priors = {w_name: outw_prior, b_name: outb_prior}
        lifted_module = pyro.random_module("module", net, priors)()

        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)
            lhat = nnf.log_softmax(logits, dim=-1)
            cond_model = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

    def guide(self, x_data, y_data=None):

        net = self.basenet 
        w, b, w_name, b_name = self._last_layer(net)

        outw_mu = torch.randn_like(w)
        outw_sigma = torch.randn_like(w)
        outw_mu_param = pyro.param("outw_mu", outw_mu)
        outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param)

        outb_mu = torch.randn_like(b)
        outb_sigma = torch.randn_like(b)
        outb_mu_param = pyro.param("outb_mu", outb_mu)
        outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)

        priors = {w_name: outw_prior, b_name: outb_prior}
        lifted_module = pyro.random_module("module", net, priors)()

        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)
            probs = nnf.softmax(logits, dim=-1)

        return probs

    def forward(self, inputs, n_samples=10, seeds=None, training=False):
            
        if seeds:
            if len(seeds) != n_samples:
                raise ValueError("Number of seeds should match number of samples.")
        else:
            seeds = list(range(n_samples))

        preds = []  

        for seed in seeds:
            pyro.set_rng_seed(seed)
            guide_trace = poutine.trace(self.guide).get_trace(inputs)   
            preds.append(guide_trace.nodes['_RETURN']['value'])

        if DEBUG:
            print("\nlearned variational params:\n")
            print(pyro.get_param_store().get_all_param_names())
            print(list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))
            print("\n", pyro.get_param_store()["model.0.weight_loc"][0][:5])
            print(guide_trace.nodes['module$$$model.0.weight']["fn"].loc[0][:5])
            print("posterior sample: ", 
              guide_trace.nodes['module$$$model.0.weight']['value'][5][0][0])
        
        output_probs = torch.stack(preds).mean(0)
        return output_probs


def set_params_updates(model, feature_extract):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    return params_to_update

NN = torchvisionNN()
BNN = torchvisionBNN(NN)

model_nn, input_size = NN.initialize_model(model_name=args.model, num_classes=num_classes, 
                                        feature_extract=True, use_pretrained=True)
# Initialize the model for this run
model_bnn, input_size = BNN.initialize_model(model_name=args.model, num_classes=num_classes, 
                                        feature_extract=True, use_pretrained=True)

# Print the model we just instantiated
# print(model_ft)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_nn = model_nn.to(device)
model_bnn = model_bnn.to(device)

params_to_update = set_params_updates(model_nn, feature_extract=True)
set_params_updates(model_bnn, feature_extract=True)

optimizer_nn = optim.Adam(params_to_update, lr=0.001)
optimizer_bnn = pyro.optim.Adam({"lr":0.001})
criterion = nn.CrossEntropyLoss()

# Train and evaluate

if args.train is True:

    model_nn, hist = NN.train_model(model_nn, dataloaders_dict, criterion, optimizer_nn, 
                                 num_epochs=args.nn_epochs)
    model_bnn, hist = BNN.train_svi(model_bnn, dataloaders_dict, criterion, optimizer_bnn, 
                                 num_epochs=args.bnn_epochs)

    save_weights_nn(model=model_nn, path=model_nn.name+"/", filename=model_nn.name+"_weights.pt")
    save_weights_bnn(model=model_bnn, path=model_bnn.name+"/", filename=model_bnn.name+"_svi_weights.pt")

else:
    load_weights_nn(model=model_nn, path=model_nn.name+"/", filename=model_nn.name+"_weights.pt")
    load_weights_bnn(model=model_bnn, path=model_bnn.name+"/", filename=model_bnn.name+"_svi_weights.pt")


x_attack = attack(net=model_nn, x_test=x_test, y_test=y_test,
                  device=args.device, method=args.attack_method, filename=net.name)

attack_evaluation(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, 
                    device=args.device)