

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
softplus = torch.nn.Softplus()

from utils_torchvision import * 
from adversarialAttacks import *

DEBUG=False

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg11_bn = models.vgg11_bn(pretrained=True)

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="resnet")
parser.add_argument("--dataset_name", type=str, default="animals10")
parser.add_argument("--train", type=eval, default="True")
parser.add_argument("--nn_epochs", type=int, default=20)
parser.add_argument("--bnn_epochs", type=int, default=20)
parser.add_argument("--attack_method", type=str, default="fgsm")
args = parser.parse_args()

dataloaders_dict, batch_size, num_classes = load_data(dataset_name=args.dataset_name)

class torchvisionNN(PyroModule):

    def train(self, dataloaders, criterion, optimizer, num_epochs=25, 
                    is_inception=False):
        since = time.time()
        model = self.basenet

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
        self.name = "finetuned_"+str(model_name)

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

        self.basenet = model_ft
        self.input_size = input_size

        return model_ft, input_size

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def to(self, device):
        self.basenet = self.basenet.to(device)

    def save(self):
   
        path=TESTS+self.name+"/"
        filename=self.name+"_weights.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        print("\nSaving: ", path + filename)
        # print(f"\nlearned params = {self.basenets.state_dict().keys()}")
        torch.save(self.basenet.state_dict(), path + filename)

    def load(self):

        path=TESTS+self.name+"/"
        filename=self.name+"_weights.pt"
        print("\nLoading ", path + filename)

        self.basenet.load_state_dict(torch.load(path + filename))

    def forward(self, inputs, *args, **kwargs):
        return self.basenet.forward(inputs)

    def zero_grad(self, *args, **kwargs):
        return self.basenet.zero_grad(*args, **kwargs)

    def attack(self, dataloader, method, device, hyperparams=None):

        net = self
        adversarial_attack = []
        original_images_list = []
        print(f"\n\nProducing {method} attacks", end="\t")
        
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            for idx, image in enumerate(inputs):
                image = image.unsqueeze(0)
                label = labels[idx].argmax(-1).unsqueeze(0)

                if method == "fgsm":
                    perturbed_image = fgsm_attack(net=net, image=image, label=label, 
                                                  hyperparams=hyperparams)
                elif method == "pgd":
                    perturbed_image = pgd_attack(net=net, image=image, label=label, 
                                                  hyperparams=hyperparams)

                original_images_list.append(image.squeeze(0))
                adversarial_attack.append(perturbed_image)

        path = TESTS+self.name+"/"
        filename = self.name+"_"+str(method)+"_attack.pkl"

        adversarial_attack = torch.cat(adversarial_attack)

        save_to_pickle(data=adversarial_attack, path=path, filename=filename)

        idxs = np.random.choice(len(original_images_list), 10, replace=False)
        original_images_plot = torch.stack([original_images_list[i].permute(1, 2, 0) for i in idxs])
        perturbed_images_plot = torch.stack([adversarial_attack[i].permute(1, 2, 0) for i in idxs])
        plot_grid_attacks(original_images=original_images_plot.detach().cpu(), 
                          perturbed_images=perturbed_images_plot.detach().cpu(), 
                          filename=self.name+"_"+str(method)+".png", savedir=path)

        return adversarial_attack

    def evaluate_attack(self, dataloader, attack, device, n_samples=None):

        if device=="cuda":
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        print(f"\nEvaluating against the attacks", end="")
        if n_samples:
            print(f" with {n_samples} defence samples")

        random.seed(0)
        torch.manual_seed(0)
        pyro.set_rng_seed(0)
        
        with torch.no_grad():

            original_outputs = []
            original_correct = 0.0
            adversarial_outputs = []
            adversarial_correct = 0.0

            for idx, (images, labels) in enumerate(dataloader):

                images, labels = images.to(device), labels.to(device)
                attacks = attack[idx:idx+len(images)]

                out = self.forward(images, n_samples)
                original_correct += ((out.argmax(-1) == labels.argmax(-1)).sum().item())
                original_outputs.append(out)

                out = self.forward(attacks, n_samples)
                adversarial_correct += ((out.argmax(-1) == labels.argmax(-1)).sum().item())
                adversarial_outputs.append(out)

            original_accuracy = 100 * original_correct / len(dataloader.dataset)
            adversarial_accuracy = 100 * adversarial_correct / len(dataloader.dataset)
            print(f"\ntest accuracy = {original_accuracy}\tadversarial accuracy = {adversarial_accuracy}",
                  end="\t")

            original_outputs = torch.cat(original_outputs)
            adversarial_outputs = torch.cat(adversarial_outputs)
            softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

            # print(original_outputs.shape, adversarial_outputs.shape)

        return original_accuracy, adversarial_accuracy, softmax_rob


class torchvisionBNN(torchvisionNN):

    def __init__(self):
        super(torchvisionBNN, self).__init__()
        self.inference = "svi"

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):

        model_ft, input_size = super(torchvisionBNN, self).initialize_model(model_name, num_classes,
                                                     feature_extract, use_pretrained)

        self.model_name = model_name
        self.basenet = model_ft
        self.name = "finetuned_"+str(model_name)+"_svi"
        return model_ft, input_size

    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs=25, 
                    is_inception=False):

        return self.torchvisionNN.train_model(model, dataloaders, criterion, optimizer, 
                                                num_epochs, is_inception)

    def train_svi(self, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):

        print("\nLearning variational params:\n")
        print(pyro.get_param_store().get_all_param_names())

        random.seed(0)
        pyro.set_rng_seed(0)

        model = self.basenet

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
                    model.eval()  # Set model to evaluate mode

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
                        
                        loss += svi.step(x_data=inputs, y_data=labels)
                        outputs = self.forward(inputs)

                        _, preds = torch.max(outputs, 1)

                        if DEBUG:
                            print("\n", pyro.get_param_store()["outw_mu"])    

                    running_loss += loss * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if DEBUG:
                    print(list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))

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

    def forward(self, inputs, n_samples=10, seeds=None, out_prob=False):
            
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

        output_probs = torch.stack(preds)
        # print(output_probs.mean(0).sum(1))
        return output_probs if out_prob else output_probs.mean(0)

    def save(self):
        path=TESTS+self.name+"/"
        filename=self.name+"_weights.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        param_store = pyro.get_param_store()
        print("\nSaving: ", path + filename)
        print(f"\nlearned params = {param_store.get_all_param_names()}")
        param_store.save(path + filename)

    def load(self):
  
        path=TESTS+self.name+"/"
        filename=self.name+"_weights.pt"
        print("\nLoading ", path + filename)

        param_store = pyro.get_param_store()
        param_store.load(path + filename)
        for key, value in param_store.items():
            param_store.replace_param(key, value, value)

    def attack(self, dataloader, method, device, hyperparams=None, n_samples=10):

        net = self
        print(f"\n\nProducing {method} attacks with {n_samples} attack samples")
        adversarial_attack = []
        original_images_list = []
        
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            for idx, image in enumerate(inputs):
                image = image.unsqueeze(0)
                label = labels[idx].argmax(-1).unsqueeze(0)

                if method == "fgsm":
                    perturbed_image = fgsm_attack(net=net, image=image, label=label, 
                                                  hyperparams=hyperparams, n_samples=n_samples)
                elif method == "pgd":
                    perturbed_image = pgd_attack(net=net, image=image, label=label, 
                                                  hyperparams=hyperparams, n_samples=n_samples)

                original_images_list.append(image.squeeze(0))
                adversarial_attack.append(perturbed_image)

        adversarial_attack = torch.cat(adversarial_attack)

        path = TESTS+self.name+"/" 
        filename = self.name+"_"+str(method)+"_attackSamp="+str(n_samples)+"_attack.pkl"
        save_to_pickle(data=adversarial_attack, path=path, filename=filename)

        idxs = np.random.choice(len(original_images_list), 10, replace=False)
        original_images_plot = torch.stack([original_images_list[i].permute(1, 2, 0) for i in idxs])
        perturbed_images_plot = torch.stack([adversarial_attack[i].permute(1, 2, 0) for i in idxs])
        plot_grid_attacks(original_images=original_images_plot.detach().cpu(), 
                          perturbed_images=perturbed_images_plot.detach().cpu(), 
                          filename=self.name+"_"+str(method)+".png", savedir=path)

        return adversarial_attack

    def evaluate_attack(self, dataloader, attack, device, n_samples=10):

        self.basenet.to(device) # fixed layers in BNN

        super(torchvisionBNN, self).evaluate_attack(dataloader=dataloader, attack=attack, 
                                                     device=device, n_samples=n_samples)


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

model_nn = torchvisionNN()
model_bnn = torchvisionBNN()

model_nn.initialize_model(model_name=args.model, num_classes=num_classes, 
                                            feature_extract=True, use_pretrained=True)
# Initialize the model for this run
model_bnn.initialize_model(model_name=args.model, num_classes=num_classes, 
                                            feature_extract=True, use_pretrained=True)

# Print the model we just instantiated
# print(model_nn.basenet, model_bnn.basenet)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_nn.to(device)
model_bnn.to(device)

nn_name = model_nn.name
bnn_name = model_bnn.name

params_nn = set_params_updates(model_nn.basenet, feature_extract=True)
optimizer_nn = optim.Adam(params_nn, lr=0.001)
optimizer_bnn = pyro.optim.Adam({"lr":0.001})
criterion = nn.CrossEntropyLoss()

# Train and evaluate
if args.train is True:

    # model_nn.train(dataloaders_dict, criterion, optimizer_nn, num_epochs=args.nn_epochs)
    model_bnn.train_svi(dataloaders_dict, criterion, optimizer_bnn, num_epochs=args.bnn_epochs)

    # model_nn.save()
    model_bnn.save()

else:
    model_nn.load()
    model_bnn.load()

nn_attack = model_nn.attack(dataloader=dataloaders_dict["test"], 
    method=args.attack_method, device=device)
bnn_attack = model_bnn.attack(dataloader=dataloaders_dict["test"], 
    method=args.attack_method, n_samples=10, device=device)

model_nn.evaluate_attack(dataloader=dataloaders_dict["test"], 
    attack=nn_attack, device=device)
model_bnn.evaluate_attack(dataloader=dataloaders_dict["test"], 
    attack=bnn_attack, n_samples=10, device=device)