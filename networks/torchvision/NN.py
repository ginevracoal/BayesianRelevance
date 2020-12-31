import time
import copy
import os
from tqdm import tqdm
from utils.savedir import *

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf
import torchvision.models as models
from torchvision import datasets, models, transforms
from pyro.nn import PyroModule
softplus = torch.nn.Softplus()

class torchvisionNN(PyroModule):

    def __init__(self, model_name, dataset_name):
        super(torchvisionNN, self).__init__()

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.name = "finetuned_"+str(model_name)+"_"+str(dataset_name)

    def train(self, dataloaders, criterion, optimizer, device, num_iters=25, is_inception=False):
        since = time.time()
        model = self.basenet
        self.to(device)

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_iters):
            print('Epoch {}/{}'.format(epoch, num_iters - 1))
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
        self.basenet.load_state_dict(best_model_wts)
        return self.basenet, val_acc_history

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific. 
        model_ft = None
        input_size = 0
        self.num_classes=num_classes

        if model_name == "resnet":

            model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":

            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "vgg":

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

    def save(self, num_iters):
   
        path=TESTS+self.name+"/"
        filename=self.name+"_iters="+str(num_iters)+"_weights.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        print("\nSaving: ", path + filename)
        self.to("cpu")
        # print(f"\nlearned params = {self.basenets.state_dict().keys()}")
        torch.save(self.basenet.state_dict(), path + filename)

    def load(self, num_iters, device):

        path=TESTS+self.name+"/"
        filename=self.name+"_iters="+str(num_iters)+"_weights.pt"
        print("\nLoading ", path + filename)

        self.basenet.load_state_dict(torch.load(path + filename))
        self.to(device)

    def forward(self, inputs, *args, **kwargs):
        return self.basenet.forward(inputs)

    def zero_grad(self, *args, **kwargs):
        return self.basenet.zero_grad(*args, **kwargs)