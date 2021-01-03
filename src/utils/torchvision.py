import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

import pyro
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

from utils.savedir import *
from attacks.plot import plot_grid_attacks

torch.manual_seed(0)


def set_params_updates(model, feature_extract):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    print("\nParams to learn:")

    count = 0
    params_to_update = []

    for name,param in model.named_parameters():
        if param.requires_grad == True:
            if feature_extract:
                params_to_update.append(param)
            print("\t", name)
            count += param.numel()

    print("Total n. of params =", count)

    return params_to_update
    

class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def load_data(dataset_name, batch_size=None, img_size=224, debug=False):
    """
    Builds a dictionary of torch training, validation and test dataloaders from the chosen dataset.
    In debugging mode all dataloaders are cut to 100 randomly chosen points.
    """

    if dataset_name=="animals10":
        data_dir = DATA+"animals10/"
        dirs = os.listdir(data_dir)

        translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant",
                     "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat",
                     "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel",
                     "dog": "cane", "cavallo": "horse", "elephant" : "elefante",
                     "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto",
                     "cow": "mucca", "ragno": "spider", "squirrel": "scoiattolo"}

        names = {}
        for key, val in translate.items():
            if key in dirs:
                names[key] = val

        encodings = {
            "dog":0,
            "elephant":1,
            "butterfly":2,
            "chicken":3,
            "cat":4,
            "cow":5,
            "spider":6,
            "squirrel":7,
            "sheep":8,
            "horse":9
        }

        num_classes = 10

        dataset = datasets.ImageFolder(data_dir, transform = transforms.Compose([
                                                                transforms.Resize((224,224)), 
                                                                transforms.ToTensor(),
                                                                ]))


        val_size = int(0.1 * len(dataset))
        test_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size - test_size
        train_subset, val_subset, test_set = random_split(dataset, [train_size, val_size, test_size])
        
        # Augment and normalize

        # stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        stats = [0.,0.,0.],[1.,1.,1.]

        train_set = TransformDataset(train_subset, transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(img_size, padding=None),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stats, inplace=True),

            ]))

        val_set = TransformDataset(val_subset, transform = transforms.Normalize(*stats, inplace=True))

        test_set = TransformDataset(test_set, transform = transforms.Normalize(*stats, inplace=True))

        if batch_size is None:
            batch_size = len(train_set)

        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

        dataloaders_dict = {'train': train_dataloader, 
                           'val':val_dataloader,
                           'test':test_dataloader}

    elif dataset_name=="hymenoptera":

        data_dir = DATA+"hymenoptera_data"
        num_classes = 2

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                             for x in ['train', 'test']}

        if batch_size is None:
            batch_size = len(train_set)     

        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, 
                            shuffle=True, num_workers=4) for x in ['train', 'test']}

    elif dataset_name=="imagenette":

        data_dir = DATA+"imagenette2-320"
        num_classes = 10

        transform = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
        train_set = datasets.ImageFolder(data_dir+"/train", transform=transform)
        test_set = datasets.ImageFolder(data_dir+"/test", transform=transform)

        val_size = int(0.1 * len(train_set))
        train_size = len(train_set) - val_size
        train_subset, val_subset = random_split(train_set, [train_size, val_size])
        
        stats = [0.,0.,0.],[1.,1.,1.]

        train_set = TransformDataset(train_subset, transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(img_size, padding=None),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stats, inplace=True),

            ]))

        val_set = TransformDataset(val_subset, transform = transforms.Normalize(*stats, inplace=True))
        test_set = TransformDataset(test_set, transform = transforms.Normalize(*stats, inplace=True))

        if batch_size is None:
            batch_size = len(train_set)
        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4)
        val_dataloader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4)

        dataloaders_dict = {'train': train_dataloader, 'val':val_dataloader, 'test':test_dataloader}


    elif dataset_name=="imagewoof":

        data_dir = DATA+"imagewoof2-320"
        num_classes = 10

        transform = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
        train_set = datasets.ImageFolder(data_dir+"/train", transform=transform)
        test_set = datasets.ImageFolder(data_dir+"/test", transform=transform)

        stats = [0.,0.,0.],[1.,1.,1.]

        train_set = TransformDataset(train_set, transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomCrop(img_size, padding=None),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stats, inplace=True),
            ]))

        test_set = TransformDataset(test_set, transform = transforms.Normalize(*stats, inplace=True))

        if batch_size is None:
            batch_size = len(train_set)
        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4)

        dataloaders_dict = {'train': train_dataloader, 'test':test_dataloader}

    else:
        raise NotImplementedError

    if debug:

        for phase in ['train','val','test']:
            dataset = dataloaders_dict[phase].dataset
            dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), 100, replace=False))
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
            dataloaders_dict[phase]=dataloader

    print("\ntrain dataset length =", len(dataloaders_dict['train'].dataset), end="\t")
    print("val dataset length =", len(dataloaders_dict['val'].dataset), end="\t")
    print("test dataset length =", len(dataloaders_dict['test'].dataset), end="\t")
    print("img_size =", dataloaders_dict['train'].dataset[0][0].shape, end="\n")

    return dataloaders_dict, num_classes

