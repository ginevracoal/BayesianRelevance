import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from torch.utils.data import DataLoader, random_split, Dataset
import torch
from torchvision import datasets, transforms
from savedir import *

PREPROCESS=True
TEST=False
torch.manual_seed(0)


def load_data(dataset_name):

    if dataset_name=="animals10":
        data_dir = "./data/animals10/"
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

        img_size = 224
        batch_size = 64
        num_classes = 10


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

        dataset = datasets.ImageFolder(data_dir, transform = transforms.Compose([
                                                                transforms.Resize((224,224)), 
                                                                transforms.ToTensor(),
                                                                ]))

        print("\nimg_size =",dataset[0][0].shape," img_label =", dataset[0][1])

        val_size = int(0.1 * len(dataset))
        test_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size - test_size
        train_subset, val_subset, test_set = random_split(dataset, [train_size, val_size, test_size])
        
        # Augment and normalize

        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_set = TransformDataset(train_subset, transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(250, padding=50, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stats, inplace=True),

            ]))

        val_set = TransformDataset(train_subset, transform = transforms.Normalize(*stats, inplace=True))

        test_set = TransformDataset(train_subset, transform = transforms.Normalize(*stats, inplace=True))

        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

        dataloadersdict = {'train': train_dataloader, 
                           'val':val_dataloader,
                           'test':test_dataloader}

    elif dataset_name=="hymenoptera":

        raise NotImplementedError

        # data_dir = "./data/hymenoptera_data"
        # num_classes = 2
        # batch_size = 64
        # img_size = 224

        # # Data augmentation and normalization for training
        # # Just normalization for validation
        # data_transforms = {
        #     'train': transforms.Compose([
        #         transforms.RandomResizedCrop(img_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        #     'val': transforms.Compose([
        #         transforms.Resize(img_size),
        #         transforms.CenterCrop(img_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        # }

        # print("Initializing Datasets and Dataloaders...")

        # # Create training and validation datasets
        # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        #                      for x in ['train', 'val']}
        # # Create training and validation dataloaders
        # dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, 
        #                     shuffle=True, num_workers=4) for x in ['train', 'val']}


    return dataloadersdict, batch_size, num_classes


def save_weights_nn(model, path, filename):

    path=TESTS+path
    os.makedirs(os.path.dirname(path), exist_ok=True)

    print("\nSaving: ", path + filename)
    print(f"\nlearned params = {model.state_dict().keys()}")
    torch.save(model.state_dict(), path + filename)

def save_weights_bnn(model, path, filename):

    path=TESTS+path
    os.makedirs(os.path.dirname(path), exist_ok=True)

    param_store = pyro.get_param_store()
    print("\nSaving: ", path + filename)
    print(f"\nlearned params = {param_store.get_all_param_names()}")
    param_store.save(path + filename)

def load_weights_nn(model, path, filename):

    path=TESTS+path
    print("\nLoading ", path + filename)

    self.load_state_dict(torch.load(path + filename))
    print(f"\nloaded params = {list(self.state_dict().keys())}")

def load_weights_bnn(model, path, filename):

    path=TESTS+path
    print("\nLoading ", path + filename)

    param_store = pyro.get_param_store()
    param_store.load(path + filename)
    for key, value in param_store.items():
        param_store.replace_param(key, value.to(device), value)

    print(f"\nloaded params = {param_store.get_all_param_names()}")
