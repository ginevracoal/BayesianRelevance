import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import collections 

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

import pyro
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

from utils.savedir import *
from utils.seeding import *
from attacks.plot import plot_grid_attacks


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

def subset_dataloader(dataloaders_dict, batch_size, num_classes, n_inputs):

    im_idxs_dict={}

    for phase in dataloaders_dict.keys():
        dataset = dataloaders_dict[phase].dataset
        n_samples = min(n_inputs, len(dataset))

        im_idxs_dict[phase]=np.random.choice(len(dataset), n_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, im_idxs_dict[phase])
        dataloaders_dict[phase] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        
    return dataloaders_dict, im_idxs_dict

def balanced_subset_dataloader(dataloaders_dict, batch_size, num_classes, n_inputs):

    im_idxs_dict={}
    
    for phase in dataloaders_dict.keys():
        dataset = dataloaders_dict[phase].dataset
        n_samples = min(n_inputs, len(dataset))
        samples_per_class = int(n_samples/num_classes)

        sampled_idxs = []
        for target in range(num_classes+1):

            while len(sampled_idxs) < target*samples_per_class:
                idx = np.random.randint(0, len(dataset))
                if dataset[idx][1]==(target-1):
                    sampled_idxs.append(idx)

        im_idxs_dict[phase]=sampled_idxs[:n_samples]
        dataset = torch.utils.data.Subset(dataset, im_idxs_dict[phase])        
        dataloaders_dict[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloaders_dict, im_idxs_dict

def transform_data(train_set, val_set, test_set, img_size):
    
    stats = [0.,0.,0.],[1.,1.,1.]

    train_set = TransformDataset(train_set, transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(img_size, padding=None),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats, inplace=True),

        ]))

    val_set = TransformDataset(val_set, transform = transforms.Normalize(*stats, inplace=True))
    test_set = TransformDataset(test_set, transform = transforms.Normalize(*stats, inplace=True))

    return train_set, val_set, test_set

def load_data(dataset_name, phases=['train','val','test'], batch_size=128, n_inputs=None, 
              img_size=224, num_workers=0):
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
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        
        train_set, val_set, test_set = transform_data(train_set, val_set, test_set, img_size)
  
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

        # todo: add validation set

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                             for x in ['train', 'test']}
        train_set = image_datasets['train']
        test_set = image_datasets['test']

        val_size = int(0.1 * len(train_set))
        train_size = len(train_set) - val_size
        train_set, val_set = random_split(train_set, [train_size, val_size])

    elif dataset_name=="imagenette":

        data_dir = DATA+"imagenette2-320"
        num_classes = 10

        transform = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
        train_set = datasets.ImageFolder(data_dir+"/train", transform=transform)
        test_set = datasets.ImageFolder(data_dir+"/test", transform=transform)

        val_size = int(0.1 * len(train_set))
        train_size = len(train_set) - val_size
        train_set, val_set = random_split(train_set, [train_size, val_size])
        
        train_set, val_set, test_set = transform_data(train_set, val_set, test_set, img_size)

    elif dataset_name=="imagewoof":

        data_dir = DATA+"imagewoof2-320"
        num_classes = 10

        transform = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
        train_set = datasets.ImageFolder(data_dir+"/train", transform=transform)
        test_set = datasets.ImageFolder(data_dir+"/test", transform=transform)

        val_size = int(0.1 * len(train_set))
        train_size = len(train_set) - val_size
        train_set, val_set = random_split(train_set, [train_size, val_size])

        train_set, val_set, test_set = transform_data(train_set, val_set, test_set, img_size)

    else:
        raise NotImplementedError

    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    dataloaders_dict = {'train': train_dataloader, 'val':val_dataloader, 'test':test_dataloader}
    print("\nimg shape =", dataloaders_dict['train'].dataset[0][0].shape, end="\t")
    
    dataloaders_dict = {phase:dataloaders_dict[phase] for phase in phases}
    im_idxs_dict = {phase:None for phase in phases}

    if n_inputs is not None:
        # dataloaders_dict, im_idxs_dict = subset_dataloader(dataloaders_dict, batch_size, num_classes, n_inputs)
        dataloaders_dict, im_idxs_dict = balanced_subset_dataloader(dataloaders_dict, batch_size, num_classes, n_inputs)

    for phase in phases:
        print(phase, "dataset length =", len(dataloaders_dict[phase].dataset), end="\t")
    print()

    return dataloaders_dict, num_classes, im_idxs_dict

