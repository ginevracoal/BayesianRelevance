import os
import math
import time
import random
import numpy as np
import pickle as pkl
from utils.savedir import *

import torch
import keras
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist, fashion_mnist
from sklearn.datasets import make_moons
from pandas import DataFrame
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt


def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))


################
# data loaders #
################

def data_loaders(dataset_name, batch_size, n_inputs=None, channels="first", shuffle=False):
    random.seed(0)
    # batch_size = 256

    if dataset_name == "cifar":

        data_dir="../../cifar-10/"

        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        target_transform = torchvision.transforms.Compose([lambda x:torch.tensor([x]), 
                                                           lambda x:F.one_hot(x,10),
                                                           lambda x:x.squeeze()])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train,
                                                target_transform=target_transform)
        trainset = torch.utils.data.Subset(trainset, range(0, n_inputs)) if n_inputs else trainset
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test,
                                                    target_transform=target_transform)
        testset = torch.utils.data.Subset(testset, range(0, n_inputs)) if n_inputs else testset
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        input_shape = next(iter(train_loader))[0].shape[1:]
        # print(next(iter(train_loader))[0].shape)
        # print(next(iter(train_loader))[1].shape)
        num_classes = 10
    else:

        x_train, y_train, x_test, y_test, input_shape, num_classes = \
            load_dataset(dataset_name=dataset_name, n_inputs=n_inputs, channels=channels)

        train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=batch_size, 
                                  shuffle=shuffle)
        test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=batch_size, 
                                 shuffle=shuffle)

    return train_loader, test_loader, input_shape, num_classes

def classwise_data_loaders(dataset_name, batch_size, n_inputs=None, shuffle=False):
    random.seed(0)
    x_train, y_train, x_test, y_test, input_shape, num_classes = \
        load_dataset(dataset_name=dataset_name)

    train_loaders = []
    test_loaders = []

    for label in range(num_classes):
        label_idxs = y_train.argmax(1)==label
        x_train_label = x_train[label_idxs]
        y_train_label = y_train[label_idxs]

        label_idxs = y_test.argmax(1)==label
        x_test_label = x_test[label_idxs]
        y_test_label = y_test[label_idxs]

        if n_inputs:
            x_train_label = x_train_label[:n_inputs]
            y_train_label = y_train_label[:n_inputs]
            x_test_label = x_test_label[:n_inputs]
            y_test_label = y_test_label[:n_inputs]

        train_loader = DataLoader(dataset=list(zip(x_train_label, y_train_label)), 
                                  batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(dataset=list(zip(x_test_label, y_test_label)), 
                                 batch_size=batch_size, shuffle=shuffle)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders, input_shape, num_classes


def load_half_moons(channels="first", n_samples=30000):
    x, y = make_moons(n_samples=n_samples, shuffle=False, noise=0.1, random_state=0)
    x, y = (x.astype('float32'), y.astype('float32'))
    x = (x-np.min(x))/(np.max(x)-np.min(x))

    # train-test split
    split_size = int(0.8 * len(x))
    x_train, y_train = x[:split_size], y[:split_size]
    x_test, y_test = x[split_size:], y[split_size:]

    # image-like representation for compatibility with old code
    n_channels = 1
    n_coords = 2
    if channels == "first":
        x_train = x_train.reshape(x_train.shape[0], n_channels, n_coords, 1)
        x_test = x_test.reshape(x_test.shape[0], n_channels, n_coords, 1)

    elif channels == "last":
        x_train = x_train.reshape(x_train.shape[0], 1, n_coords, n_channels)
        x_test = x_test.reshape(x_test.shape[0], 1, n_coords, n_channels)
    input_shape = x_train.shape[1:]

    # binary one hot encoding
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def load_fashion_mnist(channels, img_rows=28, img_cols=28):
    print("\nLoading fashion mnist.")

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if channels == "first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    elif channels == "last":
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    input_shape = x_train.shape[1:]
    num_classes = 10
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def load_mnist(channels, img_rows=28, img_cols=28):

    print("\nLoading mnist.")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if channels == "first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    elif channels == "last":
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    input_shape = x_train.shape[1:]
    num_classes = 10
    return x_train, y_train, x_test, y_test, input_shape, num_classes

def labels_to_onehot(integer_labels, n_classes=None):
    n_rows = len(integer_labels)
    n_cols = n_classes if n_classes else integer_labels.max() + 1 
    onehot = np.zeros((n_rows, n_cols), dtype='uint8')
    onehot[np.arange(n_rows), integer_labels] = 1
    return onehot

def onehot_to_labels(y):
    if type(y) is np.ndarray:
        return np.argmax(y, axis=1)
    elif type(y) is torch.Tensor:
        return torch.max(y, 1)[1]

# def load_cifar(channels, img_rows=32, img_cols=32):
#     x_train = None
#     y_train = []

#     data_dir="../../cifar-10/"

#     for batch in range(1, 6):
#         data_dic = unpickle(data_dir + "data_batch_{}".format(batch))
#         if batch == 1:
#             x_train = data_dic['data']
#         else:
#             x_train = np.vstack((x_train, data_dic['data']))
#         y_train += data_dic['labels']

#     test_data_dic = unpickle(data_dir + "test_batch")
#     x_test = test_data_dic['data']
#     y_test = test_data_dic['labels']

#     x_train = x_train.reshape((len(x_train), 3, img_rows, img_cols))
#     x_train = np.rollaxis(x_train, 1, 4)
#     y_train = np.array(y_train)

#     x_test = x_test.reshape((len(x_test), 3, img_rows, img_cols))
#     x_test = np.rollaxis(x_test, 1, 4)
#     y_test = np.array(y_test)

#     input_shape = x_train.shape[1:]

#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     x_train /= 255
#     x_test /= 255

#     if channels == "first":
#         x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
#         x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)

#     elif channels == "last":
#         x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
#         x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

#     y_train = keras.utils.to_categorical(y_train, 10)
#     y_test = keras.utils.to_categorical(y_test, 10)

#     input_shape = x_train.shape[1:]
#     num_classes = 10
#     return x_train, y_train, x_test, y_test, input_shape, num_classes

def load_dataset(dataset_name, n_inputs=None, channels="first", shuffle=False):

    if dataset_name == "mnist":
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_mnist(channels)
    # elif dataset_name == "cifar":
    #     x_train, y_train, x_test, y_test, input_shape, num_classes = load_cifar(channels)
    elif dataset_name == "fashion_mnist":
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_fashion_mnist(channels)
    elif dataset_name == "half_moons":
        x_train, y_train, x_test, y_test, input_shape, num_classes = load_half_moons()
    else:
        raise AssertionError("\nDataset not available.")

    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

    if n_inputs:
        x_train, y_train, _ = balanced_subset(x_train, y_train, num_classes, n_inputs)
        x_test, y_test, _ = balanced_subset(x_test, y_test, num_classes, n_inputs)

    print('x_train shape =', x_train.shape, '\nx_test shape =', x_test.shape)
    print('y_train shape =', y_train.shape, '\ny_test shape =', y_test.shape)

    if shuffle is True:
        idxs = np.random.permutation(len(x_train))
        x_train, y_train = (x_train[idxs], y_train[idxs])
        idxs = np.random.permutation(len(x_test))
        x_test, y_test = (x_test[idxs], y_test[idxs])

    return x_train, y_train, x_test, y_test, input_shape, num_classes

def balanced_subset(inputs, labels, num_classes, subset_size):

    n_samples = min(subset_size, len(inputs))
    samples_per_class = int(n_samples/num_classes)

    sampled_idxs = []
    for target in range(num_classes+1):

        while len(sampled_idxs) < target*samples_per_class:
            idx = np.random.randint(0, len(inputs))

            if labels[idx].argmax(-1)==(target-1):
                sampled_idxs.append(idx)

    return inputs[sampled_idxs], labels[sampled_idxs], sampled_idxs

############
# pickling #
############


def save_to_pickle(data, path, filename):

    full_path=os.path.join(path, filename+".pkl")
    print("\nSaving pickle: ", full_path)
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(path, exist_ok=True)
    with open(full_path, 'wb') as f:
        pkl.dump(data, f)

def load_from_pickle(path, filename):

    full_path=os.path.join(path, filename+".pkl")
    print("\nLoading from pickle: ", full_path)
    with open(full_path, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    return data
    
def unpickle(file):
    """ Load byte data from file"""
    with open(file, 'rb') as f:
        data = pkl.load(f, encoding='latin-1')
    return data

def plot_loss_accuracy(dict, path):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8))
    ax1.plot(dict['loss'])
    ax1.set_title("loss")
    ax2.plot(dict['accuracy'])
    ax2.set_title("accuracy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)