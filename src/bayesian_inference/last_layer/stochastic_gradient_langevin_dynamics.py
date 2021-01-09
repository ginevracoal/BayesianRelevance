import time
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required

from utils.data import load_from_pickle, save_to_pickle

BURNIN=10

class SGLD(Optimizer):
    """
    SGLD optimiser based on pytorch's SGD.
    Note that the weight decay is specified in terms of the gaussian prior sigma.
    From https://github.com/JavierAntoran/Bayesian-Neural-Networks/tree/master/src/Stochastic_Gradient_Langevin_Dynamics
    """

    def __init__(self, params, lr=required, norm_sigma=0, addnoise=True):

        weight_decay = 1 / (norm_sigma ** 2)

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay, addnoise=addnoise)

        super(SGLD, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:

            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if group['addnoise']:
                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) / np.sqrt(group['lr'])
                    p.data.add_(-group['lr'], 0.5 * d_p + langevin_noise)
                else:
                    p.data.add_(-group['lr'], 0.5 * d_p)

        return loss


def train(bayesian_network, dataloaders, device, num_iters, is_inception=False):

    bayesian_network.to(device)
    bayesian_network.posterior_samples=[]
    model = bayesian_network.last_layer

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = SGLD(params=model.parameters(), lr=0.001, norm_sigma=0.1, addnoise=True)

    best_acc = 0.0
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())

    since = time.time()
    num_iters = BURNIN+num_iters
    for epoch in range(num_iters):
        print('Epoch {}/{}'.format(epoch, num_iters - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                features = bayesian_network.rednet(inputs).squeeze()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        logits, aux_logits = model(features)
                        loss1 = criterion(logits, labels)
                        loss2 = criterion(aux_logits, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        logits = model(features)
                        loss = criterion(logits, labels)

                    _, preds = torch.max(logits, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                minibatch_size = inputs.size(0)
                running_loss += loss.item() * minibatch_size
                running_corrects += torch.sum(preds == labels.data)

            num_minibatches =  len(dataloaders[phase].dataset)
            epoch_loss = running_loss / num_minibatches
            epoch_acc = running_corrects.double() / num_minibatches

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            val_acc_history.append(epoch_acc)

            if epoch >= BURNIN and phase == 'train':
                iter_weights = copy.deepcopy(model.state_dict())
                bayesian_network.posterior_samples.append(iter_weights)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return bayesian_network, val_acc_history

def forward(bayesian_network, inputs, n_samples, sample_idxs=None):

    if n_samples>len(bayesian_network.posterior_samples):
        raise ValueError("Too many samples.")

    if sample_idxs:
        if len(sample_idxs) != n_samples:
            raise ValueError("Number of sample_idxs should match number of samples.")
    else:
        sample_idxs = list(range(n_samples))

    logits = []  

    for idx in sample_idxs:

        last_layer_copy = copy.deepcopy(bayesian_network.last_layer)
        last_layer_copy.load_state_dict(bayesian_network.posterior_samples[idx])
        # last_layer_copy = bayesian_network.posterior_samples[idx] ### v2 ###
        features = bayesian_network.rednet(inputs).squeeze()
        logits.append(last_layer_copy(features))

    logits = torch.stack(logits).unsqueeze(1) # second dim = n. images = 1
    return logits

def save(bayesian_network, num_iters, path, filename):

    print("\nSaving in: ", path)

    for idx, sampled_weights in enumerate(bayesian_network.posterior_samples):
        save_to_pickle(sampled_weights, path, filename+"_"+str(idx)+".pkl")    

def load(bayesian_network, num_iters, path, filename):

    print("\nLoading from: ", path)

    posterior_samples = []

    for idx in range(num_iters):

        sampled_weights = load_from_pickle(path+filename+"_"+str(idx)+".pkl")
        posterior_samples.append(sampled_weights)

    bayesian_network.posterior_samples=posterior_samples

# def load(bayesian_network, num_iters, path, filename): ### v2 ###

#     print("\nLoading from: ", path)

#     posterior_samples = []

#     for idx in range(num_iters):

#         last_layer_copy = copy.deepcopy(bayesian_network.last_layer)
#         sampled_weights = load_from_pickle(path+filename+"_"+str(idx)+".pkl")
#         last_layer_copy.load_state_dict(sampled_weights)        
#         posterior_samples.append(last_layer_copy)

#     bayesian_network.posterior_samples=posterior_samples

