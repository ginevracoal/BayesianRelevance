import os
import lrp
import copy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as nnf

from utils.savedir import *
from utils.seeding import set_seed
from utils.data import load_from_pickle, save_to_pickle

cmap_name = "RdBu_r"
DEBUG=False

def compute_explanations(x_test, network, rule, n_samples=None, layer_idx=-1):

    print("\nLRP layer idx =", layer_idx)

    if n_samples is None:

        explanations = []

        for x in tqdm(x_test):
            x.requires_grad=True
            # Forward pass
            y_hat = network.forward(x.unsqueeze(0), explain=True, rule=rule, layer_idx=layer_idx)
            y_hat = nnf.softmax(y_hat, dim=-1)

            # Choose argmax
            y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]]
            y_hat = y_hat.sum()

            # Backward pass (compute explanation)
            y_hat.backward()
            explanations.append(x.grad.detach().cpu().numpy())

        return np.array(explanations)

    else:

        avg_explanations = []
        for x in tqdm(x_test):

            explanations = []
            for j in range(n_samples):

                # Forward pass
                x_copy = copy.deepcopy(x.detach()).unsqueeze(0)
                x_copy.requires_grad = True
                y_hat = network.forward(inputs=x_copy, n_samples=1, sample_idxs=[j], 
                                        explain=True, rule=rule, layer_idx=layer_idx)
                y_hat = nnf.softmax(y_hat, dim=-1)

                # Choose argmax
                y_hat = y_hat[torch.arange(x_copy.shape[0]), y_hat.max(1)[1]]
                y_hat = y_hat.sum()

                # Backward pass (compute explanation)
                y_hat.backward()
                explanations.append(x_copy.grad.detach().cpu().numpy())

            avg_explanations.append(np.array(explanations).mean(0).squeeze(0))

        return np.array(avg_explanations)

def compute_posterior_explanations(x_test, network, rule, n_samples, layer_idx):

    posterior_explanations = []
    for x in tqdm(x_test):

        explanations = []
        for j in range(n_samples):

            # Forward pass
            x_copy = copy.deepcopy(x.detach()).unsqueeze(0)
            x_copy.requires_grad = True
            y_hat = network.forward(inputs=x_copy, n_samples=1, sample_idxs=[j], 
                                    explain=True, rule=rule, layer_idx=layer_idx)
            y_hat = nnf.softmax(y_hat, dim=-1)

            # Choose argmax
            y_hat = y_hat[torch.arange(x_copy.shape[0]), y_hat.max(1)[1]]
            y_hat = y_hat.sum()

            # Backward pass (compute explanation)
            y_hat.backward()
            explanations.append(x_copy.grad.squeeze(1).detach().cpu().numpy())

        posterior_explanations.append(np.array(explanations))

    return np.array(posterior_explanations)        

def compute_vanishing_norm_idxs(inputs, n_samples_list, norm="linfty"):

    if inputs.shape[0] != len(n_samples_list):
        raise ValueError("First dimension should equal the length of `n_samples_list`")

    inputs=np.transpose(inputs, (1, 0, 2, 3, 4))
    vanishing_norm_idxs = []
    non_null_idxs = []

    print("\nvanishing norms:\n")
    count_van_images = 0
    count_incr_images = 0
    count_null_images = 0

    for idx, image in enumerate(inputs):

        if norm == "linfty":
            inputs_norm = np.max(np.abs(image[0]))
        elif norm == "l2":
            inputs_norm = np.linalg.norm(image[0])  
        else:
            raise ValueError("Wrong norm name")
        
        if inputs_norm != 0.0:
            if DEBUG:
                print("idx =",idx, end="\t")
            count_samples_idx = 0
            for samples_idx, n_samples in enumerate(n_samples_list):

                if norm == "linfty":
                    new_inputs_norm = np.max(np.abs(image[samples_idx]))
                elif norm == "l2":
                    new_inputs_norm = np.linalg.norm(image[samples_idx])

                if new_inputs_norm <= inputs_norm:
                    if DEBUG:
                        print(new_inputs_norm, end="\t")
                    inputs_norm = copy.deepcopy(new_inputs_norm)
                    count_samples_idx += 1

            if count_samples_idx == len(n_samples_list):
                vanishing_norm_idxs.append(idx)
                non_null_idxs.append(idx)
                if DEBUG:
                    print("\tcount=", count_van_images)
                count_van_images += 1
            else: 
                non_null_idxs.append(idx)
                count_incr_images += 1

            if DEBUG:
                print("\n")

        else:
            count_null_images += 1

    print(f"vanishing norms = {100*count_van_images/len(inputs)} %")
    print(f"increasing norms = {100*count_incr_images/len(inputs)} %")
    print(f"null norms = {100*count_null_images/len(inputs)} %")
    print("\nvanishing norms idxs = ", vanishing_norm_idxs)
    return vanishing_norm_idxs, non_null_idxs

def save_lrp(explanations, path, filename, layer_idx):
    savedir = os.path.join(path, lrp_savedir(layer_idx))
    save_to_pickle(explanations, path=savedir, filename=filename)

def load_lrp(path, filename, layer_idx):
    savedir = os.path.join(path, lrp_savedir(layer_idx))
    explanations = load_from_pickle(path=savedir, filename=filename)
    return explanations