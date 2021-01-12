import os
import lrp
import copy
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.colors as colors 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from utils.savedir import *

cmap_name = "RdBu_r"

def compute_explanations(x_test, network, rule, n_samples=None): 

    if n_samples is None:

        explanations = []

        for x in tqdm(x_test):
            x.requires_grad=True
            # Forward pass
            y_hat = network.forward(x.unsqueeze(0), explain=True, rule=rule)

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
                y_hat = network.forward(inputs=x_copy, n_samples=1, sample_idxs=[j], explain=True, rule=rule)

                # Choose argmax
                y_hat = y_hat[torch.arange(x_copy.shape[0]), y_hat.max(1)[1]]
                y_hat = y_hat.sum()

                # Backward pass (compute explanation)
                y_hat.backward()
                explanations.append(x_copy.grad.detach().cpu().numpy())

            avg_explanations.append(np.array(explanations).mean(0).squeeze(0))

        return np.array(avg_explanations)

def plot_explanations(images, explanations, rule, savedir, filename):

    if images.shape != explanations.shape:
        print(images.shape, "!=", explanations.shape)
        raise ValueError

    cmap = plt.cm.get_cmap(cmap_name)
    # vmax = max([max(explanations.flatten()), 0.00001])
    # vmin = min([min(explanations.flatten()), -0.00001])
    # norm = colors.TwoSlopeNorm(vcenter=0., vmax=vmax, vmin=vmin)

    rows = 2
    cols = min(len(explanations), 6)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4))
    fig.tight_layout()

    for idx in range(cols):

        image = np.squeeze(images[idx])
        expl = np.squeeze(explanations[idx])

        if len(image.shape) == 1:
            image = np.expand_dims(image, axis=0)
            expl = np.expand_dims(expl, axis=0)

        axes[0, idx].imshow(image)
        im = axes[1, idx].imshow(expl, cmap=cmap)#, norm=norm)

    # fig.subplots_adjust(right=0.83)
    # cbar_ax = fig.add_axes([0.88, 0.14, 0.03, 0.3])
    # cbar = fig.colorbar(im, ax=axes[1, :].ravel().tolist(), cax=cbar_ax)
    # cbar.set_label('Relevance', labelpad=10)

    plt.show()
    os.makedirs(os.path.dirname(savedir), exist_ok=True)
    plt.savefig(os.path.join(savedir,filename+".png"))

def plot_attacks_explanations(images, explanations, attacks, attacks_explanations, rule, savedir, filename):

    if images.shape != explanations.shape:
        print(images.shape, "!=", explanations.shape)
        raise ValueError

    cmap = plt.cm.get_cmap(cmap_name)
    vmax_expl = max([max(explanations.flatten()), 0.000001])
    vmin_expl = min([min(explanations.flatten()), -0.000001])
    norm_expl = colors.TwoSlopeNorm(vcenter=0., vmax=vmax_expl, vmin=vmin_expl)
    vmax_atk_expl = max([max(attacks_explanations.flatten()), 0.000001])
    vmin_atk_expl = min([min(attacks_explanations.flatten()), -0.000001])
    norm_atk_expl = colors.TwoSlopeNorm(vcenter=0., vmax=vmax_atk_expl, vmin=vmin_atk_expl)

    rows = 4
    cols = min(len(explanations), 6)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4))
    fig.tight_layout()

    for idx in range(cols):

        image = np.squeeze(images[idx])
        expl = np.squeeze(explanations[idx])
        attack = np.squeeze(attacks[idx])
        attack_expl = np.squeeze(attacks_explanations[idx])

        if len(image.shape) == 1:
            image = np.expand_dims(image, axis=0)
            expl = np.expand_dims(expl, axis=0)
            attack = np.expand_dims(attack, axis=0)
            attack_expl = np.expand_dims(attacks_expl, axis=0)

        axes[0, idx].imshow(image)
        expl = axes[1, idx].imshow(expl, cmap=cmap, norm=norm_expl)
        axes[2, idx].imshow(attack)
        atk_expl = axes[3, idx].imshow(attack_expl, cmap=cmap, norm=norm_atk_expl)

    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.88, 0.54, 0.03, 0.2])
    cbar = fig.colorbar(expl, ax=axes[0, :].ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Relevance', labelpad=10)
    cbar_ax = fig.add_axes([0.88, 0.14, 0.03, 0.2])
    cbar = fig.colorbar(atk_expl, ax=axes[2, :].ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Relevance', labelpad=10)

    plt.show()
    os.makedirs(os.path.dirname(savedir), exist_ok=True)
    plt.savefig(os.path.join(savedir,filename+".png"))

def plot_vanishing_explanations(images, samples_explanations, n_samples_list, rule, savedir, filename):

    if images.shape != samples_explanations[0].shape:
        print(images.shape, "!=", samples_explanations[0].shape)
        raise ValueError

    vanishing_idxs=compute_vanishing_norm_idxs(samples_explanations, n_samples_list, norm="linfty")   

    if len(vanishing_idxs)<=1:
        raise ValueError("Not enough examples.")

    rows = min(len(n_samples_list), 5)+1
    cols = min(len(vanishing_idxs), 6)

    chosen_idxs = np.random.choice(vanishing_idxs, cols)

    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    fig.tight_layout()

    for col_idx in range(cols):

        cmap = plt.cm.get_cmap(cmap_name)
        vmax = max(samples_explanations[:, chosen_idxs[col_idx]].flatten())
        vmin = min(samples_explanations[:, chosen_idxs[col_idx]].flatten())
        print(vmin, vmax)
        norm = colors.TwoSlopeNorm(vcenter=0., vmax=vmax, vmin=vmin)

        for samples_idx, n_samples in enumerate(n_samples_list):

            image = np.squeeze(images[chosen_idxs[col_idx]])
            image = np.expand_dims(image, axis=0) if len(image.shape) == 1 else image

            expl = np.squeeze(samples_explanations[samples_idx, chosen_idxs[col_idx]])
            expl = np.expand_dims(expl, axis=0) if len(expl.shape) == 1 else expl

            axes[0, col_idx].imshow(image)
            im = axes[samples_idx+1, col_idx].imshow(expl, cmap=cmap, norm=norm)

        # fig.subplots_adjust(right=0.83)
        # cbar_ax = fig.add_axes([0.88, 0.12, 0.02, 0.6])
        # cbar = fig.colorbar(im, ax=axes[samples_idx+1, :].ravel().tolist(), cax=cbar_ax)
        # cbar.set_label('Relevance', labelpad=10)

    plt.show()
    os.makedirs(os.path.dirname(savedir), exist_ok=True)
    plt.savefig(os.path.join(savedir,filename+".png"))

def compute_vanishing_norm_idxs(inputs, n_samples_list, norm="linfty"):

    if inputs.shape[0] != len(n_samples_list):
        raise ValueError("First dimension should equal the length of `n_samples_list`")

    inputs=np.transpose(inputs, (1, 0, 2, 3, 4))
    vanishing_norm_idxs = []

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
            print("idx =",idx, end="\t")
            count_samples_idx = 0
            for samples_idx, n_samples in enumerate(n_samples_list):

                if norm == "linfty":
                    new_inputs_norm = np.max(np.abs(image[samples_idx]))
                elif norm == "l2":
                    new_inputs_norm = np.linalg.norm(image[samples_idx])

                if new_inputs_norm <= inputs_norm:
                    print(new_inputs_norm, end="\t")
                    inputs_norm = copy.deepcopy(new_inputs_norm)
                    count_samples_idx += 1

            if count_samples_idx == len(n_samples_list):
                vanishing_norm_idxs.append(idx)
                print("\tcount=", count_van_images)
                count_van_images += 1
            else: 
                count_incr_images += 1

            print("\n")

        else:
            count_null_images += 1

    print(f"vanishing norms = {100*count_van_images/len(inputs)} %")
    print(f"increasing norms = {100*count_incr_images/len(inputs)} %")
    print(f"null norms = {100*count_null_images/len(inputs)} %")
    print("\nvanishing norms idxs = ", vanishing_norm_idxs)
    return vanishing_norm_idxs
