import os
import lrp
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.colors as colors 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import copy

from attacks.plot import compute_vanishing_norm_idxs

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

    cmap = plt.cm.get_cmap('RdBu')
    vmax = max([max(explanations.flatten()), 0.00001])
    vmin = min([min(explanations.flatten()), -0.00001])
    norm = colors.TwoSlopeNorm(vcenter=0., vmax=vmax, vmin=vmin)

    rows = 2
    cols = min(len(explanations), 10)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4))
    fig.tight_layout()

    for idx in range(cols):

        image = np.squeeze(images[idx])
        expl = np.squeeze(explanations[idx])

        if len(image.shape) == 1:
            image = np.expand_dims(image, axis=0)
            expl = np.expand_dims(expl, axis=0)

        axes[0, idx].imshow(image)
        im = axes[1, idx].imshow(expl, cmap=cmap, norm=norm)

    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.88, 0.14, 0.03, 0.3])
    cbar = fig.colorbar(im, ax=axes[1, :].ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Relevance', labelpad=10)

    plt.show()
    os.makedirs(os.path.dirname(savedir+"/"), exist_ok=True)
    plt.savefig(savedir+filename)

def plot_attacks_explanations(images, explanations, attacks, attacks_explanations, rule, savedir, filename):

    if images.shape != explanations.shape:
        print(images.shape, "!=", explanations.shape)
        raise ValueError

    cmap = plt.cm.get_cmap('RdBu')
    vmax_expl = max([max(explanations.flatten()), 0.000001])
    vmin_expl = min([min(explanations.flatten()), -0.000001])
    norm_expl = colors.TwoSlopeNorm(vcenter=0., vmax=vmax_expl, vmin=vmin_expl)
    vmax_atk_expl = max([max(attacks_explanations.flatten()), 0.000001])
    vmin_atk_expl = min([min(attacks_explanations.flatten()), -0.000001])
    norm_atk_expl = colors.TwoSlopeNorm(vcenter=0., vmax=vmax_atk_expl, vmin=vmin_atk_expl)

    rows = 4
    cols = min(len(explanations), 10)
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
    os.makedirs(os.path.dirname(savedir+"/"), exist_ok=True)
    plt.savefig(savedir+filename)    

def plot_vanishing_explanations(images, samples_explanations, n_samples_list, rule, savedir, filename):

    if images.shape != samples_explanations[0].shape:
        print(images.shape, "!=", samples_explanations[0].shape)
        raise ValueError

    vanishing_idxs=compute_vanishing_norm_idxs(samples_explanations, n_samples_list, norm="linfty")    

    rows = min(2*len(n_samples_list), 2*4)
    cols = min(len(vanishing_idxs), 10)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.tight_layout()

    for samples_idx, n_samples in enumerate(n_samples_list):

        cmap = plt.cm.get_cmap('RdBu')
        vmax = max([max(samples_explanations.flatten()), 0.000001])
        vmin = min([min(samples_explanations.flatten()), -0.000001])
        norm = colors.TwoSlopeNorm(vcenter=0., vmax=vmax, vmin=vmin)

        for col_idx in range(cols):

            image = np.squeeze(images[col_idx])
            image = np.expand_dims(image, axis=0) if len(image.shape) == 1 else image

            expl = np.squeeze(samples_explanations[samples_idx, col_idx])
            expl = np.expand_dims(expl, axis=0) if len(expl.shape) == 1 else expl
            print(image.shape, expl.shape)

            axes[2*samples_idx, col_idx].imshow(image)
            im = axes[2*samples_idx+1, col_idx].imshow(expl, cmap=cmap, norm=norm)

        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.88, 0.54, 0.03, 0.2])
        cbar = fig.colorbar(im, ax=axes[2*samples_idx+1, :].ravel().tolist(), cax=cbar_ax)
        cbar.set_label('Relevance', labelpad=10)

    plt.show()
    os.makedirs(os.path.dirname(savedir+"/"), exist_ok=True)
    plt.savefig(savedir+filename)    

