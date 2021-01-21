import os
import lrp
import copy
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.colors as colors 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from utils.savedir import *
from utils.seeding import set_seed
from utils.lrp import *

relevance_cmap = "RdBu_r"

def plot_explanations(images, explanations, rule, savedir, filename, layer_idx=-1):

    savedir = os.path.join(savedir, lrp_savedir(layer_idx))

    if images.shape != explanations.shape:
        print(images.shape, "!=", explanations.shape)
        raise ValueError

    cmap = plt.cm.get_cmap(relevance_cmap)

    rows = 2
    cols = min(len(explanations), 6)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4))
    fig.tight_layout()

    set_seed(0)
    idxs = np.random.choice(len(explanations), cols)

    for idx, col in enumerate(range(cols)):

        image = np.squeeze(images[idx])
        expl = np.squeeze(explanations[idx])

        if len(image.shape) == 1:
            image = np.expand_dims(image, axis=0)
            expl = np.expand_dims(expl, axis=0)

        axes[0, col].imshow(image)
        im = axes[1, col].imshow(expl, cmap=cmap)

    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir,filename+".png"))

def plot_attacks_explanations(images, explanations, attacks, attacks_explanations, explanations_attacks,
                              rule, savedir, filename, layer_idx=-1):

    images_cmap='Greys'

    set_seed(0)
    idxs = np.random.choice(len(images), 6)
    images = images[idxs].detach().cpu().numpy()
    explanations = explanations[idxs].detach().cpu().numpy()
    attacks = attacks[idxs].detach().cpu().numpy()
    attacks_explanations = attacks_explanations[idxs].detach().cpu().numpy()
    explanations_attacks = explanations_attacks[idxs].detach().cpu().numpy()

    if images.shape != explanations.shape:
        print(images.shape, "!=", explanations.shape)
        raise ValueError

    cmap = plt.cm.get_cmap(relevance_cmap)

    vmax_expl = max([max(explanations.flatten()), 0.000001])
    vmin_expl = min([min(explanations.flatten()), -0.000001])
    norm_expl = colors.TwoSlopeNorm(vcenter=0., vmax=vmax_expl, vmin=vmin_expl)

    vmax_atk_expl = max([max(attacks_explanations.flatten()), 0.000001])
    vmin_atk_expl = min([min(attacks_explanations.flatten()), -0.000001])
    norm_atk_expl = colors.TwoSlopeNorm(vcenter=0., vmax=vmax_atk_expl, vmin=vmin_atk_expl)

    vmax_expl_atk = max([max(explanations_attacks.flatten()), 0.000001])
    vmin_expl_atk = min([min(explanations_attacks.flatten()), -0.000001])
    norm_expl_atk = colors.TwoSlopeNorm(vcenter=0., vmax=vmax_atk_expl, vmin=vmin_atk_expl)

    rows = 5
    cols = min(len(explanations), 6)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6), dpi=150)
    fig.tight_layout()

    for idx in range(cols):

        image = np.squeeze(images[idx])
        expl = np.squeeze(explanations[idx])
        attack = np.squeeze(attacks[idx])
        attack_expl = np.squeeze(attacks_explanations[idx])
        expl_attack = np.squeeze(explanations_attacks[idx])

        if len(image.shape) == 1:
            image = np.expand_dims(image, axis=0)
            expl = np.expand_dims(expl, axis=0)
            attack = np.expand_dims(attack, axis=0)
            attack_expl = np.expand_dims(attack_expl, axis=0)
            expl_attack = np.expand_dims(expl_attack, axis=0)

        axes[0, idx].imshow(image, cmap=images_cmap)
        expl = axes[1, idx].imshow(expl, cmap=cmap, norm=norm_expl)
        axes[2, idx].imshow(attack, cmap=images_cmap)
        atk_expl = axes[3, idx].imshow(attack_expl, cmap=cmap, norm=norm_atk_expl)
        axes[4, idx].imshow(expl_attack, cmap=images_cmap)

        axes[0,0].set_ylabel("images")
        axes[1,0].set_ylabel("lrp")
        axes[2,0].set_ylabel("images attacks")
        axes[3,0].set_ylabel("lrp of the attacks")
        axes[4,0].set_ylabel("attacks to lrp")

    fig.subplots_adjust(right=0.85)

    cbar_ax = fig.add_axes([0.9, 0.63, 0.01, 0.13])
    cbar = fig.colorbar(expl, ax=axes[0, :].ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Relevance', labelpad=-70)

    cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.13])
    cbar = fig.colorbar(atk_expl, ax=axes[2, :].ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Relevance', labelpad=-60)

    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir,filename+".png"))

def plot_vanishing_explanations(images, samples_explanations, n_samples_list, rule, savedir, filename,
                                layer_idx=-1):
    
    savedir = os.path.join(savedir, lrp_savedir(layer_idx))

    if images.shape != samples_explanations[0].shape:
        print(images.shape, "!=", samples_explanations[0].shape)
        raise ValueError

    vanishing_idxs=compute_vanishing_norm_idxs(samples_explanations, n_samples_list, norm="linfty")[0]   

    if len(vanishing_idxs)<=1:
        raise ValueError("Not enough examples.")

    rows = min(len(n_samples_list), 5)+1
    cols = min(len(vanishing_idxs), 6)

    set_seed(0)
    chosen_idxs = np.random.choice(vanishing_idxs, cols)

    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    fig.tight_layout()

    for col_idx in range(cols):

        cmap = plt.cm.get_cmap(relevance_cmap)
        vmax = max(samples_explanations[:, chosen_idxs[col_idx]].flatten())
        vmin = min(samples_explanations[:, chosen_idxs[col_idx]].flatten())
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

    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir, filename+".png"))