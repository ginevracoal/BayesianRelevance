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
DEBUG=False

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

def relevant_subset(images, pxl_idxs, lrp_rob_method):

    flat_images = images.reshape(*images.shape[:1], -1)
    images_rel = np.zeros(flat_images.shape)

    if lrp_rob_method=="imagewise":
        # different selection of pixels for each image

        for image_idx, im_pxl_idxs in enumerate(pxl_idxs):
            images_rel[image_idx,im_pxl_idxs] = flat_images[image_idx,im_pxl_idxs]

    elif lrp_rob_method=="pixelwise":
        # same pxls for all the images
        
        images_rel[:,pxl_idxs] = flat_images[:,pxl_idxs]

    else:
        raise NotImplementedError
    
    images_rel = images_rel.reshape(images.shape)
    return images_rel

def plot_attacks_explanations(images, explanations, attacks, attacks_explanations, 
                              predictions, attacks_predictions, successful_attacks_idxs, failed_attacks_idxs,
                              labels, pxl_idxs, lrp_rob_method, rule, savedir, filename, layer_idx=-1):

    if DEBUG:
        print(images.shape, explanations.shape, attacks.shape, attacks_explanations.shape)
        print(predictions.shape, attacks_predictions.shape)
        print(successful_attacks_idxs.shape, failed_attacks_idxs.shape)

    if len(successful_attacks_idxs)<3 or len(failed_attacks_idxs)<3:
        return None

    images_cmap='Greys'

    set_seed(0)
    chosen_successful_idxs = np.random.choice(successful_attacks_idxs, 3)
    chosen_failed_idxs =  np.random.choice(failed_attacks_idxs, 3)
    im_idxs = np.concatenate([chosen_successful_idxs, chosen_failed_idxs])
    print("im_idxs =", im_idxs)

    if DEBUG:
        print("successful_attacks_idxs", successful_attacks_idxs)
        print("failed_attacks_idxs", failed_attacks_idxs)
        print("chosen_successful_idxs", chosen_successful_idxs)
        print("chosen_failed_idxs", chosen_failed_idxs)
        print("idxs", im_idxs)

    images = images[im_idxs].detach().cpu().numpy()
    explanations = explanations[im_idxs].detach().cpu().numpy()
    attacks = attacks[im_idxs].detach().cpu().numpy()
    attacks_explanations = attacks_explanations[im_idxs].detach().cpu().numpy()
    predictions = predictions[im_idxs].detach().cpu().numpy()
    attacks_predictions = attacks_predictions[im_idxs].detach().cpu().numpy()
    labels = labels[im_idxs].detach().cpu().numpy()

    if images.shape != explanations.shape:
        print(images.shape, "!=", explanations.shape)
        raise ValueError

    selected_pxl_idxs = pxl_idxs if lrp_rob_method=="pixelwise" else pxl_idxs[im_idxs]

    images_rel = relevant_subset(images, selected_pxl_idxs, lrp_rob_method)
    attacks_rel = relevant_subset(attacks, selected_pxl_idxs, lrp_rob_method)
    explanations = relevant_subset(explanations, selected_pxl_idxs, lrp_rob_method)
    attacks_explanations = relevant_subset(attacks_explanations, selected_pxl_idxs, lrp_rob_method)

    images_rel = np.ma.masked_where(images_rel == 0., images_rel)
    attacks_rel = np.ma.masked_where(attacks_rel == 0., attacks_rel)

    cmap = plt.cm.get_cmap(relevance_cmap)

    vmax_expl = max([max(explanations.flatten()), 0.000001])
    vmin_expl = min([min(explanations.flatten()), -0.000001])
    norm_expl = colors.TwoSlopeNorm(vcenter=0., vmax=vmax_expl, vmin=vmin_expl)

    vmax_atk_expl = max([max(attacks_explanations.flatten()), 0.000001])
    vmin_atk_expl = min([min(attacks_explanations.flatten()), -0.000001])
    norm_atk_expl = colors.TwoSlopeNorm(vcenter=0., vmax=vmax_atk_expl, vmin=vmin_atk_expl)

    rows = 4
    cols = min(len(explanations)+1, 7)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6), dpi=150)
    fig.tight_layout()

    fig.text(0.17, 0.97, "Successful attacks")
    fig.text(0.74, 0.97, "Failed attacks")

    for im_idx, axis_idx in enumerate([0,1,2,4,5,6]):

        image = np.squeeze(images[im_idx])
        image_rel = np.squeeze(images_rel[im_idx])
        expl = np.squeeze(explanations[im_idx])
        attack = np.squeeze(attacks[im_idx])
        attack_rel = np.squeeze(attacks_rel[im_idx])
        attack_expl = np.squeeze(attacks_explanations[im_idx])

        if len(image.shape) == 1:
            image = np.expand_dims(image, axis=0)
            image_rel = np.expand_dims(image_rel, axis=0)
            expl = np.expand_dims(expl, axis=0)
            attack = np.expand_dims(attack, axis=0)
            attack_rel = np.expand_dims(attack_rel, axis=0)
            attack_expl = np.expand_dims(attack_expl, axis=0)

        axes[0, axis_idx].imshow(image, cmap=images_cmap)
        axes[0, axis_idx].imshow(image_rel)
        axes[0, axis_idx].set_xlabel(f"label={labels[im_idx]}\nprediction={predictions[im_idx]}")
        expl = axes[1, axis_idx].imshow(expl, cmap=cmap, norm=norm_expl)
        axes[2, axis_idx].imshow(attack, cmap=images_cmap)
        axes[2, axis_idx].imshow(attack_rel)
        axes[2, axis_idx].set_xlabel(f"prediction={attacks_predictions[im_idx]}")
        atk_expl = axes[3, axis_idx].imshow(attack_expl, cmap=cmap, norm=norm_atk_expl)

        axes[0,0].set_ylabel("images")
        axes[1,0].set_ylabel("lrp(images)")
        axes[2,0].set_ylabel("im. attacks")
        axes[3,0].set_ylabel("lrp(attacks)")

    for idx in range(4):
        axes[idx,3].set_axis_off()

    # fig.subplots_adjust(right=0.9)

    cbar_ax = fig.add_axes([0.5, 0.56, 0.01, 0.15])
    cbar = fig.colorbar(expl, ax=axes[0, :].ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Relevance', labelpad=-60)

    cbar_ax = fig.add_axes([0.5, 0.07, 0.01, 0.15])
    cbar = fig.colorbar(atk_expl, ax=axes[2, :].ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Relevance', labelpad=-60)

    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir,filename+"_layeridx="+str(layer_idx)+".png"))

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


def plot_attacks_explanations_layers(images, explanations, attacks, attacks_explanations, 
                                      predictions, attacks_predictions, successful_attacks_idxs, failed_attacks_idxs,
                                      labels, pxl_idxs, learnable_layers_idxs, lrp_rob_method, rule, 
                                      savedir, filename, layer_idx=-1):

    n_layers = len(pxl_idxs)

    images_cmap='Greys'
    cmap = plt.cm.get_cmap(relevance_cmap)

    set_seed(0)
    im_idx = np.random.choice(successful_attacks_idxs, 1)
    print("im_idx =", im_idx)

    image = images[im_idx].detach().cpu().numpy()
    attack = attacks[im_idx].detach().cpu().numpy()
    prediction = predictions[im_idx].argmax().detach().cpu().numpy()
    attack_prediction = attacks_predictions[im_idx].argmax().detach().cpu().numpy()
    label = labels[im_idx].item()

    explanations = explanations[:,im_idx]
    attack_explanations = attacks_explanations[:,im_idx]

    for layer_idx in range(len(learnable_layers_idxs)):
        layer_pxl_idxs = pxl_idxs[layer_idx]
        selected_pxl_idxs = layer_pxl_idxs if lrp_rob_method=="pixelwise" else layer_pxl_idxs[im_idx]

        explanations[layer_idx] = relevant_subset(explanations[layer_idx], selected_pxl_idxs, lrp_rob_method)
        attack_explanations[layer_idx] = relevant_subset(attack_explanations[layer_idx], selected_pxl_idxs, lrp_rob_method)
       
    vmax_expl = max([max(explanations.flatten()), 0.000001])
    vmin_expl = min([min(explanations.flatten()), -0.000001])
    norm_expl = colors.TwoSlopeNorm(vcenter=0., vmax=vmax_expl, vmin=vmin_expl)

    vmax_atk_expl = max([max(attack_explanations.flatten()), 0.000001])
    vmin_atk_expl = min([min(attack_explanations.flatten()), -0.000001])
    norm_atk_expl = colors.TwoSlopeNorm(vcenter=0., vmax=vmax_atk_expl, vmin=vmin_atk_expl)

    for layer_idx in range(len(learnable_layers_idxs)):
        layer_pxl_idxs = pxl_idxs[layer_idx]
        selected_pxl_idxs = layer_pxl_idxs if lrp_rob_method=="pixelwise" else layer_pxl_idxs[im_idx]

        image_rel = relevant_subset(image, selected_pxl_idxs, lrp_rob_method)
        attack_rel = relevant_subset(attack, selected_pxl_idxs, lrp_rob_method)
        image_rel = np.ma.masked_where(image_rel == 0., image_rel)
        attack_rel = np.ma.masked_where(attack_rel == 0., attack_rel)

    rows = 2
    cols = n_layers+1
    fig, axes = plt.subplots(rows, cols, figsize=(8, 4), dpi=150)
    fig.tight_layout()

    # axes[0,0].set_ylabel("Image")
    # axes[1,0].set_ylabel("Attack")
    fig.text(0.035, 0.63, f"Image", ha='center', rotation=90, weight='bold')
    fig.text(0.035, 0.25, f"Attack", ha='center', rotation=90, weight='bold')

    axes[0, 0].imshow(np.squeeze(image), cmap=images_cmap)
    axes[0, 0].imshow(np.squeeze(image_rel))
    # axes[0, 0].set_xlabel(f"Label={label}\nPrediction={prediction}")
    fig.text(0.13, 0.5, f"Label={label}\nPrediction={prediction}", ha='center')

    axes[1, 0].imshow(np.squeeze(attack), cmap=images_cmap)
    axes[1, 0].imshow(np.squeeze(attack_rel))
    # axes[1, 0].set_xlabel(f"Prediction={attack_prediction}")
    fig.text(0.13, 0.05, f"Prediction={attack_prediction}", ha='center')

    x_positions=[0.365, 0.58, 0.81]

    for col_idx, layer_idx in enumerate(learnable_layers_idxs):

        # axes[1,col_idx+1].set_xlabel("Layer idx="+str(layer_idx))
        fig.text(x_positions[col_idx], 0.05, "Layer idx="+str(layer_idx), ha='center', weight='bold')

        expl = np.squeeze(explanations[col_idx])
        attack_expl = np.squeeze(attack_explanations[col_idx])
        expl = axes[0, col_idx+1].imshow(expl, cmap=cmap, norm=norm_expl)
        atk_expl = axes[1, col_idx+1].imshow(attack_expl, cmap=cmap, norm=norm_atk_expl)

    for col_idx in range(len(learnable_layers_idxs)+1):
        axes[0,col_idx].set_axis_off()
        axes[1,col_idx].set_axis_off()

    # plt.tick_params(
    # axis='x',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    # labelbottom=True) #

    fig.subplots_adjust(right=0.88)

    cbar_ax = fig.add_axes([0.91, 0.61, 0.01, 0.32])
    cbar = fig.colorbar(expl, ax=axes[0, :].ravel().tolist(), cax=cbar_ax)
    cbar.set_label('LRP', labelpad=-40)

    cbar_ax = fig.add_axes([0.91, 0.12, 0.01, 0.32])
    cbar = fig.colorbar(atk_expl, ax=axes[1, :].ravel().tolist(), cax=cbar_ax)
    cbar.set_label('LRP', labelpad=-40)

    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir,filename+"_layeridx="+str(layer_idx)+".png"))