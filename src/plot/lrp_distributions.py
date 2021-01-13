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


def select_informative_pixels(lrp_heatmaps, topk):

    print(f"\nTop {topk} most informative pixels:")

    if len(lrp_heatmaps.shape)==4:
        print(f"\n(n. images, image shape) = {lrp_heatmaps.shape[0], lrp_heatmaps.shape[1:]}")
        squeeze_dim=1

    elif len(lrp_heatmaps.shape)==5:
        print(f"\n(samples_list_size, n. images, image shape) = {lrp_heatmaps.shape[0], lrp_heatmaps.shape[1], lrp_heatmaps.shape[2:]}")
        squeeze_dim=2

    else:
        raise ValueError("Wrong array shape.")

    flat_lrp_heatmaps = lrp_heatmaps.reshape(*lrp_heatmaps.shape[:squeeze_dim], -1)
    lrp_sum = flat_lrp_heatmaps.sum(0).sum(0) if len(flat_lrp_heatmaps.shape)>2 else flat_lrp_heatmaps.sum(0)

    chosen_pxl_idxs = np.argsort(np.abs(lrp_sum))[-topk:]
    chosen_images_lrp = flat_lrp_heatmaps[..., chosen_pxl_idxs] 

    print("out shape =", chosen_images_lrp.shape)
    print("\nchosen pixels idxs =", chosen_pxl_idxs)

    return chosen_images_lrp, chosen_pxl_idxs

def stripplot_lrp_values(lrp_heatmaps_list, n_samples_list, savedir, filename):

    matplotlib.rc('font', **{'weight': 'bold', 'size': 12})
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150, facecolor='w', edgecolor='k')    
    sns.set_style("darkgrid")

    lrp_heatmaps_components = []
    plot_samples = []
    for samples_idx, n_samples in enumerate(n_samples_list):
        
        print("\nsamples = ", n_samples, end="\t")
        print(f"min = {lrp_heatmaps_list[samples_idx].min():.4f}", end="\t")
        print(f"max = {lrp_heatmaps_list[samples_idx].max():.4f}")

        flat_lrp_heatmap = np.array(lrp_heatmaps_list[samples_idx]).flatten()
        lrp_heatmaps_components.extend(flat_lrp_heatmap)
        plot_samples.extend(np.repeat(n_samples, len(flat_lrp_heatmap)))

    df = pd.DataFrame(data={"lrp_heatmaps": lrp_heatmaps_components, "samples": plot_samples})

    sns.stripplot(x="samples", y="lrp_heatmaps", data=df, linewidth=-0.1, ax=ax, 
                  jitter=0.2, alpha=0.4, palette="gist_heat")

    ax.set_ylabel("")
    ax.set_xlabel("")

    fig.text(0.5, 0.01, "Samples involved in the expectations ($w \sim p(w|D)$)", ha='center')
    fig.text(0.03, 0.5, r"LRP heatmaps components components", 
             va='center', rotation='vertical')

    savedir=os.path.join(savedir, LRP_DIR)
    os.makedirs(savedir, exist_ok=True)
    fig.savefig(os.path.join(savedir, filename+".png"))


def lrp_labels_distributions(lrp_heatmaps, labels, num_classes, n_samples_list, savedir, filename, topk=None):

    savedir = os.path.join(savedir, LRP_DIR, "labels_distributions")
    os.makedirs(savedir, exist_ok=True)

    if topk is not None:
        flat_lrp_heatmaps, _ = select_informative_pixels(lrp_heatmaps, topk)
    else:
        flat_lrp_heatmaps = lrp_heatmaps.reshape(*lrp_heatmaps.shape[:2], -1)

    ### dataframe

    lrp_list = []
    labels_list = []
    samples_list = []

    for samples_idx, n_samples in enumerate(n_samples_list):
        
        print("\nsamples = ", n_samples, end="\t")
        print(f"min = {flat_lrp_heatmaps[samples_idx].min():.4f}", end="\t")
        print(f"max = {flat_lrp_heatmaps[samples_idx].max():.4f}")
        
        for im_idx, label in enumerate(labels):

            lrp_heatmap = flat_lrp_heatmaps[samples_idx, im_idx, :]
            lrp_list.extend(lrp_heatmap)
            samples_list.extend(np.repeat(n_samples, len(lrp_heatmap)))
            labels_list.extend(np.repeat(label, len(lrp_heatmap)))

    df = pd.DataFrame(data={"lrp": lrp_list, "samples": samples_list, "label":labels_list})
    print(df.head())

    ### plot

    for label in range(num_classes):

        matplotlib.rc('font', **{'weight': 'bold', 'size': 12})
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150, facecolor='w', edgecolor='k')    
        sns.set_style("darkgrid")

        colors = ["teal","skyblue","red"]
        for idx, n_samples in enumerate(n_samples_list):

            label_df = df.loc[df['label'] == label]
            label_df = label_df.loc[label_df['samples'] == n_samples]
            sns.distplot(label_df["lrp"], color=colors[idx], label=f"{n_samples} samples", ax=ax, kde=False)
            ax.set_yscale('log')

        plt.legend()
        os.makedirs(savedir, exist_ok=True)
        fig.savefig(os.path.join(savedir, filename+"_label="+str(label)+".png"))
        plt.close(fig)


def lrp_samples_distributions(lrp_heatmaps, labels, num_classes, n_samples_list, savedir, filename):

    savedir = os.path.join(savedir, LRP_DIR, "samples_distributions")
    os.makedirs(savedir, exist_ok=True)
    
    flat_lrp_heatmaps = lrp_heatmaps.reshape(*lrp_heatmaps.shape[:2], -1)

    ### dataframe

    lrp_list=[]
    labels_list=[]
    samples_list=[]

    for samples_idx, n_samples in enumerate(n_samples_list):

        print("\nsamples =", n_samples, end="\t")
        print(f"min = {flat_lrp_heatmaps[samples_idx].min():.4f}", end=" \t")
        print(f"max = {flat_lrp_heatmaps[samples_idx].max():.4f}")

        for pixel_idx in range(flat_lrp_heatmaps.shape[-1]):
            pixel_lrp = flat_lrp_heatmaps[samples_idx, :, pixel_idx]

            lrp_list.extend(pixel_lrp)
            samples_list.extend(np.repeat(n_samples, len(pixel_lrp)))
            labels_list.extend(labels)

        df = pd.DataFrame(data={"lrp": lrp_list, "samples": samples_list,
                               "label":labels_list})

    ### plot 

    for samples_idx, n_samples in enumerate(n_samples_list):

        samp_df = df.loc[df['samples'] == n_samples]

        sns.set_style("darkgrid")
        matplotlib.rc('font', **{'weight': 'bold', 'size': 12})
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150, facecolor='w', edgecolor='k')    

        for label in range(num_classes):

            label_df = samp_df.loc[samp_df['label'] == label]
            sns.distplot(label_df["lrp"], label=f"class={label}", ax=ax, kde=False)
            ax.set_yscale('log')

        plt.legend()
        fig.savefig(os.path.join(savedir, filename+"_samp="+str(n_samples)+".png"))
        plt.close(fig)

def lrp_pixels_distributions(lrp_heatmaps, labels, num_classes, n_samples_list, savedir, filename, topk=10):
    
    savedir = os.path.join(savedir, LRP_DIR, "pixels_distributions")
    os.makedirs(savedir, exist_ok=True)

    ths = 200
    lrp_list = []
    labels_list = []
    samples_list = []
    pixel_idx_list = []

    flat_lrp_heatmaps, chosen_pxl_idxs = select_informative_pixels(lrp_heatmaps, topk)

    ### dataframe

    lrp_list=[]
    labels_list=[]
    samples_list=[]
    pixel_idxs_list=[]

    for samples_idx, n_samples in enumerate(n_samples_list):

        print("\nsamples =", n_samples, end="\t")
        print(f"min = {flat_lrp_heatmaps[samples_idx].min():.4f}", end=" \t")
        print(f"max = {flat_lrp_heatmaps[samples_idx].max():.4f}")

        for array_idx, pixel_idx in enumerate(chosen_pxl_idxs):

            pixel_lrp = flat_lrp_heatmaps[samples_idx, :, array_idx]
            lrp_list.extend(pixel_lrp)
            samples_list.extend(np.repeat(n_samples, len(pixel_lrp)))
            pixel_idxs_list.extend(np.repeat(pixel_idx, len(pixel_lrp)))
            labels_list.extend(labels)

        df = pd.DataFrame(data={"lrp": lrp_list, "samples": samples_list,
                               "label":labels_list, "pixel_idx":pixel_idxs_list})

    ### plot 

    for array_idx, pixel_idx in enumerate(chosen_pxl_idxs):

        sns.set_style("darkgrid")
        matplotlib.rc('font', **{'weight': 'bold', 'size': 12})
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150, facecolor='w', edgecolor='k')    

        pxl_df = df.loc[df['pixel_idx'] == pixel_idx]

        for samples_idx, n_samples in enumerate(n_samples_list):

            samp_df = pxl_df.loc[pxl_df['samples'] == n_samples]
            sns.distplot(samp_df["lrp"], label=f"samples ={n_samples}", ax=ax, kde=False)
            ax.set_yscale('log')

        plt.legend()
        fig.savefig(os.path.join(savedir, filename+"_pixel_idx="+str(pixel_idx)+".png"))
        plt.close(fig)

