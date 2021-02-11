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

def stripplot_lrp_values(lrp_heatmaps_list, n_samples_list, savedir, filename, layer_idx=-1):

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

    savedir = os.path.join(savedir, lrp_savedir(layer_idx))
    os.makedirs(savedir, exist_ok=True)
    fig.savefig(os.path.join(savedir, filename+".png"))


def lrp_labels_distributions(lrp_heatmaps, labels, num_classes, n_samples_list, savedir, filename, topk=None,
                            layer_idx=-1):

    savedir = os.path.join(savedir, lrp_savedir(layer_idx), "labels_distributions")
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


def lrp_samples_distributions(lrp_heatmaps, labels, num_classes, n_samples_list, savedir, 
                              filename, layer_idx=-1):

    savedir = os.path.join(savedir, lrp_savedir(layer_idx), "samples_distributions")
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

def lrp_pixels_distributions(lrp_heatmaps, labels, num_classes, n_samples, savedir, filename, topk=1,
                             layer_idx=-1):
    
    savedir = os.path.join(savedir, lrp_savedir(layer_idx), "pixels_distributions")
    os.makedirs(savedir, exist_ok=True) 

    ### dataframe

    lrp_list=[]
    labels_list=[]
    samples_list=[]
    pixel_idxs_list=[]

    for im_idx in range(len(lrp_heatmaps)):

        image_lrp_heatmaps = np.expand_dims(lrp_heatmaps[im_idx,:], axis=1)
        image_label = labels[im_idx]
        flat_lrp_heatmaps, chosen_pxl_idxs = select_informative_pixels(image_lrp_heatmaps, topk)

        for samples_idx in range(n_samples):
            for array_idx, pixel_idx in enumerate(chosen_pxl_idxs):

                pixel_lrp = flat_lrp_heatmaps[samples_idx, array_idx][0]

                lrp_list.append(pixel_lrp)
                samples_list.append(n_samples)
                pixel_idxs_list.append(pixel_idx)
                labels_list.append(image_label)

        df = pd.DataFrame(data={"lrp":lrp_list, "samples":samples_list,
                               "label":labels_list, "pixel_idx":pixel_idxs_list})

        ### plot 

        for array_idx, pixel_idx in enumerate(chosen_pxl_idxs):

            sns.set_style("darkgrid")
            matplotlib.rc('font', **{'weight': 'bold', 'size': 12})
            fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150, facecolor='w', edgecolor='k')    

            pxl_df = df.loc[df['pixel_idx'] == pixel_idx]

            for samples_idx in range(n_samples):

                samp_df = pxl_df.loc[pxl_df['samples'] == n_samples]
                sns.distplot(samp_df["lrp"], ax=ax, kde=False)
                ax.set_yscale('log')

        fig.savefig(os.path.join(savedir, filename+"_im_idx="+str(im_idx)+".png"))
        plt.close(fig)

def lrp_imagewise_robustness_distributions(det_lrp_robustness, bay_lrp_robustness, mode_lrp_robustness,
                                            det_successful_lrp_robustness, det_failed_lrp_robustness,
                                           bay_successful_lrp_robustness, bay_failed_lrp_robustness,
                                           mode_successful_lrp_robustness, mode_failed_lrp_robustness,
                                           n_samples_list, n_original_images, savedir, filename):

    print("\n=== Percentage of successful/failed attacks ===")

    print("\ndeterministic attack:")
    print(f"det. eval\t {len(det_successful_lrp_robustness)/n_original_images} successful \t{len(det_failed_lrp_robustness)/n_original_images} failed")
    
    print("\nbayesian attack:")
    for idx, n_samples in enumerate(n_samples_list):
        print(f"bay. eval samp={n_samples}\t {len(bay_successful_lrp_robustness[idx])/n_original_images} successful \t{len(bay_failed_lrp_robustness[idx])/n_original_images} failed")

    print("\nmode attack:")
    for idx, n_samples in enumerate(n_samples_list):
        print(f"bay. eval samp={n_samples} \t {len(mode_successful_lrp_robustness[idx])/n_original_images} successful \t{len(mode_failed_lrp_robustness[idx])/n_original_images} failed")
    print(f"mode eval \t{len(mode_successful_lrp_robustness[-1])/n_original_images} successful \t{len(mode_failed_lrp_robustness[-1])/n_original_images} failed")

    ## Successful vs failed

    os.makedirs(savedir, exist_ok=True) 

    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'weight': 'bold', 'size': 12})
    fig, ax = plt.subplots(3, 2, figsize=(10, 6), sharex=True, dpi=150, facecolor='w', edgecolor='k') 

    fig.text(0.3, 0.91, f"Successful attacks", ha='center')

    sns.distplot(det_successful_lrp_robustness, ax=ax[0,0], label="det atk", kde=True)
    sns.distplot(mode_successful_lrp_robustness[-1], ax=ax[0,0], label="mode atk", kde=True)
    for samp_idx, n_samples in enumerate(n_samples_list):
        sns.distplot(mode_successful_lrp_robustness[samp_idx], ax=ax[1,0], label="samp="+str(n_samples)+" mode atk", kde=True)
        sns.distplot(bay_successful_lrp_robustness[samp_idx], ax=ax[2,0], label="samp="+str(n_samples)+" bay atk", kde=True)

    fig.text(0.7, 0.91, "Failed attacks", ha='center')

    sns.distplot(det_failed_lrp_robustness, ax=ax[0,1], label="det atk", kde=True)
    sns.distplot(mode_failed_lrp_robustness[-1], ax=ax[0,1], label="mode atk", kde=True)
    for samp_idx, n_samples in enumerate(n_samples_list):
        sns.distplot(mode_failed_lrp_robustness[samp_idx], ax=ax[1,1], label="samp="+str(n_samples)+" mode atk", kde=True)
        sns.distplot(bay_failed_lrp_robustness[samp_idx], ax=ax[2,1], label="samp="+str(n_samples)+" bay atk", kde=True)

    # print(mode_failed_lrp_robustness[-1])

    ax[2,0].set_xlabel("LRP robustness")
    ax[2,1].set_xlabel("LRP robustness")
    # ax[2,0].set_xlim(0,1)
    # ax[2,1].set_xlim(0,1)

    ax[0,1].set_ylabel("det. attack\ndet. interpretation", rotation=270, labelpad=40)
    ax[0,1].yaxis.set_label_position("right")
    ax[1,1].set_ylabel("det. attack\nbay. interpretation", rotation=270, labelpad=40)
    ax[1,1].yaxis.set_label_position("right")
    ax[2,1].set_ylabel("bay. attack\nbay. interpretation", rotation=270, labelpad=40)
    ax[2,1].yaxis.set_label_position("right")

    ax[0,1].legend()
    ax[1,1].legend()
    ax[2,1].legend()

    fig.savefig(os.path.join(savedir, filename+"_succ_vs_failed.png"))
    plt.close(fig)

    ### All images

    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'weight': 'bold', 'size': 10})
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True, dpi=150, facecolor='w', edgecolor='k') 
    fig.tight_layout()

    ax[0].set_ylabel("det. attack\ndet. interpretation", rotation=270, labelpad=40)
    ax[0].yaxis.set_label_position("right")
    ax[1].set_ylabel("det. attack\nbay. interpretation", rotation=270, labelpad=40)
    ax[1].yaxis.set_label_position("right")
    ax[2].set_ylabel("bay. attack\nbay. interpretation", rotation=270, labelpad=40)
    ax[2].yaxis.set_label_position("right")

    sns.distplot(det_lrp_robustness, ax=ax[0], label="Det")
    sns.distplot(mode_lrp_robustness[-1], ax=ax[0], label="Mode")

    for samp_idx, n_samples in enumerate(n_samples_list):
        sns.distplot(mode_lrp_robustness[samp_idx], ax=ax[1], label="Bay samp="+str(n_samples))
        sns.distplot(bay_lrp_robustness[samp_idx], ax=ax[2], label="Bay samp="+str(n_samples))

    ax[2].set_xlabel("LRP robustness")
    ax[2].set_xlabel("LRP robustness")


    ax[0].legend()
    ax[1].legend()
    ax[2].legend()


    fig.savefig(os.path.join(savedir, filename+"_all_images.png"))
    plt.close(fig)

def lrp_catplot_robustness_distributions(det_lrp_robustness, bay_lrp_robustness, mode_lrp_robustness,
                                           n_samples_list, n_original_images, savedir, filename):

    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'weight': 'bold', 'size': 10})
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True, dpi=150, facecolor='w', edgecolor='k') 
    fig.tight_layout()

    lrp_robustness=[]
    net_type=[]
    atk_type=[]

    lrp_robustness.extend(det_lrp_robustness)
    net_type.extend(np.repeat("deterministic", len(det_lrp_robustness)))
    atk_type.extend(np.repeat("deterministic", len(det_lrp_robustness)))

    lrp_robustness.extend(mode_lrp_robustness[-1])
    net_type.extend(np.repeat("mode", len(mode_lrp_robustness[-1])))
    atk_type.extend(np.repeat("mode", len(mode_lrp_robustness[-1])))

    for samp_idx, n_samples in enumerate(n_samples_list):
        lrp_robustness.extend(bay_lrp_robustness[samp_idx])
        net_type.extend(np.repeat("bayesian", len(bay_lrp_robustness[samp_idx])))
        atk_type.extend(np.repeat("bayesian", len(bay_lrp_robustness[samp_idx])))

        lrp_robustness.extend(mode_lrp_robustness[samp_idx])
        net_type.extend(np.repeat("bayesian", len(mode_lrp_robustness[samp_idx])))
        atk_type.extend(np.repeat("mode", len(mode_lrp_robustness[samp_idx])))

    df = pd.DataFrame({"lrp_robustness":lrp_robustness, "net_type":net_type, "atk_type":atk_type})

    print(df)

    sns.stripplot(data=df, x="lrp_robustness", y="net_type", hue="atk_type", ax=ax)

    fig.savefig(os.path.join(savedir, filename+"_catplot.png"))
    plt.close(fig)


def lrp_robustness_scatterplot(adversarial_robustness, bayesian_adversarial_robustness,
                               mode_adversarial_robustness, 
                               lrp_robustness, bayesian_lrp_robustness, mode_lrp_robustness,
                               n_samples_list, savedir, filename):

    os.makedirs(savedir, exist_ok=True)
    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'weight': 'bold', 'size': 8})

    fig, ax = plt.subplots(4, 3, figsize=(10, 6), 
                           gridspec_kw={'width_ratios': [1, 2, 1], 'height_ratios': [1, 3, 3, 1]}, 
                           sharex=False, sharey=False, dpi=150, facecolor='w', edgecolor='k') 
    alpha=0.5

    ### scatterplot

    ax[2,1].set_xlabel('Softmax robustness')
    ax[1,0].set_ylabel('LRP robustness')
    ax[2,0].set_ylabel('LRP robustness')

    tot_num_images = len(adversarial_robustness)

    sns.scatterplot(x=adversarial_robustness, y=lrp_robustness, ax=ax[1,1], label='deterministic', alpha=alpha)
    sns.scatterplot(x=mode_adversarial_robustness, y=mode_lrp_robustness, label='posterior mode', 
                    ax=ax[1,1], alpha=alpha)

    for idx, n_samples in enumerate(n_samples_list):
        sns.scatterplot(x=bayesian_adversarial_robustness[idx], y=bayesian_lrp_robustness[idx], 
                        label='posterior samp='+str(n_samples), ax=ax[2,1], alpha=alpha)

    ### degenerate softmax robustness

    ax[2,0].set_xlabel('Softmax rob. = 0.')

    im_idxs = np.where(adversarial_robustness==0.)[0]
    sns.distplot(lrp_robustness[im_idxs], ax=ax[1,0], vertical=True, 
                 label=f"{100*len(lrp_robustness[im_idxs])/tot_num_images}% images")

    im_idxs = np.where(mode_adversarial_robustness==0.)[0]
    sns.distplot(mode_lrp_robustness[im_idxs], vertical=True, color="darkorange", ax=ax[1,0],
                 label=f"{100*len(mode_lrp_robustness[im_idxs])/tot_num_images}% images")

    for sample_idx, n_samples in enumerate(n_samples_list):
        im_idxs = np.where(bayesian_adversarial_robustness[sample_idx]==0.)[0]
        sns.distplot(bayesian_lrp_robustness[sample_idx][im_idxs], vertical=True, ax=ax[2,0],
                     label=f"{100*len(bayesian_lrp_robustness[sample_idx][im_idxs])/tot_num_images}% images")

    ax[2,2].set_xlabel('Softmax rob. = 1.')

    im_idxs = np.where(adversarial_robustness==1.)[0]
    sns.distplot(lrp_robustness[im_idxs], ax=ax[1,2], vertical=True,
                 label=f"{100*len(lrp_robustness[im_idxs])/tot_num_images}% images")

    im_idxs = np.where(mode_adversarial_robustness==1.)[0]
    sns.distplot(mode_lrp_robustness[im_idxs], vertical=True, color="darkorange", ax=ax[1,2],
                 label=f"{100*len(mode_lrp_robustness[im_idxs])/tot_num_images}% images")

    for sample_idx, n_samples in enumerate(n_samples_list):
        im_idxs = np.where(bayesian_adversarial_robustness[sample_idx]==1.)[0]
        sns.distplot(bayesian_lrp_robustness[sample_idx][im_idxs], vertical=True, ax=ax[2,2],
                     label=f"{100*len(bayesian_lrp_robustness[sample_idx][im_idxs])/tot_num_images}% images")

    ### softmax robustness distributions

    sns.distplot(adversarial_robustness, ax=ax[0,1], vertical=False)
    sns.distplot(mode_adversarial_robustness, ax=ax[0,1], vertical=False)
    
    for idx, n_samples in enumerate(n_samples_list):
        sns.distplot(bayesian_adversarial_robustness[idx], ax=ax[3,1], vertical=False)

    ax[1,0].set_ylim(0,1)
    ax[2,0].set_ylim(0,1)
    ax[1,2].set_ylim(0,1)
    ax[2,2].set_ylim(0,1)
    ax[0,1].set_xlim(0,1)
    ax[3,1].set_xlim(0,1)

    ax[1,0].legend()
    ax[2,0].legend()
    ax[1,2].legend()
    ax[2,2].legend()

    for idx in [0,1,2]:
        ax[0,idx].set_axis_off()
        ax[3,idx].set_axis_off()
    ax[0,0].set_axis_off()
    ax[0,2].set_axis_off()

    fig.savefig(os.path.join(savedir, filename+".png"))
    plt.close(fig)    

# def plot_wasserstein_dist(bay_wass_dist_layers, mode_wass_dist_layers, increasing_n_samples, filename, savedir):

#     os.makedirs(savedir, exist_ok=True)
#     sns.set_style("darkgrid")
#     matplotlib.rc('font', **{'weight': 'bold', 'size': 10})

#     fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True, dpi=150, facecolor='w', edgecolor='k') 
#     alpha=0.5

#     ax.set_xlabel('Layer idx')
#     ax.set_ylabel('Wass. dist.')

#     layers_idxs = np.arange(len(bay_wass_dist_layers))+1

#     print(layers_idxs)
#     print(bay_wass_dist_layers[:,0].flatten().shape)

#     for sample_idx, n_samples in enumerate(increasing_n_samples):

#         sns.lineplot(x=layers_idxs, y=bay_wass_dist_layers[:,sample_idx].flatten(), ax=ax, 
#                     label="bayesian samp="+str(n_samples))
#         sns.lineplot(x=layers_idxs, y=mode_wass_dist_layers[:,sample_idx].flatten(), ax=ax, 
#                     label="mode samp="+str(n_samples))

#     sns.lineplot(x=layers_idxs, y=mode_wass_dist_layers, ax=ax, label="mode vs mode")

#     ax.legend()
#     fig.savefig(os.path.join(savedir, filename+".png"))
#     plt.close(fig)

def lrp_layers_robustness_distributions(
        det_lrp_robustness, det_successful_lrp_robustness, det_failed_lrp_robustness,
        bay_lrp_robustness, bay_successful_lrp_robustness, bay_failed_lrp_robustness,
        n_samples_list, topk_list,
        n_original_images, learnable_layers_idxs, savedir, filename):

    os.makedirs(savedir, exist_ok=True) 

    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'weight': 'bold', 'size': 10})

    # cmap = cm.get_cmap('Blues', len(topk_list)+5)
    # hex_colors_blues = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    # cmap = cm.get_cmap('Greens', len(topk_list)+5)
    # hex_colors_greens = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    det_col =  plt.cm.get_cmap('rocket', 100)(np.linspace(0, 1, 10))[7]
    bay_col = plt.cm.get_cmap('crest', 100)(np.linspace(0, 1, len(n_samples_list)+1))[1:]
    clip=None # (0,1)
    alphas = np.linspace(0.7, 0.4, num=len(topk_list))

    ### Successful vs failed
    
    fig, ax = plt.subplots(len(learnable_layers_idxs), 2, figsize=(10, 6), sharex=True, dpi=150, 
                            facecolor='w', edgecolor='k') 
    fig.tight_layout()
    fig.subplots_adjust(top=0.98)
    fig.text(0.3, 0.98, "Successful attacks", ha='center')
    fig.text(0.75, 0.98, "Failed attacks", ha='center')
    fig.subplots_adjust(bottom=0.1)
    ax[len(learnable_layers_idxs)-1,0].set_xlabel("LRP robustness", weight='bold')
    ax[len(learnable_layers_idxs)-1,1].set_xlabel("LRP robustness", weight='bold')
    # ax[len(learnable_layers_idxs)-1,0].set_xlim(-0.1,1.1)
    # ax[len(learnable_layers_idxs)-1,1].set_xlim(-0.1,1.1)

    for row_idx, layer_idx in enumerate(learnable_layers_idxs):
        for topk_idx, topk in enumerate(topk_list):

            sns.kdeplot(det_successful_lrp_robustness[topk_idx][row_idx], ax=ax[row_idx,0], color=det_col, 
                         label=f"Det. topk={topk}", alpha=alphas[topk_idx], fill=True, linewidth=0, clip=clip)
            
            for samp_idx, n_samples in enumerate(n_samples_list):
                sns.kdeplot(bay_successful_lrp_robustness[topk_idx][row_idx][samp_idx], ax=ax[row_idx,0],
                            color=bay_col[samp_idx], 
                            alpha=alphas[topk_idx], label=f"Bay. samp={n_samples} topk={topk}", fill=True,
                            linewidth=0, clip=clip)

            sns.kdeplot(det_failed_lrp_robustness[topk_idx][row_idx], ax=ax[row_idx,1], color=det_col, 
                         label=f"Det. topk={topk}", alpha=alphas[topk_idx], fill=True, linewidth=0, clip=clip)
            
            for samp_idx, n_samples in enumerate(n_samples_list):
                sns.kdeplot(bay_failed_lrp_robustness[topk_idx][row_idx][samp_idx], ax=ax[row_idx,1], 
                            color=bay_col[samp_idx], 
                            alpha=alphas[topk_idx], label=f"Bay. samp={n_samples} topk={topk}", fill=True,
                            linewidth=0, clip=clip)
        
        ax[row_idx,1].yaxis.set_label_position("right")
        ax[row_idx,1].set_ylabel("layer idx="+str(layer_idx), rotation=270, labelpad=15, weight='bold', size=8)


    ax[0,0].legend(loc="upper left")
    # plt.setp(ax[0,0].get_legend().get_texts(), fontsize='8')
    plt.legend(frameon=False)
    plt.subplots_adjust(hspace=0.1)
    fig.savefig(os.path.join(savedir, filename+"_succ_vs_failed.png"))
    plt.close(fig)

    ### All images

    if len(n_samples_list) > 1 and len(topk_list) > 1: # split cols 

        alpha = 0.7
        fig, ax = plt.subplots(len(learnable_layers_idxs), 2, figsize=(6, 5), sharex=True, dpi=150, 
                                facecolor='w', edgecolor='k', sharey=True) 
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1) 

        for row_idx, layer_idx in enumerate(learnable_layers_idxs):

            ax[row_idx,1].yaxis.set_label_position("right")
            ax[row_idx,1].set_ylabel("Layer idx="+str(layer_idx), rotation=270, labelpad=10, weight='bold', size=8)

            for topk_idx, topk in enumerate(topk_list):

                sns.kdeplot(det_lrp_robustness[topk_idx][row_idx], ax=ax[row_idx, topk_idx], label=f"Det.", 
                            color=det_col, alpha=alpha, fill=True, linewidth=0, clip=clip)

                for samp_idx, n_samples in enumerate(n_samples_list):
                    sns.kdeplot(bay_lrp_robustness[topk_idx][row_idx][samp_idx], ax=ax[row_idx, topk_idx], 
                                color=bay_col[samp_idx], 
                                label=f"Bay. samp={n_samples}",  alpha=alpha,
                               fill=True, linewidth=0, clip=clip)

                    ax[len(learnable_layers_idxs)-1, topk_idx].set_xlabel("LRP robustness") 

                ax[0, topk_idx].set_title('topk='+str(topk),fontdict={'fontsize':10, 'fontweight':'bold'})

        fig.subplots_adjust(top=0.9)
        ax[0,1].legend(bbox_to_anchor=(0.3, 1.45))

        fig.savefig(os.path.join(savedir, filename+"_all_images.png"))
        plt.close(fig)

    else:

        fig, ax = plt.subplots(len(learnable_layers_idxs), 1, figsize=(5, 5), sharex=True, dpi=150, 
                                facecolor='w', edgecolor='k') 
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1)
        ax[len(learnable_layers_idxs)-1].set_xlabel("LRP robustness", weight='bold')

        for row_idx, layer_idx in enumerate(learnable_layers_idxs):
            for topk_idx, topk in enumerate(topk_list):

                sns.kdeplot(det_lrp_robustness[topk_idx][row_idx], ax=ax[row_idx], label=f"Det. topk={topk}", 
                            color=det_col, alpha=alphas[topk_idx], fill=True, linewidth=0, clip=clip)

                for samp_idx, n_samples in enumerate(n_samples_list):
                    sns.kdeplot(bay_lrp_robustness[topk_idx][row_idx][samp_idx], ax=ax[row_idx], color=bay_col[samp_idx], 
                                label=f"Bay. samp={n_samples} topk={topk}",  alpha=alphas[topk_idx],
                               fill=True, linewidth=0, clip=clip)
                
            ax[row_idx].yaxis.set_label_position("right")
            ax[row_idx].set_ylabel("layer idx="+str(layer_idx), rotation=270, labelpad=15, weight='bold', size=8)

        fig.subplots_adjust(top=0.9)
        ax[0].legend(bbox_to_anchor=(0.7, 1.45))
        # ax[0].legend(loc="upper left")
        # plt.setp(ax[0].get_legend().get_texts(), fontsize='8')
        # plt.legend(frameon=False)
        # plt.subplots_adjust(hspace=0.1)
        fig.savefig(os.path.join(savedir, filename+"_all_images.png"))
        plt.close(fig)


def lrp_layers_robustness_scatterplot(det_successful_lrp_robustness, det_failed_lrp_robustness,
                                       bay_successful_lrp_robustness, bay_failed_lrp_robustness,
                                       det_successful_lrp_norm, det_failed_lrp_norm,
                                       bay_successful_lrp_norm, bay_failed_lrp_norm,
                                       n_samples_list, topk_list,
                                       n_original_images, n_learnable_layers, savedir, filename):

    ### Successful vs failed

    os.makedirs(savedir, exist_ok=True) 

    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'weight': 'bold', 'size': 10})
    fig, ax = plt.subplots(n_learnable_layers, 2, figsize=(10, 8), sharex=True, dpi=150, facecolor='w', edgecolor='k') 
    fig.tight_layout()
    
    # cmap = cm.get_cmap('Blues', n_learnable_layers+3)
    # hex_colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    fig.subplots_adjust(top=0.97)
    fig.text(0.3, 0.98, "Successful attacks", ha='center')
    fig.text(0.75, 0.98, "Failed attacks", ha='center')
    alpha=0.4

    for layer_idx in range(n_learnable_layers):
        for topk_idx, topk in enumerate(topk_list):

            # color = hex_colors[layer_idx+3]
            legend = 'brief' if layer_idx==0 else False

            sns.scatterplot(det_successful_lrp_robustness[topk_idx][layer_idx], 
                            det_successful_lrp_norm[topk_idx][layer_idx],
                            ax=ax[layer_idx,0], label="deterministic", alpha=alpha, legend=legend)
            
            for samp_idx, n_samples in enumerate(n_samples_list):

                sns.scatterplot(bay_successful_lrp_robustness[topk_idx][layer_idx][samp_idx], 
                                bay_successful_lrp_norm[topk_idx][layer_idx][samp_idx], 
                                ax=ax[layer_idx,0], label=f"bayesian {n_samples} samples", 
                                alpha=alpha, legend=legend)

            sns.scatterplot(det_failed_lrp_robustness[topk_idx][layer_idx], 
                            det_failed_lrp_norm[topk_idx][layer_idx], 
                            ax=ax[layer_idx,1], alpha=alpha, legend=False)
            
            for samp_idx, n_samples in enumerate(n_samples_list):
                sns.scatterplot(bay_failed_lrp_robustness[topk_idx][layer_idx][samp_idx], 
                                bay_failed_lrp_norm[topk_idx][layer_idx][samp_idx], 
                                ax=ax[layer_idx,1], alpha=alpha, legend=False)
        
        ax[layer_idx,1].yaxis.set_label_position("right")
        ax[layer_idx,1].set_ylabel("idx="+str(layer_idx+1), rotation=270, labelpad=15)

    for col_idx in range(2):
        ax[n_learnable_layers-1,col_idx].set_xlabel("LRP robustness")

    ax[int(n_learnable_layers/2),0].set_ylabel("LRP diff. norm")

    plt.legend(frameon=False)
    plt.setp(ax[0,0].get_legend().get_texts(), fontsize='8')
    # plt.setp(ax[0,1].get_legend().get_texts(), fontsize='8')
    plt.subplots_adjust(hspace=0.1)
    # fig.subplots_adjust(left=0.3)
    # ax[0,0].legend(bbox_to_anchor=(-0.5, 0.5))
    # ax[0,1].legend(bbox_to_anchor=(-1.6, -0.5))

    fig.savefig(os.path.join(savedir, filename+".png"))
    plt.close(fig)


def lrp_layers_mode_robustness(det_lrp_robustness, bay_lrp_robustness, mode_lrp_robustness,
                               n_samples_list, topk_list, learnable_layers_idxs, savedir, filename):

    ### dataframe

    df_lrp_robustness= []
    df_layer_idx = []
    df_model_type = []
    df_atk_type = []
    df_n_samples = []
    df_topk = []

    for topk_idx, topk in enumerate(topk_list):
        for layer_idx, layer in enumerate(learnable_layers_idxs):

            subset_lrp_rob = det_lrp_robustness[topk_idx, layer_idx]
            df_lrp_robustness.extend(subset_lrp_rob)
            df_layer_idx.extend(np.repeat(layer, len(subset_lrp_rob)))
            df_model_type.extend(np.repeat("deterministic", len(subset_lrp_rob)))
            df_atk_type.extend(np.repeat("deterministic", len(subset_lrp_rob)))
            df_n_samples.extend(np.repeat(None, len(subset_lrp_rob)))
            df_topk.extend(np.repeat(topk, len(subset_lrp_rob)))

            subset_lrp_rob = mode_lrp_robustness[topk_idx, layer_idx, -1]
            df_lrp_robustness.extend(subset_lrp_rob)
            df_layer_idx.extend(np.repeat(layer, len(subset_lrp_rob)))
            df_model_type.extend(np.repeat("mode", len(subset_lrp_rob)))
            df_atk_type.extend(np.repeat("mode", len(subset_lrp_rob)))
            df_n_samples.extend(np.repeat(None, len(subset_lrp_rob)))
            df_topk.extend(np.repeat(topk, len(subset_lrp_rob)))

            for samp_idx, n_samples in enumerate(n_samples_list):

                subset_lrp_rob = bay_lrp_robustness[topk_idx, layer_idx, samp_idx]
                df_lrp_robustness.extend(subset_lrp_rob)
                df_layer_idx.extend(np.repeat(layer, len(subset_lrp_rob)))
                df_model_type.extend(np.repeat("bayesian", len(subset_lrp_rob)))
                df_atk_type.extend(np.repeat("bayesian", len(subset_lrp_rob)))
                df_n_samples.extend(np.repeat(None, len(subset_lrp_rob)))
                df_topk.extend(np.repeat(topk, len(subset_lrp_rob)))

                subset_lrp_rob = mode_lrp_robustness[topk_idx, layer_idx, samp_idx]
                df_lrp_robustness.extend(subset_lrp_rob)
                df_layer_idx.extend(np.repeat(layer, len(subset_lrp_rob)))
                df_model_type.extend(np.repeat("bayesian", len(subset_lrp_rob)))
                df_atk_type.extend(np.repeat("mode", len(subset_lrp_rob)))
                df_n_samples.extend(np.repeat(None, len(subset_lrp_rob)))
                df_topk.extend(np.repeat(topk, len(subset_lrp_rob)))


    df = pd.DataFrame(data={"LRP Rob.":df_lrp_robustness, "Layer idx":df_layer_idx,
                            "Net.":df_model_type, "Atk.":df_atk_type, "Samp.":df_n_samples, "topk":df_topk})

    ### plot

    os.makedirs(savedir, exist_ok=True) 
    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'size': 10}) #'weight': 'bold', 

    det_col =  plt.cm.get_cmap('rocket', 100)(np.linspace(0, 1, 10))
    bay_col = plt.cm.get_cmap('crest', 100)(np.linspace(0, 1, len(n_samples_list)+1))[1:]

    fig, ax = plt.subplots(len(learnable_layers_idxs), 2, figsize=(8, 6), sharex=True, dpi=150, 
                            facecolor='w', edgecolor='k', sharey=True) 
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1) 

    ax[0, 0].set_title('Deterministic Nets', fontdict={'fontsize':10, 'fontweight':'bold'})
    ax[0, 1].set_title('Bayesian Nets', fontdict={'fontsize':10, 'fontweight':'bold'})

    x="topk"
    y="LRP Rob."

    for row_idx, layer_idx in enumerate(learnable_layers_idxs):

        ax[row_idx,0].set_ylabel("LRP robustness")

        ax[row_idx,1].yaxis.set_label_position("right")
        ax[row_idx,1].set_ylabel("Layer idx="+str(layer_idx), rotation=270, labelpad=10, weight='bold', size=8)

        subset_df = df.loc[(df['Net.'] == "deterministic") & (df['Atk.'] == "deterministic")]
        sns.lineplot(data=subset_df, x=x, y=y, ax=ax[row_idx, 0], style="Atk.",
                    color=det_col[7], ci=100)

        subset_df = df.loc[(df['Net.'] == "mode") & (df['Atk.'] == "mode")]
        g = sns.lineplot(data=subset_df, x=x, y=y, dashes=[(2,2)], style="Atk.",
                        ax=ax[row_idx, 0], color=det_col[4], ci=100)

        for samp_idx, n_samples in enumerate(n_samples_list):

            subset_df = df.loc[(df['Net.'] == "bayesian") & (df['Atk.'] == "bayesian")]
            sns.lineplot(data=subset_df, x=x, y=y, style="Atk.",
                        ax=ax[row_idx, 1], 
                        color=bay_col[samp_idx], ci=100)

            subset_df = df.loc[(df['Net.'] == "bayesian") & (df['Atk.'] == "mode")]
            g = sns.lineplot(data=subset_df, x=x, y=y,  dashes=[(2,2)],
                        ax=ax[row_idx, 1], style="Atk.",
                        color=bay_col[samp_idx], ci=100)

        g.set_xticks(topk_list)
        g.set_xticklabels(topk_list)

    fig.subplots_adjust(top=0.9)
    # ax[0,1].legend(bbox_to_anchor=(0.3, 1.45))

    fig.savefig(os.path.join(savedir, filename+"_all_images.png"))
    plt.close(fig)


# def lrp_mode_wasserstein_robustness(det_lrp_robustness, bay_lrp_robustness, mode_lrp_robustness,
#                                     n_samples_list, topk_list, learnable_layers_idxs, savedir, filename):
#     from scipy.stats import wasserstein_distance

#     ### dataframe

#     df_lrp_robustness_dist= []
#     df_layer_idx = []
#     df_model_type = []
#     df_atk_type = []
#     df_n_samples = []
#     df_topk = []

#     for topk_idx, topk in enumerate(topk_list):
#         for layer_idx, layer in enumerate(learnable_layers_idxs):

#             # subset_det_lrp_rob = det_lrp_robustness[topk_idx, layer_idx]
#             # df_lrp_robustness.extend(subset_lrp_rob)
#             # df_layer_idx.extend(np.repeat(layer, len(subset_lrp_rob)))
#             # df_model_type.extend(np.repeat("deterministic", len(subset_lrp_rob)))
#             # df_atk_type.extend(np.repeat("deterministic", len(subset_lrp_rob)))
#             # df_n_samples.extend(np.repeat(None, len(subset_lrp_rob)))
#             # df_topk.extend(np.repeat(topk, len(subset_lrp_rob)))

#             subset_lrp_rob = mode_lrp_robustness[topk_idx, layer_idx, -1]
#             df_lrp_robustness_dist.extend(wasserstein_distance(subset_lrp_rob, subset_det_lrp_rob))
#             df_layer_idx.extend(np.repeat(layer, len(subset_lrp_rob)))
#             df_model_type.extend(np.repeat("mode", len(subset_lrp_rob)))
#             df_atk_type.extend(np.repeat("mode", len(subset_lrp_rob)))
#             df_n_samples.extend(np.repeat(None, len(subset_lrp_rob)))
#             df_topk.extend(np.repeat(topk, len(subset_lrp_rob)))

#             for samp_idx, n_samples in enumerate(n_samples_list):

#                 subset_lrp_rob = bay_lrp_robustness[topk_idx, layer_idx, samp_idx]
#                 df_lrp_robustness_dist.extend(wasserstein_distance(subset_lrp_rob, subset_det_lrp_rob))
#                 df_layer_idx.extend(np.repeat(layer, len(subset_lrp_rob)))
#                 df_model_type.extend(np.repeat("bayesian", len(subset_lrp_rob)))
#                 df_atk_type.extend(np.repeat("bayesian", len(subset_lrp_rob)))
#                 df_n_samples.extend(np.repeat(None, len(subset_lrp_rob)))
#                 df_topk.extend(np.repeat(topk, len(subset_lrp_rob)))

#                 subset_lrp_rob = mode_lrp_robustness[topk_idx, layer_idx, samp_idx]
#                 df_lrp_robustness_dist.extend(wasserstein_distance(subset_lrp_rob, subset_det_lrp_rob))
#                 df_layer_idx.extend(np.repeat(layer, len(subset_lrp_rob)))
#                 df_model_type.extend(np.repeat("bayesian", len(subset_lrp_rob)))
#                 df_atk_type.extend(np.repeat("mode", len(subset_lrp_rob)))
#                 df_n_samples.extend(np.repeat(None, len(subset_lrp_rob)))
#                 df_topk.extend(np.repeat(topk, len(subset_lrp_rob)))


#     df = pd.DataFrame(data={"LRP Rob. Wass. dist":df_lrp_robustness, "Layer idx":df_layer_idx,
#                             "Net.":df_model_type, "Atk.":df_atk_type, "Samp.":df_n_samples, "topk":df_topk})

#     ### plot

#     os.makedirs(savedir, exist_ok=True) 
#     sns.set_style("darkgrid")
#     matplotlib.rc('font', **{'size': 10}) #'weight': 'bold', 

#     det_col =  plt.cm.get_cmap('rocket', 100)(np.linspace(0, 1, 10))
#     bay_col = plt.cm.get_cmap('crest', 100)(np.linspace(0, 1, len(n_samples_list)+1))[1:]

#     fig, ax = plt.subplots(len(learnable_layers_idxs), 2, figsize=(8, 6), sharex=True, dpi=150, 
#                             facecolor='w', edgecolor='k', sharey=True) 
#     fig.tight_layout()
#     fig.subplots_adjust(bottom=0.1) 

#     ax[0, 0].set_title('Deterministic Nets', fontdict={'fontsize':10, 'fontweight':'bold'})
#     ax[0, 1].set_title('Bayesian Nets', fontdict={'fontsize':10, 'fontweight':'bold'})

#     x="topk"
#     y="LRP Rob. Wass. dist"

#     for row_idx, layer_idx in enumerate(learnable_layers_idxs):

#         ax[row_idx,0].set_ylabel("LRP robustness")

#         ax[row_idx,1].yaxis.set_label_position("right")
#         ax[row_idx,1].set_ylabel("Layer idx="+str(layer_idx), rotation=270, labelpad=10, weight='bold', size=8)

#         subset_df = df.loc[(df['Net.'] == "deterministic") & (df['Atk.'] == "deterministic")]
#         sns.lineplot(data=subset_df, x=x, y=y, ax=ax[row_idx, 0], style="Atk.",
#                     color=det_col[7], ci=100)

#         subset_df = df.loc[(df['Net.'] == "mode") & (df['Atk.'] == "mode")]
#         g = sns.lineplot(data=subset_df, x=x, y=y, dashes=[(2,2)], style="Atk.",
#                         ax=ax[row_idx, 0], color=det_col[4], ci=100)

#         for samp_idx, n_samples in enumerate(n_samples_list):

#             subset_df = df.loc[(df['Net.'] == "bayesian") & (df['Atk.'] == "bayesian")]
#             sns.lineplot(data=subset_df, x=x, y=y, style="Atk.",
#                         ax=ax[row_idx, 1], 
#                         color=bay_col[samp_idx], ci=100)

#             subset_df = df.loc[(df['Net.'] == "bayesian") & (df['Atk.'] == "mode")]
#             g = sns.lineplot(data=subset_df, x=x, y=y,  dashes=[(2,2)],
#                         ax=ax[row_idx, 1], style="Atk.",
#                         color=bay_col[samp_idx], ci=100)

#         g.set_xticks(topk_list)
#         g.set_xticklabels(topk_list)

#     fig.subplots_adjust(top=0.9)
#     # ax[0,1].legend(bbox_to_anchor=(0.3, 1.45))

#     fig.savefig(os.path.join(savedir, filename+"_all_images.png"))
#     plt.close(fig)
