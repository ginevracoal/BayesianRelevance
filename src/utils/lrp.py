import os
import lrp
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.colors as colors 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

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
            x.requires_grad=True


            for i in range(n_samples):

                # Forward pass
                x_copy = copy.deepcopy(x)
                # x_copy.requires_grad = True
                y_hat = network.forward(inputs=x_copy, n_samples=1, sample_idxs=[i], explain=True, rule=rule)

                # Choose argmax
                y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]]
                y_hat = y_hat.sum()

                # Backward pass (compute explanation)
                y_hat.backward()
                avg_explanations.append(x.grad.detach().cpu().numpy())

        avg_explanations = torch.stack(avg_explanations,0).mean(0)
        return np.array(avg_explanations)

def plot_explanations(images, explanations, rule, savedir, filename):

    if images.shape != explanations.shape:
        raise ValueError

    cmap = plt.cm.get_cmap('RdBu')
    # vmax = max([max(explanations.flatten()), 0.000001])
    # vmin = min([min(explanations.flatten()), -0.000001])
    # norm = colors.TwoSlopeNorm(vcenter=0., vmax=vmax, vmin=vmin)

    rows = 2
    cols = min(len(explanations), 10)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4))

    for idx in range(cols):

        image = np.squeeze(images[idx])
        expl = np.squeeze(explanations[idx])

        if len(image.shape) == 1:
            image = np.expand_dims(image, axis=0)
            expl = np.expand_dims(expl, axis=0)

        axes[0, idx].imshow(image)
        axes[1, idx].imshow(expl, cmap=cmap)#, norm=norm)

    # fig.subplots_adjust(right=0.83)
    # cbar_ax = fig.add_axes([0.88, 0.14, 0.03, 0.75])
    # cbar = fig.colorbar(image, ax=ax.ravel().tolist(), cax=cbar_ax)
    # cbar.set_label('Relevance', labelpad=-60)

    plt.show()
    os.makedirs(os.path.dirname(savedir+"/"), exist_ok=True)
    plt.savefig(savedir+filename)

def plot_attacks_explanations(images, explanations, attacks, attacks_explanations, rule, savedir, filename):

    if images.shape != explanations.shape:
        raise ValueError

    cmap = plt.cm.get_cmap('RdBu')
    # vmax = max([max(explanations.flatten()), 0.000001])
    # vmin = min([min(explanations.flatten()), -0.000001])
    # norm = colors.TwoSlopeNorm(vcenter=0., vmax=vmax, vmin=vmin)

    rows = 4
    cols = min(len(explanations), 10)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4))

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
        axes[1, idx].imshow(expl, cmap=cmap)#, norm=norm)
        axes[2, idx].imshow(attack)
        axes[3, idx].imshow(attack_expl, cmap=cmap)#, norm=norm)

    # fig.subplots_adjust(right=0.83)
    # cbar_ax = fig.add_axes([0.88, 0.14, 0.03, 0.75])
    # cbar = fig.colorbar(image, ax=ax.ravel().tolist(), cax=cbar_ax)
    # cbar.set_label('Relevance', labelpad=-60)

    plt.show()
    os.makedirs(os.path.dirname(savedir+"/"), exist_ok=True)
    plt.savefig(savedir+filename)    