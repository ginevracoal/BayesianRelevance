import os
import copy
import numpy as np
import matplotlib.pyplot as plt

def plot_grid_attacks(original_images, perturbed_images, filename, savedir):

    fig, axes = plt.subplots(2, len(original_images), figsize = (12,4))

    for i in range(0, len(original_images)):

        original_image = original_images[i].permute(1,2,0) if len(original_images[i].shape) > 2 else original_images[i]
        perturbed_image = perturbed_images[i].permute(1,2,0) if len(perturbed_images[i].shape) > 2 else perturbed_images[i]

        axes[0, i].imshow(original_image)
        axes[1, i].imshow(perturbed_image)
        
    os.makedirs(os.path.dirname(savedir+"/"), exist_ok=True)
    plt.savefig(os.path.join(savedir, filename+".png"))


