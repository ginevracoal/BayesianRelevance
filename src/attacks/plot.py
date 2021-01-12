import os
import copy
import numpy as np
import matplotlib.pyplot as plt

def plot_grid_attacks(original_images, perturbed_images, filename, savedir):

    fig, axes = plt.subplots(2, len(original_images), figsize = (12,4))

    for i in range(0, len(original_images)):
        axes[0, i].imshow(original_images[i])
        axes[1, i].imshow(perturbed_images[i])

    plt.show()
    os.makedirs(os.path.dirname(savedir+"/"), exist_ok=True)
    plt.savefig(savedir+filename)


