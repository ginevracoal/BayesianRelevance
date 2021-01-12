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
