import sys
sys.path.append(".")
from directories import *
from lossGradients import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from utils import load_dataset
from bnn import saved_bnns
from reducedBNN import saved_redBNNs
from laplaceRedBNN import saved_baseNNs, saved_LaplaceRedBNNs



def build_dataframe(dataset, model_type):
    """
    Build dataframe for plots by evaluating saved models on the test set.
    """

    _, test_loader, inp_shape, out_size = \
        data_loaders(dataset_name=args.dataset, batch_size=128, n_inputs=args.inputs,
                     shuffle=False)

    # load base net and test (return softmax)


    # load other net and test (return softmax)

    # fill dataframe



def stripplot_softmax_values(dataframe):
    """
    Plot distribution of softmax values over classes.
    category = class
    color = model_type
    """

    matplotlib.rc('font', **{'weight': 'bold', 'size': 12})
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150, facecolor='w', edgecolor='k')    
    sns.set_style("darkgrid")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])


def main(args):

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot softmax values")
    parser.add_argument("--inputs", default=100, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist, fashion_mnist, cifar")
    parser.add_argument("--model", default="fullBNN", type=str, help="fullBNN, redBNN, laplBNN")
    parser.add_argument("--device", default='cuda', type=str, help='cpu, cuda')   
    main(args=parser.parse_args())