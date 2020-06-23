"""
Plot softmax distributions from different classes and posterior predictive networks
"""

import sys
import matplotlib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from utils_data import *
from model_bnn import saved_BNNs
from model_redBNN import saved_redBNNs
from utils_models import load_test_net


alpha=0.8
figsize=(8, 4)


def _build_df_allSoftmax(data_loader, net, device, load_dir, n_samples_list, load_df=False):

    n_inputs = len(data_loader.dataset)

    filename = "df_allSoftmax_inp="+str(n_inputs)
    filename = filename+"_incr_samples.csv" if len(n_samples_list)>1 else filename+".csv"

    if load_df:
        df = pd.read_csv(load_dir+str(net.name)+"/"+filename)

    else:
        df = pd.DataFrame(columns=["n_samples", "prediction", "softmax_values", "class_idx"])
        row_count = 0

        with torch.no_grad():
            for n_samples in n_samples_list:

                correct_predictions = 0.0
                for x_batch, y_batch in data_loader:
                
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device).argmax(-1)
                    probs = net.forward(x_batch, n_samples=n_samples).mean(0)

                    for image_idx, softmax_vector in enumerate(probs): 

                        pred = softmax_vector.argmax(-1)
                        label = y_batch[image_idx]
                        prediction = "correct" if pred==label else "incorrect"
                        
                        max_softmax_value = softmax_vector.max().item()

                        for class_idx, softmax_value in enumerate(softmax_vector):

                            row_dict={"n_samples":int(n_samples), "prediction":prediction, 
                                      "softmax_values":softmax_value.item(), 
                                      "class_idx":int(class_idx)}
                            df.loc[row_count] = pd.Series(row_dict)
                            row_count += 1

        os.makedirs(os.path.dirname(TESTS+str(net.name)+"/"), exist_ok=True)
        df.to_csv(TESTS+str(net.name)+"/"+filename, index = False, header=True)
    
    print(df.head())
    return df

def _build_df_maxSoftmax(data_loader, net, device, load_dir, n_samples_list, load_df=False):

    n_inputs = len(data_loader.dataset)

    filename = "df_maxSoftmax_inp="+str(n_inputs)
    filename = filename+"_incr_samples.csv" if len(n_samples_list)>1 else filename+".csv"

    if load_df:
        # df = pd.read_csv(load_dir+str(net.name)+"/"+filename)

    else:
        df = pd.DataFrame(columns=["n_samples", "prediction", "class_idx", 
                                   "max_softmax_value"])
        row_count = 0

        with torch.no_grad():
            for n_samples in n_samples_list:

                correct_predictions = 0.0
                for x_batch, y_batch in data_loader:
                
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device).argmax(-1)
                    probs = net.forward(x_batch, n_samples=n_samples).mean(0)

                    for image_idx, softmax_vector in enumerate(probs): 

                        pred = softmax_vector.argmax(-1)
                        label = y_batch[image_idx]
                        prediction = "correct" if pred==label else "incorrect"
                        
                        max_softmax_value = softmax_vector.max().item()
                        class_idx = softmax_vector.argmax()
                        row_dict={"n_samples":int(n_samples), "prediction":prediction, 
                                  "class_idx":int(class_idx),
                                  "max_softmax_value":max_softmax_value}
                        df.loc[row_count] = pd.Series(row_dict)
                        row_count += 1

        # os.makedirs(os.path.dirname(TESTS+str(net.name)+"/"), exist_ok=True)
        # df.to_csv(TESTS+str(net.name)+"/"+filename, index = False, header=True)
    
    print(df.head())
    return df

def plot_softmax_vs_classes(data_loader, net, n_samples_list, device, 
                            allSoftmax=False, maxSoftmax=True, load_dir=TESTS):

    n_inputs = len(data_loader.dataset)

    if allSoftmax:
        df = _build_df_allSoftmax(data_loader=data_loader, net=net, n_samples_list=[100], 
                       device=device, load_dir=load_dir)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
        sns.boxenplot(x="class_idx", y="softmax_values", data=df, color="grey")   
        sns.stripplot(x="class_idx", y="softmax_values", hue="prediction", data=df, alpha=alpha,
                      palette= "gist_heat", jitter=0.1, dodge=True)
        os.makedirs(os.path.dirname(TESTS), exist_ok=True)
        plt.savefig(TESTS +str(net.name)+"/plot_softmax_class_inp="+str(n_inputs)+"_allSoftmax.png")

    if maxSoftmax:
        df = _build_df_maxSoftmax(data_loader=data_loader, net=net, n_samples_list=[100], 
                       device=device, load_dir=load_dir)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
        sns.boxenplot(x="class_idx", y="max_softmax_value", data=df, color="grey")   
        sns.stripplot(x="class_idx", y="max_softmax_value", hue="prediction", data=df, alpha=alpha,
                      palette= "gist_heat", jitter=0.1, dodge=True)
        os.makedirs(os.path.dirname(TESTS), exist_ok=True)
        plt.savefig(TESTS +str(net.name)+"/plot_softmax_class_inp="+str(n_inputs)+"_maxSoftmax.png")


def plot_softmax_vs_samples(data_loader, net, n_samples_list, device, 
                            allSoftmax=False, maxSoftmax=True, load_dir=TESTS):
    
    n_inputs = len(data_loader.dataset)

    if allSoftmax:
        df = _build_df_allSoftmax(data_loader=data_loader, net=net, n_samples_list=[1,50,100], 
                       device=device, load_dir=load_dir)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
        sns.boxenplot(x="n_samples", y="softmax_values", data=df, color="grey")   
        sns.stripplot(x="n_samples", y="softmax_values", hue="prediction", data=df, alpha=alpha,
                      palette= "gist_heat", jitter=0.1, dodge=True)
        os.makedirs(os.path.dirname(TESTS), exist_ok=True)
        plt.savefig(TESTS +str(net.name)+"/plot_softmax_samples_inp="+str(n_inputs)+"_allSoftmax.png")
    
    if maxSoftmax:
        df = _build_df_maxSoftmax(data_loader=data_loader, net=net, n_samples_list=[1,50,100], 
                       device=device, load_dir=load_dir)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
        sns.boxenplot(x="n_samples", y="max_softmax_value", data=df, color="grey")   
        sns.stripplot(x="n_samples", y="max_softmax_value", hue="prediction", data=df, alpha=alpha,
                      palette= "gist_heat", jitter=0.1, dodge=True)
        os.makedirs(os.path.dirname(TESTS), exist_ok=True)
        plt.savefig(TESTS +str(net.name)+"/plot_softmax_samples_inp="+str(n_inputs)+"_maxSoftmax.png")


def main(args):

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    load_dir = DATA if args.load_dir=="DATA" else TESTS
    test_loader, net = load_test_net(model_idx=args.model_idx, model_type=args.model_type, 
                                     device=args.device, load_dir=load_dir, n_inputs=args.n_inputs)

    plot_softmax_vs_classes(data_loader=test_loader, net=net, n_samples_list=[100], 
                            device=args.device, load_dir=load_dir)

    plot_softmax_vs_samples(data_loader=test_loader, net=net, n_samples_list=[1,50,100], 
                            device=args.device, load_dir=load_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inputs", default=100, type=int)
    parser.add_argument("--model_idx", default=0, type=int)
    parser.add_argument("--model_type", default="redBNN", type=str, help="fullBNN, redBNN, laplBNN")
    # parser.add_argument("--load_df", default=False, type=eval, help="load dataframe")
    parser.add_argument("--load_dir", default="TESTS", type=str, help="DATA, TESTS")
    parser.add_argument("--device", default='cuda', type=str, help='cpu, cuda')   
    main(args=parser.parse_args())