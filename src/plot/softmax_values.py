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


def _build_df_allSoftmax(data_loader, net, device, n_samples_list):

    n_inputs = len(data_loader.dataset)

    filename = "df_allSoftmax_inp="+str(n_inputs)
    filename = filename+"_incr_samples.csv" if len(n_samples_list)>1 else filename+".csv"

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

def _build_df_maxSoftmax(data_loader, net, device, n_samples_list):

    n_inputs = len(data_loader.dataset)

    filename = "df_maxSoftmax_inp="+str(n_inputs)
    filename = filename+"_incr_samples.csv" if len(n_samples_list)>1 else filename+".csv"

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
                              "class_idx":int(class_idx), "max_softmax_value":max_softmax_value}
                    df.loc[row_count] = pd.Series(row_dict)
                    row_count += 1
    
    print(df.head())
    return df

def plot_allSoftmax(model_type, data_loader, net, n_samples_list, device, load_dir=TESTS):

    n_inputs = len(data_loader.dataset)
    df = _build_df_allSoftmax(data_loader=data_loader, net=net, n_samples_list=n_samples_list, 
                                    device=device)

    for x, vs in [("class_idx","Classes"),("n_samples","Samples")]:
        
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
        plt.title(f"{model_type} inp={n_inputs} n_samp={n_samples_list}")        
        sns.boxenplot(x=x, y="softmax_values", data=df, color="grey")   
        sns.stripplot(x=x, y="softmax_values", hue="prediction", data=df, alpha=alpha,
                      palette= "gist_heat", jitter=0.1, dodge=True)
        path = TESTS+str(net.name)+"/"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path+model_type+"_allSoftmax_vs"+vs+"_inp="+str(n_inputs)+".png")

def plot_maxSoftmax(model_type, data_loader, net, n_samples_list, device, load_dir=TESTS):

    n_inputs = len(data_loader.dataset)
    df = _build_df_maxSoftmax(data_loader=data_loader, net=net, n_samples_list=n_samples_list, 
                                    device=device)

    for x, vs in [("class_idx","Classes"),("n_samples","Samples")]:
        
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300, facecolor='w', edgecolor='k')
        plt.title(f"{model_type} inp={n_inputs} n_samp={n_samples_list}")        
        sns.boxenplot(x=x, y="max_softmax_value", data=df, color="grey")   
        sns.stripplot(x=x, y="max_softmax_value", hue="prediction", data=df, alpha=alpha,
                      palette= "gist_heat", jitter=0.1, dodge=True)
        path = TESTS+str(net.name)+"/"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path+model_type+"_maxSoftmax_vs"+vs+"_inp="+str(n_inputs)+".png")


# def main(args):

#     n_samples_list=[1,50,100]

#     if args.device=="cuda":
#         torch.set_default_tensor_type('torch.cuda.FloatTensor')

#     load_dir = DATA if args.load_dir=="DATA" else TESTS
#     test_loader, net = load_test_net(model_idx=args.model_idx, model_type=args.model_type, 
#                                      device=args.device, load_dir=load_dir, n_inputs=args.n_inputs)

#     if args.allSoftmax:
#         plot_allSoftmax(data_loader=test_loader, net=net, n_samples_list=n_samples_list, 
#                         device=args.device, model_type=args.model_type)

#     if args.maxSoftmax:
#         plot_maxSoftmax(data_loader=test_loader, net=net, n_samples_list=n_samples_list, 
#                         device=args.device, model_type=args.model_type)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_inputs", default=500, type=int)
#     parser.add_argument("--model_idx", default=0, type=int)
#     parser.add_argument("--model_type", default="fullBNN", type=str, help="fullBNN, redBNN, laplBNN")
#     parser.add_argument("--load_dir", default="DATA", type=str, help="DATA, TESTS")
#     parser.add_argument("--allSoftmax", default=False, type=eval)
#     parser.add_argument("--maxSoftmax", default=True, type=eval)
#     parser.add_argument("--device", default='cuda', type=str, help='cpu, cuda')   
#     main(args=parser.parse_args())