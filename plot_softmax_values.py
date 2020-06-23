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
# from model_laplaceRedBNN import saved_baseNNs, saved_LaplaceRedBNNs
from utils_models import load_data_net


def catplot_softmax_distr(model_idx, model_type, n_inputs, n_samples_list, device, 
                          load_dir, load=False):
    """
    Plot distribution of softmax values over classes.
    category = class
    color = n_samples
    """

    ### collect data
    dataset_name, test_loader, net = load_data_net(model_idx=model_idx, model_type=model_type, 
                                                   device=device, load_dir=DATA,
                                                   n_inputs=n_inputs)

    if load:
        df = pd.read_csv(load_dir+str(net.name)+"/df_softmax_distr_inp="+str(n_inputs)+".csv")

    else:

        df = pd.DataFrame(columns=["n_samples", "prediction", "softmax_value", "class_idx"])
        row_count = 0

        with torch.no_grad():
            for n_samples in n_samples_list:

                correct_predictions = 0.0
                for x_batch, y_batch in test_loader:
                
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device).argmax(-1)
                    probs = net.forward(x_batch, n_samples=n_samples).mean(0)

                    for image_idx, softmax_vector in enumerate(probs): 

                        pred = softmax_vector.argmax(-1)
                        label = y_batch[image_idx]
                        prediction = "correct" if pred==label else "incorrect"
                        
                        for class_idx, softmax_value in enumerate(softmax_vector):

                            row_dict={"n_samples":int(n_samples), "prediction":prediction, 
                                      "softmax_value":softmax_value.item(), "class_idx":int(class_idx)}
                            df.loc[row_count] = pd.Series(row_dict)
                            row_count += 1

        os.makedirs(os.path.dirname(TESTS+str(net.name)+"/"), exist_ok=True)
        df.to_csv(TESTS+str(net.name)+"/df_softmax_distr_inp="+str(n_inputs)+".csv", 
                  index = False, header=True)
    
    print(df.head())

    ### plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300, facecolor='w', edgecolor='k')

    sns.boxplot(x="class_idx", y="softmax_value", data=df, whis=np.inf,)    
    sns.stripplot(x="class_idx", y="softmax_value", hue="prediction", data=df, alpha=0.6,
                  palette="gist_heat", jitter=0.1, dodge=True)

    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS +str(net.name)+"/catplot_softmax_distr_inp="+str(n_inputs)+".png")


def main(args):

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    catplot_softmax_distr(model_idx=args.model_idx, model_type=args.model_type, load_dir=TESTS,
                          n_samples_list=[5,10,50], n_inputs=args.n_inputs, device=args.device,
                          load=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inputs", default=100, type=int)
    parser.add_argument("--model_idx", default=0, type=int)
    parser.add_argument("--model_type", default="redBNN", type=str, help="fullBNN, redBNN, laplBNN")
    parser.add_argument("--device", default='cuda', type=str, help='cpu, cuda')   
    main(args=parser.parse_args())