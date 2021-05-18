import os
import argparse
import numpy as np

import torch
import torchvision
from torch import nn
import torch.nn.functional as nnf
import torch.optim as torchopt
import torch.nn.functional as F

from utils.data import *
from utils.networks import *
from utils.savedir import *
from utils.seeding import *

from networks.baseNN import *
from networks.fullBNN import *
from networks.redBNN import *

from utils.lrp import *
from attacks.gradient_based import evaluate_attack
from attacks.run_attacks import *

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=500, type=int, help="Number of test points")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--topk", default=30, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--n_samples", default=100, type=int)
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--lrp_method", default="avg_heatmap", type=str, help="avg_prediction, avg_heatmap")
parser.add_argument("--load", type=eval, default="False", help="If True load dataframe else build it.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  

args = parser.parse_args()
lrp_robustness_method = "imagewise"
n_inputs=100 if args.debug else args.n_inputs

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Load models and attacks

m = baseNN_settings["model_"+str(args.model_idx)]

x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=m["dataset"], shuffle=False, n_inputs=n_inputs)[2:]
det_model_savedir = get_model_savedir(model="baseNN", dataset=m["dataset"], architecture=m["architecture"], 
                                            debug=args.debug, model_idx=args.model_idx)
detnet = baseNN(inp_shape, num_classes, *list(m.values()))
detnet.load(savedir=det_model_savedir, device=args.device)

det_attacks = load_attack(method=args.attack_method, model_savedir=det_model_savedir)

det_predictions, det_atk_predictions, det_softmax_robustness, det_successful_idxs, det_failed_idxs = \
            evaluate_attack(net=detnet, x_test=x_test, x_attack=det_attacks, y_test=y_test, 
                            device=args.device, return_classification_idxs=True)

m = fullBNN_settings["model_"+str(args.model_idx)]
x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], shuffle=False, n_inputs=n_inputs)[2:]

bay_model_savedir = get_model_savedir(model="fullBNN", dataset=m["dataset"], architecture=m["architecture"], 
                                                        model_idx=args.model_idx, debug=args.debug)

bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
bayesnet.load(savedir=bay_model_savedir, device=args.device)

bay_attacks = load_attack(method=args.attack_method, model_savedir=bay_model_savedir, n_samples=args.n_samples)

bay_predictions, bay_atk_predictions, bay_softmax_robustness, bay_successful_idxs, bay_failed_idxs = \
            evaluate_attack(net=bayesnet, n_samples=args.n_samples, x_test=x_test, x_attack=bay_attacks, y_test=y_test, 
                            device=args.device, return_classification_idxs=True)

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)

### Load or fill the dataframe

plot_savedir = os.path.join(bay_model_savedir, str(args.attack_method))
filename="rules_robustness_"+m["dataset"]+"_"+str(bayesnet.inference)+"_"+str(args.attack_method)+\
            "_images="+str(n_inputs)+"_samples="+str(args.n_samples)+"_topk="+str(args.topk)

if args.load:
    df = load_from_pickle(path=plot_savedir, filename=filename)

else:
    df = pd.DataFrame()
    rules_list = ['epsilon','gamma'] if m["architecture"] in ['conv','conv2'] else ['epsilon','gamma','alpha1beta0']

    for rule in rules_list: 
        for layer_idx in detnet.learnable_layers_idxs:

            savedir = get_lrp_savedir(model_savedir=det_model_savedir, attack_method=args.attack_method, rule=rule, 
                                        layer_idx=layer_idx)
            det_lrp = load_from_pickle(path=savedir, filename="det_lrp")
            det_attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

            savedir = get_lrp_savedir(model_savedir=bay_model_savedir, attack_method=args.attack_method, 
                                      rule=rule, layer_idx=layer_idx, lrp_method=args.lrp_method)
            bay_lrp = load_from_pickle(path=savedir, filename="bay_lrp_samp="+str(args.n_samples))
            bay_attack_lrp = load_from_pickle(path=savedir, filename="bay_attack_lrp_samp="+str(args.n_samples))

            det_robustness, det_pxl_idxs = lrp_robustness(original_heatmaps=det_lrp, adversarial_heatmaps=det_attack_lrp, 
                                          topk=args.topk, method=lrp_robustness_method)
            bay_robustness, bay_pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp, adversarial_heatmaps=bay_attack_lrp, 
                                          topk=args.topk, method=lrp_robustness_method)

            for robustness in det_robustness:
                df = df.append({'rule':rule, 'layer_idx':layer_idx, 'model':'Det.', 
                                'robustness':robustness}, ignore_index=True)

            for robustness in bay_robustness:
                df = df.append({'rule':rule, 'layer_idx':layer_idx, 'model':f'Bay. samp={args.n_samples}', 
                                'robustness':robustness}, ignore_index=True)

    save_to_pickle(data=df, path=plot_savedir, filename=filename)

### Plots

def plot_rules_robustness(df, n_samples, learnable_layers_idxs, savedir, filename):

    os.makedirs(savedir, exist_ok=True) 
    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'size': 10})

    det_col =  plt.cm.get_cmap('rocket', 100)(np.linspace(0, 1, 13))[3:]
    bay_col = plt.cm.get_cmap('crest', 100)(np.linspace(0, 1, 10))[3:]
    palettes = [det_col, bay_col]

    fig, ax = plt.subplots(len(learnable_layers_idxs), 2, figsize=(4.5, 5), sharex=True, sharey=True, dpi=150, 
                            facecolor='w', edgecolor='k') 
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)

    for col_idx, model in enumerate(list(df.model.unique())):
        palette = {"epsilon":palettes[col_idx][2], "gamma":palettes[col_idx][4], "alpha1beta0":palettes[col_idx][6]}

        for row_idx, layer_idx in enumerate(learnable_layers_idxs):

            temp_df = df[df['layer_idx']==layer_idx]
            temp_df = temp_df[temp_df['model']==model]

            sns.boxplot(data=temp_df, ax=ax[row_idx, col_idx], x='rule', y='robustness', orient='v', hue='rule', 
                        palette=palette, dodge=False)

            # for patch in ax[row_idx, col_idx].artists:
            #     r, g, b, a = patch.get_facecolor()
            #     patch.set_facecolor((r, g, b, .75))
            #     patch.set_edgecolor((r, g, b, a))

            for i, patch in enumerate(ax[row_idx, col_idx].artists):
                
                r, g, b, a = patch.get_facecolor()
                col = (r, g, b, a) 
                patch.set_facecolor((r, g, b, .7))
                patch.set_edgecolor(col)

                for j in range(i*6, i*6+6):
                    line = ax[row_idx, col_idx].lines[j]
                    line.set_color(col)
                    line.set_mfc(col)
                    line.set_mec(col)
                    # line.set_linewidth(0.5)

            ax[0, col_idx].xaxis.set_label_position("top")
            ax[0, col_idx].set_xlabel(model, weight='bold', size=10)
            ax[row_idx, col_idx].set_ylabel("")
            ax[1, 0].set_ylabel("LRP robustness")
            ax[row_idx, 1].yaxis.set_label_position("right")
            ax[row_idx, 1].set_ylabel("Layer idx="+str(layer_idx), rotation=270, labelpad=15, weight='bold', size=8)
            ax[row_idx, col_idx].get_legend().remove()
            ax[row_idx, col_idx].set_xlabel("")
            ax[2, col_idx].set_xlabel("LRP rule")

            ax[row_idx, col_idx].set_xticklabels([r'$\epsilon$',r'$\gamma$',r'$\alpha\beta$'])

            # print([t.get_text() for t in ax[row_idx, col_idx].get_xticklabels()])
            # ax[row_idx, col_idx].set_xticklabels([textwrap.fill(t.get_text(), 10)  for t in ax[row_idx, col_idx].get_xticklabels()])

    fig.subplots_adjust(left=0.15)
    # ax[0].legend(bbox_to_anchor=(0.7, 1.45))
    # ax[0].legend(loc="upper left")
    # plt.setp(ax[0].get_legend().get_texts(), fontsize='8')
    # plt.legend(frameon=False)
    # plt.subplots_adjust(hspace=0.1)
    print("\nSaving: ", os.path.join(savedir, filename+".png"))                                        
    fig.savefig(os.path.join(savedir, filename+".png"))
    plt.close(fig)

plot_rules_robustness(df=df,
                      n_samples=args.n_samples,
                      learnable_layers_idxs=detnet.learnable_layers_idxs,
                      savedir=plot_savedir,
                      filename=filename)
