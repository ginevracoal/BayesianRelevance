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

from utils.lrp import *
from attacks.gradient_based import evaluate_attack
from attacks.run_attacks import *

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from plot.lrp_distributions import significance_symbol
from scipy.stats import mannwhitneyu as stat_test


parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=500, type=int, help="Number of test points")
parser.add_argument("--topk", default=20, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--n_samples", default=100, type=int)
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--lrp_method", default="avg_heatmap", type=str, help="avg_prediction, avg_heatmap")
parser.add_argument("--load", type=eval, default="False", help="If True load dataframe else build it.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  

args = parser.parse_args()
lrp_robustness_method = "imagewise"
n_inputs=100 if args.debug else args.n_inputs
alternative = 'less'

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

learnable_layers_idxs = [38]

det_savedir = '../experiments/baseNN/cifar_resnet/'
det_attacks = load_attack(method=args.attack_method, model_savedir=det_savedir)

adv_savedir = '../experiments/advNN/cifar_resnet_atk=fgsm/'
adv_attacks = load_attack(method=args.attack_method, model_savedir=adv_savedir)

bay_savedir = '../experiments/fullBNN/cifar_resnet/'
bay_attacks = load_attack(method=args.attack_method, model_savedir=bay_savedir, n_samples=args.n_samples)

plot_savedir = os.path.join(bay_savedir, str(args.attack_method))
filename="rules_robustness_cifar_svi_"+str(args.attack_method)+"_images="+str(n_inputs)\
            +"_samples="+str(args.n_samples)+"_topk="+str(args.topk)

if args.load:
    df = load_from_pickle(path=plot_savedir, filename=filename)

else:
    df = pd.DataFrame()
    rules_list = ['epsilon','gamma','alpha1beta0']

    for rule in rules_list: 
        for layer_idx in learnable_layers_idxs:

            savedir = get_lrp_savedir(model_savedir=det_savedir, attack_method=args.attack_method, 
                                    layer_idx=layer_idx, rule=rule)
            det_lrp = load_from_pickle(path=savedir, filename="det_lrp")
            det_attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

            savedir = get_lrp_savedir(model_savedir=adv_savedir, attack_method=args.attack_method, 
                                    layer_idx=layer_idx, rule=rule)
            adv_lrp = load_from_pickle(path=savedir, filename="det_lrp")
            adv_attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

            savedir = get_lrp_savedir(model_savedir=bay_savedir, attack_method=args.attack_method, 
                                        layer_idx=layer_idx, rule=rule)
            bay_lrp = load_from_pickle(path=savedir, filename="bay_lrp_samp="+str(args.n_samples))
            bay_attack_lrp = load_from_pickle(path=savedir, filename="bay_attack_lrp_samp="+str(args.n_samples))

            det_robustness, det_pxl_idxs = lrp_robustness(original_heatmaps=det_lrp, adversarial_heatmaps=det_attack_lrp, 
                                          topk=args.topk, method=lrp_robustness_method)
            adv_robustness, adv_pxl_idxs = lrp_robustness(original_heatmaps=adv_lrp, adversarial_heatmaps=adv_attack_lrp, 
                                          topk=args.topk, method=lrp_robustness_method)
            bay_robustness, bay_pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp, adversarial_heatmaps=bay_attack_lrp, 
                                          topk=args.topk, method=lrp_robustness_method)

            _, adv_p = stat_test(x=det_robustness, y=adv_robustness, alternative=alternative)
            _, bay_p = stat_test(x=det_robustness, y=bay_robustness, alternative=alternative)
            _, p = stat_test(x=adv_robustness, y=bay_robustness, alternative=alternative)
            print("\np values =", adv_p, bay_p, p)  

            for im_idx in range(len(det_robustness)):

                df = df.append({'rule':rule, 'layer_idx':layer_idx, 'model':'Adv - Det', 
                                'robustness_diff':adv_robustness[im_idx]-det_robustness[im_idx],
                                'p_value':p, 'adv_p_value':adv_p, 'bay_p_value':bay_p},
                                ignore_index=True)

                df = df.append({'rule':rule, 'layer_idx':layer_idx, 'model':f'Bay - Det', #samp={args.n_samples}', 
                                'robustness_diff':bay_robustness[im_idx]-det_robustness[im_idx],
                                'p_value':p, 'adv_p_value':adv_p, 'bay_p_value':bay_p},
                                ignore_index=True)

    save_to_pickle(data=df, path=plot_savedir, filename=filename)

### Plots

def plot_rules_robustness(df, n_samples, learnable_layers_idxs, savedir, filename):

    print(df.head())

    os.makedirs(savedir, exist_ok=True) 
    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'size': 9})

    det_col =  plt.cm.get_cmap('rocket', 100)(np.linspace(0, 1, 13))[3:]
    bay_col = plt.cm.get_cmap('crest', 100)(np.linspace(0, 1, 10))[3:]
    palettes = [det_col, bay_col]

    fig, ax = plt.subplots(len(learnable_layers_idxs), 2, figsize=(3, 2), sharex=True, sharey=True, dpi=150, 
                            facecolor='w', edgecolor='k') 
    fig.tight_layout()

    for col_idx, model in enumerate(list(df['model'].unique())):
        palette = {"epsilon":palettes[col_idx][2], "gamma":palettes[col_idx][4], "alpha1beta0":palettes[col_idx][6]}

        for row_idx, layer_idx in enumerate(learnable_layers_idxs):

            temp_df = df[df['layer_idx']==layer_idx]
            y = min(temp_df['robustness_diff'])*1.2

            temp_df = temp_df[temp_df['model']==model]

            sns.boxplot(data=temp_df, ax=ax[col_idx], x='rule', y='robustness_diff', orient='v', hue='rule', 
                        palette=palette, dodge=False, flierprops={'markersize':3})

            for rule, x in zip(temp_df['rule'].unique(), [-0.3, 0.7, 1.7]):
                rule_df = temp_df[temp_df['rule']==rule]

                p_value = rule_df['p_value'].unique()[0]
                assert len(rule_df['p_value'].unique())==1
                significance = significance_symbol(p_value)
                # if significance!='n.s.':
                #     y = min(temp_df['robustness_diff'])-0.12
                ax[col_idx].text(x=x, y=y, s=significance, weight='bold', size=8, color=palette[rule])

            for i, patch in enumerate(ax[col_idx].artists):
                
                r, g, b, a = patch.get_facecolor()
                col = (r, g, b, a) 
                patch.set_facecolor((r, g, b, .7))
                patch.set_edgecolor(col)

                for j in range(i*6, i*6+6):
                    line = ax[col_idx].lines[j]
                    line.set_color(col)
                    line.set_mfc(col)
                    line.set_mec(col)

            ax[col_idx].xaxis.set_label_position("top")
            ax[col_idx].set_xlabel(model, weight='bold', size=9)
            ax[col_idx].set_ylabel("")
            ax[0].set_ylabel("LRP robustness diff.")
            ax[1].yaxis.set_label_position("right")
            ax[1].set_ylabel("Layer idx="+str(layer_idx), rotation=270, labelpad=10, weight='bold', size=9)
            ax[col_idx].get_legend().remove()
            ax[col_idx].set_xlabel("")
            ax[0].set_xlabel("Adv - Det")
            ax[1].set_xlabel("Bay - Det")
            ax[col_idx].text(x=0.4, y=-0.48, s="LRP rule", weight='bold')
            ax[col_idx].set_xticklabels([r'$\epsilon$',r'$\gamma$',r'$\alpha\beta$'])

    plt.subplots_adjust(hspace=0.05)
    plt.subplots_adjust(wspace=0.05)
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(bottom=0.2)
    print("\nSaving: ", os.path.join(savedir, filename+".png"))                                        
    fig.savefig(os.path.join(savedir, filename+".png"))
    plt.close(fig)

plot_rules_robustness(df=df,
                      n_samples=args.n_samples,
                      learnable_layers_idxs=learnable_layers_idxs,
                      savedir=os.path.join(TESTS,'figures/rules_robustness'),
                      filename=filename)
