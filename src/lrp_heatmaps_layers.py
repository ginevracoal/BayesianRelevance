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
from plot.lrp_heatmaps import plot_attacks_explanations_layers
from attacks.gradient_based import evaluate_attack
from attacks.run_attacks import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=500, type=int, help="Number of test points")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--topk", default=500, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="baseNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--n_samples", default=50, type=int)
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--lrp_method", default="avg_heatmap", type=str, help="avg_prediction, avg_heatmap")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--normalize", default=True, type=eval, help="Normalize lrp heatmaps.")
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

if args.model=="baseNN":

    m = baseNN_settings["model_"+str(args.model_idx)]

    x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=m["dataset"], 
                                                                shuffle=False, n_inputs=n_inputs)[2:]
    model_savedir = get_model_savedir(model="baseNN", dataset=m["dataset"], architecture=m["architecture"], 
                                                debug=args.debug, model_idx=args.model_idx)
    detnet = baseNN(inp_shape, num_classes, *list(m.values()))
    detnet.load(savedir=model_savedir, device=args.device)

    attacks = load_attack(method=args.attack_method, model_savedir=model_savedir)

    predictions, atk_predictions, softmax_robustness, successful_idxs, failed_idxs = evaluate_attack(net=detnet, 
                x_test=x_test, x_attack=attacks, y_test=y_test, device=args.device, return_classification_idxs=True)

    learnable_layers_idxs = detnet.learnable_layers_idxs

elif args.model=="fullBNN":

    m = fullBNN_settings["model_"+str(args.model_idx)]
    x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], shuffle=False, n_inputs=n_inputs)[2:]

    model_savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                                                            model_idx=args.model_idx, debug=args.debug)

    bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
    bayesnet.load(savedir=model_savedir, device=args.device)

    attacks = load_attack(method=args.attack_method, model_savedir=model_savedir, n_samples=args.n_samples)

    predictions, atk_predictions, softmax_robustness, successful_idxs, failed_idxs = evaluate_attack(net=bayesnet, 
                                    n_samples=n_samples, x_test=x_test, x_attack=attacks, y_test=y_test, 
                                    device=args.device, return_classification_idxs=True)

    learnable_layers_idxs = bayesnet.learnable_layers_idxs

else:
    raise NotImplementedError

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)

explanations_layers = []
attacks_explanations_layers = []
pxl_idxs_layers=[]

for layer_idx in detnet.learnable_layers_idxs:

    ### Load explanations

    if args.model=="baseNN":

        savedir = get_lrp_savedir(model_savedir=model_savedir, attack_method=args.attack_method, 
                                  layer_idx=layer_idx)
        lrp = load_from_pickle(path=savedir, filename="det_lrp")
        attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

    else:

        savedir = get_lrp_savedir(model_savedir=model_savedir, attack_method=args.attack_method, 
                                  layer_idx=layer_idx, lrp_method=args.lrp_method)
        lrp.append(load_from_pickle(path=savedir, filename="bay_lrp_samp="+str(n_samples)))
        attack_lrp.append(load_from_pickle(path=savedir, filename="bay_attack_lrp_samp="+str(n_samples)))

    if args.normalize:  
        for im_idx in range(lrp.shape[0]):
            lrp[im_idx] = normalize(lrp[im_idx])
            attack_lrp[im_idx] = normalize(attack_lrp[im_idx])

    pxl_idxs = lrp_robustness(original_heatmaps=lrp, adversarial_heatmaps=attack_lrp, 
                              topk=args.topk, method=lrp_robustness_method)[1]

    explanations_layers.append(lrp.detach().cpu().numpy())
    attacks_explanations_layers.append(attack_lrp.detach().cpu().numpy())
    pxl_idxs_layers.append(pxl_idxs)

explanations_layers=np.array(explanations_layers)
attacks_explanations_layers=np.array(attacks_explanations_layers)
pxl_idxs_layers=np.array(pxl_idxs_layers)

### Plots

lrp_method=None if args.model=="baseNN" else args.lrp_method
savedir = get_lrp_savedir(model_savedir=model_savedir, attack_method=args.attack_method, lrp_method=lrp_method)

filename=args.rule+"_layers_heatmaps_"+m["dataset"]+"_images="+str(n_inputs)+\
         "_samples="+str(args.n_samples)+"_atk="+str(args.attack_method)
if args.normalize:
    filename+="_norm"

plot_attacks_explanations_layers(images=images, 
                                 attacks=attacks, 
                                 explanations=explanations_layers, 
                                 attacks_explanations=attacks_explanations_layers, 
                                 predictions=predictions, 
                                 attacks_predictions=atk_predictions, 
                                 successful_attacks_idxs=successful_idxs, 
                                 failed_attacks_idxs=failed_idxs,
                                 labels=labels, 
                                 pxl_idxs=pxl_idxs_layers, 
                                 learnable_layers_idxs=learnable_layers_idxs,
                                 lrp_rob_method=lrp_robustness_method, 
                                 rule=args.rule, savedir=savedir, filename=filename)
