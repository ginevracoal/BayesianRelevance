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

from utils.lrp import *
from plot.lrp_heatmaps import plot_vanishing_explanations
import plot.lrp_distributions as plot_lrp
from attacks.gradient_based import evaluate_attack
from attacks.run_attacks import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=100, type=int, help="Number of test points")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="fullBNN", type=str, help="fullBNN")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--layer_idx", default=-1, type=int, help="Layer idx for LRP computation.")
parser.add_argument("--redBNN_layer_idx", default=-1, type=int, help="Bayesian layer idx in redBNN.")
parser.add_argument("--normalize", default=False, type=eval, help="Normalize lrp heatmaps.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

n_inputs=100 if args.debug else args.n_inputs
n_samples_list=[2,10,50]

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Load models and attacks

model = baseNN_settings["model_"+str(args.model_idx)]

x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=model["dataset"], shuffle=False, n_inputs=n_inputs)[2:]
model_savedir = get_model_savedir(model="baseNN", dataset=model["dataset"], architecture=model["architecture"], 
                                  debug=args.debug, model_idx=args.model_idx)
detnet = baseNN(inp_shape, num_classes, *list(model.values()))
detnet.load(savedir=model_savedir, device=args.device)
det_attack = load_attack(method=args.attack_method, model_savedir=model_savedir)

if args.model=="fullBNN":

    m = fullBNN_settings["model_"+str(args.model_idx)]

    model_savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                                model_idx=args.model_idx, debug=args.debug)

    bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
    bayesnet.load(savedir=model_savedir, device=args.device)


elif args.model=="redBNN":

    m = redBNN_settings["model_"+str(args.model_idx)]
    base_m = baseNN_settings["model_"+str(m["baseNN_idx"])]

    x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], shuffle=False, n_inputs=n_inputs)[2:]

    basenet = baseNN(inp_shape, out_size, *list(base_m.values()))
    basenet_savedir = get_model_savedir(model="baseNN", dataset=m["dataset"], 
                      architecture=m["architecture"], debug=args.debug, model_idx=m["baseNN_idx"])
    basenet.load(savedir=basenet_savedir, device=args.device)

    hyp = get_hyperparams(m)

    layer_idx=args.redBNN_layer_idx+basenet.n_learnable_layers+1 if args.redBNN_layer_idx<0 else args.redBNN_layer_idx
    bayesnet = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=basenet, hyperparams=hyp,
                      layer_idx=layer_idx)
    model_savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                          debug=args.debug, model_idx=args.model_idx, layer_idx=layer_idx)
    bayesnet.load(savedir=model_savedir, device=args.device)

else:
    raise NotImplementedError

bay_attack=[]
for n_samples in n_samples_list:

    bay_attack.append(load_attack(method=args.attack_method, model_savedir=model_savedir, 
                                  n_samples=n_samples))

mode_attack = load_attack(method=args.attack_method, model_savedir=model_savedir, 
                          n_samples=n_samples, atk_mode=True)

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)

layer_idx=args.layer_idx+detnet.n_layers+1 if args.layer_idx<0 else args.layer_idx
savedir = get_lrp_savedir(model_savedir=model_savedir, attack_method=args.attack_method, 
                          layer_idx=layer_idx, normalize=args.normalize)

### Deterministic explanations

if args.load:
    det_lrp = load_from_pickle(path=savedir, layer_idx=layer_idx, filename="det_lrp")
    det_attack_lrp = load_from_pickle(path=savedir, layer_idx=layer_idx, filename="det_attack_lrp")

else:

    det_lrp = compute_explanations(images, detnet, layer_idx=layer_idx, rule=args.rule, normalize=args.normalize)
    det_attack_lrp = compute_explanations(det_attack, detnet, layer_idx=layer_idx, rule=args.rule, normalize=args.normalize)

    save_to_pickle(det_lrp, path=savedir, filename="det_lrp")
    save_to_pickle(det_attack_lrp, path=savedir, filename="det_attack_lrp")


### Bayesian explanations

bay_lrp=[]
bay_attack_lrp=[]
mode_attack_lrp=[]

if args.load:

    for n_samples in n_samples_list:
        bay_lrp.append(load_from_pickle(path=savedir, filename="bay_lrp_samp="+str(n_samples)))
        bay_attack_lrp.append(load_from_pickle(path=savedir, filename="bay_attack_lrp_samp="+str(n_samples)))

    mode_lrp = load_from_pickle(path=savedir, filename="mode_lrp_avg_post_samp="+str(n_samples))

    for samp_idx, n_samples in enumerate(n_samples_list):
        mode_attack_lrp.append(load_from_pickle(path=savedir, filename="mode_attack_lrp_samp="+str(n_samples)))
    
    mode_attack_lrp.append(load_from_pickle(path=savedir, filename="mode_attack_lrp_avg_post_samp="+str(n_samples)))

else:

    for samp_idx, n_samples in enumerate(n_samples_list):

        bay_lrp.append(compute_explanations(images, bayesnet, rule=args.rule, layer_idx=layer_idx, 
                                                n_samples=n_samples, normalize=args.normalize))
        bay_attack_lrp.append(compute_explanations(bay_attack[samp_idx], bayesnet, layer_idx=layer_idx,
                                                   rule=args.rule, n_samples=n_samples, normalize=args.normalize))

        save_to_pickle(bay_lrp[samp_idx], path=savedir, filename="bay_lrp_samp="+str(n_samples))
        save_to_pickle(bay_attack_lrp[samp_idx], path=savedir, filename="bay_attack_lrp_samp="+str(n_samples))
    
    mode_lrp = compute_explanations(images, bayesnet, rule=args.rule, layer_idx=layer_idx, 
                                    n_samples=n_samples, avg_posterior=True, normalize=args.normalize)
    save_to_pickle(mode_lrp, path=savedir, filename="mode_lrp_avg_post_samp="+str(n_samples))

    for samp_idx, n_samples in enumerate(n_samples_list):
        mode_attack_lrp.append(compute_explanations(mode_attack, bayesnet, rule=args.rule, layer_idx=layer_idx, 
                                                        n_samples=n_samples, normalize=args.normalize))
        save_to_pickle(mode_attack_lrp[samp_idx], path=savedir, filename="mode_attack_lrp_samp="+str(n_samples))

    mode_attack_lrp.append(compute_explanations(mode_attack, bayesnet, rule=args.rule, layer_idx=layer_idx,
                                                n_samples=n_samples, avg_posterior=True, normalize=args.normalize))
    save_to_pickle(mode_attack_lrp[samp_idx+1], path=savedir, filename="mode_attack_lrp_avg_post_samp="+str(n_samples))

### plots

# plot_vanishing_explanations(images_plt, samples_explanations, n_samples_list=bayesian_samples,
#     rule=args.rule, savedir=savedir, filename=args.rule+"_vanishing_explanations")
# plot_lrp.stripplot_lrp_values(samples_explanations, n_samples_list=bayesian_samples, 
#                  savedir=savedir, filename=args.rule+"_explanations_components")

# plot_lrp.lrp_pixels_distributions(post_explanations, labels=labels_post_exp, num_classes=num_classes, 
#                          n_samples=post_samples, savedir=savedir, filename=filename+"_lrp_pixel_distr")
# plot_lrp.lrp_samples_distributions(samples_explanations, labels=labels_plt, num_classes=num_classes,
#                 n_samples_list=bayesian_samples, savedir=savedir, filename=filename+"_lrp_pixel_distr")
# plot_lrp.lrp_labels_distributions(samples_explanations, labels=labels_plt, num_classes=num_classes,
#                 n_samples_list=bayesian_samples, savedir=savedir, 
#                 filename=filename+"_lrp_pixel_distr")
