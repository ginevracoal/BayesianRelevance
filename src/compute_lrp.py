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
import attacks.gradient_based as grad_based
import attacks.deeprobust as deeprobust


parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=1000, type=int, help="Number of test points")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="fullBNN", type=str, help="fullBNN")
parser.add_argument("--attack_library", type=str, default="grad_based", help="grad_based, deeprobust")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--layer_idx", default=-1, type=int, help="Layer idx for LRP computation.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

n_samples_list=[1,5] if args.debug else [5,10,50]
n_inputs=100 if args.debug else args.n_inputs

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

atk_lib = eval(args.attack_library)
attack = atk_lib.attack
load_attack = atk_lib.load_attack

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Load models and attacks

model = baseNN_settings["model_"+str(args.model_idx)]

_, _, x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=model["dataset"], 
                                                            shuffle=False, n_inputs=n_inputs)
model_savedir = get_savedir(model="baseNN", dataset=model["dataset"], architecture=model["architecture"], 
                      debug=args.debug, model_idx=args.model_idx)
detnet = baseNN(inp_shape, num_classes, *list(model.values()))
detnet.load(savedir=model_savedir, device=args.device)

det_attack = load_attack(method=args.attack_method, filename=detnet.name, savedir=model_savedir)

if args.model=="fullBNN":

    m = fullBNN_settings["model_"+str(args.model_idx)]

    model_savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                                model_idx=args.model_idx, debug=args.debug)

    bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
    bayesnet.load(savedir=model_savedir, device=args.device)

    bay_attack=[]
    for n_samples in n_samples_list:

        bay_attack.append(load_attack(method=args.attack_method, filename=bayesnet.name, savedir=model_savedir, 
                                      n_samples=n_samples))

    mode_attack = load_attack(method=args.attack_method, filename=bayesnet.name+"_mode", savedir=model_savedir, 
                              n_samples=n_samples)

else:
    raise NotImplementedError

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)
savedir = os.path.join(model_savedir, "lrp/pkl/")

### Deterministic explanations

if args.load:
    det_lrp = load_from_pickle(path=savedir, filename="det_lrp")
    det_attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

else:

    det_lrp = compute_explanations(images, detnet, rule=args.rule)
    det_attack_lrp = compute_explanations(det_attack, detnet, rule=args.rule)

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
    
    mode_attack_lrp.append(load_from_pickle(path=savedir, filename="mode_lrp_avg_post_samp="+str(n_samples)))

else:

    for samp_idx, n_samples in enumerate(n_samples_list):

        bay_lrp.append(compute_explanations(images, bayesnet, rule=args.rule, n_samples=n_samples))
        bay_attack_lrp.append(compute_explanations(bay_attack[samp_idx], bayesnet, 
                                                   rule=args.rule, n_samples=n_samples))

        save_to_pickle(bay_lrp[samp_idx], path=savedir, filename="bay_lrp_samp="+str(n_samples))
        save_to_pickle(bay_attack_lrp[samp_idx], path=savedir, filename="bay_attack_lrp_samp="+str(n_samples))
    
    mode_lrp = compute_explanations(images, bayesnet, rule=args.rule, n_samples=n_samples, avg_posterior=True)
    save_to_pickle(mode_lrp, path=savedir, filename="mode_lrp_avg_post_samp="+str(n_samples))

    for samp_idx, n_samples in enumerate(n_samples_list):
        mode_attack_lrp.append(compute_explanations(mode_attack, bayesnet, rule=args.rule, n_samples=n_samples))
        save_to_pickle(mode_attack_lrp[samp_idx], path=savedir, filename="mode_attack_lrp_samp="+str(n_samples))

    mode_attack_lrp.append(compute_explanations(mode_attack, bayesnet, rule=args.rule, 
                                                n_samples=n_samples, avg_posterior=True))
    save_to_pickle(mode_attack_lrp[samp_idx+1], path=savedir, filename="mode_lrp_avg_post_samp"+str(n_samples))

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
