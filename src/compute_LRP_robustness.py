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
from attacks.gradient_based import *
from attacks.gradient_based import load_attack
from utils.lrp import *
from plot.lrp_heatmaps import *
import plot.lrp_distributions as plot_lrp

from networks.baseNN import *
from networks.fullBNN import *
from networks.redBNN import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=1000, type=int, help="Number of test points")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="fullBNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--n_samples", default=50, type=int, help="Number of bayesian samples.")
parser.add_argument("--layer_idx", default=-1, type=int, help="Layer idx for LRP computation.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--explain_attacks", default=False, type=eval)
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
args = parser.parse_args()

n_inputs=10 if args.debug else args.n_inputs
topk=3 if args.debug else 100
n_samples=2 if args.debug else args.n_samples

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Load models

model = baseNN_settings["model_"+str(args.model_idx)]

_, _, x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=model["dataset"], 
                                                        shuffle=True, n_inputs=n_inputs)
model_savedir = get_savedir(model="baseNN", dataset=model["dataset"], architecture=model["architecture"], 
                      debug=args.debug, model_idx=args.model_idx)
detnet = baseNN(inp_shape, num_classes, *list(model.values()))
detnet.load(savedir=model_savedir, device=args.device)

if args.model=="fullBNN":

    m = fullBNN_settings["model_"+str(args.model_idx)]

    model_savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                            model_idx=args.model_idx, debug=args.debug)

    bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)

# elif args.model=="redBNN":

#     m = redBNN_settings["model_"+str(args.model_idx)]
#     base_m = baseNN_settings["model_"+str(m["baseNN_idx"])]

#     _, _, x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=m["dataset"], 
#                                                                 n_inputs=n_inputs)

#     savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
#                           debug=args.debug, model_idx=args.model_idx)
#     basenet = baseNN(inp_shape, num_classes, *list(base_m.values()))
#     basenet_savedir = get_savedir(model="baseNN", dataset=m["dataset"], 
#                       architecture=m["architecture"], debug=args.debug, model_idx=m["baseNN_idx"])
#     basenet.load(savedir=basenet_savedir, device=args.device)

#     hyp = get_hyperparams(m)
#     net = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=basenet, hyperparams=hyp)

else:
    raise NotImplementedError

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)
savedir = os.path.join(model_savedir, "lrp/robustness/")

### Deterministic explanations

filename = args.rule+"_deterministic_explanations"

if args.load:
    det_lrp_robustness = load_from_pickle(path=savedir, filename=filename+"_lrp_robustness").to(args.device)
    pxl_idxs = load_from_pickle(path=savedir, filename=filename+"_lrp_robustness_pxl_idxs")

else:
    lrp = compute_explanations(images, detnet, rule=args.rule)
    lrp_attack = attack(net=detnet, x_test=lrp, y_test=y_test, savedir=savedir,
                      device=args.device, method=args.attack_method, filename=detnet.name, save=False)
    det_lrp_robustness, pxl_idxs = lrp_robustness(original_heatmaps=lrp, adversarial_heatmaps=lrp_attack, 
                                                  topk=topk)
    save_to_pickle(det_lrp_robustness, path=savedir, filename=filename+"_lrp_robustness")
    save_to_pickle(pxl_idxs, path=savedir, filename=filename+"_lrp_robustness_pxl_idxs")

### Bayesian explanations

post_filename = args.rule+"_posterior_explanations"

if args.load:
    bay_lrp_robustness = load_from_pickle(path=savedir, filename=post_filename+"_lrp_robustness").to(args.device)

else:
    post_lrp = compute_posterior_explanations(images, bayesnet, rule=args.rule, n_samples=n_samples)

    bay_lrp_robustness = []
    for sample_idx in range(n_samples):
        post_lrp_attack = attack(net=bayesnet, x_test=post_lrp[:,sample_idx], y_test=y_test, savedir=savedir,
                          device=args.device, method=args.attack_method, filename=bayesnet.name, 
                          n_samples=1, sample_idxs=[sample_idx], save=False)
        bay_lrp_robustness.append(lrp_robustness(original_heatmaps=post_lrp[:,sample_idx], pxl_idxs=pxl_idxs,
                                            adversarial_heatmaps=post_lrp_attack, topk=topk)[0])

    bay_lrp_robustness = torch.stack(bay_lrp_robustness)
    save_to_pickle(bay_lrp_robustness, path=savedir, filename=post_filename+"_lrp_robustness")

### Plot distributions

det_lrp_robustness = det_lrp_robustness.detach().cpu().numpy()
bay_lrp_robustness = bay_lrp_robustness.detach().cpu().numpy()
plot_lrp.lrp_robustness_distributions(lrp_robustness=det_lrp_robustness, 
        posterior_lrp_robustness=bay_lrp_robustness, savedir=savedir, filename=args.rule+"_lrp_robustness")