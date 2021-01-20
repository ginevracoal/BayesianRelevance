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
parser.add_argument("--topk", default=50, type=int, help="Top k most relevant pixels.")
parser.add_argument("--n_samples", default=10, type=int, help="Number of posterior samples.")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="fullBNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--layer_idx", default=-1, type=int, help="Layer idx for LRP computation.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--explain_attacks", default=False, type=eval)
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
args = parser.parse_args()

n_inputs=10 if args.debug else args.n_inputs
topk=3 if args.debug else args.topk
n_samples=2 if args.debug else args.n_samples

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Load models

model = baseNN_settings["model_"+str(args.model_idx)]

_, _, x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=model["dataset"], 
                                                            shuffle=False, n_inputs=n_inputs)
model_savedir = get_savedir(model="baseNN", dataset=model["dataset"], architecture=model["architecture"], 
                      debug=args.debug, model_idx=args.model_idx)
detnet = baseNN(inp_shape, num_classes, *list(model.values()))
detnet.load(savedir=model_savedir, device=args.device)

if args.model=="fullBNN":

    m = fullBNN_settings["model_"+str(args.model_idx)]

    model_savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                                model_idx=args.model_idx, debug=args.debug)

    bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
    bayesnet.load(savedir=model_savedir, device=args.device)

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
# ## todo: load net

else:
    raise NotImplementedError

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)
savedir = os.path.join(model_savedir, "lrp/robustness/")

atk_filename = args.attack_method+"_softmax_robustness"
lrp_filename = args.rule+"_lrp_robustness"

### Deterministic explanations

if args.load:
    det_softmax_robustness = load_from_pickle(path=savedir, filename="det_"+atk_filename)
    det_lrp_robustness = load_from_pickle(path=savedir, filename="det_"+lrp_filename)
    pxl_idxs = load_from_pickle(path=savedir, filename=lrp_filename+"_pxl_idxs")

else:
    x_attack = attack(net=detnet, x_test=images, y_test=y_test, savedir=savedir,
                        device=args.device, method=args.attack_method, filename=detnet.name, save=False)
    det_softmax_robustness = attack_evaluation(net=detnet, x_test=images, x_attack=x_attack, 
                                               y_test=y_test, device=args.device)[2]

    lrp = compute_explanations(images, detnet, rule=args.rule)
    lrp_attack = attack(net=detnet, x_test=lrp, y_test=y_test, savedir=savedir,
                      device=args.device, method=args.attack_method, filename=detnet.name, save=False)
    det_lrp_robustness, pxl_idxs = lrp_robustness(original_heatmaps=lrp, adversarial_heatmaps=lrp_attack, 
                                                  topk=topk)

    save_to_pickle(det_softmax_robustness, path=savedir, filename="det_"+atk_filename)
    save_to_pickle(det_lrp_robustness, path=savedir, filename="det_"+lrp_filename)
    save_to_pickle(pxl_idxs, path=savedir, filename=lrp_filename+"_pxl_idxs")

### Bayesian explanations

if args.load:
    bay_softmax_robustness = load_from_pickle(path=savedir, filename="bay_"+atk_filename)    
    bay_lrp_robustness = load_from_pickle(path=savedir, filename="bay_"+lrp_filename)

else:
    
    post_lrp = compute_posterior_explanations(images, bayesnet, rule=args.rule, n_samples=n_samples)

    bay_softmax_robustness = []
    bay_lrp_robustness = []

    for sample_idx in range(n_samples):

        x_attack = attack(net=bayesnet, x_test=images, y_test=y_test, savedir=savedir,
                          n_samples=1, sample_idxs=[sample_idx],
                          device=args.device, method=args.attack_method, filename=bayesnet.name, save=False)
        softmax_rob = attack_evaluation(net=bayesnet, x_test=images, x_attack=x_attack, sample_idxs=[sample_idx],
                                        y_test=y_test, device=args.device, n_samples=1)[2] 
        bay_softmax_robustness.append(softmax_rob)

        post_lrp_attack = attack(net=bayesnet, x_test=post_lrp[:,sample_idx], y_test=y_test, savedir=savedir,
                          device=args.device, method=args.attack_method, filename=bayesnet.name, 
                          n_samples=1, sample_idxs=[sample_idx], save=False)
        bay_lrp_robustness.append(lrp_robustness(original_heatmaps=post_lrp[:,sample_idx], pxl_idxs=pxl_idxs,
                                            adversarial_heatmaps=post_lrp_attack, topk=topk)[0])
    bay_lrp_robustness = torch.stack(bay_lrp_robustness)
    bay_softmax_robustness = torch.stack(bay_softmax_robustness)

    save_to_pickle(bay_softmax_robustness, path=savedir, filename="bay_"+atk_filename)
    save_to_pickle(bay_lrp_robustness, path=savedir, filename="bay_"+lrp_filename)

### Plot

det_softmax_robustness = det_softmax_robustness.detach().cpu().numpy()
bay_softmax_robustness = bay_softmax_robustness.detach().cpu().numpy()
det_lrp_robustness = det_lrp_robustness.detach().cpu().numpy()
bay_lrp_robustness = bay_lrp_robustness.detach().cpu().numpy()

filename=args.rule+"_lrp_robustness_"+m["dataset"]+"_images="+str(n_inputs)+\
         "_samples="+str(n_samples)+"_pxls="+str(topk)+"_atk="+str(args.attack_method)

plot_lrp.lrp_robustness_distributions(lrp_robustness=det_lrp_robustness, 
                                      bayesian_lrp_robustness=bay_lrp_robustness, 
                                      savedir=savedir, filename="dist_"+filename)

plot_lrp.lrp_robustness_scatterplot(adversarial_robustness=det_softmax_robustness, 
                                    bayesian_adversarial_robustness=bay_softmax_robustness,
                                    lrp_robustness=det_lrp_robustness, 
                                    bayesian_lrp_robustness=bay_lrp_robustness,
                                    savedir=savedir, filename="scatterplot_"+filename)