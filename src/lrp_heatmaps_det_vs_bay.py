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
from plot.lrp_heatmaps import plot_heatmaps_det_vs_bay
from attacks.gradient_based import evaluate_attack
from attacks.run_attacks import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=500, type=int, help="Number of test points")
parser.add_argument("--model_idx", default=2, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--topk", default=100, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--n_samples", default=100, type=int)
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--lrp_method", default="avg_heatmap", type=str, help="avg_prediction, avg_heatmap")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--normalize", default=False, type=eval, help="Normalize lrp heatmaps.")
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

### Load explanations

layer_idx = detnet.learnable_layers_idxs[-1]

savedir = get_lrp_savedir(model_savedir=bay_model_savedir, attack_method=args.attack_method, 
                          layer_idx=layer_idx, lrp_method=args.lrp_method)

det_lrp = load_from_pickle(path=savedir, filename="det_lrp")
det_attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

bay_lrp = load_from_pickle(path=savedir, filename="bay_lrp_samp="+str(args.n_samples))
bay_attack_lrp = load_from_pickle(path=savedir, filename="bay_attack_lrp_samp="+str(args.n_samples))

if args.normalize:  
    for im_idx in range(det_lrp.shape[0]):
        det_lrp[im_idx] = normalize(det_lrp[im_idx])
        det_attack_lrp[im_idx] = normalize(det_attack_lrp[im_idx])
        bay_lrp[im_idx] = normalize(bay_lrp[im_idx])
        bay_attack_lrp[im_idx] = normalize(bay_attack_lrp[im_idx])

det_robustness, det_pxl_idxs = lrp_robustness(original_heatmaps=det_lrp, adversarial_heatmaps=det_attack_lrp, 
                              topk=args.topk, method=lrp_robustness_method)
bay_robustness, bay_pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp, adversarial_heatmaps=bay_attack_lrp, 
                              topk=args.topk, method=lrp_robustness_method)

### Select failed attack

set_seed(15)
shared_failed_idxs = np.intersect1d(det_failed_idxs, bay_failed_idxs)
shared_failed_idxs = shared_failed_idxs[np.where(bay_robustness[shared_failed_idxs]!=1.)]

im_idx = shared_failed_idxs[np.argmin(det_robustness[shared_failed_idxs])]
# im_idx = shared_failed_idxs[np.argmax(bay_robustness[shared_failed_idxs])]
# im_idx = np.random.choice(shared_failed_idxs, 1)

print("det LRP robustness =", det_robustness[im_idx])
print("bay LRP robustness =", bay_robustness[im_idx])

### Plots

savedir = get_lrp_savedir(model_savedir=bay_model_savedir, attack_method=args.attack_method, lrp_method=args.lrp_method)

filename=args.rule+"_heatmaps_det_vs_bay_"+m["dataset"]+"_topk="+str(args.topk)+"_failed_atk="+str(args.attack_method)

if args.normalize:
    filename+="_norm"

plot_heatmaps_det_vs_bay(image=images[im_idx].detach().cpu().numpy(), 
                         det_attack=det_attacks[im_idx].detach().cpu().numpy(), 
                         bay_attack=bay_attacks[im_idx].detach().cpu().numpy(), 
                         det_prediction=det_predictions[im_idx],
                         bay_prediction=bay_predictions[im_idx],
                         label=labels[im_idx],
                         det_explanation=det_lrp[im_idx],
                         det_attack_explanation=det_attack_lrp[im_idx],
                         bay_explanation=bay_lrp[im_idx],
                         bay_attack_explanation=bay_attack_lrp[im_idx],
                         # det_pxl_idxs=det_pxl_idxs[im_idx], 
                         # bay_pxl_idxs=bay_pxl_idxs[im_idx], 
                         lrp_rob_method=lrp_robustness_method, 
                         topk=args.topk, rule=args.rule, savedir=savedir, filename=filename)
