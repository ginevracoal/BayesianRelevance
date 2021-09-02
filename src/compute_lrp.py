import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn.functional as nnf
import torch.optim as torchopt
import torchvision

from networks.advNN import *
from networks.baseNN import *
from networks.fullBNN import *
from utils.data import *
from utils.model_settings import *
from utils.savedir import *
from utils.seeding import *

import plot.lrp_distributions as plot_lrp

from attacks.gradient_based import evaluate_attack
from attacks.run_attacks import *
from utils.lrp import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="baseNN", type=str, help="baseNN, fullBNN, advNN")
parser.add_argument("--n_inputs", default=500, type=int, help="Number of test points")
parser.add_argument("--n_samples", default=100, type=int)
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--lrp_method", default="avg_heatmap", type=str, help="avg_prediction, avg_heatmap")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation: epsilon, gamma, alpha1beta0")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

n_inputs=100 if args.debug else args.n_inputs
n_samples=5 if args.debug else args.n_samples

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if args.model in ["baseNN", "advNN"]:

    if args.model=="baseNN":

        model = baseNN_settings["model_"+str(args.model_idx)]

        x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=model["dataset"], shuffle=False, n_inputs=n_inputs)[2:]
        
        det_model_savedir = get_model_savedir(model="baseNN", dataset=model["dataset"], architecture=model["architecture"], 
                                          debug=args.debug, model_idx=args.model_idx)
        detnet = baseNN(inp_shape, num_classes, *list(model.values()))
        detnet.load(savedir=det_model_savedir, device=args.device)
        det_attack = load_attack(method=args.attack_method, model_savedir=det_model_savedir)

    elif args.model=="advNN":

        model = baseNN_settings["model_"+str(args.model_idx)]

        x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=model["dataset"], shuffle=False, n_inputs=n_inputs)[2:]

        det_model_savedir = get_model_savedir(model=args.model, dataset=model["dataset"], architecture=model["architecture"], 
                                    debug=args.debug, model_idx=args.model_idx, attack_method='fgsm')
        detnet = advNN(inp_shape, num_classes, *list(model.values()), attack_method='fgsm')
        detnet.load(savedir=det_model_savedir, device=args.device)

        det_attack = load_attack(method=args.attack_method, model_savedir=det_model_savedir)

    ### Deterministic explanations

    images = x_test.to(args.device).detach()
    labels = y_test.argmax(-1).to(args.device)

    for layer_idx in detnet.learnable_layers_idxs:

        savedir = get_lrp_savedir(model_savedir=det_model_savedir, attack_method=args.attack_method, 
                                  layer_idx=layer_idx, rule=args.rule)

        det_lrp = compute_explanations(images, detnet, layer_idx=layer_idx, rule=args.rule, device=args.device, 
                                        method=args.lrp_method)
        det_attack_lrp = compute_explanations(det_attack, detnet, layer_idx=layer_idx, rule=args.rule, device=args.device,
                                                method=args.lrp_method)

        save_to_pickle(det_lrp, path=savedir, filename="det_lrp")
        save_to_pickle(det_attack_lrp, path=savedir, filename="det_attack_lrp")

elif args.model=="fullBNN":

    m = fullBNN_settings["model_"+str(args.model_idx)]

    x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=m["dataset"], shuffle=False, n_inputs=n_inputs)[2:]

    bay_model_savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                                model_idx=args.model_idx, debug=args.debug)

    bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
    bayesnet.load(savedir=bay_model_savedir, device=args.device)

    bay_attack = load_attack(method=args.attack_method, model_savedir=bay_model_savedir, n_samples=n_samples)

    ### Bayesian explanations

    images = x_test.to(args.device).detach()
    labels = y_test.argmax(-1).to(args.device)

    for layer_idx in bayesnet.basenet.learnable_layers_idxs:

        savedir = get_lrp_savedir(model_savedir=bay_model_savedir, attack_method=args.attack_method, 
                                  layer_idx=layer_idx, rule=args.rule, lrp_method=args.lrp_method)

        bay_lrp = compute_explanations(images, bayesnet, rule=args.rule, layer_idx=layer_idx, 
                                            n_samples=n_samples, method=args.lrp_method, device=args.device)
        bay_attack_lrp = compute_explanations(bay_attack, bayesnet, layer_idx=layer_idx, 
                                device=args.device, rule=args.rule, n_samples=n_samples, method=args.lrp_method)

        save_to_pickle(bay_lrp, path=savedir, filename="bay_lrp_samp="+str(n_samples))
        save_to_pickle(bay_attack_lrp, path=savedir, filename="bay_attack_lrp_samp="+str(n_samples))
            
else:
    raise NotImplementedError