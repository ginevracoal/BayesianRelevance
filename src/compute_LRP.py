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


parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=100, type=int, help="number of input points")
parser.add_argument("--model_idx", default=0, type=int, help="choose model idx from pre defined settings")
parser.add_argument("--model_type", default="baseNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--rule", default="epsilon", type=str)
parser.add_argument("--n_samples", default=10, type=int)
parser.add_argument("--train", default=True, type=eval)
parser.add_argument("--attack", default=True, type=eval)
parser.add_argument("--debug", default=False, type=eval)
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

bayesian_samples=[1, 10, 50]
n_inputs=args.n_inputs
rel_path=TESTS

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


if args.model_type=="baseNN":

    model = baseNN_settings["model_"+str(args.model_idx)]

    _, _, x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=model["dataset"], 
                                                                         n_inputs=n_inputs)
    
    images = x_test[:args.n_inputs].to(args.device)
    net = baseNN(inp_shape, out_size, *list(model.values()))
    net.load(device=args.device, rel_path=rel_path)

    attacks = load_attack(method=args.attack_method, filename=net.name, rel_path=rel_path)
    attacks = attacks[:args.n_inputs].detach().to(args.device)

    explanations = compute_explanations(images, net, rule=args.rule)
    attacks_explanations = compute_explanations(attacks, net, rule=args.rule)

    images = images.detach().cpu().numpy()
    attacks = attacks.detach().cpu().numpy()

    # plot_explanations(images, explanations, rule=args.rule, savedir=rel_path+net.name+"/", 
    #                                                 filename=args.rule+"_explanations.png")
    plot_attacks_explanations(images, explanations, attacks, attacks_explanations,
                                rule=args.rule, savedir=rel_path+net.name+"/", 
                                filename=args.rule+"_explanations.png")

else:

    if args.model_type=="fullBNN":

        m = fullBNN_settings["model_"+str(args.model_idx)]

        _, _, x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], 
                                                                             n_inputs=n_inputs)

        net = BNN(m["dataset"], *list(m.values())[1:], inp_shape, out_size)
   
    # elif args.model_type=="redBNN":

    #     m = redBNN_settings["model_"+str(args.model_idx)]
    #     _, _, x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], 
    #                                                                          n_inputs=n_inputs)
    #     basenet = baseNN(dataset_name=m["dataset"], input_shape=inp_shape, output_size=out_size,
    #               epochs=m["baseNN_epochs"], lr=m["baseNN_lr"], hidden_size=m["hidden_size"], 
    #               activation=m["activation"], architecture=m["architecture"])        
    #     basenet.load(rel_path=rel_path, device=args.device)

    #     hyp = get_hyperparams(m)
    #     net = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=basenet, hyperparams=hyp)

    else:
        raise NotImplementedError

    images = x_test[:args.n_inputs].to(args.device)
    net.load(device=args.device, rel_path=rel_path)

    samples_explanations=[]
    
    for n_samples in bayesian_samples:
        attacks = load_attack(method=args.attack_method, filename=net.name, 
                                n_samples=n_samples, rel_path=rel_path)
        attacks = attacks[:args.n_inputs].detach().to(args.device)

        explanations = compute_explanations(images, net, rule=args.rule, n_samples=n_samples)
        attacks_explanations = compute_explanations(attacks, net, rule=args.rule, n_samples=n_samples)

        images_plt = images.detach().cpu().numpy()
        attacks_plt = attacks.detach().cpu().numpy()

        # plot_explanations(images_plt, explanations, rule=args.rule, savedir=rel_path+net.name+"/", 
        #                                     filename=args.rule+"_explanations_"+str(n_samples)+".png")
        # plot_explanations(attacks_plt, attacks_explanations, rule=args.rule, savedir=rel_path+net.name+"/", 
        #                                     filename=args.rule+"_atk_explanations_"+str(n_samples)+".png")
        # plot_attacks_explanations(images, explanations, attacks, attacks_explanations,
        #                             rule=args.rule, savedir=rel_path+net.name+"/", 
        #                             filename=args.rule+"_explanations.png")

        samples_explanations.append(explanations)

    samples_explanations = np.array(samples_explanations)

    plot_vanishing_explanations(images_plt, samples_explanations, n_samples_list=bayesian_samples,
        rule=args.rule, savedir=rel_path+net.name+"/", filename=args.rule+"_vanishing_explanations.png")
