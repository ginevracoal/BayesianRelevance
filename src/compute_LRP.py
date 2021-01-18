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
from plot.lrp_distributions import lrp_labels_distributions, lrp_samples_distributions, lrp_pixels_distributions


parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=1000, type=int, help="number of input points")
parser.add_argument("--model_idx", default=0, type=int, help="choose model idx from pre defined settings")
parser.add_argument("--model", default="baseNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--rule", default="epsilon", type=str)
parser.add_argument("--n_samples", default=10, type=int)
parser.add_argument("--load", default=False, type=eval)
parser.add_argument("--explain_attacks", default=False, type=eval)
parser.add_argument("--debug", default=False, type=eval)
parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
args = parser.parse_args()

bayesian_samples=[1] if args.debug else [1, 10, 50]
n_inputs=100 if args.debug else args.n_inputs

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


if args.model=="baseNN":

    model = baseNN_settings["model_"+str(args.model_idx)]

    _, _, x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=model["dataset"], 
                                                            shuffle=True, n_inputs=n_inputs)
    savedir = get_savedir(model=args.model, dataset=model["dataset"], architecture=model["architecture"], 
                          debug=args.debug, model_idx=args.model_idx)

    images = x_test.to(args.device)
    net = baseNN(inp_shape, num_classes, *list(model.values()))
    net.load(savedir=savedir, device=args.device)

    if args.load:
        explanations = load_lrp(path=savedir, filename=args.rule+"_explanations")

    else:
        explanations = compute_explanations(images, net, rule=args.rule)
        save_lrp(explanations, path=savedir, filename=args.rule+"_explanations")

    images = images.detach().cpu().numpy()
    plot_explanations(images, explanations, rule=args.rule, savedir=savedir,
                     filename=args.rule+"_explanations")

    if args.explain_attacks:

        if args.load:
            attacks_explanations = load_lrp(path=savedir, filename=args.rule+"_attacks_explanations")
    
        else:
            attacks = load_attack(method=args.attack_method, filename=net.name, savedir=savedir)
            attacks = attacks[:args.n_inputs].detach().to(args.device)
            attacks_explanations = compute_explanations(attacks, net, rule=args.rule)
            save_lrp(attacks_explanations, path=savedir, filename=args.rule+"_attacks_explanations")
            attacks = attacks.detach().cpu().numpy()
            plot_attacks_explanations(images, explanations, attacks, attacks_explanations,
                                rule=args.rule, savedir=savedir, filename=args.rule+"_attacks_explanations")

else:

    if args.model=="fullBNN":

        m = fullBNN_settings["model_"+str(args.model_idx)]

        _, _, x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=m["dataset"], 
                                                                    shuffle=True, n_inputs=n_inputs)
        savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                                model_idx=args.model_idx, debug=args.debug)
        plots_savedir = os.path.join(savedir,"lrp")

        net = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
   
    # elif args.model=="redBNN":

    #     m = redBNN_settings["model_"+str(args.model_idx)]
    #     _, _, x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=m["dataset"], 
    #                                                                          n_inputs=n_inputs)
    #     basenet = baseNN(dataset_name=m["dataset"], input_shape=inp_shape, output_size=num_classes,
    #               epochs=m["baseNN_epochs"], lr=m["baseNN_lr"], hidden_size=m["hidden_size"], 
    #               activation=m["activation"], architecture=m["architecture"])        
    #     basenet.load(rel_path=rel_path, device=args.device)

    #     hyp = get_hyperparams(m)
    #     net = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=basenet, hyperparams=hyp)

    else:
        raise NotImplementedError

    images = x_test.to(args.device)
    labels = y_test.to(args.device)
    images_plt = images.detach().cpu().numpy()
    labels_plt = labels.argmax(-1).detach().cpu().numpy()
    net.load(savedir=savedir, device=args.device)

    if args.load:
        samples_explanations = load_lrp(path=savedir, filename=args.rule+"_explanations")

        if args.explain_attacks:
            samples_attacks_explanations = load_lrp(path=savedir, filename=args.rule+"_attacks_explanations")

    else:

        samples_explanations=[]
        samples_attacks_explanations=[]
        
        for n_samples in bayesian_samples:

            explanations = compute_explanations(images, net, rule=args.rule, n_samples=n_samples)
            samples_explanations.append(explanations)

            if args.explain_attacks:
                attacks = load_attack(method=args.attack_method, filename=net.name, savedir=savedir,
                                        n_samples=n_samples)
                attacks = attacks[:args.n_inputs].detach().to(args.device)
                attacks_explanations = compute_explanations(attacks, net, rule=args.rule, n_samples=n_samples)
                attacks_plt = attacks.detach().cpu().numpy()

                plot_explanations(attacks_plt, attacks_explanations, rule=args.rule, savedir=savedir, 
                                                    filename=args.rule+"_atk_explanations_"+str(n_samples))
                plot_attacks_explanations(images, explanations, attacks, attacks_explanations,
                                            rule=args.rule, savedir=savedir, filename=args.rule+"_explanations")
                samples_attacks_explanations.append(attacks_explanations)
        
        samples_explanations = np.array(samples_explanations)
        save_lrp(samples_explanations, path=savedir, filename=args.rule+"_explanations")

        if args.explain_attacks:
            save_lrp(samples_attacks_explanations, path=savedir, 
                            filename=args.rule+"_attacks_explanations")

    plot_vanishing_explanations(images_plt, samples_explanations, n_samples_list=bayesian_samples,
        rule=args.rule, savedir=savedir, filename=args.rule+"_vanishing_explanations")
    # stripplot_lrp_values(samples_explanations, n_samples_list=bayesian_samples, 
    #                  savedir=savedir, filename=args.rule+"_explanations_components")

    # lrp_samples_distributions(samples_explanations, labels=labels_plt, num_classes=num_classes,
    #                 n_samples_list=bayesian_samples, savedir=savedir, filename=args.rule+"_lrp_pixel_distr")
    lrp_labels_distributions(samples_explanations, labels=labels_plt, num_classes=num_classes,
                    n_samples_list=bayesian_samples, savedir=savedir, 
                    filename=args.rule+"_lrp_pixel_distr")

    if args.explain_attacks:
        samples_attacks_explanations = np.array(samples_attacks_explanations)
        plot_vanishing_explanations(attacks_plt, samples_attacks_explanations, n_samples_list=bayesian_samples,
            rule=args.rule, savedir=savedir, filename=args.rule+"_vanishing_explanations")
