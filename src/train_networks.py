import os
import torch
import argparse
import numpy as np

from utils.data import *
from utils import savedir
from utils.seeding import *
import attacks.gradient_based as grad_based
import attacks.deeprobust as deeprobust

from networks.baseNN import *
from networks.fullBNN import *
from networks.advNN import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="baseNN", type=str, help="baseNN, fullBNN, advNN")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
# parser.add_argument("--attack_iters", default=3, type=int, help="Number of iterations in iterative attacks.")
# parser.add_argument("--epsilon", default=0.2, type=int, help="Strength of a perturbation.")
# parser.add_argument("--attack_lrp_rule", default='epsilon', type=str, help="LRP rule used for the attacks.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

# attack_hyperparams={'epsilon':args.epsilon, 'iters':args.attack_iters, 'lrp_rule':args.attack_lrp_rule}

n_inputs=100 if args.debug else None

print("PyTorch Version: ", torch.__version__)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if args.model=="baseNN":

    model = baseNN_settings["model_"+str(args.model_idx)]

    train_loader, test_loader, inp_shape, out_size = data_loaders(dataset_name=model["dataset"], n_inputs=n_inputs,
                                                                  batch_size=128, shuffle=True)

    savedir = get_model_savedir(model=args.model, dataset=model["dataset"], architecture=model["architecture"], 
                         baseiters=None, debug=args.debug, model_idx=args.model_idx)
    
    net = baseNN(inp_shape, out_size, *list(model.values()))

    if args.load:
        net.load(savedir=savedir, device=args.device)
    
    else:
        net.train(train_loader=train_loader, savedir=savedir, device=args.device)

    net.evaluate(test_loader=test_loader, device=args.device)

elif args.model=="advNN":

    model = baseNN_settings["model_"+str(args.model_idx)]

    train_loader, test_loader, inp_shape, out_size = data_loaders(dataset_name=model["dataset"], n_inputs=n_inputs,
                                                                  batch_size=128, shuffle=True)

    savedir = get_model_savedir(model=args.model, dataset=model["dataset"], architecture=model["architecture"], 
                         baseiters=None, debug=args.debug, model_idx=args.model_idx, attack_method=args.attack_method)
    
    net = advNN(inp_shape, out_size, *list(model.values()), attack_method=args.attack_method)

    if args.load:
        net.load(savedir=savedir, device=args.device)
    
    else:
        net.train(train_loader=train_loader, savedir=savedir, device=args.device, hyperparams=attack_hyperparams)

    net.evaluate(test_loader=test_loader, device=args.device)


else:

    if args.model=="fullBNN":

        m = fullBNN_settings["model_"+str(args.model_idx)]
        batch_size = 4000 if m["inference"] == "hmc" else 128 
        # num_workers = 0 if args.device=="cuda" else 4

        train_loader, test_loader, inp_shape, out_size = data_loaders(dataset_name=m["dataset"], n_inputs=n_inputs,
                                                                      batch_size=batch_size, shuffle=True)

        savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                              debug=args.debug, model_idx=args.model_idx)

        net = BNN(m["dataset"], *list(m.values())[1:], inp_shape, out_size)

    else:
        raise NotImplementedError

    if args.debug:
        bayesian_attack_samples=[1,5]
    else:
        if m["inference"]=="svi":
            bayesian_attack_samples=[5,10,50]

        elif m["inference"]=="hmc":
            bayesian_attack_samples=[5,10,50]

    if args.load:
        net.load(savedir=savedir, device=args.device)

    else:
        net.train(train_loader=train_loader, savedir=savedir, device=args.device)

    for n_samples in bayesian_attack_samples:
        net.evaluate(test_loader=test_loader, device=args.device, n_samples=n_samples)
