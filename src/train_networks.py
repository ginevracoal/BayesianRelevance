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
from networks.redBNN import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="baseNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--bayesian_layer_idx", default=-1, type=int, help="Index for the Bayesian layer in redBNN.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

n_inputs=100 if args.debug else None

print("PyTorch Version: ", torch.__version__)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if args.model=="baseNN":

    model = baseNN_settings["model_"+str(args.model_idx)]

    train_loader, test_loader, inp_shape, out_size = data_loaders(dataset_name=model["dataset"], n_inputs=n_inputs,
                                                                  batch_size=128, shuffle=True)

    savedir = get_savedir(model=args.model, dataset=model["dataset"], architecture=model["architecture"], 
                         baseiters=None, debug=args.debug, model_idx=args.model_idx)
    
    net = baseNN(inp_shape, out_size, *list(model.values()))

    if args.load:
        net.load(savedir=savedir, device=args.device)
    
    else:
        net.train(train_loader=train_loader, savedir=savedir, device=args.device)

    net.evaluate(test_loader=test_loader, device=args.device)

else:

    if args.model=="fullBNN":

        m = fullBNN_settings["model_"+str(args.model_idx)]

        train_loader, test_loader, inp_shape, out_size = data_loaders(dataset_name=m["dataset"], n_inputs=n_inputs,
                                                                      batch_size=128, shuffle=True)

        savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                              debug=args.debug, model_idx=args.model_idx)

        net = BNN(m["dataset"], *list(m.values())[1:], inp_shape, out_size)

    elif args.model=="redBNN":
        
        m = redBNN_settings["model_"+str(args.model_idx)]
        base_m = baseNN_settings["model_"+str(m["baseNN_idx"])]

        train_loader, test_loader, inp_shape, out_size = data_loaders(dataset_name=m["dataset"], n_inputs=n_inputs,
                                                                  batch_size=128, shuffle=True)

        savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                              debug=args.debug, model_idx=args.model_idx)
        basenet = baseNN(inp_shape, out_size, *list(base_m.values()))
        basenet_savedir = get_savedir(model="baseNN", dataset=m["dataset"], 
                          architecture=m["architecture"], debug=args.debug, model_idx=m["baseNN_idx"])
        basenet.load(savedir=basenet_savedir, device=args.device)

        hyp = get_hyperparams(m)
        net = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=basenet, hyperparams=hyp,
                     layer_idx=args.bayesian_layer_idx)

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
        batch_size = 4000 if m["inference"] == "hmc" else 128 
        num_workers = 0 if args.device=="cuda" else 4
        net.train(train_loader=train_loader, savedir=savedir, device=args.device)

    for n_samples in bayesian_attack_samples:
        net.evaluate(test_loader=test_loader, device=args.device, n_samples=n_samples)
