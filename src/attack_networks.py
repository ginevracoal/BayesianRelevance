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
parser.add_argument("--attack_library", type=str, default="grad_based", help="grad_based, deeprobust")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--atk_inputs", default=1000, type=int, help="Number of test points to be attacked.")
parser.add_argument("--bayesian_layer_idx", default=-1, type=int, help="Index for the Bayesian layer in redBNN.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

n_inputs=200 if args.debug else None
atk_inputs=200 if args.debug else args.atk_inputs

print("PyTorch Version: ", torch.__version__)

atk_lib = eval(args.attack_library)
attack = atk_lib.attack
load_attack = atk_lib.load_attack
save_attack = atk_lib.save_attack
evaluate_attack = grad_based.evaluate_attack

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if args.model=="baseNN":

    model = baseNN_settings["model_"+str(args.model_idx)]

    x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=model["dataset"], n_inputs=atk_inputs)[2:]

    savedir = get_model_savedir(model=args.model, dataset=model["dataset"], architecture=model["architecture"], 
                         baseiters=None, debug=args.debug, model_idx=args.model_idx)
    
    net = baseNN(inp_shape, out_size, *list(model.values()))
    net.load(savedir=savedir, device=args.device)

    if args.load:
        x_attack = load_attack(method=args.attack_method, model_savedir=savedir)
    
    else:
        x_attack = attack(net=net, x_test=x_test, y_test=y_test,
                          device=args.device, method=args.attack_method)
        save_attack(x_test, x_attack, method=args.attack_method, model_savedir=savedir)

    evaluate_attack(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)

else:

    if args.model=="fullBNN":

        m = fullBNN_settings["model_"+str(args.model_idx)]

        x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], n_inputs=atk_inputs)[2:]

        savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                              debug=args.debug, model_idx=args.model_idx)

        net = BNN(m["dataset"], *list(m.values())[1:], inp_shape, out_size)

    elif args.model=="redBNN":
        
        m = redBNN_settings["model_"+str(args.model_idx)]
        base_m = baseNN_settings["model_"+str(m["baseNN_idx"])]

        x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], n_inputs=atk_inputs)[2:]

        savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                              debug=args.debug, model_idx=args.model_idx)
        basenet = baseNN(inp_shape, out_size, *list(base_m.values()))
        basenet_savedir = get_model_savedir(model="baseNN", dataset=m["dataset"], 
                          architecture=m["architecture"], debug=args.debug, model_idx=m["baseNN_idx"])
        basenet.load(savedir=basenet_savedir, device=args.device)

        hyp = get_hyperparams(m)
        net = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=basenet, hyperparams=hyp,
                     layer_idx=args.bayesian_layer_idx)

    else:
        raise NotImplementedError

    net.load(savedir=savedir, device=args.device)

    if args.debug:
        bayesian_attack_samples=[1,5]
    else:
        if m["inference"]=="svi":
            bayesian_attack_samples=[5,10,50]

        elif m["inference"]=="hmc":
            bayesian_attack_samples=[5,10,50]
    
    if args.load:

        for n_samples in bayesian_attack_samples:

            x_attack = load_attack(method=args.attack_method, model_savedir=savedir, n_samples=n_samples)
            evaluate_attack(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, 
                              device=args.device, n_samples=n_samples)

        mode_attack = load_attack(method=args.attack_method, model_savedir=savedir, 
                                  n_samples=n_samples, atk_mode=True)
        evaluate_attack(net=net, x_test=x_test, x_attack=mode_attack, y_test=y_test, 
                          device=args.device, n_samples=n_samples, avg_posterior=True)

    else:
        batch_size = 4000 if m["inference"] == "hmc" else 128 
        num_workers = 0 if args.device=="cuda" else 4

        for n_samples in bayesian_attack_samples:
            x_attack = attack(net=net, x_test=x_test, y_test=y_test, device=args.device,
                              method=args.attack_method, n_samples=n_samples)
            save_attack(x_test, x_attack, method=args.attack_method, 
                             model_savedir=savedir, n_samples=n_samples)
            evaluate_attack(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, 
                              device=args.device, n_samples=n_samples)

        mode_attack = attack(net=net, x_test=x_test, y_test=y_test, device=args.device,
                          method=args.attack_method, n_samples=n_samples, avg_posterior=True)
        save_attack(x_test, mode_attack, method=args.attack_method,   
                         model_savedir=savedir, n_samples=n_samples, atk_mode=True)
        evaluate_attack(net=net, x_test=x_test, x_attack=mode_attack, y_test=y_test, 
                          device=args.device, n_samples=n_samples, avg_posterior=True)

