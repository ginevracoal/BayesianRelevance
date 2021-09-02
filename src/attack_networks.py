import argparse
import numpy as np
import os
import torch

from attacks.gradient_based import evaluate_attack
from attacks.run_attacks import *
from networks.advNN import *
from networks.baseNN import *
from networks.fullBNN import *
from utils import savedir
from utils.data import *
from utils.seeding import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="baseNN", type=str, help="baseNN, fullBNN, advNN")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings.")
parser.add_argument("--n_inputs", default=500, type=int, help="Number of test points to be attacked.")
parser.add_argument("--n_samples", default=100, type=int)
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--attack_iters", default=10, type=int, help="Number of iterations in iterative attacks.")
parser.add_argument("--attack_lrp_rule", default='epsilon', type=str, help="LRP rule used for the attacks.")
parser.add_argument("--epsilon", default=0.2, type=int, help="Strength of a perturbation.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

MODE_ATKS = False

n_inputs=100 if args.debug else args.n_inputs
n_samples=5 if args.debug else args.n_samples
# attack_iters=10 if args.attack_method=='beta' else 10

attack_hyperparams={'epsilon':args.epsilon, 'iters':args.attack_iters, 'lrp_rule':args.attack_lrp_rule}

print("PyTorch Version: ", torch.__version__)

if args.attack_method=="deepfool":
  args.device="cpu"

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if args.model=="baseNN":

    model = baseNN_settings["model_"+str(args.model_idx)]
    x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=model["dataset"], n_inputs=n_inputs)[2:]

    savedir = get_model_savedir(model=args.model, dataset=model["dataset"], architecture=model["architecture"], 
                                debug=args.debug, model_idx=args.model_idx)
    
    net = baseNN(inp_shape, out_size, *list(model.values()))
    net.load(savedir=savedir, device=args.device)

    if args.load:
        x_attack = load_attack(method=args.attack_method, model_savedir=savedir)
    
    else:
        x_attack = attack(net=net, x_test=x_test, y_test=y_test, hyperparams=attack_hyperparams,
                          device=args.device, method=args.attack_method)
        save_attack(x_test, x_attack, method=args.attack_method, model_savedir=savedir)

    evaluate_attack(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)

elif args.model=="advNN":

    model = baseNN_settings["model_"+str(args.model_idx)]

    x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=model["dataset"], n_inputs=n_inputs)[2:]

    savedir = get_model_savedir(model=args.model, dataset=model["dataset"], architecture=model["architecture"], 
                                debug=args.debug, model_idx=args.model_idx, attack_method='fgsm')
    
    net = advNN(inp_shape, out_size, *list(model.values()), attack_method='fgsm')
    net.load(savedir=savedir, device=args.device)

    if args.load:
        x_attack = load_attack(method=args.attack_method, model_savedir=savedir)
    
    else:
        x_attack = attack(net=net, x_test=x_test, y_test=y_test, hyperparams=attack_hyperparams,
                          device=args.device, method=args.attack_method)
        save_attack(x_test, x_attack, method=args.attack_method, model_savedir=savedir)

    evaluate_attack(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)

else:

    if args.model=="fullBNN":

        m = fullBNN_settings["model_"+str(args.model_idx)]

        x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], n_inputs=n_inputs)[2:]

        savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                              debug=args.debug, model_idx=args.model_idx)

        net = BNN(m["dataset"], *list(m.values())[1:], inp_shape, out_size)

    else:
        raise NotImplementedError

    net.load(savedir=savedir, device=args.device)

    if args.load:

        x_attack = load_attack(method=args.attack_method, model_savedir=savedir, n_samples=n_samples)
        evaluate_attack(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, 
                          device=args.device, n_samples=n_samples)

        if MODE_ATKS:
            if m["inference"]=="svi":
                mode_attack = load_attack(method=args.attack_method, model_savedir=savedir, 
                                          n_samples=n_samples, atk_mode=True)
                evaluate_attack(net=net, x_test=x_test, x_attack=mode_attack, y_test=y_test, 
                                  device=args.device, n_samples=n_samples, avg_posterior=True)

    else:
        batch_size = 4000 if m["inference"] == "hmc" else 128 
        num_workers = 0 if args.device=="cuda" else 4

        x_attack = attack(net=net, x_test=x_test, y_test=y_test, device=args.device,
                          method=args.attack_method, n_samples=n_samples, hyperparams=attack_hyperparams)
        save_attack(x_test, x_attack, method=args.attack_method, 
                         model_savedir=savedir, n_samples=n_samples)
        evaluate_attack(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, 
                          device=args.device, n_samples=n_samples)

        if MODE_ATKS:
            if m["inference"]=="svi":
                mode_attack = attack(net=net, x_test=x_test, y_test=y_test, device=args.device, hyperparams=attack_hyperparams,
                                  method=args.attack_method, n_samples=n_samples, avg_posterior=True)
                save_attack(x_test, mode_attack, method=args.attack_method,   
                                 model_savedir=savedir, n_samples=n_samples, atk_mode=True)
                evaluate_attack(net=net, x_test=x_test, x_attack=mode_attack, y_test=y_test, 
                                  device=args.device, n_samples=n_samples, avg_posterior=True)
