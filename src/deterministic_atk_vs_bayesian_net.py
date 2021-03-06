import os
import torch
import argparse
import numpy as np

from utils.data import *
from utils import savedir
from utils.seeding import *
from attacks.gradient_based import *

from networks.baseNN import *
from networks.fullBNN import *
from networks.redBNN import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="fullBNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--model_idx", default=0, type=int, help="choose model idx from pre defined settings")
parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
parser.add_argument("--load", default=True, type=eval)
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--atk_inputs", default=1000, type=int, help="number of input points")
parser.add_argument("--layer_idx", default=-1, type=int)
parser.add_argument("--debug", default=False, type=eval)
parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
args = parser.parse_args()

n_inputs=100 if args.debug else None
atk_inputs=100 if args.debug else args.atk_inputs

print("PyTorch Version: ", torch.__version__)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


model = baseNN_settings["model_"+str(args.model_idx)]
x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=model["dataset"], n_inputs=atk_inputs)[2:]
savedir = get_savedir(model="baseNN", dataset=model["dataset"], architecture=model["architecture"], 
                     baseiters=None, debug=args.debug, model_idx=args.model_idx)

net = baseNN(inp_shape, out_size, *list(model.values()))

if args.load:
    net.load(savedir=savedir, device=args.device)
    x_attack = load_attack(method=args.attack_method, filename=net.name, savedir=savedir)

else:
    x_train, y_train, _, _, inp_shape, out_size = load_dataset(dataset_name=model["dataset"], n_inputs=n_inputs)
    train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=128, shuffle=True)
    net.train(train_loader=train_loader, savedir=savedir, device=args.device)
    x_attack = attack(net=net, x_test=x_test, y_test=y_test, savedir=savedir,
                  device=args.device, method=args.attack_method, filename=net.name)

attack_evaluation(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)

if args.model=="fullBNN":

    m = fullBNN_settings["model_"+str(args.model_idx)] 

    x_train, y_train, _, _, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], n_inputs=n_inputs)
    x_test, y_test = load_dataset(dataset_name=m["dataset"], n_inputs=atk_inputs)[2:4]

    savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                          debug=args.debug, model_idx=args.model_idx)

    net = BNN(m["dataset"], *list(m.values())[1:], inp_shape, out_size)


elif args.model=="redBNN":
    m = redBNN_settings["model_"+str(args.model_idx)]
    base_m = baseNN_settings["model_"+str(m["baseNN_idx"])]

    x_train, y_train, _, _, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], n_inputs=n_inputs)
    x_test, y_test = load_dataset(dataset_name=m["dataset"], n_inputs=atk_inputs)[2:4]

    savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                          debug=args.debug, model_idx=args.model_idx)
    basenet = baseNN(inp_shape, out_size, *list(base_m.values()))
    basenet_savedir = get_savedir(model="baseNN", dataset=m["dataset"], 
                      architecture=m["architecture"], debug=args.debug, model_idx=m["baseNN_idx"])
    basenet.load(savedir=basenet_savedir, device=args.device)

    hyp = get_hyperparams(m)
    net = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=basenet, hyperparams=hyp,
                 layer_idx=args.layer_idx)

else:
    raise NotImplementedError

if args.debug:
    bayesian_defence_samples=[1]
else:
    if m["inference"]=="svi":
        bayesian_defence_samples=[1,10,50]

    elif m["inference"]=="hmc":
        bayesian_defence_samples=[1,5,10]

if args.load:
    net.load(savedir=savedir, device=args.device)
else:
    batch_size = int(len(x_train)/max(bayesian_attack_samples)) if m["inference"] == "hmc" else 128 
    num_workers = 0 if args.device=="cuda" else 4
    train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=batch_size, 
                              num_workers=num_workers, shuffle=True)
    net.train(train_loader=train_loader, savedir=savedir, device=args.device)

for n_samples in bayesian_defence_samples:

    attack_evaluation(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, 
                      device=args.device, n_samples=n_samples)

