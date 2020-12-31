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

parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=60000, type=int, help="number of input points")
parser.add_argument("--model_idx", default=0, type=int, help="choose model idx from pre defined settings")
parser.add_argument("--model_type", default="baseNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--train", default=True, type=eval)
parser.add_argument("--attack", default=True, type=eval)
parser.add_argument("--debug", default=False, type=eval)
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

bayesian_attack_samples=[3] if args.debug else [1,20,50]
n_inputs=100 if args.debug else args.n_inputs
rel_path=TESTS

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if args.model_type=="baseNN":

    model = baseNN_settings["model_"+str(args.model_idx)]

    x_train, y_train, x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=model["dataset"], 
                                                                         n_inputs=n_inputs)
    
    net = baseNN(inp_shape, out_size, *list(model.values()))

    if args.train:
        train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=128, shuffle=False)
        net.train(train_loader=train_loader, device=args.device)
    else:
        net.load(device=args.device, rel_path=rel_path)

    x_attack = attack(net=net, x_test=x_test, y_test=y_test,
                      device=args.device, method=args.attack_method, filename=net.name)

    attack_evaluation(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, device=args.device)

else:

    if args.model_type=="fullBNN":

        m = fullBNN_settings["model_"+str(args.model_idx)]

        x_train, y_train, x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], 
                                                                             n_inputs=n_inputs)
                            
        net = BNN(m["dataset"], *list(m.values())[1:], inp_shape, out_size)
   
    elif args.model_type=="redBNN":

        m = redBNN_settings["model_"+str(args.model_idx)]
        x_train, y_train, x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], 
                                                                             n_inputs=n_inputs)
        basenet = baseNN(dataset_name=m["dataset"], input_shape=inp_shape, output_size=out_size,
                  epochs=m["baseNN_epochs"], lr=m["baseNN_lr"], hidden_size=m["hidden_size"], 
                  activation=m["activation"], architecture=m["architecture"])        
        basenet.load(rel_path=rel_path, device=args.device)

        hyp = get_hyperparams(m)
        net = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=basenet, hyperparams=hyp)

    else:
        raise NotImplementedError

    if args.train is True:
        batch_size = 5000 if m["inference"] == "hmc" else 128
        train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=batch_size, shuffle=False)
        net.train(train_loader=train_loader, device=args.device)
    else:
        net.load(n_inputs=m["BNN_inputs"], device=args.device, rel_path=rel_path)
    
    for attack_samples in bayesian_attack_samples:
        x_attack = attack(net=net, x_test=x_test, y_test=y_test, device=args.device, 
                        method=args.attack_method, filename=net.name, n_samples=attack_samples)

        defence_samples = attack_samples
        attack_evaluation(net=net, x_test=x_test, x_attack=x_attack, y_test=y_test, 
                          device=args.device, n_samples=defence_samples)


