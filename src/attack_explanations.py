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
parser.add_argument("--lrp_method", default="avg_heatmap", type=str, help="avg_prediction, avg_heatmap")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation: epsilon, gamma, alpha1beta0")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--epsilon", default=0.2, type=int, help="Strenght of a perturbation.")
parser.add_argument("--n_inputs", default=500, type=int, help="Number of test points to be attacked.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

MODE_ATKS = False

n_inputs=100 if args.debug else args.n_inputs
bayesian_attack_samples=[100]
hyperparams={'epsilon':args.epsilon}

print("PyTorch Version: ", torch.__version__)

if args.attack_method=="deepfool":
  args.device="cpu"

if args.device=="cuda":
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

model = baseNN_settings["model_"+str(args.model_idx)]
x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=model["dataset"], shuffle=False, n_inputs=n_inputs)[2:]
images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)

if args.model=="baseNN":

	model = baseNN_settings["model_"+str(args.model_idx)]
	model_savedir = get_model_savedir(model=args.model, dataset=model["dataset"], architecture=model["architecture"], 
								debug=args.debug, model_idx=args.model_idx)
	net = baseNN(inp_shape, out_size, *list(model.values()))
	net.load(savedir=model_savedir, device=args.device)

	for layer_idx in net.learnable_layers_idxs:

		print("\nlayer_idx =", layer_idx)

		lrp_savedir = get_lrp_savedir(model_savedir=model_savedir, attack_method=args.attack_method, 
								  layer_idx=layer_idx, rule=args.rule)
		lrp = load_from_pickle(path=lrp_savedir, filename="det_lrp")

		lrp_attack = attack(net=net, x_test=lrp, y_test=y_test, hyperparams=hyperparams,
							device=args.device, method=args.attack_method)
		save_to_pickle(lrp_attack, path=lrp_savedir, filename="det_lrp_attack")

elif args.model=="advNN":

	model = baseNN_settings["model_"+str(args.model_idx)]
	model_savedir = get_model_savedir(model=args.model, dataset=model["dataset"], architecture=model["architecture"], 
								debug=args.debug, model_idx=args.model_idx, attack_method='fgsm')
	
	net = advNN(inp_shape, out_size, *list(model.values()), attack_method='fgsm')
	net.load(savedir=model_savedir, device=args.device)

	for layer_idx in net.learnable_layers_idxs:

		print("\nlayer_idx =", layer_idx)

		lrp_savedir = get_lrp_savedir(model_savedir=model_savedir, attack_method=args.attack_method, 
								  layer_idx=layer_idx, rule=args.rule)
		lrp = load_from_pickle(path=lrp_savedir, filename="det_lrp")

		lrp_attack = attack(net=net, x_test=lrp, y_test=y_test, hyperparams=hyperparams,
							device=args.device, method=args.attack_method)
		save_to_pickle(lrp_attack, path=lrp_savedir, filename="det_lrp_attack")

elif args.model=="fullBNN":

	m = fullBNN_settings["model_"+str(args.model_idx)]
	model_savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
						  debug=args.debug, model_idx=args.model_idx)
	net = BNN(m["dataset"], *list(m.values())[1:], inp_shape, out_size)
	net.load(savedir=model_savedir, device=args.device)

	batch_size = 4000 if m["inference"] == "hmc" else 128 
	num_workers = 0 if args.device=="cuda" else 4

	for layer_idx in net.basenet.learnable_layers_idxs:

		lrp_savedir = get_lrp_savedir(model_savedir=model_savedir, attack_method=args.attack_method, 
								  layer_idx=layer_idx, rule=args.rule, lrp_method=args.lrp_method)

		for n_samples in bayesian_attack_samples:

			print(f"\n--- Layer_idx = {layer_idx} samp = {n_samples} ---")

			lrp = load_from_pickle(path=lrp_savedir, filename="bay_lrp_samp="+str(n_samples))

			lrp_attack = attack(net=net, x_test=lrp, y_test=y_test, device=args.device, hyperparams=hyperparams,
							  method=args.attack_method, n_samples=n_samples)
			save_to_pickle(lrp_attack, path=lrp_savedir, filename="bay_lrp_attack_samp="+str(n_samples))

else:
	raise NotImplementedError
