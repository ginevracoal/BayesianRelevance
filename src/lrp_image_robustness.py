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

from networks.baseNN import *
from networks.fullBNN import *
from networks.redBNN import *

from utils.lrp import *
from plot.lrp_heatmaps import *
import plot.lrp_distributions as plot_lrp
import attacks.gradient_based as grad_based
import attacks.deeprobust as deeprobust

parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=1000, type=int, help="Number of test points")
parser.add_argument("--topk", default=200, type=int, help="Top k most relevant pixels.")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="fullBNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--attack_library", type=str, default="grad_based", help="grad_based, deeprobust")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--layer_idx", default=-1, type=int, help="Layer idx for LRP computation.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

lrp_robustness_method = "imagewise"
n_samples_list=[1,5] if args.debug else [5, 10, 50]
n_inputs=100 if args.debug else args.n_inputs
topk=100 if args.debug else args.topk

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

atk_lib = eval(args.attack_library)
attack = atk_lib.attack
load_attack = atk_lib.load_attack
evaluate_attack = atk_lib.evaluate_attack

if args.device=="cuda":
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Load models and attacks

model = baseNN_settings["model_"+str(args.model_idx)]

_, _, x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=model["dataset"], 
															shuffle=False, n_inputs=n_inputs)
model_savedir = get_savedir(model="baseNN", dataset=model["dataset"], architecture=model["architecture"], 
					  debug=args.debug, model_idx=args.model_idx)
detnet = baseNN(inp_shape, num_classes, *list(model.values()))
detnet.load(savedir=model_savedir, device=args.device)

det_attack = load_attack(method=args.attack_method, filename=detnet.name, savedir=model_savedir)

if args.model=="fullBNN":

	m = fullBNN_settings["model_"+str(args.model_idx)]

	model_savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
								model_idx=args.model_idx, debug=args.debug)

	bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
	bayesnet.load(savedir=model_savedir, device=args.device)

	bay_attack=[]
	for n_samples in n_samples_list:
		bay_attack.append(load_attack(method=args.attack_method, filename=bayesnet.name, savedir=model_savedir, 
						  n_samples=n_samples))


else:
	raise NotImplementedError

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)
savedir = os.path.join(model_savedir, "lrp/pkl/")

### Load explanations

det_lrp = load_from_pickle(path=savedir, filename="det_lrp")
det_attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

bay_attack=[]
bay_lrp=[]
bay_attack_lrp=[]
for n_samples in n_samples_list:
	bay_attack.append(load_from_pickle(path=savedir, filename="bay_attack_samp="+str(n_samples)))
	bay_lrp.append(load_from_pickle(path=savedir, filename="bay_lrp_samp="+str(n_samples)))
	bay_attack_lrp.append(load_from_pickle(path=savedir, filename="bay_attack_lrp_samp="+str(n_samples)))

mode_attack = load_from_pickle(path=savedir, filename="mode_attack_samp="+str(n_samples))
mode_lrp = load_from_pickle(path=savedir, filename="mode_lrp_samp="+str(n_samples))
mode_attack_lrp = load_from_pickle(path=savedir, filename="mode_attack_lrp_samp="+str(n_samples))


### Evaluate explanations

det_preds, det_atk_preds, det_softmax_robustness, det_successful_idxs = evaluate_attack(net=detnet, x_test=images, 
				x_attack=det_attack, y_test=y_test, device=args.device, return_successful_idxs=True)
det_softmax_robustness = det_softmax_robustness.detach().cpu().numpy()

det_lrp_robustness, det_lrp_pxl_idxs = lrp_robustness(original_heatmaps=det_lrp, 
													  adversarial_heatmaps=det_attack_lrp, 
													  topk=topk, method=lrp_robustness_method)

bay_softmax_robustness=[]
bay_successful_idxs=[]
bay_lrp_robustness=[]
bay_lrp_pxl_idxs=[]
bay_preds=[]
bay_atk_preds=[]

for samp_idx, n_samples in enumerate(n_samples_list):

	preds, atk_preds, softmax_rob, successf_idxs = evaluate_attack(net=bayesnet, x_test=images, 
												   x_attack=bay_attack[samp_idx], y_test=y_test, device=args.device, 
												   n_samples=n_samples, return_successful_idxs=True)
	bay_softmax_robustness.append(softmax_rob.detach().cpu().numpy())
	bay_successful_idxs.append(successf_idxs)
	bay_preds.append(preds)
	bay_atk_preds.append(atk_preds)

	bay_lrp_rob, lrp_pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp[samp_idx], 
											   adversarial_heatmaps=bay_attack_lrp[samp_idx], 
											   topk=topk, method=lrp_robustness_method)
	bay_lrp_robustness.append(bay_lrp_rob)
	bay_lrp_pxl_idxs.append(lrp_pxl_idxs)

mode_preds, mode_atk_preds, mode_softmax_robustness, mode_successful_idxs = evaluate_attack(net=bayesnet, x_test=images, 
				x_attack=mode_attack, y_test=y_test, device=args.device, n_samples=n_samples, return_successful_idxs=True)
mode_softmax_robustness = mode_softmax_robustness.detach().cpu().numpy()

mode_lrp_robustness, mode_pxl_idxs = lrp_robustness(original_heatmaps=mode_lrp, 
													adversarial_heatmaps=mode_attack_lrp, 
													topk=topk, method=lrp_robustness_method)

### Plots

savedir = os.path.join(model_savedir, "lrp/robustness/")

plot_attacks_explanations(images=images, 
						  explanations=det_lrp, 
						  attacks=det_attack, 
						  attacks_explanations=det_attack_lrp, 
						  predictions=det_preds.argmax(-1),
						  attacks_predictions=det_atk_preds.argmax(-1),
						  successful_attacks_idxs=det_successful_idxs,
						  labels=labels, lrp_method=lrp_robustness_method,
						  rule=args.rule, savedir=savedir, pxl_idxs=det_lrp_pxl_idxs,
						  filename=lrp_robustness_method+"_det_lrp_attacks", layer_idx=-1)

for samp_idx, n_samples in enumerate(n_samples_list):

	plot_attacks_explanations(images=images, 
							  explanations=bay_lrp[samp_idx], 
							  attacks=bay_attack[samp_idx], 
							  attacks_explanations=bay_attack_lrp[samp_idx],
							  predictions=bay_preds[samp_idx].argmax(-1),
							  attacks_predictions=bay_atk_preds[samp_idx].argmax(-1),
							  successful_attacks_idxs=bay_successful_idxs[samp_idx],
							  labels=labels, lrp_method=lrp_robustness_method,
							  rule=args.rule, savedir=savedir, pxl_idxs=bay_lrp_pxl_idxs[samp_idx],
							  filename=lrp_robustness_method+"_bay_lrp_attacks_samp="+str(n_samples), layer_idx=-1)

plot_attacks_explanations(images=images, 
						  explanations=mode_lrp, 
						  attacks=mode_attack, 
						  attacks_explanations=mode_attack_lrp, 
						  predictions=mode_preds.argmax(-1),
						  attacks_predictions=mode_atk_preds.argmax(-1),
						  successful_attacks_idxs=mode_successful_idxs,
						  labels=labels, lrp_method=lrp_robustness_method,
						  rule=args.rule, savedir=savedir, pxl_idxs=mode_pxl_idxs,
						  filename=lrp_robustness_method+"_mode_lrp_attacks_samp="+str(n_samples), layer_idx=-1)

filename=args.rule+"_lrp_robustness"+m["dataset"]+"_images="+str(n_inputs)+\
		 "_samples="+str(n_samples)+"_pxls="+str(topk)+"_atk="+str(args.attack_method)

plot_lrp.lrp_robustness_distributions(lrp_robustness=det_lrp_robustness, 
									  bayesian_lrp_robustness=bay_lrp_robustness, 
									  mode_lrp_robustness=mode_lrp_robustness,
									  n_samples_list=n_samples_list,
									  savedir=savedir, filename="dist_"+filename)

plot_lrp.lrp_robustness_scatterplot(adversarial_robustness=det_softmax_robustness, 
									bayesian_adversarial_robustness=bay_softmax_robustness,
									mode_adversarial_robustness=mode_softmax_robustness,
									lrp_robustness=det_lrp_robustness, 
									bayesian_lrp_robustness=bay_lrp_robustness,
									mode_lrp_robustness=mode_lrp_robustness,
									n_samples_list=n_samples_list,
									savedir=savedir, filename="scatterplot_"+filename)