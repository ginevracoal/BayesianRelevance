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
parser.add_argument("--topk", default=300, type=int, help="Top k most relevant pixels.")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="fullBNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--attack_library", type=str, default="grad_based", help="grad_based, deeprobust")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--normalize", default=False, type=eval, help="Normalize lrp heatmaps.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

lrp_robustness_method = "imagewise"
n_samples_list=[1,5] if args.debug else [5, 10, 50]
n_inputs=200 if args.debug else args.n_inputs
topk=args.topk

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
model_savedir = get_model_savedir(model="baseNN", dataset=model["dataset"], architecture=model["architecture"], 
											debug=args.debug, model_idx=args.model_idx)
detnet = baseNN(inp_shape, num_classes, *list(model.values()))
detnet.load(savedir=model_savedir, device=args.device)

n_layers = detnet.n_layers

det_attack = load_attack(method=args.attack_method, model_savedir=model_savedir)

if args.model=="fullBNN":

		m = fullBNN_settings["model_"+str(args.model_idx)]

		model_savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
																model_idx=args.model_idx, debug=args.debug)

		bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
		bayesnet.load(savedir=model_savedir, device=args.device)

		bay_attack=[]
		for n_samples in n_samples_list:
				bay_attack.append(load_attack(method=args.attack_method, model_savedir=model_savedir, 
													n_samples=n_samples))

else:
		raise NotImplementedError

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)

det_successful_lrp_robustness_layers=[]
det_failed_lrp_robustness_layers=[]
bay_successful_lrp_robustness_layers=[]
bay_failed_lrp_robustness_layers=[]

det_successful_norm_layers=[]
det_failed_norm_layers=[]
bay_successful_norm_layers=[]
bay_failed_norm_layers=[]

for layer_idx in range(n_layers):
	layer_idx+=1

	savedir = get_lrp_savedir(model_savedir=model_savedir, attack_method=args.attack_method, 
                          	  layer_idx=layer_idx, normalize=args.normalize)

	### Load explanations

	det_lrp = load_from_pickle(path=savedir, filename="det_lrp")
	det_attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

	bay_lrp=[]
	bay_attack_lrp=[]
	for n_samples in n_samples_list:
		bay_lrp.append(load_from_pickle(path=savedir, filename="bay_lrp_samp="+str(n_samples)))
		bay_attack_lrp.append(load_from_pickle(path=savedir, filename="bay_attack_lrp_samp="+str(n_samples)))

	### Evaluate explanations

	det_preds, det_atk_preds, det_softmax_robustness, det_successful_idxs, det_failed_idxs = evaluate_attack(net=detnet, 
					x_test=images, x_attack=det_attack, y_test=y_test, device=args.device, return_classification_idxs=True)
	det_softmax_robustness = det_softmax_robustness.detach().cpu().numpy()

	det_lrp_robustness, det_lrp_pxl_idxs = lrp_robustness(original_heatmaps=det_lrp, 
														  adversarial_heatmaps=det_attack_lrp, 
														  topk=topk, method=lrp_robustness_method)

	succ_det_lrp_robustness, succ_det_lrp_pxl_idxs = lrp_robustness(original_heatmaps=det_lrp[det_successful_idxs], 
																	adversarial_heatmaps=det_attack_lrp[det_successful_idxs], 
																	topk=topk, method=lrp_robustness_method)
	fail_det_lrp_robustness, fail_det_lrp_pxl_idxs = lrp_robustness(original_heatmaps=det_lrp[det_failed_idxs], 
																	adversarial_heatmaps=det_attack_lrp[det_failed_idxs], 
																	topk=topk, method=lrp_robustness_method)
	
	det_successful_norm = lrp_distances(det_lrp[det_successful_idxs], det_attack_lrp[det_successful_idxs], 
										axis_norm=1).detach().cpu().numpy()
	det_failed_norm = lrp_distances(det_lrp[det_failed_idxs], det_attack_lrp[det_failed_idxs], 
									axis_norm=1).detach().cpu().numpy()

	bay_preds=[]
	bay_atk_preds=[]
	bay_softmax_robustness=[]
	bay_successful_idxs=[]
	bay_failed_idxs=[]

	bay_lrp_robustness=[]
	bay_lrp_pxl_idxs=[]
	succ_bay_lrp_robustness=[]
	succ_bay_lrp_pxl_idxs=[]
	fail_bay_lrp_robustness=[]
	fail_bay_lrp_pxl_idxs=[]

	bay_successful_norm=[]
	bay_failed_norm=[]

	for samp_idx, n_samples in enumerate(n_samples_list):

			preds, atk_preds, softmax_rob, successf_idxs, failed_idxs = evaluate_attack(net=bayesnet, x_test=images, 
														 x_attack=bay_attack[samp_idx], y_test=y_test, device=args.device, 
														 n_samples=n_samples, return_classification_idxs=True)
			bay_softmax_robustness.append(softmax_rob.detach().cpu().numpy())
			bay_successful_idxs.append(successf_idxs)
			bay_failed_idxs.append(failed_idxs)
			bay_preds.append(preds)
			bay_atk_preds.append(atk_preds)

			bay_lrp_rob, lrp_pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp[samp_idx], 
													 adversarial_heatmaps=bay_attack_lrp[samp_idx], 
													 topk=topk, method=lrp_robustness_method)
			bay_lrp_robustness.append(bay_lrp_rob)
			bay_lrp_pxl_idxs.append(lrp_pxl_idxs)

			bay_lrp_rob, succ_lrp_pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp[samp_idx][successf_idxs], 
														 adversarial_heatmaps=bay_attack_lrp[samp_idx][successf_idxs], 
														 topk=topk, method=lrp_robustness_method)
			succ_bay_lrp_robustness.append(bay_lrp_rob)
			succ_bay_lrp_pxl_idxs.append(succ_lrp_pxl_idxs)

			failed_idxs = np.setdiff1d(np.arange(len(images)), successf_idxs)
			bay_lrp_rob, fail_lrp_pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp[samp_idx][failed_idxs], 
														 adversarial_heatmaps=bay_attack_lrp[samp_idx][failed_idxs], 
														 topk=topk, method=lrp_robustness_method)
			fail_bay_lrp_robustness.append(bay_lrp_rob)
			fail_bay_lrp_pxl_idxs.append(fail_lrp_pxl_idxs)

			bay_successful_norm.append(lrp_distances(bay_lrp[samp_idx][successf_idxs], 
													bay_attack_lrp[samp_idx][successf_idxs], 
													axis_norm=1).detach().cpu().numpy())
			bay_failed_norm.append(lrp_distances(bay_lrp[samp_idx][failed_idxs], 
												bay_attack_lrp[samp_idx][failed_idxs], 
												axis_norm=1).detach().cpu().numpy())

	det_successful_lrp_robustness_layers.append(succ_det_lrp_robustness)
	det_failed_lrp_robustness_layers.append(fail_det_lrp_robustness)
	bay_successful_lrp_robustness_layers.append(succ_bay_lrp_robustness)
	bay_failed_lrp_robustness_layers.append(fail_bay_lrp_robustness)

	det_successful_norm_layers.append(det_successful_norm)
	det_failed_norm_layers.append(det_failed_norm)
	bay_successful_norm_layers.append(bay_successful_norm)
	bay_failed_norm_layers.append(bay_failed_norm)


### Plots

savedir = os.path.join(model_savedir, "lrp/robustness/")

filename=args.rule+"_lrp_robustness_"+m["dataset"]+"_images="+str(n_inputs)+\
		  "_samples="+str(n_samples)+"_pxls="+str(topk)+"_atk="+str(args.attack_method)

if args.normalize:
	filename=filename+"_norm"

plot_lrp.lrp_imagewise_layers_robustness_distributions(
										det_successful_lrp_robustness=det_successful_lrp_robustness_layers,
										det_failed_lrp_robustness=det_failed_lrp_robustness_layers,
										bay_successful_lrp_robustness=bay_successful_lrp_robustness_layers,
										bay_failed_lrp_robustness=bay_failed_lrp_robustness_layers,
										n_samples_list=n_samples_list,
										n_original_images=len(images),
										n_layers=n_layers,
										savedir=savedir, 
										filename="dist_"+filename+"_layers")

# if args.normalize is False:
plot_lrp.lrp_layers_robustness_scatterplot(
											det_successful_lrp_robustness=det_successful_lrp_robustness_layers,
											det_failed_lrp_robustness=det_failed_lrp_robustness_layers,
											bay_successful_lrp_robustness=bay_successful_lrp_robustness_layers,
											bay_failed_lrp_robustness=bay_failed_lrp_robustness_layers,
											det_successful_lrp_norm=det_successful_norm_layers,
											det_failed_lrp_norm=det_failed_norm_layers,
											bay_successful_lrp_norm=bay_successful_norm_layers,
											bay_failed_lrp_norm=bay_failed_norm_layers,
											n_samples_list=n_samples_list,
											n_original_images=len(images),
											n_layers=n_layers,
											savedir=savedir, 
											filename="scatterplot_"+filename+"_layers")