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
from networks.advNN import *
from networks.fullBNN import *

from utils.lrp import *
from plot.lrp_heatmaps import *
import plot.lrp_distributions as plot_lrp
from attacks.gradient_based import evaluate_attack
from attacks.run_attacks import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=500, type=int, help="Number of test points")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm")
parser.add_argument("--lrp_method", default="avg_heatmap", type=str, help="avg_prediction, avg_heatmap")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--normalize", default=False, type=eval, help="Normalize lrp heatmaps.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

lrp_robustness_method = "imagewise"
n_samples_list=[5] if args.debug else [100]
topk_list = [20]
n_inputs=100 if args.debug else args.n_inputs

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if args.device=="cuda":
		torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Load models and attacks

## baseNN

model = baseNN_settings["model_"+str(args.model_idx)]

x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=model["dataset"], 
														shuffle=False, n_inputs=n_inputs)[2:]
det_model_savedir = get_model_savedir(model="baseNN", dataset=model["dataset"], architecture=model["architecture"], 
											debug=args.debug, model_idx=args.model_idx)
detnet = baseNN(inp_shape, num_classes, *list(model.values()))
detnet.load(savedir=det_model_savedir, device=args.device)

det_attack = load_attack(method=args.attack_method, model_savedir=det_model_savedir)

## advNN

adv_model_savedir = get_model_savedir(model="advNN", dataset=model["dataset"], architecture=model["architecture"], 
                            debug=args.debug, model_idx=args.model_idx, attack_method='fgsm')
advnet = advNN(inp_shape, num_classes, *list(model.values()), attack_method='fgsm')
advnet.load(savedir=adv_model_savedir, device=args.device)

adv_attack = load_attack(method=args.attack_method, model_savedir=adv_model_savedir)

## fullBNN

m = fullBNN_settings["model_"+str(args.model_idx)]

bay_model_savedir = get_model_savedir(model="fullBNN", dataset=m["dataset"], architecture=m["architecture"], 
														model_idx=args.model_idx, debug=args.debug)

bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
bayesnet.load(savedir=bay_model_savedir, device=args.device)

bay_attack=[]
for n_samples in n_samples_list:
		bay_attack.append(load_attack(method=args.attack_method, model_savedir=bay_model_savedir, 
											n_samples=n_samples))

### plot

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)

det_lrp_robustness_topk=[]
adv_lrp_robustness_topk=[]
bay_lrp_robustness_topk=[]

det_failed_idxs_topk=[]
adv_failed_idxs_topk=[]
bay_failed_idxs_topk=[]

det_norm_topk=[]
adv_norm_topk=[]
bay_norm_topk=[]

det_softmax_robustness_topk=[]
adv_softmax_robustness_topk=[]
bay_softmax_robustness_topk=[]

for topk in topk_list:

	det_lrp_robustness_layers=[]
	adv_lrp_robustness_layers=[]
	bay_lrp_robustness_layers=[]

	det_failed_idxs_layers=[]
	adv_failed_idxs_layers=[]
	bay_failed_idxs_layers=[]

	det_norm_layers=[]
	adv_norm_layers=[]
	bay_norm_layers=[]

	det_softmax_robustness_layers=[]
	adv_softmax_robustness_layers=[]
	bay_softmax_robustness_layers=[]

	for layer_idx in detnet.learnable_layers_idxs:

		### Load explanations
		savedir = get_lrp_savedir(model_savedir=det_model_savedir, attack_method=args.attack_method, 
									rule=args.rule, layer_idx=layer_idx)
		det_lrp = load_from_pickle(path=savedir, filename="det_lrp")
		det_attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

		savedir = get_lrp_savedir(model_savedir=adv_model_savedir, attack_method=args.attack_method, 
									rule=args.rule, layer_idx=layer_idx)
		adv_lrp = load_from_pickle(path=savedir, filename="det_lrp")
		adv_attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

		savedir = get_lrp_savedir(model_savedir=bay_model_savedir, attack_method=args.attack_method, 
	                          	  rule=args.rule, layer_idx=layer_idx, lrp_method=args.lrp_method)
		bay_lrp=[]
		bay_attack_lrp=[]
		for n_samples in n_samples_list:
			bay_lrp.append(load_from_pickle(path=savedir, filename="bay_lrp_samp="+str(n_samples)))
			bay_attack_lrp.append(load_from_pickle(path=savedir, filename="bay_attack_lrp_samp="+str(n_samples)))

		n_images = det_lrp.shape[0]
		if det_attack_lrp.shape[0]!=n_images or bay_lrp[0].shape[0]!=n_inputs or bay_attack_lrp[0].shape[0]!=n_inputs:
			print("det_lrp.shape[0] =", det_lrp.shape[0])
			print("det_attack_lrp.shape[0] =", det_attack_lrp.shape[0])
			print("bay_lrp[0].shape[0] =", bay_lrp[0].shape[0])
			print("bay_attack_lrp[0].shape[0] =", bay_attack_lrp[0].shape[0])
			raise ValueError("Inconsistent n_inputs")

		### Normalize heatmaps

		if args.normalize:
			for im_idx in range(det_lrp.shape[0]):
				det_lrp[im_idx] = normalize(det_lrp[im_idx])
				det_attack_lrp[im_idx] = normalize(det_attack_lrp[im_idx])

				adv_lrp[im_idx] = normalize(adv_lrp[im_idx])
				adv_attack_lrp[im_idx] = normalize(adv_attack_lrp[im_idx])

				for samp_idx in range(len(n_samples_list)):
					bay_lrp[samp_idx][im_idx] = normalize(bay_lrp[samp_idx][im_idx])
					bay_attack_lrp[samp_idx][im_idx] = normalize(bay_attack_lrp[samp_idx][im_idx])

		### Evaluate explanations

		det_preds, det_atk_preds, det_softmax_robustness, det_successful_idxs, det_failed_idxs = evaluate_attack(net=detnet, 
						x_test=images, x_attack=det_attack, y_test=y_test, device=args.device, return_classification_idxs=True)
		det_softmax_robustness = det_softmax_robustness.detach().cpu().numpy()

		det_lrp_robustness, det_lrp_pxl_idxs = lrp_robustness(original_heatmaps=det_lrp, 
															  adversarial_heatmaps=det_attack_lrp, 
															  topk=topk, method=lrp_robustness_method)
		
		det_norm = lrp_distances(det_lrp, det_attack_lrp, axis_norm=1).detach().cpu().numpy()

		adv_preds, adv_atk_preds, adv_softmax_robustness, adv_successful_idxs, adv_failed_idxs = evaluate_attack(net=advnet, 
						x_test=images, x_attack=adv_attack, y_test=y_test, device=args.device, return_classification_idxs=True)
		adv_softmax_robustness = adv_softmax_robustness.detach().cpu().numpy()

		adv_lrp_robustness, adv_lrp_pxl_idxs = lrp_robustness(original_heatmaps=adv_lrp, 
															  adversarial_heatmaps=adv_attack_lrp, 
															  topk=topk, method=lrp_robustness_method)
		
		adv_norm = lrp_distances(adv_lrp, adv_attack_lrp, axis_norm=1).detach().cpu().numpy()

		bay_preds=[]
		bay_atk_preds=[]
		bay_softmax_robustness=[]
		bay_failed_idxs=[]
		bay_lrp_robustness=[]
		bay_norm=[]

		for samp_idx, n_samples in enumerate(n_samples_list):

			preds, atk_preds, softmax_rob, successf_idxs, failed_idxs = evaluate_attack(net=bayesnet, x_test=images, 
														 x_attack=bay_attack[samp_idx], y_test=y_test, device=args.device, 
														 n_samples=n_samples, return_classification_idxs=True)
			bay_preds.append(preds)
			bay_atk_preds.append(atk_preds)
			bay_softmax_robustness.append(softmax_rob.detach().cpu().numpy())
			bay_failed_idxs.append(failed_idxs)

			robustness, pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp[samp_idx], 
												 adversarial_heatmaps=bay_attack_lrp[samp_idx], 
												 topk=topk, method=lrp_robustness_method)
			bay_lrp_robustness.append(robustness)

			bay_norm.append(lrp_distances(bay_lrp[samp_idx], 
											bay_attack_lrp[samp_idx], 
											axis_norm=1).detach().cpu().numpy())

		det_lrp_robustness_layers.append(det_lrp_robustness)
		adv_lrp_robustness_layers.append(adv_lrp_robustness)
		bay_lrp_robustness_layers.append(bay_lrp_robustness)

		det_failed_idxs_layers.append(det_failed_idxs)
		adv_failed_idxs_layers.append(adv_failed_idxs)
		bay_failed_idxs_layers.append(bay_failed_idxs)

		det_norm_layers.append(det_norm)
		adv_norm_layers.append(adv_norm)
		bay_norm_layers.append(bay_norm)

		det_softmax_robustness_layers.append(det_softmax_robustness)
		adv_softmax_robustness_layers.append(adv_softmax_robustness)
		bay_softmax_robustness_layers.append(bay_softmax_robustness)

	det_lrp_robustness_topk.append(det_lrp_robustness_layers)
	adv_lrp_robustness_topk.append(adv_lrp_robustness_layers)
	bay_lrp_robustness_topk.append(bay_lrp_robustness_layers)

	det_failed_idxs_topk.append(det_failed_idxs_layers)
	adv_failed_idxs_topk.append(adv_failed_idxs_layers)
	bay_failed_idxs_topk.append(bay_failed_idxs_layers)

	det_norm_topk.append(det_norm_layers)
	adv_norm_topk.append(adv_norm_layers)
	bay_norm_topk.append(bay_norm_layers)

	det_softmax_robustness_topk.append(det_softmax_robustness_layers)
	adv_softmax_robustness_topk.append(adv_softmax_robustness_layers)
	bay_softmax_robustness_topk.append(bay_softmax_robustness_layers)

### Plots

savedir = get_lrp_savedir(model_savedir=bay_model_savedir, attack_method=args.attack_method, 
                      	  rule=args.rule, lrp_method=args.lrp_method)

filename="lrp_robustness_"+m["dataset"]+"_"+str(bayesnet.inference)+"_images="+str(n_inputs)+"_rule="+str(args.rule)\
		  +"_samples="+str(n_samples)+"_atk="+str(args.attack_method)+"_model_idx="+str(args.model_idx)

if args.normalize:
	filename+="_norm"

plot_lrp.lrp_layers_robustness_differences(
						det_lrp_robustness=det_lrp_robustness_topk,
						adv_lrp_robustness=adv_lrp_robustness_topk,
						bay_lrp_robustness=bay_lrp_robustness_topk,
						n_samples_list=n_samples_list,
						topk_list=topk_list,
						n_original_images=len(images),
						learnable_layers_idxs=detnet.learnable_layers_idxs,
						savedir=os.path.join(TESTS,'figures/layers_robustness'), 
						filename="diff_"+filename+"_layers")
