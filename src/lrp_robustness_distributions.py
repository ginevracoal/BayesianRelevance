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
from attacks.gradient_based import evaluate_attack
from attacks.run_attacks import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=500, type=int, help="Number of test points")
parser.add_argument("--topk", default=30, type=int, help="Top k most relevant pixels.")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="fullBNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--lrp_method", default="avg_heatmap", type=str, help="avg_prediction, avg_heatmap")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--redBNN_layer_idx", default=-1, type=int, help="Bayesian layer idx in redBNN.")
parser.add_argument("--normalize", default=False, type=eval, help="Normalize lrp heatmaps.")
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

lrp_robustness_method = "imagewise"
n_samples_list=[100]
n_inputs=100 if args.debug else args.n_inputs
topk=args.topk

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

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

det_attack = load_attack(method=args.attack_method, model_savedir=model_savedir)

if args.model=="fullBNN":

	m = fullBNN_settings["model_"+str(args.model_idx)]

	model_savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
								model_idx=args.model_idx, debug=args.debug)

	bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
	bayesnet.load(savedir=model_savedir, device=args.device)

elif args.model=="redBNN":

    m = redBNN_settings["model_"+str(args.model_idx)]
    base_m = baseNN_settings["model_"+str(m["baseNN_idx"])]

    x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=m["dataset"], shuffle=False, n_inputs=n_inputs)[2:]

    basenet = baseNN(inp_shape, out_size, *list(base_m.values()))
    basenet_savedir = get_model_savedir(model="baseNN", dataset=m["dataset"], 
                      					architecture=m["architecture"], debug=args.debug, model_idx=m["baseNN_idx"])
    basenet.load(savedir=basenet_savedir, device=args.device)

    hyp = get_hyperparams(m)
    layer_idx=args.redBNN_layer_idx+basenet.n_learnable_layers+1 if args.redBNN_layer_idx<0 else args.redBNN_layer_idx
    bayesnet = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=basenet, hyperparams=hyp,
                      layer_idx=layer_idx)
    model_savedir = get_model_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                          debug=args.debug, model_idx=args.model_idx, layer_idx=layer_idx)
    bayesnet.load(savedir=model_savedir, device=args.device)

else:
	raise NotImplementedError

bay_attack=[]
for n_samples in n_samples_list:
	bay_attack.append(load_attack(method=args.attack_method, model_savedir=model_savedir, 
					  n_samples=n_samples))

mode_attack = load_attack(method=args.attack_method, model_savedir=model_savedir, 
						  n_samples=n_samples, atk_mode=True)

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)

for layer_idx in detnet.learnable_layers_idxs:

	savedir = get_lrp_savedir(model_savedir=model_savedir, attack_method=args.attack_method, 
	                          layer_idx=layer_idx, lrp_method=args.lrp_method)

	### Load explanations

	det_lrp = load_from_pickle(path=savedir, filename="det_lrp")
	det_attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

	bay_lrp=[]
	bay_attack_lrp=[]
	for n_samples in n_samples_list:
		bay_lrp.append(load_from_pickle(path=savedir, filename="bay_lrp_samp="+str(n_samples)))
		bay_attack_lrp.append(load_from_pickle(path=savedir, filename="bay_attack_lrp_samp="+str(n_samples)))

	mode_lrp = load_from_pickle(path=savedir, filename="mode_lrp_avg_post_samp="+str(n_samples))

	mode_attack_lrp=[]
	for samp_idx, n_samples in enumerate(n_samples_list):
	    mode_attack_lrp.append(load_from_pickle(path=savedir, filename="mode_attack_lrp_samp="+str(n_samples)))
	mode_attack_lrp.append(load_from_pickle(path=savedir, filename="mode_attack_lrp_avg_post_samp="+str(n_samples)))
	# mode_attack_lrp = np.array(mode_attack_lrp)

	### Normalize heatmaps

	if args.normalize:
		for im_idx in range(det_lrp.shape[0]):
			det_lrp[im_idx] = normalize(det_lrp[im_idx])
			det_attack_lrp[im_idx] = normalize(det_attack_lrp[im_idx])
			mode_lrp[im_idx] = normalize(mode_lrp[im_idx])

			for samp_idx in range(len(n_samples_list)):
				bay_lrp[samp_idx][im_idx] = normalize(bay_lrp[samp_idx][im_idx])
				bay_attack_lrp[samp_idx][im_idx] = normalize(bay_attack_lrp[samp_idx][im_idx])
				mode_attack_lrp[samp_idx][im_idx] = normalize(mode_attack_lrp[samp_idx][im_idx])

			mode_attack_lrp[samp_idx+1][im_idx] = normalize(mode_attack_lrp[samp_idx+1][im_idx])

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

	for samp_idx, n_samples in enumerate(n_samples_list):

		preds, atk_preds, softmax_rob, successf_idxs, failed_idxs = evaluate_attack(net=bayesnet, x_test=images, 
													   x_attack=bay_attack[samp_idx], y_test=y_test, device=args.device, 
													   n_samples=n_samples, return_classification_idxs=True)
		bay_softmax_robustness.append(softmax_rob.detach().cpu().numpy())
		bay_successful_idxs.append(successf_idxs)
		bay_failed_idxs.append(failed_idxs)
		bay_preds.append(preds)
		bay_atk_preds.append(atk_preds)

		robustness, pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp[samp_idx], 
												   adversarial_heatmaps=bay_attack_lrp[samp_idx], 
												   topk=topk, method=lrp_robustness_method)
		bay_lrp_robustness.append(robustness)
		bay_lrp_pxl_idxs.append(pxl_idxs)

		robustness, pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp[samp_idx][successf_idxs], 
												   adversarial_heatmaps=bay_attack_lrp[samp_idx][successf_idxs], 
												   topk=topk, method=lrp_robustness_method)
		succ_bay_lrp_robustness.append(robustness)
		succ_bay_lrp_pxl_idxs.append(pxl_idxs)

		robustness, pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp[samp_idx][failed_idxs], 
												   adversarial_heatmaps=bay_attack_lrp[samp_idx][failed_idxs], 
												   topk=topk, method=lrp_robustness_method)
		fail_bay_lrp_robustness.append(robustness)
		fail_bay_lrp_pxl_idxs.append(pxl_idxs)

	mode_preds=[]
	mode_atk_preds=[]
	mode_softmax_robustness=[]
	mode_successful_idxs=[]
	mode_failed_idxs=[]
	mode_lrp_robustness=[]
	mode_lrp_pxl_idxs=[]
	succ_mode_lrp_robustness=[]
	succ_mode_lrp_pxl_idxs=[]
	fail_mode_lrp_robustness=[]
	fail_mode_lrp_pxl_idxs=[]

	for samp_idx, n_samples in enumerate(n_samples_list):

		preds, atk_preds, softmax_rob, succ_idxs, fail_idxs = evaluate_attack(net=bayesnet, 
														   x_test=images, x_attack=mode_attack,
														   y_test=y_test, device=args.device, n_samples=n_samples, 
														   return_classification_idxs=True)

		mode_preds.append(preds) 
		mode_atk_preds.append(atk_preds)
		mode_softmax_robustness.append(softmax_rob.detach().cpu().numpy()) 
		mode_successful_idxs.append(succ_idxs)
		mode_failed_idxs.append(failed_idxs)

		robustness, pxl_idxs = lrp_robustness(original_heatmaps=mode_lrp, 
											  adversarial_heatmaps=mode_attack_lrp[samp_idx], 
											  topk=topk, method=lrp_robustness_method)
		mode_lrp_robustness.append(robustness)
		mode_lrp_pxl_idxs.append(pxl_idxs)

		robustness, pxl_idxs = lrp_robustness(original_heatmaps=mode_lrp[succ_idxs], 
											  adversarial_heatmaps=mode_attack_lrp[samp_idx][succ_idxs], 
											  topk=topk, method=lrp_robustness_method)
		succ_mode_lrp_robustness.append(robustness)
		succ_mode_lrp_pxl_idxs.append(pxl_idxs)

		robustness, pxl_idxs = lrp_robustness(original_heatmaps=mode_lrp[fail_idxs], 
											  adversarial_heatmaps=mode_attack_lrp[samp_idx][fail_idxs], 
											  topk=topk, method=lrp_robustness_method)
		fail_mode_lrp_robustness.append(robustness) 
		fail_mode_lrp_pxl_idxs.append(pxl_idxs)

	preds, atk_preds, softmax_rob, succ_idxs, fail_idxs = evaluate_attack(net=bayesnet, 
													   x_test=images, x_attack=mode_attack, avg_posterior=True,
													   y_test=y_test, device=args.device, n_samples=n_samples, 
													   return_classification_idxs=True)
	mode_preds.append(preds) 
	mode_atk_preds.append(atk_preds)
	mode_softmax_robustness.append(softmax_rob.detach().cpu().numpy()) 
	mode_successful_idxs.append(succ_idxs)
	mode_failed_idxs.append(failed_idxs)

	robustness, pxl_idxs = lrp_robustness(original_heatmaps=mode_lrp, 
										  adversarial_heatmaps=mode_attack_lrp[samp_idx+1], 
										  topk=topk, method=lrp_robustness_method)
	mode_lrp_robustness.append(robustness)
	mode_lrp_pxl_idxs.append(pxl_idxs)

	robustness, pxl_idxs = lrp_robustness(original_heatmaps=mode_lrp[succ_idxs], 
										  adversarial_heatmaps=mode_attack_lrp[samp_idx+1][succ_idxs], 
										  topk=topk, method=lrp_robustness_method)
	succ_mode_lrp_robustness.append(robustness)
	succ_mode_lrp_pxl_idxs.append(pxl_idxs)

	robustness, pxl_idxs = lrp_robustness(original_heatmaps=mode_lrp[fail_idxs], 
										  adversarial_heatmaps=mode_attack_lrp[samp_idx+1][fail_idxs], 
										  topk=topk, method=lrp_robustness_method)
	fail_mode_lrp_robustness.append(robustness) 
	fail_mode_lrp_pxl_idxs.append(pxl_idxs)


	### Plots

	filename = lrp_robustness_method
	if args.normalize:
		filename+="_norm"

	plot_attacks_explanations(images=images, 
							  explanations=det_lrp, 
							  attacks=det_attack, 
							  attacks_explanations=det_attack_lrp, 
							  predictions=det_preds.argmax(-1),
							  attacks_predictions=det_atk_preds.argmax(-1),
							  successful_attacks_idxs=det_successful_idxs,
							  failed_attacks_idxs=det_failed_idxs,
							  labels=labels, lrp_rob_method=lrp_robustness_method,
							  rule=args.rule, savedir=savedir, pxl_idxs=det_lrp_pxl_idxs,
							  filename="det_lrp_attacks_"+filename, 
							  layer_idx=layer_idx)

	for samp_idx, n_samples in enumerate(n_samples_list):

		plot_attacks_explanations(images=images, 
								  explanations=bay_lrp[samp_idx], 
								  attacks=bay_attack[samp_idx], 
								  attacks_explanations=bay_attack_lrp[samp_idx],
								  predictions=bay_preds[samp_idx].argmax(-1),
								  attacks_predictions=bay_atk_preds[samp_idx].argmax(-1),
								  successful_attacks_idxs=bay_successful_idxs[samp_idx],
								  failed_attacks_idxs=bay_failed_idxs[samp_idx],
								  labels=labels, lrp_rob_method=lrp_robustness_method,
								  rule=args.rule, savedir=savedir, pxl_idxs=bay_lrp_pxl_idxs[samp_idx],
								  filename="bay_lrp_attacks_samp="+str(n_samples)+"_"+filename, 
								  layer_idx=layer_idx)

	mode_vs_mode_idx = samp_idx+1 # lrp mode computations 

	filename=args.rule+"_lrp_robustness"+m["dataset"]+"_images="+str(n_inputs)+\
			 "_samples="+str(n_samples)+"_pxls="+str(topk)+"_atk="+str(args.attack_method)+"_layeridx="+str(layer_idx)
	if args.normalize:
		filename+="_norm"
			 
	plot_lrp.lrp_imagewise_robustness_distributions(
				det_successful_lrp_robustness=succ_det_lrp_robustness,
				det_failed_lrp_robustness=fail_det_lrp_robustness,
				bay_successful_lrp_robustness=succ_bay_lrp_robustness,
				bay_failed_lrp_robustness=fail_bay_lrp_robustness,
				mode_successful_lrp_robustness=succ_mode_lrp_robustness,
				mode_failed_lrp_robustness=fail_mode_lrp_robustness,
				n_samples_list=n_samples_list,
				n_original_images=len(images),
				savedir=savedir, 
				filename="dist_"+filename)

	# if layer_idx == detnet.learnable_layers_idxs[-1]:
	plot_lrp.lrp_robustness_scatterplot(
				adversarial_robustness=det_softmax_robustness, 
				bayesian_adversarial_robustness=bay_softmax_robustness,
				mode_adversarial_robustness=mode_softmax_robustness[mode_vs_mode_idx],
				lrp_robustness=det_lrp_robustness, 
				bayesian_lrp_robustness=bay_lrp_robustness,
				mode_lrp_robustness=mode_lrp_robustness[mode_vs_mode_idx],
				n_samples_list=n_samples_list,
				savedir=savedir, 
				filename="scatterplot_"+filename)
