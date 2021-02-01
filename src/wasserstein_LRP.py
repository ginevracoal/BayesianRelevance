import os
import torch
import argparse
import numpy as np
import torchvision

from utils.data import *
from utils.networks import *
from utils.savedir import *
from utils.seeding import *
from attacks.gradient_based import *
from attacks.gradient_based import load_attack
from utils.lrp import *
from plot.lrp_heatmaps import *
import plot.lrp_distributions as plot_lrp

from networks.baseNN import *
from networks.fullBNN import *
from networks.redBNN import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=1000, type=int, help="Number of test points")
parser.add_argument("--topk", default=300, type=int, help="Top k most relevant pixels.")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="fullBNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--lrp_method", default="pixelwise", type=str, help="intersection, union, average")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--layer_idx", default=-1, type=int, help="Layer idx for LRP computation.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--explain_attacks", default=False, type=eval)
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")  
args = parser.parse_args()

n_samples_list=[2,5] if args.debug else [5,10,50]
n_inputs=500 if args.debug else args.n_inputs
topk=100 if args.debug else args.topk

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if args.device=="cuda":
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Load models

model = baseNN_settings["model_"+str(args.model_idx)]

_, _, x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=model["dataset"], 
															shuffle=True, n_inputs=n_inputs)
model_savedir = get_savedir(model="baseNN", dataset=model["dataset"], architecture=model["architecture"], 
					  debug=args.debug, model_idx=args.model_idx)
detnet = baseNN(inp_shape, num_classes, *list(model.values()))
detnet.load(savedir=model_savedir, device=args.device)

if args.model=="fullBNN":

	m = fullBNN_settings["model_"+str(args.model_idx)]

	model_savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
								model_idx=args.model_idx, debug=args.debug)

	bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
	bayesnet.load(savedir=model_savedir, device=args.device)

else:
	raise NotImplementedError

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)
savedir = os.path.join(model_savedir, "lrp/wasserstein/pkl/")

### Deterministic explanations

if args.load:
	det_attack = load_from_pickle(path=savedir, filename="det_attack")
	det_lrp = load_from_pickle(path=savedir, filename="det_lrp")
	det_attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")

else:

	det_attack = attack(net=detnet, x_test=images, y_test=y_test, device=args.device, method=args.attack_method)
	det_lrp = compute_explanations(images, detnet, rule=args.rule)
	det_attack_lrp = compute_explanations(det_attack, detnet, rule=args.rule)

	save_to_pickle(det_attack, path=savedir, filename="det_attack")
	save_to_pickle(det_lrp, path=savedir, filename="det_lrp")
	save_to_pickle(det_attack_lrp, path=savedir, filename="det_attack_lrp")

det_preds, det_atk_preds, det_softmax_robustness, det_successful_idxs = attack_evaluation(net=detnet, x_test=images, 
				x_attack=det_attack, y_test=y_test, device=args.device, return_successful_idxs=True)
det_lrp_robustness, det_lrp_pxl_idxs = lrp_robustness(original_heatmaps=det_lrp, 
											adversarial_heatmaps=det_attack_lrp, 
											  topk=topk, method=args.lrp_method)

succ_wess_dist = lrp_wasserstein_distance(det_lrp[det_successful_idxs], 
										  det_attack_lrp[det_successful_idxs], 
										  det_lrp_pxl_idxs)
det_failed_idxs = np.setdiff1d(np.arange(len(images)), det_successful_idxs)
fail_wess_dist = lrp_wasserstein_distance(det_lrp[det_failed_idxs], 
										  det_attack_lrp[det_failed_idxs], 
										  det_lrp_pxl_idxs)

### Bayesian explanations

bay_attack=[]
bay_lrp=[]
bay_attack_lrp=[]

if args.load:

	for n_samples in n_samples_list:
		bay_attack.append(load_from_pickle(path=savedir, filename="bay_attack_samp="+str(n_samples)))
		bay_lrp.append(load_from_pickle(path=savedir, filename="bay_lrp_samp="+str(n_samples)))
		bay_attack_lrp.append(load_from_pickle(path=savedir, filename="bay_attack_lrp_samp="+str(n_samples)))

	mode_attack = load_from_pickle(path=savedir, filename="mode_attack_samp="+str(n_samples))
	mode_lrp = load_from_pickle(path=savedir, filename="mode_lrp_samp="+str(n_samples))
	mode_attack_lrp = load_from_pickle(path=savedir, filename="mode_attack_lrp_samp="+str(n_samples))

else:

	for samp_idx, n_samples in enumerate(n_samples_list):

		bay_attack.append(attack(net=bayesnet, x_test=images, y_test=y_test, n_samples=n_samples,
							device=args.device, method=args.attack_method))
		bay_lrp.append(compute_explanations(images, bayesnet, rule=args.rule, n_samples=n_samples))
		bay_attack_lrp.append(compute_explanations(bay_attack[samp_idx], bayesnet, 
												   rule=args.rule, n_samples=n_samples))

		save_to_pickle(bay_attack[samp_idx], path=savedir, filename="bay_attack_samp="+str(n_samples))
		save_to_pickle(bay_lrp[samp_idx], path=savedir, filename="bay_lrp_samp="+str(n_samples))
		save_to_pickle(bay_attack_lrp[samp_idx], path=savedir, filename="bay_attack_lrp_samp="+str(n_samples))
	
	mode_attack = attack(net=bayesnet, x_test=images, y_test=y_test, n_samples=n_samples,
					  device=args.device, method=args.attack_method, avg_posterior=True)
	mode_lrp = compute_explanations(images, bayesnet, rule=args.rule, n_samples=n_samples, avg_posterior=True)
	mode_attack_lrp = compute_explanations(mode_attack, bayesnet, rule=args.rule, 
											n_samples=n_samples, avg_posterior=True)
	
	save_to_pickle(mode_attack, path=savedir, filename="mode_attack_samp="+str(n_samples))
	save_to_pickle(mode_lrp, path=savedir, filename="mode_lrp_samp="+str(n_samples))
	save_to_pickle(mode_attack_lrp, path=savedir, filename="mode_attack_lrp_samp="+str(n_samples))

### Evaluations

bay_softmax_robustness=[]
bay_successful_idxs=[]
bay_lrp_robustness=[]
bay_lrp_pxl_idxs=[]
bay_preds=[]
bay_atk_preds=[]

for samp_idx, n_samples in enumerate(n_samples_list):

	preds, atk_preds, softmax_rob, successf_idxs = attack_evaluation(net=bayesnet, x_test=images, x_attack=bay_attack[samp_idx],
						   y_test=y_test, device=args.device, n_samples=n_samples, return_successful_idxs=True)
	bay_softmax_robustness.append(softmax_rob.detach().cpu().numpy())
	bay_successful_idxs.append(successf_idxs)
	bay_preds.append(preds)
	bay_atk_preds.append(atk_preds)

	bay_lrp_rob, lrp_pxl_idxs = lrp_robustness(original_heatmaps=bay_lrp[samp_idx], 
												  adversarial_heatmaps=bay_attack_lrp[samp_idx], 
												  topk=topk, method=args.lrp_method)
	bay_lrp_robustness.append(bay_lrp_rob)
	bay_lrp_pxl_idxs.append(lrp_pxl_idxs)


mode_preds, mode_atk_preds, mode_softmax_robustness, mode_successful_idxs = attack_evaluation(net=bayesnet, 
					   x_test=images, x_attack=mode_attack,
					   y_test=y_test, device=args.device, n_samples=n_samples, 
					   return_successful_idxs=True)
mode_lrp_robustness, mode_pxl_idxs = lrp_robustness(original_heatmaps=mode_lrp, 
												adversarial_heatmaps=mode_attack_lrp, 
												topk=topk, method=args.lrp_method)
mode_softmax_robustness = mode_softmax_robustness.detach().cpu().numpy()

succ_bay_wess_dist=[]
fail_bay_wess_dist=[]
for samp_idx, n_samples in enumerate(n_samples_list):

	succ_im_idxs = bay_successful_idxs[samp_idx]
	succ_bay_wess_dist.append(lrp_wasserstein_distance(bay_lrp[samp_idx][succ_im_idxs],
													   bay_attack_lrp[samp_idx][succ_im_idxs], 
													   bay_lrp_pxl_idxs[samp_idx]))

	failed_im_idxs = np.setdiff1d(np.arange(len(images)), succ_im_idxs)
	fail_bay_wess_dist.append(lrp_wasserstein_distance(bay_lrp[samp_idx][failed_im_idxs], 
											  bay_attack_lrp[samp_idx][failed_im_idxs], 
											  bay_lrp_pxl_idxs[samp_idx]))

succ_wess_dist=np.array(succ_wess_dist)
fail_wess_dist=np.array(fail_wess_dist)
succ_bay_wess_dist=np.array(succ_bay_wess_dist)
fail_bay_wess_dist=np.array(fail_bay_wess_dist)

### Plots

savedir = os.path.join(model_savedir, "lrp/wasserstein/")

plot_attacks_explanations(images=images, 
	                      explanations=det_lrp, 
						  attacks=det_attack, 
						  attacks_explanations=det_attack_lrp, 
						  predictions=det_preds.argmax(-1),
						  attacks_predictions=det_atk_preds.argmax(-1),
						  successful_attacks_idxs=det_successful_idxs,
						  labels=labels, lrp_method=args.lrp_method,
						  rule=args.rule, savedir=savedir, pxl_idxs=det_lrp_pxl_idxs,
						  filename=args.lrp_method+"_det_lrp_attacks", layer_idx=-1)

for samp_idx, n_samples in enumerate(n_samples_list):

	plot_attacks_explanations(images=images, 
							  explanations=bay_lrp[samp_idx], 
							  attacks=bay_attack[samp_idx], 
							  attacks_explanations=bay_attack_lrp[samp_idx],
							  predictions=bay_preds[samp_idx].argmax(-1),
							  attacks_predictions=bay_atk_preds[samp_idx].argmax(-1),
							  successful_attacks_idxs=bay_successful_idxs[samp_idx],
							  labels=labels, lrp_method=args.lrp_method,
							  rule=args.rule, savedir=savedir, pxl_idxs=bay_lrp_pxl_idxs[samp_idx],
							  filename=args.lrp_method+"_bay_lrp_attacks_samp="+str(n_samples), layer_idx=-1)

plot_attacks_explanations(images=images, 
						  explanations=mode_lrp, 
						  attacks=mode_attack, 
						  attacks_explanations=mode_attack_lrp, 
						  predictions=mode_preds.argmax(-1),
						  attacks_predictions=mode_atk_preds.argmax(-1),
						  successful_attacks_idxs=mode_successful_idxs,
						  labels=labels, lrp_method=args.lrp_method,
						  rule=args.rule, savedir=savedir, pxl_idxs=mode_pxl_idxs,
						  filename=args.lrp_method+"_mode_lrp_attacks_samp="+str(n_samples), layer_idx=-1)


filename=args.rule+"_lrp_wasserstein_"+m["dataset"]+"_images="+str(n_inputs)+\
		 "_pxls="+str(topk)+"_atk="+str(args.attack_method)

plot_lrp.plot_wasserstein_dist(det_successful_atks_wess_dist=succ_wess_dist, 
							   det_failed_atks_wess_dist=fail_wess_dist, 
							   bay_successful_atks_wess_dist=succ_bay_wess_dist, 
							   bay_failed_atks_wess_dist=fail_bay_wess_dist,
							   increasing_n_samples=n_samples_list, filename=filename, savedir=savedir)

# plot_lrp.lrp_robustness_scatterplot(adversarial_robustness=det_softmax_robustness, 
# 									bayesian_adversarial_robustness=bay_softmax_robustness,
# 									mode_adversarial_robustness=mode_softmax_robustness,
# 									lrp_robustness=det_lrp_robustness, 
# 									bayesian_lrp_robustness=bay_lrp_robustness,
# 									mode_lrp_robustness=mode_lrp_robustness,
# 									n_samples_list=n_samples_list,
# 									savedir=savedir, filename="scatterplot_"+filename)
