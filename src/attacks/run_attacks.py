import torch
import numpy as np
from tqdm import tqdm
import copy
from torch import autograd

from utils.data import *
from utils.seeding import *
from utils.savedir import *
from plot.attacks import plot_grid_attacks
from attacks.robustness_measures import softmax_robustness
from torch.autograd.gradcheck import zero_gradients

from attacks.deeprobust.fgsm import FGSM
from attacks.deeprobust.pgd import PGD
from attacks.deeprobust.cw import CarliniWagner
from attacks.deepfool import DeepFool
from attacks.beta import Beta
from attacks.topk import Topk

from deeprobust.image.attack.Nattack import NATTACK
from deeprobust.image.attack.YOPOpgd import FASTPGD

from utils.networks import relu_to_softplus


def attack(net, x_test, y_test, device, method, hyperparams={}, n_samples=None, sample_idxs=None, avg_posterior=False,
			verbose=True):

	if verbose:
		print(f"\n\nCrafting {method} attacks")

	net.to(device)
	x_test, y_test = x_test.to(device), y_test.to(device)

	if x_test.shape[1] == 3:  # todo: test on 3-channels
		data_mean = torch.empty(3)
		for i in [0, 1, 2]:
			data_mean[i], data_std[i] = x_test[:,i].mean().item(), x_test[:,i].std().item()
	else:
		data_mean, data_std = x_test.mean().item(), x_test.std().item()

	adversarial_attacks = []

	if n_samples is None or avg_posterior is True:
		net.avg_posterior=avg_posterior

		iterable_x_test = tqdm(range(len(x_test))) if verbose else range(len(x_test))
		for idx in iterable_x_test:

			image = x_test[idx].unsqueeze(0)
			label = y_test[idx].argmax(-1).unsqueeze(0)
			num_classes = len(y_test[idx])
			hyperparams['num_classes']=num_classes

			if method=='beta':
				random.seed(idx)
				hyperparams['target_image'] = random.choice(x_test).unsqueeze(0)
				hyperparams['data_mean'] = data_mean
				hyperparams['data_std'] = data_std

			perturbed_image = run_attack(net=net, image=image, label=label, method=method, 
												 device=device, hyperparams=hyperparams)
			perturbed_image = torch.clamp(perturbed_image, 0., 1.)
			adversarial_attacks.append(perturbed_image)

		del net.avg_posterior

	else:

		if sample_idxs is not None:
			if len(sample_idxs) != n_samples:
				raise ValueError("Number of sample_idxs should match number of samples.")
		else:
			sample_idxs = list(range(n_samples))

		iterable_x_test = tqdm(range(len(x_test))) if verbose else range(len(x_test))
		for idx in iterable_x_test:

			image = x_test[idx].unsqueeze(0)
			label = y_test[idx].argmax(-1).unsqueeze(0)
			num_classes = len(y_test[idx])
			hyperparams['num_classes']=num_classes

			if method=='beta':
				random.seed(idx)
				hyperparams['target_image'] = random.choice(x_test).unsqueeze(0)
				hyperparams['data_mean'] = data_mean
				hyperparams['data_std'] = data_std

			samples_attacks=[]
			for idx in sample_idxs:
				net.n_samples, net.sample_idxs = (1, [idx])


				perturbed_image = run_attack(net=net, image=image, label=label, method=method, 
													 device=device, hyperparams=hyperparams)
				perturbed_image = torch.clamp(perturbed_image, 0., 1.)
				samples_attacks.append(perturbed_image)

			adversarial_attacks.append(torch.stack(samples_attacks).mean(0))

		del net.n_samples, net.sample_idxs

	adversarial_attacks = torch.cat(adversarial_attacks)
	return adversarial_attacks

def run_attack(net, image, label, method, device, hyperparams=None):

	assert 'epsilon' in hyperparams

	if method == "fgsm":
		adversary = FGSM
		adversary_params = {'epsilon':hyperparams['epsilon'], 'order': np.inf, 'clip_max':None, 'clip_min':None}
		adv = adversary(net, device)
		perturbed_image = adv.generate(image, label, **adversary_params)

	elif method == "pgd":
		adversary = PGD
		adversary_params = {'epsilon':hyperparams['epsilon'], 'clip_max': 1.0, 'clip_min': 0.0, 'print_process': False}
		adv = adversary(net, device)
		perturbed_image = adv.generate(image, label, **adversary_params)

	elif method == "cw":
		adversary = CarliniWagner
		adversary_params = {'confidence': 1e-4, 'clip_max': 1, 'clip_min': 0, 'max_iterations': 1000,
							'initial_const': 1e-2, 'binary_search_steps': 5, 'learning_rate': 5e-3,
							'abort_early': True}
		adv = adversary(net, device)
		target_label = 1 # todo: set to random class different from true label
		perturbed_image = adv.generate(image, label, target_label=target_label, **adversary_params)

	elif method == "deepfool":
		perturbed_image=DeepFool(image, net=net, num_classes=10, overshoot=0.02, max_iter=10)[-1].squeeze(0)

	elif method == "beta":
		if hasattr(net, "basenet"):
			net.basenet = relu_to_softplus(net.basenet)
		else:
			net.model = relu_to_softplus(net.model)
			
		perturbed_image = Beta(image, model=net, target_image=hyperparams['target_image'], 
							   iters=hyperparams['iters'], lrp_rule=hyperparams['lrp_rule'],
							   data_mean=hyperparams['data_mean'], data_std=hyperparams['data_std'])

	elif method == "topk":
		perturbed_image = Topk(image, model=net, epsilon=hyperparams['epsilon'], iters=hyperparams['iters'], 
								lrp_rule=hyperparams['lrp_rule'])

	return perturbed_image

def save_attack(inputs, attacks, method, model_savedir, atk_mode=False, n_samples=None):

	filename, savedir = get_atk_filename_savedir(attack_method=method, model_savedir=model_savedir, 
												 atk_mode=atk_mode, n_samples=n_samples)
	save_to_pickle(data=attacks, path=savedir, filename=filename)

	set_seed(0)
	idxs = np.random.choice(len(inputs), 10, replace=False)
	original_images_plot = torch.stack([inputs[i].squeeze() for i in idxs])
	perturbed_images_plot = torch.stack([attacks[i].squeeze() for i in idxs])
	plot_grid_attacks(original_images=original_images_plot.detach().cpu(), 
					  perturbed_images=perturbed_images_plot.detach().cpu(), 
					  filename=filename, savedir=savedir)


def load_attack(method, model_savedir, atk_mode=False, n_samples=None):
	filename, savedir = get_atk_filename_savedir(attack_method=method, model_savedir=model_savedir, 
												 atk_mode=atk_mode, n_samples=n_samples)
	return load_from_pickle(path=savedir, filename=filename)

