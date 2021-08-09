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

from deeprobust.image.attack.Nattack import NATTACK
from deeprobust.image.attack.YOPOpgd import FASTPGD

def attack(net, x_test, y_test, device, method, hyperparams={}, n_samples=None, sample_idxs=None, avg_posterior=False,
			verbose=True):

	if verbose:
		print(f"\n\nCrafting {method} attacks")

	net.to(device)
	x_test, y_test = x_test.to(device), y_test.to(device)

	adversarial_attacks = [] 

	if n_samples is None or avg_posterior is True:
		net.avg_posterior=avg_posterior

		iterable_x_test = tqdm(range(len(x_test))) if verbose else range(len(x_test))
		for idx in iterable_x_test:

			image = x_test[idx].unsqueeze(0)
			label = y_test[idx].argmax(-1).unsqueeze(0)
			num_classes = len(y_test[idx])
			hyperparams['num_classes']=num_classes

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

	if method == "fgsm":
		adversary = FGSM
		adversary_params = {'epsilon': 0.2, 'order': np.inf, 'clip_max':None, 'clip_min':None}
		adv = adversary(net, device)
		perturbed_image = adv.generate(image, label, **adversary_params)

	elif method == "pgd":
		adversary = PGD
		adversary_params = {'epsilon': 0.2, 'clip_max': 1.0, 'clip_min': 0.0, 'print_process': False}
		adv = adversary(net, device)
		perturbed_image = adv.generate(image, label, **adversary_params)

	elif method == "cw":
		adversary = CarliniWagner
		adversary_params = {'confidence': 1e-4, 'clip_max': 1, 'clip_min': 0, 'max_iterations': 1000,
							'initial_const': 1e-2, 'binary_search_steps': 5, 'learning_rate': 5e-3,
							'abort_early': True}
		adv = adversary(net, device)
		target_label = 1 # todo: set to random different class
		perturbed_image = adv.generate(image, label, target_label=target_label, **adversary_params)

	elif method == "deepfool":
		perturbed_image=DeepFool(image, net=net, num_classes=10, overshoot=0.02, max_iter=10)[-1].squeeze(0)

	# elif method == "nattack": # runs on CPU only
	# 	adversary = NATTACK
	# 	adversary_params = {'classnum':hyperparams['num_classes']}
	# 	adv = adversary(net, "cpu")
	# 	dataloader = DataLoader(dataset=list(zip(image, label)), batch_size=1)
	# 	perturbed_image = adv.generate(dataloader=dataloader, **adversary_params)

	# 	print(perturbed_image)

	# elif method == "yopo":
	# 	adversary = FASTPGD
	# 	adversary_params = {}
	# 	adv = adversary(net, device)
	# 	perturbed_image = adv.generate(image, label, **adversary_params)

	return perturbed_image

# def attack_torchvision(network, dataloader, method, device, savedir, n_samples=None, hyperparams=None):
# 	network.to(device)
# 	network.n_samples = n_samples
# 	print(f"\n{method} attack")

# 	if method == "fgsm":
# 		adversary = FGSM
# 		adversary_params = {'epsilon': 0.2, 'order': np.inf, 'clip_max': None, 'clip_min': None}

# 	elif method == "pgd":
# 		adversary = PGD
# 		adversary_params = {'epsilon': 0.2, 'clip_max': 1.0, 'clip_min': 0.0, 'print_process': False}

# 	elif method == "cw":
# 		adversary = CarliniWagner
# 		adversary_params = {'confidence': 1e-4, 'clip_max': 1, 'clip_min': 0, 'max_iterations': 1000,
# 							'initial_const': 1e-2, 'binary_search_steps': 5, 'learning_rate': 5e-3,
# 							'abort_early': True,}

# 	elif method == "nattack":
# 		adversary = NATTACK
# 		adversary_params = {}

# 	elif method == "yopo":
# 		adversary = FASTPGD
# 		adversary_params = {}

# 	elif method == "deepfool":
# 		adversary = DeepFool
# 		adversary_params = {}

# 	adversarial_data=[]

# 	# todo: sistemare calcolo gradienti

# 	for images, labels in tqdm(dataloader):

# 		for idx, image in enumerate(images):
# 			image = image.unsqueeze(0)
# 			label = labels[idx].argmax(-1).unsqueeze(0)

# 			adversary(network, device)
# 			perturbed_image = adversary.generate(image, label, **adversary_params)
# 			perturbed_image = torch.clamp(perturbed_image, 0., 1.)
# 			adversarial_data.append(perturbed_image)

# 	adversarial_data = torch.cat(adversarial_data)
# 	# todo: salvare esternamente
# 	save_attack(savedir=savedir, adversarial_data=adversarial_data, method=method, n_samples=n_samples)
# 	return adversarial_data

# def evaluate_attack_torchvision(savedir, network, dataloader, adversarial_data, device, method, n_samples=None):
# 	""" Evaluates the network on the original dataloader and its perturbed version. 
# 	When using a Bayesian network `n_samples` should be specified for the evaluation.     
# 	"""
# 	network.to(device)
# 	print(f"\nEvaluating against the attacks.")

# 	original_images_list = []   

# 	with torch.no_grad():

# 		original_outputs = []
# 		original_correct = 0.0
# 		adversarial_outputs = []
# 		adversarial_correct = 0.0

# 		for idx, (images, labels) in enumerate(dataloader):

# 			for image in images:
# 				original_images_list.append(image)

# 			images, labels = images.to(device), labels.to(device)
# 			attacks = adversarial_data[idx:idx+len(images)]

# 			out = network(images, n_samples)
# 			original_correct += torch.sum(out.argmax(-1) == labels).item()
# 			original_outputs.append(out)

# 			out = network(attacks, n_samples)
# 			adversarial_correct += torch.sum(out.argmax(-1) == labels).item()
# 			adversarial_outputs.append(out)

# 		original_accuracy = 100 * original_correct / len(dataloader.dataset)
# 		adversarial_accuracy = 100 * adversarial_correct / len(dataloader.dataset)
# 		print(f"\ntest accuracy = {original_accuracy:.2f}\tadversarial accuracy = {adversarial_accuracy:.2f}",
# 			  end="\t")

# 		original_outputs = torch.cat(original_outputs)
# 		adversarial_outputs = torch.cat(adversarial_outputs)
# 		softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

# 	_plot_attack(savedir, original_images_list, adversarial_data, method, n_samples)

# 	return original_accuracy, adversarial_accuracy, softmax_rob

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

