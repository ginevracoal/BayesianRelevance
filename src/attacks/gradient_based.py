"""
FGSM and PGD classic & bayesian adversarial attacks 
"""

import os
import sys
import copy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as nnf
from torch.utils.data import DataLoader

from utils.data import *
from utils.seeding import *
from utils.savedir import *
from utils.networks import *
from attacks.robustness_measures import *
from plot.attacks import plot_grid_attacks


DEBUG=False

def loss_gradient_sign(net, n_samples, image, label, avg_posterior, sample_idxs=None):

	if n_samples is None or avg_posterior is True:

		image.requires_grad = True
		output = net.forward(inputs=image, avg_posterior=avg_posterior)
		
		loss = torch.nn.CrossEntropyLoss()(output, label)
		net.zero_grad()
		loss.backward()
		gradient_sign = image.grad.data.sign()

	else:

		if sample_idxs is not None:
			if len(sample_idxs) != n_samples:
				raise ValueError("Number of sample_idxs should match number of samples.")
		else:
			sample_idxs = list(range(n_samples))

		loss_gradients=[]

		for idx in sample_idxs:

			x_copy = copy.deepcopy(image)
			x_copy.requires_grad = True
			output = net.forward(inputs=x_copy, n_samples=1, sample_idxs=[idx], avg_posterior=avg_posterior)

			loss = torch.nn.CrossEntropyLoss()(output.to(dtype=torch.double), label)
			net.zero_grad()
			loss.backward()
			loss_gradient = copy.deepcopy(x_copy.grad.data[0].sign())
			loss_gradients.append(loss_gradient)

		gradient_sign = torch.stack(loss_gradients,0).mean(0)

	return gradient_sign


def fgsm_attack(net, image, label, hyperparams=None, n_samples=None, sample_idxs=None, avg_posterior=False):

	epsilon = hyperparams["epsilon"] if hyperparams is not None else 0.25
	
	gradient_sign = loss_gradient_sign(net=net, n_samples=n_samples, image=image, label=label, 
									   avg_posterior=avg_posterior, sample_idxs=sample_idxs)
	perturbed_image = image + epsilon * gradient_sign
	perturbed_image = torch.clamp(perturbed_image, 0, 1)
	return perturbed_image


def pgd_attack(net, image, label, hyperparams=None, n_samples=None, sample_idxs=None, avg_posterior=False):

	if hyperparams is not None: 
		epsilon, alpha, iters = (hyperparams["epsilon"], 2/image.max(), 40)
	else:
		epsilon, alpha, iters = (0.25, 2/225, 40)

	original_image = copy.deepcopy(image)
	
	for i in range(iters):

		gradient_sign = loss_gradient_sign(net=net, n_samples=n_samples, image=image, label=label, 
										   avg_posterior=avg_posterior, sample_idxs=sample_idxs)
		perturbed_image = image + alpha * gradient_sign
		eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
		image = torch.clamp(original_image + eta, min=0, max=1)

	perturbed_image = image.detach()
	return perturbed_image

def attack(net, x_test, y_test, device, method,
		   hyperparams=None, n_samples=None, sample_idxs=None, avg_posterior=False):

	print(f"\n\nProducing {method} attacks", end="\t")
	if n_samples:
		print(f"with {n_samples} attack samples", end="\t")
	if avg_posterior:
		print(f"on the posterior mode")

	net.to(device)
	x_test, y_test = x_test.to(device), y_test.to(device)

	adversarial_attack = []

	for idx in tqdm(range(len(x_test))):
		image = x_test[idx].unsqueeze(0)
		label = y_test[idx].argmax(-1).unsqueeze(0)

		if method == "fgsm":
			perturbed_image = fgsm_attack(net=net, image=image, label=label, 
										  hyperparams=hyperparams, n_samples=n_samples,
										  avg_posterior=avg_posterior, sample_idxs=sample_idxs)
		elif method == "pgd":
			perturbed_image = pgd_attack(net=net, image=image, label=label, 
										  hyperparams=hyperparams, n_samples=n_samples,
										  avg_posterior=avg_posterior, sample_idxs=sample_idxs)

		adversarial_attack.append(perturbed_image)

	adversarial_attack = torch.cat(adversarial_attack)
	return adversarial_attack

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

def evaluate_attack(net, x_test, x_attack, y_test, device, n_samples=None, sample_idxs=None, 
					 avg_posterior=False, return_classification_idxs=False):
	""" Evaluates the network on the original data and its adversarially perturbed version. 
	When using a Bayesian network `n_samples` should be specified for the evaluation.     
	"""

	x_test = x_test.clone()
	x_attack = x_attack.clone()

	print(f"\nEvaluating against the attacks", end="")
	if avg_posterior:
		print(" with the posterior mode")
	else:
		if n_samples:
			print(f" with {n_samples} defense samples")
	
	x_test, x_attack, y_test = x_test.to(device), x_attack.to(device), y_test.to(device)

	test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=128, shuffle=False)
	attack_loader = DataLoader(dataset=list(zip(x_attack, y_test)), batch_size=128, shuffle=False)

	with torch.no_grad():

		original_outputs = []
		original_correct = 0.0
		correct_class_idxs = []
		batch_size=0

		for batch_idx, (images, labels) in enumerate(test_loader):

			out = net.forward(images, n_samples=n_samples, sample_idxs=sample_idxs, avg_posterior=avg_posterior,
								softmax=True)
			original_correct += ((out.argmax(-1) == labels.argmax(-1)).sum().item())
			original_outputs.append(out)

			correct_idxs = np.where(out.argmax(-1).cpu() == labels.argmax(-1).cpu())[0]
			correct_class_idxs.extend(correct_idxs+batch_size*batch_idx)
			batch_size = len(images)

		if DEBUG:
			print("\nlabels", labels.argmax(-1))	
			print("det out", out.argmax(-1))
			print("correct_class_idxs", correct_class_idxs)

		adversarial_outputs = []
		adversarial_correct = 0.0
		wrong_atk_class_idxs = []
		correct_atk_class_idxs = []
		batch_size=0

		for batch_idx, (attacks, labels) in enumerate(attack_loader):
			out = net.forward(attacks, n_samples=n_samples, sample_idxs=sample_idxs, avg_posterior=avg_posterior,
								softmax=True)
			adversarial_correct += ((out.argmax(-1) == labels.argmax(-1)).sum().item())
			adversarial_outputs.append(out)

			wrong_idxs = np.where(out.argmax(-1).cpu() != labels.argmax(-1).cpu())[0]
			wrong_atk_class_idxs.extend(wrong_idxs+batch_size*batch_idx)
			correct_idxs = np.where(out.argmax(-1).cpu() == labels.argmax(-1).cpu())[0]
			correct_atk_class_idxs.extend(correct_idxs+batch_size*batch_idx)
			batch_size = len(attacks)

		successful_atk_idxs = np.intersect1d(correct_class_idxs, wrong_atk_class_idxs)
		failed_atk_idxs = np.intersect1d(correct_class_idxs, correct_atk_class_idxs)

		if DEBUG:
			print("bay out", out.argmax(-1))
			print("wrong_atk_class_idxs", wrong_atk_class_idxs)
			print("successful_atk_idxs", successful_atk_idxs)

		original_accuracy = 100 * original_correct / len(x_test)
		adversarial_accuracy = 100 * adversarial_correct / len(x_test)
		print(f"\ntest accuracy = {original_accuracy}\tadversarial accuracy = {adversarial_accuracy}",
			  end="\t")

		original_outputs = torch.cat(original_outputs)
		adversarial_outputs = torch.cat(adversarial_outputs)
		softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

	if return_classification_idxs:
		return original_outputs, adversarial_outputs, softmax_rob, successful_atk_idxs, failed_atk_idxs
	else:
		return original_outputs, adversarial_outputs, softmax_rob
