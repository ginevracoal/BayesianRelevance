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
import attacks.torchvision_gradient_based as torchvision_atks
from torch.autograd.gradcheck import zero_gradients

from deeprobust.image.attack.fgsm import FGSM
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.attack.cw import CarliniWagner
from deeprobust.image.attack.Nattack import NATTACK
from deeprobust.image.attack.YOPOpgd import FASTPGD
from deeprobust.image.attack.deepfool import DeepFool

def attack(net, x_test, y_test, device, method, *args, **kwargs):
		   # hyperparams=None, n_samples=None, sample_idxs=None, avg_posterior=False):

	print(f"\n\nCrafting {method} attacks")

	net.to(device)
	x_test, y_test = x_test.to(device), y_test.to(device)

	adversarial_attack = []

	for idx in tqdm(range(len(x_test))):
		image = x_test[idx].unsqueeze(0)
		label = y_test[idx].argmax(-1).unsqueeze(0)
		num_classes = len(y_test[idx])

		if method == "fgsm":
			adversary = FGSM
			adversary_params = {'epsilon': 0.2, 'order': np.inf, 'clip_max': None, 'clip_min': None}
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

		elif method == "nattack":
			adversary = NATTACK
			adversary_params = {}
			adv = adversary(net, device)
			perturbed_image = adv.generate()

		elif method == "yopo":
			adversary = FASTPGD
			adversary_params = {}
			adv = adversary(net, device)
			perturbed_image = adv.generate(image, label, **adversary_params)

		elif method == "deepfool":
			adversary = DeepFool
			adversary_params = {}
			adv = adversary(net, device)
			perturbed_image = adv.generate(image, label, **adversary_params)

		perturbed_image = torch.clamp(perturbed_image, 0., 1.)
		adversarial_attack.append(perturbed_image)

	adversarial_attack = torch.cat(adversarial_attack)
	return adversarial_attack

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

# def _plot_attack(savedir, original_images_list, adversarial_data, method, n_samples=None):
# 	method = method+"_deeprobust"
# 	torchvision_atks._plot_attack(savedir, original_images_list, adversarial_data, method, n_samples)

def save_attack(inputs, attacks, method, filename, savedir, n_samples=None):
	savedir = os.path.join(savedir, ATK_DIR)
	filename = _get_attacks_filename(filename, method, n_samples)
	save_to_pickle(data=attacks, path=savedir, filename=filename)

	set_seed(0)
	idxs = np.random.choice(len(inputs), 10, replace=False)
	original_images_plot = torch.stack([inputs[i].squeeze() for i in idxs])
	perturbed_images_plot = torch.stack([attacks[i].squeeze() for i in idxs])
	plot_grid_attacks(original_images=original_images_plot.detach().cpu(), 
					  perturbed_images=perturbed_images_plot.detach().cpu(), 
					  filename=filename, savedir=savedir)


def load_attack(method, filename, savedir, n_samples=None):
	savedir = os.path.join(savedir, ATK_DIR)
	filename = _get_attacks_filename(filename, method, n_samples)
	return load_from_pickle(path=savedir, filename=name)

def _get_attacks_filename(filename, method, n_samples=None):

	if n_samples:
		return filename+"_"+str(method)+"_attackSamp="+str(n_samples)+"_attack_deeprobust"
	else:
		return filename+"_"+str(method)+"_attack_deeprobust"

### Attacks from DeepRobust library ###

# def deepfool_attack(model, image, num_classes, device, overshoot=0.02, max_iter=50, *args, **kwargs):
# 	image = copy.deepcopy(image)

# 	f_image = model.forward(image, *args, **kwargs).data.cpu().numpy().flatten()
# 	output = (np.array(f_image)).flatten().argsort()[::-1]

# 	output = output[0:num_classes]
# 	label = output[0]

# 	input_shape = image.cpu().numpy().shape
# 	x = copy.deepcopy(image).requires_grad_(True)
# 	w = np.zeros(input_shape)
# 	r_tot = np.zeros(input_shape) 

# 	fs = model.forward(x, *args, **kwargs)
# 	# fs_list = [fs[0,output[k]] for k in range(num_classes)]
# 	current_pred_label = label

# 	for i in range(max_iter):

# 		pert = np.inf
# 		fs[0, output[0]].backward(retain_graph=True)
# 		grad_orig = x.grad.data.cpu().numpy().copy()

# 		for k in range(1, num_classes):
# 			zero_gradients(x)

# 			fs[0, output[k]].backward(retain_graph=True)
# 			cur_grad = x.grad.data.cpu().numpy().copy()

# 			# set new w_k and new f_k
# 			w_k = cur_grad - grad_orig
# 			f_k = (fs[0, output[k]] - fs[0, output[0]]).data.cpu().numpy()

# 			pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

# 			# determine which w_k to use
# 			if pert_k < pert:
# 				pert = pert_k
# 				w = w_k

# 		# compute r_i and r_tot
# 		# Added 1e-4 for numerical stability
# 		r_i =  (pert+1e-4) * w / np.linalg.norm(w)
# 		r_tot = np.float32(r_tot + r_i)

# 		pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).to(device)

# 		x = pert_image.detach().requires_grad_(True)
# 		fs = model.forward(x, *args, **kwargs)

# 		if (not np.argmax(fs.data.cpu().numpy().flatten()) == label):
# 			break

# 	r_tot = (1+overshoot)*r_tot

# 	return pert_image #, r_tot, i
