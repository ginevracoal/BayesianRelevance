import copy
import math
import numpy as np
import os 
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from utils.networks import change_beta


def get_beta(current_iter, iters):
	start_beta, end_beta = 10.0, 100.0
	return start_beta * (end_beta / start_beta) ** (current_iter / iters)

def clamp_image(x, mean, std):
	"""
	Helper method for clamping the adversarial example in order to ensure that it is a valid image
	"""
	upper = (1.0 - mean) / std
	lower = (0.0 - mean) / std

	if x.shape[1] == 3:  # 3-channel image
		for i in [0, 1, 2]:
			x[0][i] = torch.clamp(x[0][i], min=lower[i], max=upper[i])
	else:
		x = torch.clamp(x, min=lower[0], max=upper[0])
	return x

def Beta(image, model, target_image, data_mean, data_std, lrp_rule, iters, 
		 delta=1, gamma=1, lr=0.001, beta_growth=False):

	x = image.detach()
	x_target = target_image.detach()
	x_adv = x.clone().detach()

	# produce explanations
	x.requires_grad=True
	x_target.requires_grad=True
	x_adv.requires_grad=True

	x_probs = nnf.softmax(model.forward(x, explain=True, rule=lrp_rule), dim=-1)
	y_hat = x_probs[torch.arange(x.shape[0]), x_probs.max(1)[1]].sum()
	y_hat.backward(retain_graph=True)
	x_expl = x.grad.detach()

	x_target_probs = nnf.softmax(model.forward(x_target, explain=True, rule=lrp_rule), dim=-1)
	y_hat = x_target_probs[torch.arange(x_target.shape[0]), x_target_probs.max(1)[1]].sum()
	y_hat.backward(retain_graph=True)
	x_target_expl = x_target.grad.detach()

	# optimize
	optimizer = torch.optim.Adam([x_adv], lr=lr)

	for i in range(iters):

		if beta_growth:
			if hasattr(model, "basenet"):
				model.basenet = change_beta(model.basenet, get_beta(i, iters))
			else:
				model.model = change_beta(model.model, get_beta(i, iters))

		optimizer.zero_grad()

		# calculate loss
		x_adv_probs = nnf.softmax(model.forward(x_adv, explain=True, rule=lrp_rule), dim=-1)
		y_hat = x_adv_probs[torch.arange(x_adv.shape[0]), x_adv_probs.max(1)[1]].sum()
		y_hat.backward(retain_graph=True)
		x_adv_expl = x_adv.grad.detach()		

		loss_expl = nnf.mse_loss(x_adv_expl, x_target_expl)
		loss_output = nnf.mse_loss(x_adv_probs, x_probs.detach())
		total_loss = delta*loss_expl + gamma*loss_output

		# update adversarial example
		total_loss.backward()
		optimizer.step()

		# clamp adversarial example
		# Note: x_adv.data returns tensor which shares data with x_adv but requires
		#       no gradient. Since we do not want to differentiate the clamping,
		#       this is what we need
		x_adv.data = torch.clamp(x_adv.data, data_mean, data_std)

		# print("Iteration {}: Total Loss: {}, Expl Loss: {}, Output Loss: {}".format(i, total_loss.item(), loss_expl.item(), loss_output.item()))

	return x_adv
