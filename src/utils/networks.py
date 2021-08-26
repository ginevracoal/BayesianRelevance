import torch
import torch.nn as nn


def relu_to_softplus(model, beta):
	for child_name, child in model.named_children():
		if isinstance(child, nn.LeakyReLU):
			setattr(model, child_name, nn.Softplus(beta=beta))
		else:
			relu_to_softplus(child, beta)

	return model

def change_beta(model, beta):
	for child_name, child in model.named_children():
		if isinstance(child, nn.Softplus):
			setattr(model, child_name, nn.Softplus(beta=beta))
		else:
			change_beta(child, beta)

	return model