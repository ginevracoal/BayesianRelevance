import torch
import torch.nn.functional as nnf

from utils.lrp import select_informative_pixels

def Topk(image, model, epsilon, lrp_rule, iters, k=20, step_size=0.5, lr=0.01):

	x_orig = image.clone().detach()
	x_orig.requires_grad=True

	probs = nnf.softmax(model.forward(x_orig, explain=True, rule=lrp_rule), dim=-1)
	y_hat = probs[torch.arange(x_orig.shape[0]), probs.max(1)[1]].sum()
	y_hat.backward(retain_graph=True)
	x_orig_lrp = x_orig.grad.detach()	

	topk_pxl_idxs = select_informative_pixels(x_orig_lrp, topk=k)[1]

	x_adv = image.clone().detach()
	x_adv.requires_grad = True

	for i in range(iters):

		probs = nnf.softmax(model.forward(x_adv, explain=True, rule=lrp_rule), dim=-1)
		y_hat = probs[torch.arange(x_adv.shape[0]), probs.max(1)[1]].sum()
		y_hat.backward(retain_graph=True)
		x_adv_lrp = x_adv.grad.detach()

		x_adv_lrp.requires_grad = True

		loss = - torch.sum(x_adv_lrp.flatten()[topk_pxl_idxs])
		loss.backward()

		x_adv = x_adv + step_size * x_adv.grad.data.sign()
		x_adv = torch.clamp(x_adv, -epsilon, epsilon)

		x_adv = x_adv.detach()
		x_adv.requires_grad = True

		# print("iteration {:.0f}, loss:{:.4f}".format(i,loss))

	return x_adv
