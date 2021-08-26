import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import numpy as np
import math
import copy
import os
import random


def TargetRegion(image, model, epsilon, lrp_rule, iters, target_pxls, step_size=0.5, lr=0.01):

    x_adv = image.clone().detach()
    x_adv.requires_grad = True

    for i in range(iters):

        probs = nnf.softmax(model.forward(x_adv, explain=True, rule=lrp_rule), dim=-1)
        y_hat = probs[torch.arange(x_adv.shape[0]), probs.max(1)[1]].sum()
        y_hat.backward(retain_graph=True)
        x_adv_lrp = x_adv.grad.detach()

        x_adv_lrp.requires_grad = True

        loss = torch.sum(x_adv_lrp.flatten()[target_pxls])
        loss.backward()

        x_adv = x_adv + step_size * x_adv.grad.data.sign()
        x_adv = torch.clamp(x_adv, -epsilon, epsilon)

        x_adv = x_adv.detach()
        x_adv.requires_grad = True

        # print("iteration {:.0f}, loss:{:.4f}".format(i,loss))

    return x_adv
