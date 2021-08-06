'''
Bayesian ResNet for CIFAR10.

ResNet architecture ref:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from bayesian_torch.bayesian_torch.layers import Conv2dReparameterization
from bayesian_torch.bayesian_torch.layers import LinearReparameterization

from lrp.sequential import Sequential 
from lrp.linear import Linear
from lrp.conv import Conv2d

__all__ = [
    'ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110'
]

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -2.0


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) \
    or isinstance(m, Linear) or isinstance(m, Conv2d): # <--------------
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dReparameterization(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dReparameterization(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = Sequential(
                    Conv2dReparameterization(
                        in_channels=in_planes,
                        out_channels=self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        prior_mean=prior_mu,
                        prior_variance=prior_sigma,
                        posterior_mu_init=posterior_mu_init,
                        posterior_rho_init=posterior_rho_init,
                        bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x, explain=False, rule='epsilon'):
        kl_sum = 0
        out, kl = self.conv1(x, explain, rule)
        kl_sum += kl
        out = self.bn1(out)
        out = F.relu(out)
        out, kl = self.conv2(out, explain, rule)
        kl_sum += kl
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out, kl_sum


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = Conv2dReparameterization(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = LinearReparameterization(
            in_features=64,
            out_features=num_classes,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
        )

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return Sequential(*layers) # <-----------------------

    def forward(self, x, explain=False, rule="epsilon"): # <-----------------------
        kl_sum = 0
        out, kl = self.conv1(x, explain, rule) # <-----------------------

        # print(self.conv1.mu_kernel[0,0,0].cpu().detach(), "\t", self.conv1.rho_kernel[0,0,0].cpu().detach())
        # print(self.conv1.mu_kernel.mean().item(), self.conv1.mu_kernel.std().item())
        # print(self.conv1.rho_kernel.mean().item(), self.conv1.rho_kernel.std().item())
        # exit()

        kl_sum += kl
        out = self.bn1(out)
        out = F.relu(out)
        for l in self.layer1:
            out, kl = l(out)
        kl_sum += kl
        for l in self.layer2:
            out, kl = l(out)
        kl_sum += kl
        for l in self.layer3:
            out, kl = l(out)
        kl_sum += kl

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out, kl = self.linear(out, explain, rule) # <-----------------------
        kl_sum += kl

        return out, kl_sum


def resnet20():
    print("resnet20")
    return ResNet(BasicBlock, [3, 3, 3])

def resnet32():
    print("resnet32")
    return ResNet(BasicBlock, [5, 5, 5])

def resnet44():
    print("resnet44")
    return ResNet(BasicBlock, [7, 7, 7])

def resnet56():
    print("resnet56")
    return ResNet(BasicBlock, [9, 9, 9])

def resnet110():
    print("resnet110")
    return ResNet(BasicBlock, [18, 18, 18])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print(
        "Total layers",
        len(
            list(
                filter(lambda p: p.requires_grad and len(p.data.size()) > 1,
                       net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
