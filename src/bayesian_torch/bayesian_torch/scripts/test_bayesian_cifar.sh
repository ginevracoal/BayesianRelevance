#!/bin/bash

model='resnet20'
mode='test'
batch_size=1000
num_monte_carlo=50

python bayesian_torch/bayesian_torch/examples/main_bayesian_cifar.py --arch=$model --mode=$mode --batch-size=$batch_size --num_monte_carlo=$num_monte_carlo
