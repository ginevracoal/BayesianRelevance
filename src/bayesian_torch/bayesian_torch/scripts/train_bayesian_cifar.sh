#!/bin/bash

model=resnet20
mode='train'
batch_size=128
lr=0.001
epochs=200

python bayesian_torch/bayesian_torch/examples/main_bayesian_cifar.py --lr=$lr --arch=$model --mode=$mode --batch-size=$batch_size --epochs=$epochs
