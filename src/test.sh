#!/bin/bash

model_names=( "resnet" "alexnet" "vgg" )
dataset_names=( "animals10" "hymenoptera" "imagenette" "imagewoof" )
bayesian=( "True" "False" )
inference=( "svi" ) #"laplace"
attack_methods=( "fgsm" "pgd" )
devices=( "cpu" "cuda" )
DEBUG="True"
N_SAMPLES=1
ITERS=1

source ../venv/bin/activate

for MODEL_NAME in "${model_names[@]}"; do
	for DATASET_NAME in "${dataset_names[@]}"; do
		for BAYESIAN in "${bayesian[@]}"; do
			for INFERENCE in "${inference[@]}"; do
				for ATTACK_METHOD in "${attack_methods[@]}"; do
					for DEVICE in "${devices[@]}"; do
						python3 attack_torchvision_networks.py --model_name=$MODEL_NAME  \
						--dataset_name=$DATASET_NAME --bayesian=$BAYESIAN --inference=$INFERENCE --train="True" \
						--attack="True" --iters=$ITERS --attack_method=$ATTACK_METHOD --n_samples=$N_SAMPLES \
						--device=$DEVICE --debug="True"
					done
				done
			done
		done
	done
done

deactivate
