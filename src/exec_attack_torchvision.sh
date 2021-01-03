#!/bin/bash

MODEL_NAME="resnet" # resnet, alexnet, vgg
DATASET_NAME="animals10" # imagenette, imagewoof, animals10, hymenoptera
BAYESIAN="True"
INFERENCE="svi" # laplace, svi
TRAIN="True"
ATTACK="True"
N_SAMPLES=1 #00
ITERS=1
ATTACK_METHOD="fgsm"
DEVICE="cpu"

source ../venv/bin/activate

TIME=$(date +%H:%M:%S)

if [ "${BAYESIAN}" = "True" ]; then
    SAVEDIR="${MODEL_NAME}_${DATASET_NAME}_${INFERENCE}_iters=${ITERS}"
	OUT="${SAVEDIR}${TIME}_${MODEL_NAME}_${DATASET_NAME}_${INFERENCE}_iters=${ITERS}_${ATTACK_METHOD}.txt"
else
    SAVEDIR="${MODEL_NAME}_${DATASET_NAME}_iters=${ITERS}"
	OUT="${SAVEDIR}${TIME}_${MODEL_NAME}_${DATASET_NAME}_iters=${ITERS}_${ATTACK_METHOD}.txt"
fi

python3 attack_torchvision_networks.py --savedir=$SAVEDIR --model=$MODEL_NAME  --dataset=$DATASET_NAME \
		--bayesian=$BAYESIAN --inference=$INFERENCE --train=$TRAIN --attack=$ATTACK --iters=$ITERS \
		--attack_method=$ATTACK_METHOD --n_samples=$N_SAMPLES --device=$DEVICE &> $OUT

deactivate