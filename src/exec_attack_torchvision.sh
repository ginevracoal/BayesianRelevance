#!/bin/bash

MODEL_NAME="vgg" # resnet, alexnet, vgg
DATASET_NAME="animals10" # imagenette, imagewoof, animals10, hymenoptera
BAYESIAN="False"
INFERENCE="svi" # laplace, svi
TRAIN="True"
ATTACK="False"
N_SAMPLES=100
ITERS=2
ATTACK_METHOD="fgsm"
DEVICE="cpu"

source ../venv/bin/activate

DATE=$(date '+%Y-%m-%d')
TIME=$(date +%H:%M:%S)
TESTS="../experiments/"

if [ "${BAYESIAN}" = "True" ]; then
    SAVEDIR="${MODEL_NAME}_redBNN_${DATASET_NAME}_${INFERENCE}_iters=${ITERS}"
	OUT="${TESTS}${SAVEDIR}/${MODEL_NAME}_${DATASET_NAME}_${INFERENCE}_iters=${ITERS}_${ATTACK_METHOD}.txt"
else
    SAVEDIR="${MODEL_NAME}_baseNN_${DATASET_NAME}_iters=${ITERS}"
	OUT="${TESTS}${SAVEDIR}/${MODEL_NAME}_${DATASET_NAME}_iters=${ITERS}_${ATTACK_METHOD}.txt"
fi

mkdir -p "${TESTS}${SAVEDIR}"

echo "=== exec date time ${DATE} ${TIME} ===" >> $OUT

python3 attack_torchvision_networks.py --savedir=$SAVEDIR --model=$MODEL_NAME  --dataset=$DATASET_NAME \
		--bayesian=$BAYESIAN --inference=$INFERENCE --train=$TRAIN --attack=$ATTACK --iters=$ITERS \
		--attack_method=$ATTACK_METHOD --n_samples=$N_SAMPLES --device=$DEVICE &>> $OUT

deactivate