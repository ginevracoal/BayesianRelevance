#!/bin/bash

MODEL_NAME="resnet" # resnet, alexnet, vgg
DATASET_NAME="imagenette" # imagenette, imagewoof, animals10, hymenoptera
BAYESIAN="True"
INFERENCE="laplace" # laplace, svi, sgld
TRAIN="False"
ATTACK="True"
N_SAMPLES=50
ITERS=5
ATTACK_METHOD="fgsm"
DEVICE="cpu"
	
source ../venv/bin/activate

DATE=$(date '+%Y-%m-%d')
TIME=$(date +%H:%M:%S)
TESTS="../experiments/"

if [ "${BAYESIAN}" = "True" ]; then
    SAVEDIR="${MODEL_NAME}_redBNN_${DATASET_NAME}_${INFERENCE}_iters=${ITERS}"
else
    SAVEDIR="${MODEL_NAME}_baseNN_${DATASET_NAME}_iters=${ITERS}"
fi

OUT="${TESTS}${SAVEDIR}/out.txt"
mkdir -p "${TESTS}${SAVEDIR}"

echo "=== exec ${DATE} ${TIME} ===" >> $OUT

python3 attack_torchvision_networks.py --savedir=$SAVEDIR --model=$MODEL_NAME  --dataset=$DATASET_NAME \
		--bayesian=$BAYESIAN --inference=$INFERENCE --train=$TRAIN --attack=$ATTACK --iters=$ITERS \
		--attack_method=$ATTACK_METHOD --samples=$N_SAMPLES --device=$DEVICE &>> $OUT

deactivate