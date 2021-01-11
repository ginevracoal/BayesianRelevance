#!/bin/bash

############
# SETTINGS #
############

TRAIN="True"
ATTACK="True"
TRAIN_DEVICE="cpu"
ATK_DEVICE="cpu"

MODEL="redBNN"
ARCHITECTURE="resnet"
TRAIN_ITERS=30
BASE_ITERS=2 # redBNN only
INFERENCE="svi" # redBNN only
SAMPLES=30 # redBNN only

DATASET="animals10"
TEST_INPUTS=100 # test points to be attacked
ATTACK_METHOD="fgsm" 

########
# EXEC #
########
	
source ../venv/bin/activate

DATE=$(date '+%Y-%m-%d')
TIME=$(date +%H:%M:%S)
OUT_DIR="../experiments/logs/"
OUT_FILE="${TESTS}${DATE}_${TIME}_out.txt"
mkdir -p "${OUT_DIR}"

if [ "${MODEL}" = "baseNN" ]; then

	INFERENCE="None"; SAMPLES=0; BASE_ITERS=0
fi

if [ "${TRAIN}" = "True" ]; then

	python3 train_torchvision_networks.py --dataset=$DATASET_NAME --model=$MODEL \
			--architecture=$ARCHITECTURE --iters=$ITERS --inference=$INFERENCE \
			--samples=$SAMPLES --base_iters=$BASE_ITERS --device=$TRAIN_DEVICE &>> $OUT
fi

if [ "${ATTACK}" = "True" ]; then

	python3 attack_torchvision_networks.py --dataset=$DATASET_NAME --model=$MODEL_NAME \
			--architecture=$ARCHITECTURE --iters=$ITERS --inference=$INFERENCE \
			--samples=$N_SAMPLES --base_iters=$BASE_ITERS --attack_method=$ATTACK_METHOD \
			--device=$TRAIN_DEVICE --inputs=$TEST_INPUTS --device=$ATK_DEVICE &>> $OUT
fi

deactivate