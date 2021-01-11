#!/bin/bash

N_INPUTS=10
MODEL="baseNN" # baseNN, fullBNN, redBNN, laplRedBNN
MODEL_IDX=0 # model idx from the chosen dictionary
ATTACK_METHOD="fgsm" # fgsm, pgd

TRAIN="True" # if True trains else loads
ATTACK="True" # if True attacks else loads

DEBUG="False"
TRAIN_DEVICE="cpu"
ATK_DEVICE="cpu"

########
# EXEC #
########

DATE=$(date '+%Y-%m-%d')
TIME=$(date +%H:%M:%S)
OUT_DIR="../experiments/logs/"
OUT_FILE="${TESTS}${DATE}_${TIME}_out.txt"
mkdir -p "${OUT_DIR}"

OUT="${TESTS}${SAVEDIR}/out.txt"
mkdir -p "${TESTS}${SAVEDIR}"

if [ "${MODEL}" = "baseNN" ]; then

	INFERENCE="None"; SAMPLES=0; BASE_ITERS=0
fi

if [ "${TRAIN}" = "True" ]; then

	# python3 train_torchvision_networks.py --savedir=$SAVEDIR --model=$MODEL_NAME  --dataset=$DATASET_NAME \
	# 		--bayesian=$BAYESIAN --inference=$INFERENCE --iters=$ITERS --debug=$DEBUG \
	# 		--samples=$N_SAMPLES --device=$TRAIN_DEVICE &>> $OUT
fi

if [ "${ATTACK}" = "True" ]; then

	# python3 attack_torchvision_networks.py --savedir=$SAVEDIR --model=$MODEL_NAME  --dataset=$DATASET_NAME \
	# 		--bayesian=$BAYESIAN --inference=$INFERENCE --iters=$ITERS --debug=$DEBUG \
	# 		--attack_method=$ATTACK_METHOD --samples=$N_SAMPLES --device=$ATK_DEVICE --inputs=$TEST_INPUTS &>> $OUT
fi

deactivate