#!/bin/bash

MODEL="baseNN" # baseNN, fullBNN, redBNN, laplRedBNN
MODEL_IDX=0 # model idx from the chosen dictionary

ATTACK_METHOD="fgsm" # fgsm, pgd
ATK_INPUTS=10

TRAIN_ATK="True" # if True trains else loads
COMPUTE_LRP="True" # if True trains else loads

DEVICE="cuda"

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


if [ "${TRAIN_ATK}" = "True" ]; then

	if [ "${MODEL}" = "baseNN" ]; then

	# 	python3 train_attack_networks.py --model=$MODEL --model_idx=$MODEL_IDX \
	# 			--atk_inputs=$ATK_INPUTS --attack_method=$ATTACK_METHOD --device=$DEVICE &>> $OUT 
	# else
	# 	python3 train_attack_networks.py --model=$MODEL --model_idx=$MODEL_IDX --inference=$INFERENCE \
	# 			--atk_inputs=$ATK_INPUTS --attack_method=$ATTACK_METHOD --device=$DEVICE &>> $OUT 

	fi

fi

if [ "${COMPUTE_LRP}" = "True" ]; then

	# python3 attack_torchvision_networks.py --savedir=$SAVEDIR --model=$MODEL_NAME  --dataset=$DATASET_NAME \
	# 		--bayesian=$BAYESIAN --inference=$INFERENCE --iters=$ITERS --debug=$DEBUG \
	# 		--attack_method=$ATTACK_METHOD --samples=$N_SAMPLES --device=$ATK_DEVICE --inputs=$TEST_INPUTS &>> $OUT
fi

deactivate