#!/bin/bash

############
# SETTINGS #
############

MODEL_NAME="resnet" # resnet, alexnet, vgg
DATASET_NAME="imagenette" # imagenette, imagewoof, animals10, hymenoptera
BAYESIAN="True"
INFERENCE="svi" # laplace, svi, sgld
N_SAMPLES=10
ITERS=5 # set equal to n_samples for SGLD inference
ATTACK_METHOD="fgsm"
DEVICE="cuda"
TEST_INPUTS=100

TRAIN="False"
ATTACK="True"
DEBUG="False"

########
# EXEC #
########
	
source ../venv/bin/activate

DATE=$(date '+%Y-%m-%d')
TIME=$(date +%H:%M:%S)
TESTS="../experiments/"

if [ "${BAYESIAN}" = "True" ]; then
    SAVEDIR="${MODEL_NAME}_redBNN_${DATASET_NAME}_${INFERENCE}_iters=${ITERS}"
else
    SAVEDIR="${MODEL_NAME}_baseNN_${DATASET_NAME}_iters=${ITERS}"
fi

if [ "${DEBUG}" = "True" ]; then
	SAVEDIR="debug"
fi

OUT="${TESTS}${SAVEDIR}/out.txt"
mkdir -p "${TESTS}${SAVEDIR}"

echo "=== exec ${DATE} ${TIME} ===" >> $OUT

if [ "${TRAIN}" = "True" ]; then

	python3 train_torchvision_networks.py --savedir=$SAVEDIR --model=$MODEL_NAME  --dataset=$DATASET_NAME \
			--bayesian=$BAYESIAN --inference=$INFERENCE --iters=$ITERS --debug=$DEBUG \
			--samples=$N_SAMPLES --device=$DEVICE &>> $OUT
fi

if [ "${ATTACK}" = "True" ]; then

	python3 attack_torchvision_networks.py --savedir=$SAVEDIR --model=$MODEL_NAME  --dataset=$DATASET_NAME \
			--bayesian=$BAYESIAN --inference=$INFERENCE --iters=$ITERS --debug=$DEBUG \
			--attack_method=$ATTACK_METHOD --samples=$N_SAMPLES --device=$DEVICE --inputs=$TEST_INPUTS &>> $OUT

fi

deactivate