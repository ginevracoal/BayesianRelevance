#!/bin/bash

N_INPUTS=10
MODEL_TYPE="baseNN" # baseNN, fullBNN, redBNN, laplRedBNN
MODEL_IDX=0 # model idx from the chosen dictionary
ATTACK_METHOD="fgsm" # fgsm, pgd

TRAIN="True" # if True trains else loads
ATTACK="True" # if True attacks else loads

DEVICE="cuda" # cpu, cuda

source ../venv/bin/activate

TIME=$(date +%H:%M:%S)
TESTS="experiments/logs/"
mkdir -p $TESTS

OUT="${TESTS}${TIME}_${MODEL_NAME}_${DATASET_NAME}_iters=${ITERS}_${ATTACK_METHOD}.txt"

# todo: completare

deactivate