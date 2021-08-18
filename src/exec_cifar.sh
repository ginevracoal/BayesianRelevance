#!/bin/bash

ATTACK_METHOD="fgsm" # fgsm, pgd
TEST_INPUTS=500
ATK_SAMPLES=100
TOPK=10
DEVICE="cuda" # cpu, cuda
DEBUG="False"

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="../experiments/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"

for MODE in "train" "test"
do

	# python full_test_cifar_resnet.py --mode=$MODE --attack_method=$ATTACK_METHOD --test_inputs=$TEST_INPUTS >> $OUT
	python full_test_cifar_adversarial_resnet.py --mode=$MODE --attack_method=$ATTACK_METHOD --test_inputs=$TEST_INPUTS >> $OUT
	# python full_test_cifar_bayesian_resnet.py --mode=$MODE --attack_method=$ATTACK_METHOD --test_inputs=$TEST_INPUTS >> $OUT

done 

python lrp_rules_robustness_cifar.py --n_inputs=$TEST_INPUTS --topk=$TOPK --n_samples=$ATK_SAMPLES \
									 --attack_method=$ATTACK_METHOD --device=$DEVICE >> $OUT