#!/bin/bash

ATTACK_METHOD="topk" # fgsm, pgd, beta, topk, region
TEST_INPUTS=500
ATK_SAMPLES=100
TOPK=20
DEVICE="cuda" # cpu, cuda
DEBUG="False"

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="../experiments/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"

for MODEL_IDX in 0 #0 1 2 3
do

	for MODEL in "baseNN" "advNN" "fullBNN" 
	do

		# python train_networks.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
		# 					 	 --device=$DEVICE >> $OUT

		python attack_networks.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
								 --device=$DEVICE --n_inputs=$TEST_INPUTS >> $OUT

		for RULE in "epsilon" "gamma" "alpha1beta0"
		do
			python compute_lrp.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
						 	 		--device=$DEVICE --n_inputs=$TEST_INPUTS --rule=$RULE >> $OUT
		done

	done

	python lrp_rules_robustness.py --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
									--device=$DEVICE --n_inputs=$TEST_INPUTS --n_samples=$ATK_SAMPLES \
									--topk=$TOPK >> $OUT

done

deactivate


