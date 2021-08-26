#!/bin/bash

ATTACK_METHOD="fgsm" # fgsm, pgd
TEST_INPUTS=500
DEVICE="cuda" # cpu, cuda
DEBUG="False"
N_SAMPLES=100

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="../experiments/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"


for MODEL_IDX in 0 #1 2 3
do

	for MODEL in  "fullBNN" #"baseNN" "advNN"  "fullBNN" 
	do

	# 	python train_networks.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
	# 						 	 --device=$DEVICE >> $OUT

		python attack_networks.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
								 --device=$DEVICE --n_inputs=$TEST_INPUTS --n_samples=$N_SAMPLES >> $OUT
		
		for RULE in "epsilon" #"gamma" "alpha1beta0"
		do
			python compute_lrp.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
							  		--device=$DEVICE --n_inputs=$TEST_INPUTS --rule=$RULE --n_samples=$N_SAMPLES >> $OUT
		done

	done

	for RULE in "epsilon" 
	do
		python lrp_layers_robustness.py --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --rule=$RULE \
										--debug=$DEBUG --device=$DEVICE --n_inputs=$TEST_INPUTS >> $OUT
	done

done

deactivate


