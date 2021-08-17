#!/bin/bash

ATTACK_METHOD="pgd" # fgsm, pgd
TEST_INPUTS=500
DEVICE="cuda" # cpu, cuda
DEBUG="False"

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="../experiments/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"


for MODEL_IDX in 0 1 2 3
do

	# for MODEL in "advNN" "baseNN" "fullBNN" 
	# do

	# 	python train_networks.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
	# 						 	 --device=$DEVICE >> $OUT

	# 	python attack_networks.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
	# 							 --device=$DEVICE --n_inputs=$TEST_INPUTS >> $OUT
		
	# 	for RULE in "gamma" "alpha1beta0" "epsilon"
	# 	do
	# 		python compute_lrp.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
	# 						  		--device=$DEVICE --n_inputs=$TEST_INPUTS --rule=$RULE >> $OUT
	# 	done

	# done

	for RULE in "epsilon" 
	do
		python lrp_layers_robustness.py --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --rule=$RULE --debug=$DEBUG \
										--device=$DEVICE --n_inputs=$TEST_INPUTS >> $OUT
	done

done

deactivate

