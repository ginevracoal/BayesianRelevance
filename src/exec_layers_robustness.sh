#!/bin/bash

ATTACK_METHOD="fgsm" # fgsm, pgd
TEST_INPUTS=500
RULE="epsilon"
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

	for MODEL in  "baseNN" "advNN"  
	do

		python train_networks.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
							 	 --device=$DEVICE >> $OUT

		python attack_networks.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
								 --device=$DEVICE --n_inputs=$TEST_INPUTS >> $OUT
		
		python compute_lrp.py --model=$MODEL --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
						  		--device=$DEVICE --n_inputs=$TEST_INPUTS --rule=$RULE >> $OUT

	done

	python train_networks.py --model="fullBNN" --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
						 	 --device=$DEVICE >> $OUT

	for N_SAMPLES in 10 50 100
	do

		python attack_networks.py --model="fullBNN" --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
								 --device=$DEVICE --n_inputs=$TEST_INPUTS --n_samples=$N_SAMPLES >> $OUT
		
		python compute_lrp.py --model="fullBNN" --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --debug=$DEBUG \
						  		--device=$DEVICE --n_inputs=$TEST_INPUTS --rule=$RULE --n_samples=$N_SAMPLES >> $OUT

	done

	python lrp_robustness_diff.py --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --rule=$RULE \
									--debug=$DEBUG --device=$DEVICE --n_inputs=$TEST_INPUTS >> $OUT

	python lrp_robustness_scatterplot.py --model_idx=$MODEL_IDX --attack_method=$ATTACK_METHOD --rule=$RULE \
										--debug=$DEBUG --device=$DEVICE --n_inputs=$TEST_INPUTS >> $OUT

done

deactivate


