#! /bin/bash

python run_scripts/pope_eval.py \
--model minigpt4 \
--data_path /storage/jinhaoduan/datasets/coco2014/val2014 \
-d greedy \
--pope_type random \
--num_images 500 \
--seed 0 \
--gpu_id 0 \
--output_dir ./experiment_results/

python run_scripts/pope_eval.py \
--model minigpt4 \
--data_path /storage/jinhaoduan/datasets/coco2014/val2014 \
-d greedy \
--pope_type popular \
--num_images 500 \
--seed 0 \
--gpu_id 0 \
--output_dir ./pope_experiment_results/

python run_scripts/pope_eval.py \
--model minigpt4 \
--data_path /storage/jinhaoduan/datasets/coco2014/val2014 \
-d greedy \
--pope_type adversarial \
--num_images 500 \
--seed 0 \
--gpu_id 0 \
--output_dir ./pope_experiment_results/
