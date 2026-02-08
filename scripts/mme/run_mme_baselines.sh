#! /bin/bash

mme_task=('position', 'existance', 'color', 'count')

python run_scripts/mme_eval.py \
--model minigpt4 \
--data_path /storage/jinhaoduan/datasets/MME_Benchmark_release_version/$mme_task \
-d greedy \
--num_samples 30 \
--seed 0 \
--gpu-id 1 \
--output_dir ./experiment_results

python eval/MME_score.py \
--data_path /storage/jinhaoduan/datasets/MME_Benchmark_release_version \
--results_dir /home/jd3734@drexel.edu/workspace/Classifier-Guided-VLM-OH/experiment_results/mme/minigpt4/greedy/minigpt4_greedy_dino_box_0.4_beams_3_k_4_coco_expand_ratio_0.6_seed_0_max_tokens_128_samples_30_skip_0
