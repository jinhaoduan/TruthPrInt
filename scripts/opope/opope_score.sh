#! /bin/bash


model='minigpt4'
# caption_dir='/home/jd3734@drexel.edu/workspace/Classifier-Guided-VLM-OH/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_previous_classifier_threshold_0.9'
caption_dir='/home/jd3734@drexel.edu/workspace/Classifier-Guided-VLM-OH/experiment_results/chair/minigpt4/halc/minigpt4_halc_dino_box_0.45_beams_1_k_2_coco_expand_ratio_0.6_seed_42_max_tokens_64_samples_500_skip_0'
caption_path=$caption_dir/generated_captions.json
data_path='/storage/jinhaoduan/datasets/coco2014/val2014'


python eval/opope_eval.py \
--model $model \
--data_path $data_path \
--gt_seg_path ./pope_coco/coco_ground_truth_segmentation.json \
--output_dir $caption_dir \
--num_images 500 \
--batch_size 1 \
--generated_caption $caption_path \
--pope_type random

python eval/opope_eval.py \
--model $model \
--data_path $data_path \
--gt_seg_path ./pope_coco/coco_ground_truth_segmentation.json \
--output_dir $caption_dir \
--num_images 500 \
--batch_size 1 \
--generated_caption $caption_path \
--pope_type popular

python eval/opope_eval.py \
--model $model \
--data_path $data_path \
--gt_seg_path ./pope_coco/coco_ground_truth_segmentation.json \
--output_dir $caption_dir \
--num_images 500 \
--batch_size 1 \
--generated_caption $caption_path \
--pope_type adversarial