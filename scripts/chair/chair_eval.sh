
caption_dir_path='/home/jd3734@drexel.edu/workspace/Classifier-Guided-VLM-OH/experiment_results/chair/mplug-owl2/greedy/mplug-owl2_greedy_dino_box_0.45_beams_1_k_2_coco_expand_ratio_0.6_seed_0_max_tokens_64_samples_500_skip_0_'

python eval/caption_to_chair.py \
-c /storage/jinhaoduan/datasets/coco2014/annotations/captions_val2014.json \
-d $caption_dir_path

python eval/eval_hallucination.py \
--metric chair \
--chair_input_path $caption_dir_path/chair.json -v \
--data_dir /storage/jinhaoduan/datasets/coco2014/val2014 \
--output_dir $caption_dir_path