

python run_scripts/caption_generation.py \
    --num_samples 500 \
    --skip_num 0 \
    --verbosity \
    --decoder classifier \
    --output_dir ./experiment_results \
    --gpu-id 0 \
    --split val2014 \
    --seed 0 \
    --data_path /storage/jinhaoduan/datasets/coco2014 \
    --classifier_task classify-hallucination \
    --classifier_ckpt_path ./classifier_hidden_states_previous_hs/chair/minigpt4/minigpt4_cc_sbu_align_3k_layer16_checkpoint.pth \
    --classifier_hs_position previous \
    --classifier_threshold 2.85 \
    --max_new_tokens 64 \
    --batch_size 1 \
    --exp_nickname minigpt4_cc_sbu_align_3k_layer16