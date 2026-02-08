
python run_scripts/caption_generation.py \
    --num_samples 5000 \
    --skip_num 0 \
    --verbosity \
    --decoder classifier \
    --output_dir ./experiment_results \
    --gpu-id 2 \
    --split val2014 \
    --seed 0 \
    --data_path /storage/jinhaoduan/datasets/coco2014 \
    --classifier_hs_position previous \
    --classifier_task collect-hidden-states \
    --max_new_tokens 64 \
    --model llava-1.5 &

python run_scripts/caption_generation.py \
    --num_samples 5000 \
    --skip_num 0 \
    --verbosity \
    --decoder classifier \
    --output_dir ./experiment_results \
    --gpu-id 3 \
    --split val2014 \
    --seed 0 \
    --data_path /storage/jinhaoduan/datasets/coco2014 \
    --classifier_hs_position previous \
    --classifier_task collect-hidden-states \
    --max_new_tokens 64 \
    --model mplug-owl2 &

wait
echo "Done"