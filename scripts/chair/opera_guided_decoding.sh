

python run_scripts/caption_generation.py \
    --num_samples 500 \
    --skip_num 0 \
    --verbosity 1 \
    --decoder opera \
    --output_dir ./experiment_results \
    --gpu-id 0 \
    --split val2014 \
    --seed 42 \
    --data_path /storage/jinhaoduan/datasets/coco2014 \
    --max_new_tokens 64 \
    --beam 3 \
    --batch_size 1 # batchsize can only be 1