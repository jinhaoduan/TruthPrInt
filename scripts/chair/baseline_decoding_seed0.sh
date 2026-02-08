
SEED=0
MAX_TOKENS=64
BEAM=3
GPU_ID=1
NUM_SAMPLES=500

# Greedy
python run_scripts/caption_generation.py \
    --num_samples $NUM_SAMPLES \
    --skip_num 0 \
    --verbosity \
    --decoder greedy \
    --output_dir ./experiment_results \
    --gpu-id $GPU_ID \
    --split val2014 \
    --seed $SEED \
    --data_path /storage/jinhaoduan/datasets/coco2014 \
    --max_new_tokens $MAX_TOKENS \
    --model mplug-owl2 \
    --batch_size 1 # batchsize can only be 1


# # OPERA
# python run_scripts/caption_generation.py \
#     --num_samples $NUM_SAMPLES \
#     --skip_num 0 \
#     --verbosity 1 \
#     --decoder opera \
#     --output_dir ./experiment_results \
#     --gpu-id $GPU_ID \
#     --split val2014 \
#     --seed $SEED \
#     --data_path /storage/jinhaoduan/datasets/coco2014 \
#     --max_new_tokens $MAX_TOKENS \
#     --beam $BEAM \
#     --batch_size 1 # batchsize can only be 1

# # DOLA
# python run_scripts/caption_generation.py \
#     --num_samples $NUM_SAMPLES \
#     --skip_num 0 \
#     --verbosity 1 \
#     --decoder dola \
#     --output_dir ./experiment_results \
#     --gpu-id $GPU_ID \
#     --split val2014 \
#     --seed $SEED \
#     --data_path /storage/jinhaoduan/datasets/coco2014 \
#     --max_new_tokens $MAX_TOKENS \
#     --beam $BEAM \
#     --batch_size 1 # batchsize can only be 1

# # VCD
# python run_scripts/caption_generation.py \
#     --num_samples $NUM_SAMPLES \
#     --skip_num 0 \
#     --verbosity 1 \
#     --decoder vcd \
#     --output_dir ./experiment_results \
#     --gpu-id $GPU_ID \
#     --split val2014 \
#     --seed $SEED \
#     --data_path /storage/jinhaoduan/datasets/coco2014 \
#     --max_new_tokens $MAX_TOKENS \
#     --batch_size 1 # batchsize can only be 1

# # HALC
# python run_scripts/caption_generation.py \
#     --num_samples $NUM_SAMPLES \
#     --skip_num 0 \
#     --verbosity 1 \
#     --decoder halc \
#     --output_dir ./experiment_results \
#     --gpu-id $GPU_ID \
#     --split val2014 \
#     --seed $SEED \
#     --data_path /storage/jinhaoduan/datasets/coco2014 \
#     --max_new_tokens $MAX_TOKENS \
#     --batch_size 1 # batchsize can only be 1

# # Beam
# python run_scripts/caption_generation.py \
#     --num_samples $NUM_SAMPLES \
#     --skip_num 0 \
#     --verbosity 1 \
#     --decoder beam \
#     --output_dir ./experiment_results \
#     --gpu-id $GPU_ID \
#     --split val2014 \
#     --seed $SEED \
#     --data_path /storage/jinhaoduan/datasets/coco2014 \
#     --max_new_tokens $MAX_TOKENS \
#     --beam $BEAM \
#     --batch_size 1 # batchsize can only be 1
