import multiprocessing
import os
import time
import random
import subprocess


def worker(search_topk):
    subprocess.run(["python",
                    "eval/caption_to_chair.py",
                    "-c",
                    "/home/kongfei/data/datasets/coco2014/annotations/captions_val2014.json",
                    "-d",
                    search_topk])


def worker2(search_topk):
    subprocess.run(["python",
                    "eval/eval_hallucination.py",
                    "--metric",
                    "chair",
                    "--data_dir",
                    "/home/kongfei/data/datasets/coco2014/",
                    "--output_dir",
                    search_topk,
                    "--chair_input_path",
                    os.path.join(search_topk, "chair.json"),
                    "-v"])


def main():
    search_topks = [
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_16_br_layer2_gw_0.05_layers_[16, 17, 18, 19, 20]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_16_br_layer2_gw_0.1_layers_[16, 17, 18, 19, 20]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_16_br_layer2_gw_0.2_layers_[16, 17, 18, 19, 20]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_16_br_layer2_gw_0.5_layers_[16, 17, 18, 19, 20]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_16_br_layer2_gw_1.0_layers_[16, 17, 18, 19, 20]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_16_br_layer2_gw_2.0_layers_[16, 17, 18, 19, 20]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_16_br_layer2_gw_5.0_layers_[16, 17, 18, 19, 20]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_16_br_layer2_gw_10.0_layers_[16, 17, 18, 19, 20]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_16_br_layer2_gw_20.0_layers_[16, 17, 18, 19, 20]_",

        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_15_br_layer2_gw_0.05_layers_[11, 12, 13, 14, 15]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_15_br_layer2_gw_0.1_layers_[11, 12, 13, 14, 15]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_15_br_layer2_gw_0.2_layers_[11, 12, 13, 14, 15]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_15_br_layer2_gw_0.5_layers_[11, 12, 13, 14, 15]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_15_br_layer2_gw_1.0_layers_[11, 12, 13, 14, 15]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_15_br_layer2_gw_2.0_layers_[11, 12, 13, 14, 15]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_15_br_layer2_gw_5.0_layers_[11, 12, 13, 14, 15]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_15_br_layer2_gw_10.0_layers_[11, 12, 13, 14, 15]_",
        "/home/kongfei/models_state_dict/halc/experiment_results/chair/minigpt4/classifier/coco_split_val2014_seed_42_max_tokens_64_samples_500_skip_0_hs_position_gradient_classifier_threshold_0.9_topk_5_ch_15_br_layer2_gw_20.0_layers_[11, 12, 13, 14, 15]_",
    ]

    task_queue = multiprocessing.Queue()

    for i in search_topks:
        task_queue.put(i)

    processes = []
    while not task_queue.empty():
        task = task_queue.get()
        p = multiprocessing.Process(target=worker, args=(task,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    processes.clear()

    for i in search_topks:
        task_queue.put(i)

    while not task_queue.empty():
        task = task_queue.get()
        p = multiprocessing.Process(target=worker2, args=(task,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("所有任务完成！")


if __name__ == "__main__":
    main()
