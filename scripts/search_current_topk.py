import multiprocessing
import os
import time
import random
import subprocess


def worker(search_topk, device, device_queue):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    subprocess.run(["python",
                    "/home/kongfei/workspace/Classifier-Guided-VLM-OH/run_scripts/caption_generation.py",
                    "--num_samples",
                    "500",
                    "--skip_num",
                    "0",
                    "--verbosity",
                    "1",
                    "--decoder",
                    "classifier",
                    "--output_dir",
                    "/home/kongfei/models_state_dict/halc/experiment_results",
                    "--gpu-id",
                    "0",
                    "--split",
                    "val2014",
                    "--seed",
                    "42",
                    "--data_path",
                    "/home/kongfei/data/datasets/coco2014",
                    "--classifier_hs_position",
                    "current",
                    "--task",
                    "classify-hallucination",
                    "--classifier_ckpt_path",
                    "/home/kongfei/models_state_dict/halc/Classifier-Guided-VLM-OH/classifier_hidden_states_current_hs/chair/minigpt4/best_model_checkpoint.pth",
                    "--search_topk",
                    str(search_topk),], env=env)

    device_queue.put(device)


def main():
    search_topks = [1, 2, 3, 4, 5]
    devices = [0, 1, 2, 4]

    task_queue = multiprocessing.Queue()
    device_queue = multiprocessing.Queue()

    for i in search_topks:
        task_queue.put(i)

    for i in devices:
        device_queue.put(i)

    processes = []
    while not task_queue.empty():
        device = device_queue.get()
        task = task_queue.get()
        p = multiprocessing.Process(target=worker, args=(task, device, device_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("所有任务完成！")


if __name__ == "__main__":
    main()
