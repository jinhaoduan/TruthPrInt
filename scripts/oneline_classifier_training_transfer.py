
import fire
import json
from itertools import chain, product
import subprocess
import os
from pathlib import Path
import pandas as pd
from scripts.utils import CSVExperimentReader
from copy import deepcopy
import multiprocessing
import filelock
import sys

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def worker(params, device, device_queue):
    params_dict = deepcopy(params)
    for k in params_dict:
        params_dict[k] = str(params_dict[k])

    r0 = 0.0
    params_dict["r0"] = r0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)


    run_params = ([
        "python",
        "decoder_zoo/Classifier/classifier_training_highspecificity.py",
        "--classifier_training_hs_path",
        params["classifier_training_hs_path"],
        "--classifier_test_hs_path",
        params["classifier_test_hs_path"],
        "--random_seed",
        params["random_seed"],
        "--subspace_align",
        params["subspace_align"],
        "--train_hs_positions",
        str(params["train_hs_positions"]),
        "--hallucination_layer",
        params["hallucination_layer"],
        "--model_branch_layer",
        params["model_branch_layer"],
        "--layers_dim",
        str(params["layers_dim"]),
        "--output_path",
        params["output_path"],
        "--r0_penelty",
        params["r0_penelty"],
        "--eval_mode",
        params["eval_mode"],
        "--normalize_hs",
        params["normalize_hs"],
        ])
    run_params = [str(i) for i in run_params]
    subprocess.run(run_params, env=env)

    device_queue.put(device)


def main():
    devices = [0, 1, 2, 3]

    random_seed=[42]
    train_hs_positions = [[16]]
    hs_position = ['previous']
    hallucination_layer = [16]
    model_branch_layer=[2]
    layers_dim = [[128, 64]]
    output_path=["./highspecificity_classifier_exp/transfer"]
    r0_penelty=[0]
    normalize_hs=[True]
    # subspace_align=[0, 32, 64, 128, 256, 512, 1024, 2048]
    subspace_align=[8]
    # subspace_align=[0]
    eval_mode=[False]
    classifier_training_hs_path=['minigpt-4']
    classifier_test_hs_path=['llava-1.5', 'mplug-owl2']

    keys = [
        "random_seed",
        "train_hs_positions",
        "hs_position",
        "hallucination_layer",
        "model_branch_layer",
        "layers_dim",
        "output_path",
        "r0_penelty",
        "normalize_hs",
        "subspace_align",
        "eval_mode",
        "classifier_training_hs_path",
        "classifier_test_hs_path",
    ]

    params = []
    values = product(
        random_seed,
        train_hs_positions,
        hs_position,
        hallucination_layer,
        model_branch_layer,
        layers_dim,
        output_path,
        r0_penelty,
        normalize_hs,
        subspace_align,
        eval_mode,
        classifier_training_hs_path,
        classifier_test_hs_path,
    )

    classifier_training_hs_path=['llava-1.5']
    classifier_test_hs_path=['minigpt-4', 'mplug-owl2']
    values1 = product(
        random_seed,
        train_hs_positions,
        hs_position,
        hallucination_layer,
        model_branch_layer,
        layers_dim,
        output_path,
        r0_penelty,
        normalize_hs,
        subspace_align,
        eval_mode,
        classifier_training_hs_path,
        classifier_test_hs_path,
    )

    classifier_training_hs_path=['mplug-owl2']
    classifier_test_hs_path=['minigpt-4', 'llava-1.5']
    values2 = product(
        random_seed,
        train_hs_positions,
        hs_position,
        hallucination_layer,
        model_branch_layer,
        layers_dim,
        output_path,
        r0_penelty,
        normalize_hs,
        subspace_align,
        eval_mode,
        classifier_training_hs_path,
        classifier_test_hs_path,
    )


    params.extend([dict(zip(keys, v)) for v in values])
    params.extend([dict(zip(keys, v)) for v in values1])
    params.extend([dict(zip(keys, v)) for v in values2])

    for i in params:
        print(i)
        print("\n")
    
    print('Total number of tasks: ', len(params))

    task_queue = multiprocessing.Queue()
    device_queue = multiprocessing.Queue()

    for i in params:
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

    print('All Finished!')


if __name__ == '__main__':
    fire.Fire(main)
