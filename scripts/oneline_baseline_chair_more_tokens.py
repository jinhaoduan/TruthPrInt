import fire
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

def worker(params, device, device_queue, run_generation, run_chair, run_object_count):
    params_dict = deepcopy(params)
    del params_dict['output_dir']
    del params_dict['data_path']
    del params_dict['result_file']
    for k in params_dict:
        params_dict[k] = str(params_dict[k])

    r0 = 0.0
    params_dict["r0"] = r0
    model_output_dir = Path(params["output_dir"]) / "chair" / params["model_name"] / params["decoder"]
    chair_metrics_path = f"{params['dataset_name']}_split_{params['split']}_seed_{params['seed']}_mt_{params['max_new_tokens']}_samples_{params['num_samples']}_skip_{params['skip_num']}_beam_{params['beam']}_hp_{params['classifier_hs_position']}_ct_{params['classifier_threshold']}_topk_{params['classifier_search_topk']}_ch_{params['classifier_hallucination_layer']}_br_layer{params['classifier_branch_layer']}_gw_{float(params['classifier_guidance_weight'])}_layers_{params['classifier_hs_layer']}_ld_{params['classifier_layers_dim']}_pw_{float(params['classifier_prev_hidden_weight'])}_norm_{params['classifier_use_norm_weight']}_r0_{str(r0)}_cn_{params['classifier_preprocess_norm']}_sa_{params['subspace_align']}"
    chair_metrics_path = Path(chair_metrics_path)
    if run_generation:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        run_params = ["python",
                      "run_scripts/caption_generation.py",
                      "--model",
                      params["model_name"],
                      "--decoder",
                      params["decoder"],
                      "--classifier_r0_penelty",
                      str(r0),
                      "--gpu-id",
                      "0",
                      "--classifier_task",
                      "classify-hallucination",
                      "--chair_metrics_path",
                      str(chair_metrics_path)]

        run_params = run_params \
            + ["--skip_num",
               params["skip_num"],
               "--num_samples",
               params["num_samples"],
               "--max_new_tokens",
               params["max_new_tokens"],
               "--seed",
               params["seed"],
               "--dataset_name",
               params["dataset_name"],
               "--output_dir",
               params["output_dir"],
               "--split",
               params["split"],
               "--data_path",
               params["data_path"],
               "--classifier_prev_hidden_weight",
               params["classifier_prev_hidden_weight"],
               "--beam",
               params["beam"],
               "--classifier_hs_position",
               params["classifier_hs_position"],
               "--classifier_ckpt_path",
               params["classifier_ckpt_path"],
               "--classifier_search_topk",
               params["classifier_search_topk"],
               "--classifier_hs_layer",]\
            + params["classifier_hs_layer"]\
            + ["--classifier_layers_dim"]\
            + params["classifier_layers_dim"]\
            + ["--classifier_guidance_weight",
                params["classifier_guidance_weight"],
                "--classifier_threshold",
                params["classifier_threshold"],
                "--classifier_branch_layer",
                params["classifier_branch_layer"],
                "--classifier_hallucination_layer",
               params["classifier_hallucination_layer"]]
        if params["classifier_use_branches_mlp"]:
            run_params = run_params + ["--classifier_use_branches_mlp"]
        if params["classifier_use_norm_weight"]:
            run_params = run_params + ["--classifier_use_norm_weight"]
        if params['classifier_preprocess_norm']:
            run_params = run_params + ["--classifier_preprocess_norm"]

        run_params = [str(i) for i in run_params]
        subprocess.run(run_params, env=env)

        device_queue.put(device)

        dataset_caption_path = {
            "coco": "annotations/captions_val2014.json"
        }

        subprocess.run(["python",
                        "eval/caption_to_chair.py",
                        "-c",
                        os.path.join(params["data_path"], dataset_caption_path[params["dataset_name"]]),
                        "-d",
                        str(model_output_dir / chair_metrics_path)])

    r = {}
    if run_chair:
        subprocess.run(["python",
                        "eval/eval_hallucination.py",
                        "--metric",
                        "chair",
                        "--data_dir",
                        params["data_path"] + "/",
                        "--output_dir",
                        str(model_output_dir / chair_metrics_path),
                        "--chair_input_path",
                        str(model_output_dir / chair_metrics_path / "chair.json"),
                        "-v"])

        _r = pd.read_csv(model_output_dir / chair_metrics_path / "chair_results.txt", sep="\t")
        r.update(dict(_r.iloc[0]))
        # writer = CSVExperimentReader("./", params["result_file"], params_dict)
        # writer.write_result(r)

    if run_object_count:
        subprocess.run(["python",
                        "eval/object_counter.py",
                        str(model_output_dir / chair_metrics_path / "chair_results.json"),])

        _r = pd.read_csv(str(model_output_dir / chair_metrics_path / "object_counter.csv"))
        r.update(dict(_r.iloc[0]))
    if len(r) > 0:
        writer = CSVExperimentReader("./", params["result_file"], params_dict)
        writer.write_result(r)


def main(
    result_file: str = "baseline_results_more_tokens",
    run_generation: bool = True,
    run_chair: bool = True,
    run_object_count: bool = True,
    output_dir: str = "./baseline_experiments_more_tokens",
):
    devices = [0, 1, 2, 3]

    model_name = ["minigpt4"]
    classifier_hs_position = ["previous"]
    classifier_ckpt_path = [""]
    classifier_search_topk = [0]
    classifier_threshold = [0.0]
    classifier_hs_layer = [[16]]
    classifier_guidance_weight = [0]
    classifier_branch_layer = [0]
    classifier_hallucination_layer = [0]
    classifier_layers_dim = [[1]]
    fpr = [0]
    eval_data = [""]
    clssifier_seed = [0]
    subspace_align = [False]
    classifier_preprocess_norm = [False]
    dataset_name = ["coco"]
    seed = [42]
    max_new_tokens = [64, 128, 192, 256, 320, 384]
    num_samples = [100]
    skip_num = [0]
    split = ["val2014"]
    output_dir = [output_dir]
    data_path = ["/storage/jinhaoduan/datasets/coco2014"]
    result_file = [result_file]
    classifier_use_branches_mlp = [False]
    classifier_prev_hidden_weight = [0]
    classifier_use_norm_weight = [False]
    beam = [3]
    decoder = ["beam", "opera", "dola"]

    keys = [
        "model_name",
        "classifier_hs_position",
        "classifier_ckpt_path",
        "classifier_search_topk",
        "classifier_hs_layer",
        "classifier_guidance_weight",
        "classifier_threshold",
        "classifier_branch_layer",
        "classifier_hallucination_layer",
        "dataset_name",
        "seed",
        "max_new_tokens",
        "num_samples",
        "skip_num",
        "beam",
        "split",
        "output_dir",
        "data_path",
        "result_file",
        "classifier_use_branches_mlp",
        "classifier_prev_hidden_weight",
        "classifier_use_norm_weight",
        "classifier_layers_dim",
        "fpr",
        "eval_data",
        "classifier_seed",
        "subspace_align",
        "classifier_preprocess_norm",
        "decoder",
    ]

    params = []

    values = product(
        model_name,
        classifier_hs_position,
        classifier_ckpt_path,
        classifier_search_topk,
        classifier_hs_layer,
        classifier_guidance_weight,
        classifier_threshold,
        classifier_branch_layer,
        classifier_hallucination_layer,
        dataset_name,
        seed,
        max_new_tokens,
        num_samples,
        skip_num,
        beam,
        split,
        output_dir,
        data_path,
        result_file,
        classifier_use_branches_mlp,
        classifier_prev_hidden_weight,
        classifier_use_norm_weight,
        classifier_layers_dim,
        fpr,
        eval_data,
        clssifier_seed,
        subspace_align,
        classifier_preprocess_norm,
        decoder,
    )

    beam = [1]
    decoder = ["greedy", "vcd"]

    values1 = product(
        model_name,
        classifier_hs_position,
        classifier_ckpt_path,
        classifier_search_topk,
        classifier_hs_layer,
        classifier_guidance_weight,
        classifier_threshold,
        classifier_branch_layer,
        classifier_hallucination_layer,
        dataset_name,
        seed,
        max_new_tokens,
        num_samples,
        skip_num,
        beam,
        split,
        output_dir,
        data_path,
        result_file,
        classifier_use_branches_mlp,
        classifier_prev_hidden_weight,
        classifier_use_norm_weight,
        classifier_layers_dim,
        fpr,
        eval_data,
        clssifier_seed,
        subspace_align,
        classifier_preprocess_norm,
        decoder,
    )

    params.extend([dict(zip(keys, v)) for v in values])
    params.extend([dict(zip(keys, v)) for v in values1])

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
        if run_generation:
            device = device_queue.get()
        else:
            device = 0
        task = task_queue.get()
        p = multiprocessing.Process(target=worker, args=(task, device, device_queue, run_generation, run_chair, run_object_count))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('All Finished!')


if __name__ == '__main__':
    fire.Fire(main)
