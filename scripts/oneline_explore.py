import fire
import sys
from itertools import chain, product
sys.path.append('/home/kongfei/workspace/Classifier-Guided-VLM-OH/scripts')
import subprocess
import os
from pathlib import Path
import pandas as pd
from utils import CSVExperimentReader
from copy import deepcopy
import multiprocessing
import filelock
import json
import yaml


def training_and_eval(device_queue, params):
    out_params, exp_params, result_path = _make_training_params(params=params)
    fl = filelock.FileLock(os.path.join(
        out_params["classifier_ckpt_path"] + ".lock"
    ))

    env = os.environ.copy()

    with fl:
        device = device_queue.get()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        if params['ckpt_file'] is None and not os.path.exists(out_params["classifier_ckpt_path"]):
            run_params = ([
                "python",
                "decoder_zoo/Classifier/classifier_training.py",
                "--random_seed",
                params["seed"],
                "--model_name",
                params["model_name"],
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
                params["training_root_path"],
                "--r0_penelty",
                params["r0_penelty"],
                "--eval_mode",
                False])
            run_params = [str(i) for i in run_params]
            subprocess.run(run_params, env=env)

    if params['threshold_value'] is None:
        if os.path.exists(result_path):
            writer = CSVExperimentReader(os.path.dirname(result_path), "test_result", exp_params)
            # try:
            if writer.read_result(f"target_fpr_{params['fpr']}_{params['hallucination_layer']}") is not None:
                device_queue.put(device)
                return
            # except ValueError:
            #     os.remove(os.path.join(os.path.dirname(result_path), "test_result.csv"))

        run_params = ([
            "python",
            "decoder_zoo/Classifier/classifier_training.py",
            "--random_seed",
            params["seed"],
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
            params['training_root_path'],
            "--r0_penelty",
            params["r0_penelty"],
            "--eval_mode",
            True,
            "--eval_fpr",
            params["fpr"],
            "--eval_model_ckpt",
            out_params["classifier_ckpt_path"]])
        run_params = [str(i) for i in run_params]
        subprocess.run(run_params, env=env)

    device_queue.put(device)


def _make_training_params(params):
    threshold_value = params['threshold_value']
    fpr = params['fpr']
    training_root_path = params['training_root_path']
    ckpt_file = params['ckpt_file']
    assert (fpr is None and threshold_value is not None) or (fpr is not None and threshold_value is None)
    out_params = {}

    if params['ckpt_file'] is None:
        config_path = params['config_path']
        with open(os.path.join(config_path, f"{params['model_name']}.yaml"), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            classifier_training_hs_path = config['classifier_training_hs_path']
            classifier_test_hs_path = config['classifier_test_hs_path']
        train_name = os.path.basename(os.path.dirname(classifier_training_hs_path)) + os.path.basename(classifier_training_hs_path).split('.')[0]
        test_name = os.path.basename(os.path.dirname(classifier_test_hs_path)) + os.path.basename(classifier_test_hs_path).split('.')[0]
        ckpt_file = f"model_name_{params['model_name']}_seed_{params['seed']}_train_hs_positions_{params['train_hs_positions']}_hs_position_{params['hs_position']}_hl_{params['hallucination_layer']}_br_layer_{params['model_branch_layer']}_layers_dim_{params['layers_dim']}_r0_penelty_{params['r0_penelty']}_subspace_align_{params['subspace_align']}.pth"
        ckpt_file = os.path.join(training_root_path, train_name, test_name, ckpt_file)
        out_params["classifier_ckpt_path"] = ckpt_file

    out_params["classifier_ckpt_path"] = ckpt_file
    if params["threshold_value"] is not None:
        out_params["fpr"] = 0
        out_params["classifier_threshold"] = params["threshold_value"]
    else:
        out_params["fpr"] = fpr

    exp_params = {
        'seed': params['seed'],
        'train_hs_positions': params['train_hs_positions'],
        'hs_position': params['hs_position'],
        'hallucination_layer': params['hallucination_layer'],
        'model_branch_layer': params['model_branch_layer'],
        'layers_dim': params['layers_dim'],
        'eval_data': params['eval_data'],
        'subspace_align': params['subspace_align'],
        'model_name': params['model_name'],
    }
    _exp_params = deepcopy(exp_params)
    for k in _exp_params:
        if isinstance(_exp_params[k], (list, tuple)):
            _exp_params[k] = str(_exp_params[k])
    result_path = os.path.join(os.path.dirname(out_params["classifier_ckpt_path"]), "test_result.csv")
    out_params.update(exp_params)
    return out_params, _exp_params, result_path


def read_ckpt_threshold(params):

    out_params, exp_params, result_path = _make_training_params(params=params)

    writer = CSVExperimentReader(os.path.dirname(result_path), "test_result", exp_params)

    if params["fpr"] is not None:
        threshold = writer.read_result(f"target_fpr_{params['fpr']}_{params['hallucination_layer']}")
        if threshold is None:
            raise ValueError("Threshold not found")
        out_params["classifier_threshold"] = round(threshold[f"target_fpr_{params['fpr']}_{params['hallucination_layer']}"], 4)
    return out_params


def worker(params, device, device_queue, run_generation, run_chair, run_object_count, run_opope):
    params_dict = deepcopy(params)
    del params_dict['output_dir']
    del params_dict['data_path']
    del params_dict['result_file']
    for k in params_dict:
        params_dict[k] = str(params_dict[k])

    r0 = float(0)  # r0不再使用
    params_dict["r0"] = r0
    model_output_dir = Path(params["output_dir"]) / "chair" / params["model_name"] / "classifier"
    chair_metrics_path = f"{params['dataset_name']}_split_{params['split']}_seed_{params['seed']}_mt_{params['max_new_tokens']}_samples_{params['num_samples']}_skip_{params['skip_num']}_hp_{params['classifier_hs_position']}_ct_{params['classifier_threshold']}_topk_{params['classifier_search_topk']}_ch_{params['classifier_hallucination_layer']}_br_layer{params['classifier_branch_layer']}_gw_{float(params['classifier_guidance_weight'])}_layers_{params['classifier_hs_layer']}_ld_{params['classifier_layers_dim']}_pw_{float(params['classifier_prev_hidden_weight'])}_norm_{params['classifier_use_norm_weight']}_r0_{str(r0)}_cn_{params['classifier_preprocess_norm']}_sa_{params['subspace_align']}_bn_{params['classifier_backtracing_number']}_gmr_{float(params['classifier_gradient_mask_rate'])}"
    chair_metrics_path = Path(chair_metrics_path)
    if run_generation:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        run_params = ["python",
                      "run_scripts/caption_generation.py",
                      "--model",
                      params["model_name"],
                      "--decoder",
                      "classifier",
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
               "--classifier_backtracing_number",
               params["classifier_backtracing_number"],
               "--output_dir",
               params["output_dir"],
               "--split",
               params["split"],
               "--data_path",
               params["data_path"],
               "--classifier_prev_hidden_weight",
               params["classifier_prev_hidden_weight"],
               "--classifier_hs_position",
               params["classifier_hs_position"],
               "--classifier_ckpt_path",
               params["classifier_ckpt_path"],
               "--classifier_subspace_align",
                params["subspace_align"],
                "--classifier_gradient_mask_rate",
                params["classifier_gradient_mask_rate"],
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
        # writer = CSVExperimentReader("./", params["result_file"], params_dict)
        # writer.write_result(r)

    if run_opope:
        caption_dir = str(model_output_dir / chair_metrics_path)
        caption_path = str(model_output_dir / chair_metrics_path / "generated_captions.json")

        avg_acc = 0
        avg_recall = 0
        avg_f1 = 0
        avg_precision = 0

        for pope_type in ["random", "popular", "adversarial"]:
            subprocess.run(["python",
                            "eval/opope_eval.py",
                            "--model",
                            params["model_name"],
                            "--data_path",
                            params["data_path"] + '/val2014',
                            "--gt_seg_path",
                            "./pope_coco/coco_ground_truth_segmentation.json",
                            "--output_dir",
                            caption_dir,
                            "--num_images",
                            "100",  # HALC defaul value
                            "--batch_size",
                            "1",
                            "--generated_caption",
                            caption_path,
                            "--pope_type",
                            pope_type])
            with open(model_output_dir / chair_metrics_path / 'opope' / f"{pope_type}_result.json", "r") as f:
                _r = json.load(f)
                avg_acc += _r[f"{pope_type}_Accuracy"]
                avg_recall += _r[f"{pope_type}_Recall"]
                avg_f1 += _r[f"{pope_type}_F1 Score"]
                avg_precision += _r[f"{pope_type}_Precision"]
                r.update(_r)
        r.update({"avg_Accuracy": avg_acc / 3, "avg_Recall": avg_recall / 3, "avg_F1 Score": avg_f1 / 3, "avg_Precision": avg_precision / 3})

    if len(r) > 0:
        writer = CSVExperimentReader("./", params["result_file"], params_dict)
        writer.write_result(r)


def main(
    result_file: str = "results",
    run_generation: bool = True,
    run_chair: bool = True,
    run_object_count: bool = True,
    run_opope: bool = True,
    output_dir: str = "/home/kongfei/models_state_dict/halc/search_11_14",
    classifier_root: str = "/home/kongfei/models_state_dict/halc/search_11_14_classifier",
    config_path: str = "/home/kongfei/workspace/Classifier-Guided-VLM-OH/train_config",
):
    devices = [0, 1, 2, 3, 4, 5, 6 , 7]

    # =========================================================================================
    # Training and evaluation of the classifier
    params = []
    training_params = {
        # 'model_name': ["minigpt4", "mplug-owl2"],
        'model_name': ["minigpt4"],
        'seed': [42],
        'train_hs_positions': [[16]],
        'hs_position': ['previous'],
        'threshold_value': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'config_path': [config_path],
        'fpr': [None],
        'training_root_path': [classifier_root],
        "subspace_align": [16, 32, 64, 128],
        'ckpt_file': [None],
        'hallucination_layer': [16],
        'model_branch_layer': [2],
        'layers_dim': [[128, 64]],
        'r0_penelty': [0],
        'eval_data': ['coco'],
    }

    keys = [
        'model_name',
        'seed',
        'train_hs_positions',
        'hs_position',
        'threshold_value',
        'config_path',
        'fpr',
        'training_root_path',
        'subspace_align',
        'ckpt_file',
        'hallucination_layer',
        'model_branch_layer',
        'layers_dim',
        'r0_penelty',
        'eval_data',
    ]

    values = product(*[training_params[k] for k in training_params])

    params.extend([dict(zip(keys, v)) for v in values])

    training_task_queue = multiprocessing.Queue()
    training_device_queue = multiprocessing.Queue()

    for i in params:
        training_task_queue.put(i)

    for i in devices:
        training_device_queue.put(i)

    processes = []
    while not training_task_queue.empty():
        task = training_task_queue.get()

        p = multiprocessing.Process(target=training_and_eval, args=(training_device_queue, task))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # =========================================================================
    # read params and run the generation and evaluation
    read_params = [read_ckpt_threshold(i) for i in params]
    # for i in read_params:
    #     print(i)
    #     print('\n\n')
    # break

    classifier_hs_position = ["previous_bt_t2"]
    classifier_search_topk = [5]
    classifier_guidance_weight = [1]
    dataset_name = ["coco"]
    seed = [42]
    max_new_tokens = [64]
    num_samples = [500]
    skip_num = [0]
    split = ["val2014"]
    output_dir = [output_dir]
    data_path = ["/home/kongfei/data/datasets/coco2014"]
    result_file = [result_file]
    classifier_use_branches_mlp = [True]
    classifier_prev_hidden_weight = [0]
    classifier_use_norm_weight = [False]
    classifier_backtracing_number = [5]
    classifier_gradient_mask_rate = [0]
    classifier_preprocess_norm = [True]

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
        "classifier_backtracing_number",
        "classifier_gradient_mask_rate"
    ]

    params = []
    for _read_param in read_params:
        classifier_ckpt_path = [_read_param["classifier_ckpt_path"]]
        fpr = [_read_param["fpr"]]
        classifier_threshold = [_read_param["classifier_threshold"]]
        eval_data = [_read_param["eval_data"]]
        clssifier_seed = [_read_param["seed"]]
        classifier_hs_layer = [_read_param['train_hs_positions']]
        classifier_hallucination_layer = [_read_param['hallucination_layer']]
        classifier_branch_layer = [_read_param['model_branch_layer']]
        classifier_layers_dim = [_read_param['layers_dim']]
        subspace_align = [_read_param['subspace_align']]
        model_name = [_read_param['model_name']]

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
            classifier_backtracing_number,
            classifier_gradient_mask_rate,
        )

        params.extend([dict(zip(keys, v)) for v in values])

    # for i in params:
    #     print(i)
    #     break
    #     print("\n")

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
        p = multiprocessing.Process(target=worker, args=(task, device, device_queue, run_generation, run_chair, run_object_count, run_opope))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('所有任务完成！')


if __name__ == '__main__':
    fire.Fire(main)
