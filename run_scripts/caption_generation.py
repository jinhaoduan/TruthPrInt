import argparse
import os
import random
import sys
sys.path.append("mPLUG-Owl/mPLUG-Owl2")
sys.path.append("./")
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms

from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
import json

from types import SimpleNamespace
from decoder_zoo.Woodpecker.vis_corrector import Corrector
# from decoder_zoo.Woodpecker.config import woodpecker_args_dict
from decoder_zoo.HALC.context_density.halc import halc_assistant
from decoder_zoo.VCD.vcd_utils.vcd_add_noise import add_diffusion_noise
from decoder_zoo.Classifier.classifier_helper import ClassifierHelper

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict

import torch
from PIL import Image
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "mplug-owl2": "eval_configs/mplug-owl2_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "mplug-owl2": "USER: <|image|><question> ASSISTANT:",
}


def setup_seeds(config, seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, default="minigpt4", help="model")
parser.add_argument(
    "-d",
    "--decoder",
    type=str,
    default="classifier",
    help="Decoding strategy to use. You can choose from 'greedy', 'dola', 'halc', 'classifier'. Default is 'greedy'.",
)
parser.add_argument(
    "-g", "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
)
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="coco",
    help="Name of the dataset. Default is 'coco'.",
)
parser.add_argument(
    "--split",
    type=str,
    default="train2014",
    help="Split of the dataset. Default is 'train2014'.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/storage/jinhaoduan/datasets/coco2014",
    help="data path",
)
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")
parser.add_argument("-b", "--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-n", "--num_samples", type=int, default=10)
parser.add_argument("-m", "--max_new_tokens", type=int, default=64)
parser.add_argument(
    "-v",
    "--verbosity",
    action="store_false",
    dest="verbosity",
    default=True,
    help="Verbosity. Default: True.",
)
parser.add_argument(
    "-k",
    "--k-candidate-num",
    type=int,
    default=2,
    help="specify the k candidate number for halc.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./experiment_results",
    help="Output ditectory for saving test results. Default is './generated_chair_inputs/'.",
)
parser.add_argument(
    "-p",
    "--post-correction",
    type=str,
    default=None,
    help="Post correction method such as woodpecker, lure.",
)
parser.add_argument(
    "-e",
    "--expand-ratio",
    type=float,
    default=0.6,
    help="Expand ratio of growing contextual field.",
)
parser.add_argument(
    "--cd_alpha",
    type=float,
    default=1,
    help="Alpha param for VCD.",
)
parser.add_argument("--cd_beta", type=float, default=0.1, help="Beta param for VCD.")
parser.add_argument("--noise_step", type=int, default=500, help="Noise step for VCD.")
parser.add_argument(
    "--detector",
    type=str,
    default="dino",
    help="Detector type. Default is 'groundingdino'.",
)
parser.add_argument(
    "--debugger",
    type=int,
    default=2,
    help="0 print no debugging output; 1 only print hallucination correction; 2 print all the debugging output.",
)
parser.add_argument("--box_threshold", type=float, default=0.45, help="Box threshold for DINO.")
parser.add_argument(
    "--gt_seg_path",
    type=str,
    default="pope_coco/coco_ground_truth_segmentation.json",
    help="Input json file that contains ground truth objects in the image.",
)
parser.add_argument(
    "--generate_pope",
    action="store_true",
    default=False,
    help="Whether to generate POPE questions.",
)

parser.add_argument("--skip_num", type=int, default=0, help="Skip the first skip_num samples.")
# parameters for classifier guidance decoding
parser.add_argument("--classifier_ckpt_path",
                    type=str, default='/home/kongfei/workspace/Classifier-Guided-VLM-OH/classifier_hidden_states_previous_hs/chair/minigpt4/minigpt4_cc_sbu_align_layer16-20_previous_model_num_branch_5_model_branch_layer_2_checkpoint.pth',
                    help="Path to the classifier checkpoint.")
parser.add_argument("--classifier_task", type=str, default='classify-hallucination', choices=["collect-hidden-states", "classify-hallucination"], help="Task for the classifier.")
parser.add_argument("--classifier_hs_position", type=str, default='previous', choices=["previous", "current", "gradient", "gradient_second_token", "previous_bt", "previous_bt_t2", "gradient_bt", "gradient_bt_llm", "save_stat_bt"], help="Position of the hidden states to be used for classification, previous means using the hidden states from the previous token of the object token.")
parser.add_argument("--classifier_threshold", type=float, default=0.9, help="Threshold for the classifier confidence score.")
parser.add_argument("--exp_nickname", type=str, default='', help="Nickname for the experiment.")


parser.add_argument(
    "--classifier_fix_coco_model",
    action="store_true",
    default=False,
    help="Whether to generate POPE questions.",
)

parser.add_argument(
    "--classifier_search_topk",
    type=int,
    default=1,
)

parser.add_argument(
    "--classifier_hs_layer",
    type=int,
    nargs='+',
    default=[16],
)

parser.add_argument(
    "--classifier_guidance_weight",
    type=float,
    default=1.,
)

parser.add_argument(
    "--classifier_branch_layer",
    type=int,
    default=1,
)

parser.add_argument(
    "--classifier_use_branches_mlp",
    action="store_true",
    default=False,)

parser.add_argument(
    "--classifier_hallucination_layer",
    type=int,
    default=16,)

parser.add_argument(
    "--classifier_layers_dim",
    type=int,
    nargs='+',
    default=[128, 64],
)

parser.add_argument(
    "--classifier_prev_hidden_weight",
    type=float,
    default=0,)

parser.add_argument(
    "--classifier_use_norm_weight",
    action="store_true",
    default=False,)

parser.add_argument(
    "--classifier_r0_penelty",
    type=float,
    default=0,)

parser.add_argument(
    "--classifier_preprocess_norm",
    action="store_true",
    default=False,)

parser.add_argument(
    "--classifier_subspace_align",
    type=int,
    default=0,)

parser.add_argument(
    "--chair_metrics_path",
    type=str,
    default=None,)

parser.add_argument(
    "--classifier_backtracing_number",
    type=int,
    default=2,)

parser.add_argument(
    "--classifier_gradient_mask_rate",
    type=float,
    default=0.3,
)

# args = parser.parse_known_args()[0]
args = parser.parse_args()

# print("args.gpu_id", args.gpu_id)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)

model_name = args.model
decoding_strategy = args.decoder

seed = args.seed
setup_seeds(cfg, seed)

device = (
    torch.device(f"cuda:{int(args.gpu_id)}") if torch.cuda.is_available() else "cpu"
)
print("device", device)
# device = "cpu"

verbosity = args.verbosity
k_candidate_num = args.k_candidate_num
detector_type = args.detector
num_samples = args.num_samples
dataset_name = args.dataset_name
data_path = args.data_path
output_dir = args.output_dir
num_beams = args.beam
num_workers = args.num_workers
batch_size = args.batch_size
post_correction = args.post_correction
max_new_tokens = args.max_new_tokens
expand_ratio = args.expand_ratio
cd_alpha = args.cd_alpha
cd_beta = args.cd_beta
box_threshold = args.box_threshold
debugger = args.debugger
gt_seg_path = args.gt_seg_path
generate_pope = args.generate_pope
skip_num = args.skip_num
classifier_fix_coco_model = args.classifier_fix_coco_model
classifier_search_topk = args.classifier_search_topk
classifier_hs_layer = args.classifier_hs_layer
classifier_guidance_weight = args.classifier_guidance_weight
classifier_branch_layer = args.classifier_branch_layer
classifier_use_branches_mlp = args.classifier_use_branches_mlp
classifier_hallucination_layer = args.classifier_hallucination_layer
classifier_layers_dim = args.classifier_layers_dim
classifier_prev_hidden_weight = args.classifier_prev_hidden_weight
classifier_use_norm_weight = args.classifier_use_norm_weight
classifier_preprocess_norm = args.classifier_preprocess_norm
classifier_r0_penelty = args.classifier_r0_penelty
classifier_subspace_align = args.classifier_subspace_align
classifier_backtracing_number = args.classifier_backtracing_number
classifier_gradient_mask_rate = args.classifier_gradient_mask_rate
# ========================================
#             Model Initialization
# ========================================
print("Initializing Model")

# print("cfg", cfg)
# input()
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config)
print("device", device)
print(model.device)
model.to(device)
model.eval()

processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
    vis_processor_cfg
)


valid_decoding_strategies = [
    "greedy",
    "dola",
    "halc",
    "opera",
    "vcd",
    "classifier",
    "beam"
]


valid_post_editing_strategies = ["lure", "woodpecker"]
valid_detector = ["dino", "owlv2"]

assert (
    decoding_strategy in valid_decoding_strategies
), f"Invalid decoding strategy: {decoding_strategy}, should be in {valid_decoding_strategies}"
assert (
    post_correction in valid_post_editing_strategies or post_correction is None
), f"Invalid post correction strategy: {post_correction}, should be in {valid_post_editing_strategies}"
assert (
    detector_type in valid_detector
), f"Invalid detector type: {detector_type}, should be in {valid_detector}"

decoding_strategy = decoding_strategy
opera_decoding = False
dola_decoding = False
halc_decoding = False
vcd_decoding = False
beam_search = False
classifier_decoding = False

base_dir = os.path.join(output_dir, "chair", args.model, decoding_strategy)

print("decoding_strategy", decoding_strategy)
# config_name = f"{dataset_name}_split_{args.split}_seed_{seed}_mt_{max_new_tokens}_samples_{num_samples}_skip_{skip_num}_hp_{args.classifier_hs_position}_ct_{args.classifier_threshold}_topk_{args.classifier_search_topk}_ch_{classifier_hallucination_layer}_br_layer{classifier_branch_layer}_gw_{classifier_guidance_weight}_layers_{classifier_hs_layer}_ld_{classifier_layers_dim}_pw_{classifier_prev_hidden_weight}_norm_{classifier_use_norm_weight}_r0_{classifier_r0_penelty}_cn_{classifier_preprocess_norm}_sa_{classifier_subspace_align}_bn_{classifier_backtracing_number}_gmr_{classifier_gradient_mask_rate}"
config_name = args.chair_metrics_path
# config_name = f"{dataset_name}_split_{args.split}_seed_{seed}_mt_{max_new_tokens}_samples_{num_samples}_skip_{skip_num}_hp_{args.classifier_hs_position}_ct_{args.classifier_threshold}_topk_{args.classifier_search_topk}_ch_{classifier_hallucination_layer}_br_layer{classifier_branch_layer}_gw_{classifier_guidance_weight}_layers_{classifier_hs_layer}_ld_{classifier_layers_dim}_pw_{classifier_prev_hidden_weight}_norm_{classifier_use_norm_weight}_r0_{classifier_r0_penelty}_cn_{classifier_preprocess_norm}_sa_{classifier_subspace_align}_bn_{classifier_backtracing_number}_gmr_{classifier_gradient_mask_rate}"
# config_name = args.chair_metrics_path
# assert config_name == args.chair_metrics_path, f'{config_name} \n {args.chair_metrics_path}'
if decoding_strategy == "greedy":
    pass
elif decoding_strategy == "dola":
    dola_decoding = True
elif decoding_strategy == "halc":
    halc_decoding = True
    dola_decoding = True
    beam_search = True
elif decoding_strategy == "opera":
    beam_search = True
    opera_decoding = True
elif decoding_strategy == "vcd":
    vcd_decoding = True
elif decoding_strategy == "classifier":
    classifier_decoding = True
    if args.classifier_task == "classify-hallucination":
        assert args.classifier_ckpt_path is not None, "Classifier checkpoint path is required for classifying hallucination."
        assert args.classifier_threshold is not None, "Classifier threshold is required for classifying hallucination."
    elif args.classifier_task == "collect-hidden-states":
        assert args.classifier_hs_position is not None, "Classifier hidden states position is required for collecting hidden states."
        config_name = f"{dataset_name}_split_{args.split}_seed_{seed}_max_tokens_{max_new_tokens}_samples_{num_samples}_skip_{skip_num}_hs_position_{args.classifier_hs_position}_w_full_scores"
    else:
        raise ValueError(f"Invalid classifier task: {args.classifier_task}")
elif decoding_strategy == "beam":
    beam_search = True
else:
    raise ValueError(f"Invalid decoding strategy: {decoding_strategy}")

base_dir = os.path.join(base_dir, config_name)


if post_correction == "woodpecker":
    model_args = SimpleNamespace(**woodpecker_args_dict)
    corrector = Corrector(model_args)

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

print(f"\033[42m####### Current Decoding Strategy: {decoding_strategy} #######\033[0m")
print(f"\033[42m####### Save Path: {base_dir} #######\033[0m")

print('Arguments')
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

if verbosity:
    print("\ndecoding strategy: ", decoding_strategy)
    print("backbone model_name: ", args.model)
    print("dataset_name: ", dataset_name)
    print("data_path: ", data_path)
    print("output_dir: ", output_dir)
    print("num_samples: ", num_samples)
    print("num_beams: ", num_beams)
    print("seed: ", seed)
    print(vis_processors["eval"].transform)


mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

# annotation_file_path = os.path.join(args.data_path, f"annotations/instances_{args.split}.json")
caption_file_path = os.path.join(args.data_path, f"annotations/captions_{args.split}.json")
# with open(annotation_file_path, "r") as f:
# lines = f.readlines()
# coco_anns = json.loads(lines[0])
print('Loading caption file from', caption_file_path)
coco = COCO(caption_file_path)

img_ids = coco.getImgIds()


if generate_pope:

    num_objects = 3
    segment_results = [json.loads(q) for q in open(gt_seg_path, "r")]
    if verbosity:
        print(
            f"\nGround truth segmentation results loaded successfully, contains {len(segment_results)} classes."
        )

    # process segmentation ground truth
    processed_segment_results = []
    # Sample images which contain more than sample_num objects
    for cur_image in segment_results:
        if len(cur_image["objects"]) >= num_objects:
            processed_segment_results.append(cur_image)

    assert (
        len(processed_segment_results) >= num_samples
    ), f"The number of images that contain more than {num_objects} objects is less than {num_samples}."

    # Randomly sample num_samples images
    processed_segment_results = random.sample(processed_segment_results, num_samples)
    sampled_img_ids = [cur_image["image_id"] for cur_image in processed_segment_results]

else:

    # sample image ids
    sampled_img_ids = random.sample(img_ids, num_samples)

if classifier_fix_coco_model:
    sampled_img_ids = img_ids[:num_samples]
else:
    sampled_img_ids = sampled_img_ids[skip_num:]

print("sampled_img_ids", len(sampled_img_ids))

img_files = []
for cur_img_id in sampled_img_ids:
    cur_img = coco.loadImgs(cur_img_id)[0]
    cur_img_path = cur_img["file_name"]
    img_files.append(cur_img_path)

# img_dict = {}

# categories = coco_anns["categories"]
# category_names = [c["name"] for c in categories]
# category_dict = {int(c["id"]): c["name"] for c in categories}

# for img_info in coco_anns["images"]:
#     img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

# for ann_info in coco_anns["annotations"]:
#     img_dict[ann_info["image_id"]]["anns"].append(
#         category_dict[ann_info["category_id"]]
#     )

halc_params = {
    "context_domain": "upper",
    "contrast_weight": 0.05,
    "context_window": 4,
    "expand_ratio": expand_ratio,
    "beam_size": num_beams,
    "k_candidate_num": k_candidate_num,
    "LVLM_backbone": model_name,
    "detector": detector_type,
    "score_type": "BLIP",
    "debugger": debugger,
    "box_threshold": box_threshold,
}

halc_assistant_helper = halc_assistant(
    model,
    vis_processor=vis_processor,
    device=device,
    halc_params=halc_params,
    max_new_tokens=max_new_tokens,
)


classifier_helper = None
if classifier_decoding:
    classifier_helper = ClassifierHelper(model_name, dataset_name, 'chair',
                                         [int(img_files[img_id].split(".jpg")[0][-6:]) for img_id in range(len(img_files))],
                                         os.path.join(data_path, 'annotations'), classifier_hs_layer, classifier_search_topk,
                                         guidance_weight=classifier_guidance_weight, fix_coco_model=classifier_fix_coco_model,
                                         branch_layer=classifier_branch_layer,
                                         use_branches_mlp=classifier_use_branches_mlp,
                                         hallucination_layer=classifier_hallucination_layer,
                                         layers_dim=classifier_layers_dim,
                                         prev_hidden_weight=classifier_prev_hidden_weight,
                                         use_norm_weight=classifier_use_norm_weight,
                                         preprocess_norm=classifier_preprocess_norm,
                                         subspace_align=classifier_subspace_align,
                                         backtracing_number=classifier_backtracing_number,
                                         gradient_mask_rate=classifier_gradient_mask_rate)
    classifier_helper.task = args.classifier_task
    classifier_helper.hs_position = args.classifier_hs_position
    classifier_helper.prediction_threshold = args.classifier_threshold
    if args.classifier_task == "classify-hallucination":
        ckpt_path = args.classifier_ckpt_path
        if os.path.exists(ckpt_path):
            classifier_helper.load_model(model_path=ckpt_path, device=device)
        print(f"\033[42m####### Classifier Loaded from {ckpt_path} #######\033[0m")
        print(f"\033[42m####### Classifier Threshold is set to {args.classifier_threshold} #######\033[0m")
        print(f"\033[42m####### Hidden States Position is set to {args.classifier_hs_position} #######\033[0m")

offlight = True

for idx, img_id in tqdm(enumerate(range(len(img_files))), total=len(img_files)):
    img_file = img_files[img_id]
    if args.split in ['train2014', 'val2014']:
        img_id = int(img_file.split(".jpg")[0][-6:])
    else:
        img_id = int(img_file.split(".jpg")[0])

    # img_info = img_dict[img_id]
    # assert img_info["name"] == img_file
    # img_anns = set(img_info["anns"])
    img_save = {}
    img_save["image_id"] = img_id
    if classifier_decoding:
        classifier_helper.set_image_id(img_id)

    image_path = os.path.join(args.data_path, args.split, img_file)
    raw_image = Image.open(image_path).convert('RGB')

    if model_name == "mplug-owl2":
        max_edge = max(raw_image.size)  # We recommand you to resize to squared image for BEST performance.
        image = raw_image.resize((max_edge, max_edge))
        image_tensor = process_images([image], model.image_processor)
        image = image_tensor.to(device, dtype=torch.float16)
    else:
        image = vis_processors["eval"](raw_image).unsqueeze(0)
        image = image.to(device)

    qu = "Please describe this image in detail."
    # qu = "Please provide a very detailed description of the image."
    # qu = "Please provide a very long and detailed description of the image."
    # qu = "Generate a one sentence caption of the image."
    # qu = "Generate a short caption of the image."

    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)

    lm_early_exit_layers = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
    ]

    mature_layer = lm_early_exit_layers[-1]
    premature_layer = None
    candidate_premature_layers = lm_early_exit_layers[:-1]
    premature_layer_dist = {l: 0 for l in candidate_premature_layers}

    halc_assistant_helper.update_input(img_path=image_path, input_prompt=qu)

    image_cd = None

    if vcd_decoding:
        image_tensor_cd = add_diffusion_noise(image, args.noise_step)
        image_cd = (
            image_tensor_cd.unsqueeze(0).half().to(device)
            if image_tensor_cd is not None
            else None
        )
        cd_alpha = cd_alpha
        cd_beta = cd_beta
        print("image_cd", image_cd.shape)
        print(cd_alpha, cd_beta, args.noise_step)
        if model_name == "minigpt4":
            image_cd = image_cd.squeeze(0)

    with torch.no_grad():
        out = model.generate(
            {"image": norm(image), "prompt": qu, "img_path": image_path},
            use_nucleus_sampling=args.sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            premature_layer=premature_layer,
            candidate_premature_layers=candidate_premature_layers,
            mature_layer=mature_layer,
            beam_search=beam_search,
            dola_decoding=dola_decoding,
            opera_decoding=opera_decoding,
            vcd_decoding=vcd_decoding,
            halc_decoding=halc_decoding,
            # HALC
            halc_assistant=halc_assistant_helper,
            # OPERA
            key_position=None,
            scale_factor=args.scale_factor,
            threshold=args.threshold,
            num_attn_candidates=args.num_attn_candidates,
            penalty_weights=args.penalty_weights,
            # VCD
            images_cd=image_cd,
            cd_alpha=cd_alpha,
            cd_beta=cd_beta,
            # Classifier
            classifier_decoding=classifier_decoding,
            classifier_helper=classifier_helper,
        )

    output_text = out[0]
    print("original output text", output_text)
    sentence_list = output_text.split(".")
    sentence_filter_list = []
    for sentence in sentence_list:
        if "unk" not in sentence:
            sentence_filter_list.append(sentence)
    output_text = ".".join(sentence_filter_list)

    print("decoder output text", output_text)

    if post_correction == "woodpecker":
        decoding_strategy = "woodpecker"
        sample = {
            "img_path": image_path,
            "input_desc": output_text,
            "query": qu,
        }

        corrected_sample = corrector.correct(sample)
        output_text = corrected_sample["output"]
        print("corrected output_text", output_text)

    img_save["caption"] = output_text

    print("image_path: ", image_path)
    print("caption: ", output_text)

    generated_captions_path = os.path.join(
        base_dir,
        "generated_captions.json",
    )
    # print("generated_captions_path", generated_captions_path)
    with open(generated_captions_path, "a") as f:
        json.dump(img_save, f)
        f.write("\n")

if classifier_decoding and args.classifier_task == "collect-hidden-states":
    classifier_helper.save_hallucination_hidden_states(hs_save_path=os.path.join(base_dir, f'hallucination_hidden_states_w_confidence_w_fullscores.pth'))
