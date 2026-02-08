
import torch
import os
# os.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.chair_metrics import chair
from transformers import LlamaTokenizer
import torch.nn as nn
import pickle
from contextlib import contextmanager
from .model import MLP, MultiBranchMLP


class ClassifierHelper:

    def __init__(self, model_name, dataset_name, metric, image_ids, annotation_dir, hidden_state_position,
                 topk=None, use_branches_mlp=False, branch_layer=1,
                 guidance_weight=1, fix_coco_model=False, hallucination_layer=16,
                 layers_dim=[128, 64], prev_hidden_weight=0, use_norm_weight=False, preprocess_norm=False,
                 subspace_align=0, backtracing_number=3, gradient_mask_rate=0.7):
        assert dataset_name in ["coco", "mscoco"], "Only support mscoco."
        assert metric in ["chair"], "Only support chairnow."

        self.dataset_name = dataset_name
        self.tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat', use_fast=False)
        self.metric = metric
        self.image_ids = image_ids
        self.annotation_dir = annotation_dir
        self.layers_dim = layers_dim
        self.subspace_align = subspace_align
        self.gradient_mask_rate = gradient_mask_rate

        self.image_id = None

        self.model = None
        self.position_to_branch = [(i, j) for j, i in enumerate(hidden_state_position)]
        self.position_to_branch = dict(self.position_to_branch)
        self.prediction_threshold = None
        self.task = None
        self.hs_position = None
        self.topk = topk

        self.use_branches_mlp = use_branches_mlp
        self.branches_num = len(hidden_state_position)
        self.branch_layer = branch_layer
        self.model_name = model_name
        self.guidance_weight = guidance_weight
        self.fix_coco_model = fix_coco_model
        self.hallucination_layer = hallucination_layer
        self.prev_hidden_weight = prev_hidden_weight
        self.use_norm_weight = use_norm_weight
        self.backtracing_number = backtracing_number

        self.preprocess_norm = preprocess_norm

        self.registered_hooks = []

        self.hallucination_hidden_states = {
            'image_id': [],
            'hallucinated': [],
            'hidden_states': [],
            'node_words': [],
            'raw_words': [],
            'gt_objects': [],
            'caps': [],
            'confidence': [],
            'current_caption': [],
            'full_caption': [],
            'full_scores': [],
        }

    @ property
    def evaluator(self):
        if not hasattr(self, "_evaluator"):
            if self.metric == "chair":
                if self.fix_coco_model:
                    if os.path.exists("/home/kongfei/models_state_dict/halc/evaluator.pkl"):
                        with open("/home/kongfei/models_state_dict/halc/evaluator.pkl", 'rb') as f:
                            self._evaluator = pickle.load(f)
                    else:
                        self._evaluator = chair.CHAIR(self.image_ids, self.annotation_dir)
                        self._evaluator.get_annotations()
                        with open("/home/kongfei/models_state_dict/halc/evaluator.pkl", 'wb') as f:
                            pickle.dump(self._evaluator, f)
                else:
                    self._evaluator = chair.CHAIR(self.image_ids, self.annotation_dir)
                    self._evaluator.get_annotations()
            else:
                raise NotImplementedError(f"Invalid metric: {self.metric}")
        return self._evaluator

    def caps_to_words(self, caps):
        words, node_words, idxs, raw_words = self.evaluator.caption_to_words(caps)
        return words, node_words, idxs, raw_words

    def append_hidden_states(self, hallucinated, hidden_states, node_words, raw_words, gt_objects, caps, confidence, full_caps, full_scores):
        self.hallucination_hidden_states['image_id'].append(self.image_id)
        self.hallucination_hidden_states['hallucinated'].append(hallucinated)
        self.hallucination_hidden_states['hidden_states'].append(hidden_states)
        self.hallucination_hidden_states['node_words'].append(node_words)
        self.hallucination_hidden_states['raw_words'].append(raw_words)
        self.hallucination_hidden_states['gt_objects'].append(gt_objects)
        self.hallucination_hidden_states['caps'].append(caps)
        self.hallucination_hidden_states['confidence'].append(confidence)
        self.hallucination_hidden_states['current_caption'].append(caps)
        self.hallucination_hidden_states['full_caption'].append(full_caps)
        self.hallucination_hidden_states['full_scores'].append(full_scores)
        # print(caps, node_words, raw_words, gt_objects, hallucinated)

        # print('--------------------------------')
        # print('GT hallucinated: {}'.format(hallucinated))
        # print('node words: {}'.format(node_words))
        # print('gt objects: {}'.format(gt_objects))
        # print('current caps: {}'.format(caps))
        # print('full caps: {}'.format(full_caps))
        # print('VLM confidence: {}'.format(confidence))

    def get_groundtruth_hallucination(self, token_ids, full_token_ids):
        caps = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        full_caps = self.tokenizer.decode(full_token_ids, skip_special_tokens=True)
        node_words, raw_words, gt_objects, hallucinated = self.evaluator.compute_chair_per_cap(caps, self.image_id)

        return caps, node_words, raw_words, gt_objects, hallucinated, full_caps

    def get_pope_groundtruth_hallucination(self, token_ids, full_token_ids):
        caps = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        WORDS = ["No", "not", "no", "NO", "Yes", "yes", "YES"]
        line = caps.replace(".", "")
        line = line.replace(",", "")
        words = line.split(" ")
        node_words = []
        if any(word in WORDS for word in words) or any(
            word.endswith("n't") for word in words
        ):
            node_words = [1]

        return caps, node_words, [], [], [], []

    def set_image_id(self, image_id):
        self.image_id = image_id

    def save_hallucination_hidden_states(self, hs_save_path):
        if not os.path.exists(os.path.dirname(hs_save_path)):
            os.makedirs(os.path.dirname(hs_save_path))
        torch.save(self.hallucination_hidden_states, hs_save_path)

    def load_model(self, model_path, device):
        checkpoint = torch.load(model_path, map_location=device)
        if self.subspace_align > 0:
            input_size = self.subspace_align
        else:
            input_size = 4096
        if not self.use_branches_mlp:
            self.model = MLP(input_size=input_size)
        else:
            self.model = MultiBranchMLP(
                input_size=input_size, branch_num=self.branches_num, branch_layer=self.branch_layer,
                layers_dim=self.layers_dim)
        self.inference_mean = checkpoint['mean']
        self.inference_std = checkpoint['std']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.subspace_align > 0:
            self.model.subspace_transform = checkpoint['align_transform']
        self.model.to(device)
        self.model.eval()
        # self.prediction_threshold = 0.89
        print("load model from {}".format(model_path))

    def _predict_hallucination(self, hidden_states, position=16):
        assert self.prediction_threshold is not None, "prediction_threshold is not set, load model first"
        if self.use_branches_mlp:
            branch_idx = self.position_to_branch[position]
            # center and normalize
            _hidden_states = (hidden_states - self.inference_mean[:, branch_idx].view(1, 1, 1, -1)) / self.inference_std[:, branch_idx].view(1, 1, 1, -1)
            if self.preprocess_norm:
                _hidden_states = _hidden_states / _hidden_states.norm(dim=-1, keepdim=True)
            pred = self.model(_hidden_states, branch_idx)
        else:
            pred = self.model((hidden_states - self.inference_mean) / self.inference_std)

        return (pred > self.prediction_threshold).item(), pred

    def predict_hallucination(self, hidden_states):
        position = self.hallucination_layer
        hidden_states = hidden_states[position].unsqueeze(0)
        if hidden_states.shape[-2] > 1:
            hidden_states = hidden_states[:, :, -1, :]
        with torch.no_grad():
            return self._predict_hallucination(hidden_states, position)

    def hallucination_gradient(self, hidden_state, position=16):
        _hidden_state = hidden_state.unsqueeze(0).clone()
        _hidden_state.requires_grad = True
        with torch.enable_grad():
            _, confidence = self._predict_hallucination(_hidden_state, position)
            grad = torch.autograd.grad(confidence.squeeze(), _hidden_state, retain_graph=False)[0]
        grad = grad.squeeze(0)
        return grad.detach()

    def apply_forward_hook(self, model, prev_hs=None):
        if self.registered_hooks:
            raise ValueError("Hooks already registered")
        layers = self.position_to_branch.keys()

        def forward_hook_factory(position):
            def forward_hook(module, input, output):
                hidden_states = output[0]
                grad = self.hallucination_gradient(hidden_states, position)
                if self.gradient_mask_rate != 0:
                    flatten_grad = grad.view(-1)
                    nelement = flatten_grad.numel()
                    num_top_element = int(nelement * self.gradient_mask_rate)
                    _, top_idx = torch.topk(flatten_grad.abs(), num_top_element)
                    mask = torch.ones_like(flatten_grad)
                    mask[top_idx] = 0
                    mask = mask.view(grad.size())
                    grad = grad * mask

                if not self.use_norm_weight:
                    hidden_states = hidden_states - grad * self.guidance_weight
                else:
                    hidden_states = hidden_states - grad * self.guidance_weight / grad.norm() * hidden_states.norm()
                if self.prev_hidden_weight > 0:
                    hidden_states = self.prev_hidden_weight * prev_hs[position] + (1 - self.prev_hidden_weight) * hidden_states
                output = (hidden_states,) + output[1:]
                return output
            return forward_hook

        if self.model_name == 'minigpt4':
            for i in layers:
                # hidden_state包含input_embedding
                self.registered_hooks.append(model.layers[i - 1].register_forward_hook(forward_hook_factory(i)))

    def remove_forward_hook(self):
        for i in self.registered_hooks:
            i.remove()
        self.registered_hooks.clear()

    @contextmanager
    def gradient_hook_context(self, model, prev_hs=None):
        self.apply_forward_hook(model, prev_hs)
        yield
        self.remove_forward_hook()
