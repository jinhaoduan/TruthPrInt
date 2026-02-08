
import os
import sys
import random
sys.path.append('/home/jd3734@drexel.edu/workspace/Classifier-Guided-VLM-OH')
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from model import MultiBranchMLP
from scripts.utils import CSVExperimentReader
import fire
from copy import deepcopy
import yaml


def load_hs_result(path, hs_positions: List[int], test_size, random_seed=42, mean=None, std=None):
    hs_results = torch.load(path, map_location=torch.device('cpu'))
    hallu_labels = []
    nonhallu_labels = []
    hallu_hs = []
    nonhallu_hs = []

    for i in range(len(hs_results['hallucinated'])):
        confidence = hs_results['confidence'][i]
        selected_hs = torch.from_numpy(np.array([hs_results['hidden_states'][i][_hs_pos].squeeze(0).squeeze(0) for _hs_pos in hs_positions])).float()  # convert to float otherwise might be `inf` for meaning/summing
        if hs_results['hallucinated'][i]:
            hallu_hs.append(selected_hs)
            hallu_labels.append([1, confidence])
        else:
            nonhallu_hs.append(selected_hs)
            nonhallu_labels.append([0, confidence])

    hs = torch.stack(hallu_hs + nonhallu_hs[:len(hallu_hs)])
    labels = torch.tensor(hallu_labels + nonhallu_labels[:len(hallu_hs)])

    hs_train, hs_test, labels_train, labels_test = train_test_split(hs, labels, test_size=test_size, random_state=random_seed, shuffle=True)

    # Normalize hs_train and hs_test by the mean and std of hs_train
    if mean is None and std is None:
        mean = hs_train.mean(dim=0, keepdim=True)
        std = hs_train.std(dim=0, keepdim=True)

    hs_train = (hs_train - mean)
    hs_test = (hs_test - mean)

    hs_train = hs_train / hs_train.norm(dim=-1, keepdim=True)
    hs_test = hs_test / hs_test.norm(dim=-1, keepdim=True)

    # Convert data to tensors
    hs_train = hs_train.float()
    hs_test = hs_test.float()
    labels_train = labels_train.float()  # .unsqueeze(1)
    labels_test = labels_test.float()  # .unsqueeze(1)
    print(labels_train.shape, labels_test.shape)
    print(hs_train.shape, labels_train.shape, hs_test.shape, labels_test.shape)

    return hs_train, labels_train, hs_test, labels_test, mean, std


def _compute_subspace(data_centered, k):
    """
    Compute the subspace representation using PCA.

    Parameters:
    - data: Tensor of shape (n_samples, n_features)
    - k: Number of principal components to retain

    Returns:
    - subspace: Tensor of shape (n_features, k)
    """
    # Compute covariance matrix
    covariance_matrix = torch.matmul(data_centered.t(), data_centered) / (data_centered.shape[0] - 1)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix.double())

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top k eigenvectors
    subspace = eigenvectors[:, :k].float()

    return subspace


def subspace_alignment(source_data, target_data, k):
    """
    Perform subspace alignment between source and target domains.

    Parameters:
    - source_data: Tensor of shape (n_samples_source, n_features)
    - target_data: Tensor of shape (n_samples_target, n_features)
    - k: Number of principal components to retain

    Returns:
    - source_aligned: Projected source data in the common subspace
    - target_projected: Projected target data in the common subspace
    """
    # Compute subspaces for source and target domains
    S = _compute_subspace(source_data, k)  # Source subspace
    T = _compute_subspace(target_data, k)  # Target subspace

    # Compute the alignment matrix M
    M = torch.matmul(S.t(), T)

    # Align the source subspace
    S_aligned = torch.matmul(S, M)

    # Project data into the common subspace
    source_aligned = torch.matmul(source_data, S_aligned)
    target_projected = torch.matmul(target_data, T)

    return source_aligned, target_projected, S_aligned, T


def eval(model, test_loader, device, train_hs_positions, test_hs_positions, fpr=None):
    # Evaluate the model
    model.eval()
    all_outputs = []
    all_labels = []
    all_confidences = []
    with torch.no_grad():
        for hs_batch, labels_batch in test_loader:
            hs_batch = hs_batch.to(device)
            labels_batch = labels_batch.to(device)
            confidence_batch = labels_batch[:, 1]
            labels_batch = labels_batch[:, 0].reshape(-1, 1)
            hs_batch = hs_batch.transpose(0, 1)
            outputs = []
            labels_batchs = []
            confidence_batchs = []
            for layer_id in test_hs_positions:
                branch_idx = train_hs_positions.index(layer_id)
                _outputs = model(hs_batch[branch_idx], branch_idx)
                outputs.append(_outputs)
                labels_batchs.append(labels_batch)
                confidence_batchs.append(confidence_batch)

            all_outputs.append(outputs)
            all_labels.append(labels_batchs)
            all_confidences.append(confidence_batchs)

    all_outputs = list(zip(*all_outputs))
    all_labels = list(zip(*all_labels))
    all_confidences = list(zip(*all_confidences))

    all_outputs = [torch.cat(_outputs) for _outputs in all_outputs]
    all_labels = [torch.cat(_labels) for _labels in all_labels]
    all_confidences = [torch.cat(_confidences) for _confidences in all_confidences]

    r_dict = {}
    for layer_id in test_hs_positions:
        branch_idx = train_hs_positions.index(layer_id)
        r_dict.update(calc_metrics(all_outputs[branch_idx],
                                   all_labels[branch_idx],
                                   all_confidences[branch_idx],
                                   layer_id,
                                   fpr))
    return r_dict


def calc_metrics(all_outputs, all_labels, all_confidences, layer_id, target_fpr=None):
    alpha = 0
    all_confidences = all_confidences.reshape(-1, 1)
    all_outputs = (1 - alpha) * all_outputs + alpha * (1 - all_confidences)

    # all_labels = torch.cat(all_labels).reshape(-1, 1).cpu().numpy()
    all_outputs = all_outputs
    threshold = torch.quantile(all_outputs, 0.5)
    predictions = (all_outputs > threshold).float()

    accuracy = accuracy_score(all_labels.cpu().numpy(), predictions.float().cpu().numpy())
    auroc = roc_auc_score(all_labels.cpu().numpy(), all_outputs.cpu().numpy())

    print(f'Accuracy_{layer_id}: {accuracy}')
    print(f'AUROC_{layer_id}: {auroc}')
    # Calculate FPR under different thresholds
    thresholds = torch.arange(-1, 10, 0.01)
    fpr_values = []
    threshold_list = []
    tpr_list = []
    fpr_list = []
    for threshold in thresholds:
        predictions = (all_outputs > threshold)
        fn = ((predictions == 0) & (all_labels == 1)).sum().item()
        tp = ((predictions == 1) & (all_labels == 1)).sum().item()
        fp = ((predictions == 1) & (all_labels == 0)).sum().item()
        tn = ((predictions == 0) & (all_labels == 0)).sum().item()
        fpr = fp / (fp + tn) if (fp + tp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fp_d_pos_pred = fp / (tp + fp) if (tp + fp) > 0 else 0
        # print(f'Threshold: {threshold:.2f}, FP/(TP+FP): {fp_d_pos_pred:.4f} TPR: {tpr:.4f}, FPR: {fpr:.4f}, FP: {fp}, TP: {tp}')
        threshold_list.append(threshold)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    desired_fpr_list = [0.01]
    closest_fpr_index_list = []
    desired_tpr_list = []
    tpr_results = {}
    for desired_fpr in desired_fpr_list:
        closest_fpr_index = min(range(len(fpr_list)), key=lambda i: abs(fpr_list[i] - desired_fpr))
        closest_fpr = fpr_list[closest_fpr_index]
        tpr_at_target_fpr = tpr_list[closest_fpr_index]
        print(f'Threshold_{layer_id}: {thresholds[closest_fpr_index]} TPR at FPR={closest_fpr:.4f}: {tpr_at_target_fpr:.4f}')
        closest_fpr_index_list.append(closest_fpr_index)
        desired_tpr_list.append(tpr_at_target_fpr)

        # LR_plus = tpr_at_target_fpr / closest_fpr

        tpr_results.update({
            f'fpr_{desired_fpr}_{layer_id}': closest_fpr,
            f'tpr_at_fpr_{desired_fpr}_{layer_id}': tpr_at_target_fpr,
            f'threshold_fpr_{desired_fpr}_{layer_id}': thresholds[closest_fpr_index],
            # f'lr_plus_{desired_fpr}_{layer_id}': LR_plus,
        })

    result = {
        f'accuracy_{layer_id}': accuracy,
        f'auroc_{layer_id}': auroc,
        f'threshold_list_{layer_id}': threshold_list,
        f'tpr_list_{layer_id}': tpr_list,
        f'fpr_list_{layer_id}': fpr_list,
        **tpr_results,
    }
    if target_fpr is not None:
        closest_fpr_target_index = min(range(len(fpr_list)), key=lambda i: abs(fpr_list[i] - target_fpr))
        tpr_at_target_target_fpr = tpr_list[closest_fpr_target_index]
        print(f'Threshold_{layer_id}: {thresholds[closest_fpr_target_index]} TPR at FPR={fpr_list[closest_fpr_target_index]:.4f}: {tpr_at_target_target_fpr:.4f}')

        result[f'target_fpr_{target_fpr}_{layer_id}'] = thresholds[closest_fpr_target_index]
    return result


def filter_dict(dict):
    dict = deepcopy(dict)
    del_keys = []
    for key in dict.keys():
        if "list" in key:
            del_keys.append(key)

    for key in del_keys:
        del dict[key]

    for key in dict.keys():
        if hasattr(dict[key], 'item'):
            dict[key] = dict[key].item()
        if isinstance(dict[key], (list, tuple)):
            dict[key] = str(dict[key])

    return dict


def train(exp_params, model, train_loader, val_loader, test_loader, ckpt_save_path, device, hallucination_layer, num_epochs, train_hs_positions, mean, std, r0_penelty, align_transform):
    # Initialize the model, loss function, and optimizer
    best_test_acc = 0
    best_train_acc = 0
    best_epoch = 0
    best_tpr_at_fpr_01 = 0
    best_test_result = None
    best_train_result = None

    # Training the model
    criterion_name = 'BCE'
    if criterion_name == 'MSE':
        criterion = nn.MSELoss()
        pass
    elif criterion_name == 'BCE':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    else:
        raise ValueError(f'Unknown criterion: {criterion_name}')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for hs_batch, labels_batch in train_loader:
            # confidence_batch = labels_batch[:, 1]
            labels_batch = labels_batch[:, 0].reshape(-1, 1)
            _label_batch = labels_batch.view(-1)
            labels_batch = labels_batch.repeat(len(train_hs_positions), 1)
            hs_batch = hs_batch.transpose(0, 1)
            hs_batch = hs_batch.to(device)
            labels_batch = labels_batch.to(device)
            outputs = []
            r0_losses = []
            for _hallucination_layer, _ in enumerate(train_hs_positions):
                input = hs_batch[_hallucination_layer]
                if r0_penelty > 0:
                    input = input.clone()
                    input.requires_grad = True
                _outputs = model(input, _hallucination_layer)

                idx_label_0 = _label_batch == 0
                if idx_label_0.any():
                    if r0_penelty > 0:
                        grad = torch.autograd.grad(_outputs.mean(), input, retain_graph=True)[0]
                        grad = grad[idx_label_0]
                        r0_loss = grad.pow(2).sum(dim=1)
                        r0_losses.append(r0_loss)

                outputs.append(_outputs)

            if r0_penelty > 0:
                r0_losses = torch.cat(r0_losses)
                r0_losses = r0_losses.mean()
            else:
                r0_losses = 0
            outputs = torch.cat(outputs)
            loss = criterion(outputs, labels_batch) + r0_penelty * r0_losses
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch {epoch+1}/{num_epochs}')
        print('Testing on train set')
        train_result = eval(model, train_loader, device, train_hs_positions, train_hs_positions)
        # print('Testing on val set')
        # val_result = eval(model, val_loader, device, train_hs_positions, train_hs_positions)
        print('Testing on validation set')
        eval_result = eval(model, val_loader, device, train_hs_positions, train_hs_positions)

        if eval_result[f'tpr_at_fpr_0.01_{hallucination_layer}'] > best_tpr_at_fpr_01:
            best_tpr_at_fpr_01 = eval_result[f'tpr_at_fpr_0.01_{hallucination_layer}']
            best_epoch = epoch
            best_test_result = eval_result
            print('update best model')
            best_model_state_dict = model.state_dict()
            best_threshold_fpr_01 = eval_result[f'threshold_fpr_0.01_{hallucination_layer}']

            _exp_params = deepcopy(exp_params)
            for k in _exp_params:
                if isinstance(_exp_params[k], (list, tuple)):
                    _exp_params[k] = str(_exp_params[k])

            # Save the best model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state_dict,
                'train_result': train_result,
                'test_result': eval_result,
                'mean': mean,
                'std': std,
                'align_transform': align_transform,
                'train_hs_position': train_hs_positions,
                'best_tpr_at_fpr_01': best_tpr_at_fpr_01,
            }, ckpt_save_path)

    best_test_result_fpr = best_test_result[f"threshold_fpr_0.01_{hallucination_layer}"]
    print('ckpt_save_path:', ckpt_save_path)
    print(f'Best TPR@FPR=0.01: {best_tpr_at_fpr_01:.4f}, Best test accuracy: {best_test_acc:.4f}, Best threshold: {best_test_result_fpr:.4f}, Best epoch: {best_epoch}')


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(
    model_name="minigpt4",
    random_seed=42,
    train_hs_positions = [16],
    hs_position = 'previous',
    hallucination_layer = 16,
    model_branch_layer=2,
    layers_dim = [128, 64],
    output_path="/home/kongfei/models_state_dict/halc/search",
    r0_penelty=0,
    subspace_align=32,  # The dimension of the subspace to align to. None means disable subspace alignment. Should be less than the dimension of the hs (4096).
    eval_mode=False,
    eval_fpr=None,
    eval_model_ckpt=None,
    eval_data='coco',
    config_path="/home/kongfei/workspace/Classifier-Guided-VLM-OH/train_config",
):

    fix_seed(random_seed)
    with open(os.path.join(config_path, model_name + '.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    classifier_training_hs_path = config['classifier_training_hs_path']
    classifier_test_hs_path = config['classifier_test_hs_path']

    ckpt_file = f"model_name_{model_name}_seed_{random_seed}_train_hs_positions_{train_hs_positions}_hs_position_{hs_position}_hl_{hallucination_layer}_br_layer_{model_branch_layer}_layers_dim_{layers_dim}_r0_penelty_{r0_penelty}_subspace_align_{subspace_align}.pth"
    train_name = os.path.basename(os.path.dirname(classifier_training_hs_path)) + os.path.basename(classifier_training_hs_path).split('.')[0]
    test_name = os.path.basename(os.path.dirname(classifier_test_hs_path)) + os.path.basename(classifier_test_hs_path).split('.')[0]

    ckpt_file = os.path.join(output_path, train_name, test_name, ckpt_file)
    if not os.path.exists(os.path.dirname(ckpt_file)):
        os.makedirs(os.path.dirname(ckpt_file))

    hs_train, labels_train, hs_val, labels_val, train_mean, train_std = load_hs_result(classifier_training_hs_path,
                                                                                       hs_positions=train_hs_positions, test_size=0.2, random_seed=random_seed)
    coco_hs_test, coco_labels_test, _, _, test_mean, test_std = load_hs_result(classifier_test_hs_path,
                                                                               hs_positions=train_hs_positions, test_size=0.01, random_seed=random_seed)
    if subspace_align and subspace_align > 0:
        assert int(subspace_align) < hs_train.shape[-1] and int(subspace_align) > 0, f"Subspace align k must be less than {hs_train.shape[-1]} and greater than 0"
        # align_transform is the transform matrix for subspace alignment, used by `torch.matmul(hs: [batch_size, hidden_size], align_transform)`
        coco_hs_test, hs_train, align_transform, subspace_T = subspace_alignment(coco_hs_test.reshape(-1, coco_hs_test.shape[-1]),
                                                                                 hs_train.reshape(-1, hs_train.shape[-1]), int(subspace_align))
        coco_hs_test = coco_hs_test.reshape(-1, len(train_hs_positions), coco_hs_test.shape[-1])
        hs_train = hs_train.reshape(-1, len(train_hs_positions), hs_train.shape[-1])
        hs_val = hs_val.reshape(-1, hs_val.shape[-1])
        hs_val = torch.matmul(hs_val, subspace_T)
        hs_val = hs_val.reshape(-1, len(train_hs_positions), hs_val.shape[-1])
    else:
        align_transform = None

    exp_params = {
        'seed': random_seed,
        'train_hs_positions': train_hs_positions,
        'hs_position': hs_position,
        'hallucination_layer': hallucination_layer,
        'model_branch_layer': model_branch_layer,
        'layers_dim': layers_dim,
        'eval_data': eval_data,
        'subspace_align': subspace_align,
    }

    num_epochs = 30
    batch_size = 512

    input_size = hs_train.shape[-1]
    train_loader = DataLoader(TensorDataset(hs_train, labels_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(hs_val, labels_val), batch_size=batch_size, shuffle=False)
    coco_test_loader = DataLoader(TensorDataset(coco_hs_test, coco_labels_test), batch_size=batch_size, shuffle=False)
    device = 'cuda:0'

    model = MultiBranchMLP(input_size, branch_num=len(train_hs_positions), branch_layer=model_branch_layer, layers_dim=layers_dim).to(device)
    if not eval_mode:
        train(exp_params, model,
              train_loader,
              val_loader,
              coco_test_loader,
              ckpt_file,
              device,
              hallucination_layer,
              num_epochs,
              train_hs_positions,
              test_mean,
              test_std,
              r0_penelty,
              align_transform
              )
    else:
        checkpoint = torch.load(eval_model_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        _exp_params = deepcopy(exp_params)
        for k in _exp_params:
            if isinstance(_exp_params[k], (list, tuple)):
                _exp_params[k] = str(_exp_params[k])
        writer = CSVExperimentReader(os.path.dirname(ckpt_file), 'test_result', _exp_params)
        result = eval(model, coco_test_loader, device, train_hs_positions, train_hs_positions, eval_fpr)
        writer.write_result(filter_dict(result))


if __name__ == '__main__':
    fire.Fire(main)
