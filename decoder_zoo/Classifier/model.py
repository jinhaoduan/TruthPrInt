
import torch
import torch.nn as nn
# Define the MLP model
from copy import deepcopy


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MultiBranchMLP(nn.Module):
    def __init__(self, input_size, branch_num, branch_layer,
                 layers_dim=[128, 64], activation='relu'):
        super(MultiBranchMLP, self).__init__()
        assert activation == 'relu'
        layers = []
        activation = nn.ReLU
        _layer_dim = deepcopy(layers_dim)
        if _layer_dim[-1] != 1:
            _layer_dim.append(1)
        for i in _layer_dim:
            output_size = i
            layers.extend([nn.Linear(input_size, output_size), activation()])
            input_size = output_size
        layers.pop(-1)

        if branch_layer == 0:
            self.branches = nn.ModuleList([nn.Identity() for _ in range(branch_num)])
        else:
            self.branches = nn.ModuleList([
                nn.Sequential(*deepcopy(layers[:(branch_layer) * 2])) for _ in range(branch_num)
            ])

        if branch_layer == len(layers_dim) + 1:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(*deepcopy(layers[(branch_layer) * 2:]))

        self.subspace_transform = None

    def forward(self, x, branch_idx=0):
        if self.subspace_transform is not None:
            x = torch.matmul(x, self.subspace_transform)
        x = self.branches[branch_idx](x)
        x = self.proj(x)
        return x
