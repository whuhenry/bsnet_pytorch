import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.linear import Linear

class BSNetFC(nn.modules):
    def __init__(self, input_channels) -> None:
        super().__init__()
        self.bam = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, input_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pass