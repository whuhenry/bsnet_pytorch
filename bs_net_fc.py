import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.linear import Linear

class BSNetFC(nn.Module):
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

        self.recnet = nn.Sequential(
            nn.Linear(input_channels, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_channels), 
            nn.BatchNorm1d(input_channels),
            nn.Sigmoid(),
        )

        self.norm = nn.BatchNorm1d(input_channels)

    def forward(self, x):
        x_norm = self.norm(x)
        weight = self.bam(x_norm)
        x_attentioned = weight * x_norm
        rec_x = self.recnet(x_attentioned)
        return weight, rec_x