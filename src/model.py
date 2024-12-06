import torch
import torch.nn as nn

from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = 3 * 2 + 2

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # b, c, h, w
            nn.AdaptiveMaxPool2d((1, 1)),
            # b, c, 1, 1

            nn.Flatten(),
            # b, c

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, reference_patches, target_patches, reference_coords):
        reference_coords = reference_coords / 31.0
        reference_coords = reference_coords.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, reference_patches.size(2), reference_patches.size(3)
        )

        combined = torch.cat([reference_patches, target_patches, reference_coords], dim=1)

        target_coords = self.model(combined)

        return target_coords
