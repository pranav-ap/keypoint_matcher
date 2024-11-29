import torch
import torch.nn as nn
import torchvision

from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


def count_params(m):
    total_trainable_params = sum(
        p.numel() for p in m.parameters() if p.requires_grad
    )

    return total_trainable_params


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.Sigmoid()
        )

        self.mapper = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=38 * 5 * 5, out_features=2),
            nn.Sigmoid(),
        )

    def forward(self, reference_patches, target_patches, reference_coords):
        combined = torch.cat([reference_patches, target_patches], dim=1)
        x = self.feature_extractor(combined)

        x = x.view(x.size(0), -1)

        reference_coords = reference_coords / 31.0
        reference_coords = reference_coords.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 5, 5)

        combined = torch.cat([x, reference_coords], dim=1)
        target_coords = self.mapper(combined)

        return target_coords
