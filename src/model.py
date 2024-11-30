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
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.mapper = nn.Sequential(
            nn.Conv2d(256 + 256 + 2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, reference_patches, target_patches, reference_coords):
        reference_features = self.feature_extractor(reference_patches)
        target_features = self.feature_extractor(target_patches)

        # Normalize to [0, 1]
        reference_coords = reference_coords / 31.0
        reference_coords = reference_coords.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, reference_features.size(2), reference_features.size(3))

        combined = torch.cat([reference_features, target_features, reference_coords], dim=1)

        target_coords = self.mapper(combined)
        target_coords = target_coords.squeeze()

        return target_coords
