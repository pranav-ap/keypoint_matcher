import torch
import torch.nn as nn
import torchvision

from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        mobilenet = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2,
        )

        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False 

        for param in self.feature_extractor[-10:].parameters():
            param.requires_grad = True

        self.mapper = nn.Sequential(
            nn.Linear(1280 * 2 + 2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

    def forward(self, reference_patches, target_patches, reference_coords):
        reference_features = self.feature_extractor(reference_patches)
        target_features = self.feature_extractor(target_patches)

        reference_features = reference_features.view(reference_features.size(0), -1)
        target_features = target_features.view(target_features.size(0), -1)

        reference_coords = reference_coords / 31.0

        combined = torch.cat(
            [reference_features, target_features, reference_coords],
            dim=1
        )

        target_coords = self.mapper(combined)

        target_coords = target_coords * 31.0

        # target_coords = target_coords.round()
        # target_coords = torch.clamp(target_coords, min=0, max=31)

        return target_coords
