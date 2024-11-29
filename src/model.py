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

        self.feature_extractor = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        )

        self.feature_extractor = list(self.feature_extractor.children())[0]

        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False 

        # for param in self.feature_extractor[-6:].parameters():
        #     param.requires_grad = True

        # self.mapper = nn.Sequential(
        #     nn.Linear(576 * 2 + 2, 2),
        #     nn.Sigmoid(),
        #     # nn.BatchNorm1d(2)
        # )

        self.mapper = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  # Activation function
            nn.Flatten(),  # Flatten the tensor to feed into the Linear layer
            # nn.Linear(128 * feature_map_width * feature_map_height + 2, 2),
            # nn.Sigmoid(),
            # nn.BatchNorm1d(2)
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

        # logger.debug(f'target_coords {target_coords[0]}')

        return target_coords
