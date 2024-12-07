import torch
import torch.nn as nn

from config import config
from utils import logger
import math

torch.set_float32_matmul_precision('medium')


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model x height x width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(f"Cannot use sin/cos positional encoding with odd dimension (got dim={d_model})")
    
    pe = torch.zeros(d_model, height, width)
    
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
    

class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        # self.positional_encoding = positionalencoding2d(d_model, height=32, width=32)

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
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, reference_patches, target_patches, reference_coords):
        reference_coords = reference_coords / 31.0
        reference_coords = reference_coords.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, reference_patches.size(2), reference_patches.size(3)
        )

        # positional_encoding = self.positional_encoding[:, :height, :width]
        # reference_coords = reference_coords + positional_encoding

        combined = torch.cat([reference_patches, target_patches, reference_coords], dim=1)

        target_coords = self.model(combined)

        return target_coords
