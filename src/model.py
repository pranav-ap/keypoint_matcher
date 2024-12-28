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

        self.positional_encoding = positionalencoding2d(128, height=32, width=32).unsqueeze(0).to('cuda')

        in_channels = 3 * 2 + 2

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.model = nn.Sequential(
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
        )

        self.coords_head = nn.Sequential(
            nn.Linear(64, 2),
            nn.Tanh(),
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, 
                reference_patches, 
                target_patches, 
                reference_coords, 
                ):
        reference_coords = reference_coords / 31.0
        height, width = reference_patches.shape[2], reference_patches.shape[3]
        reference_coords = reference_coords.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

        combined = torch.cat([reference_patches, target_patches, reference_coords], dim=1)
        x = self.feature_extractor(combined)
        # x = x.to('cuda')
        x = x + self.positional_encoding.clone()
        x = self.model(x)
        # x = x.to('cuda')

        coords = self.coords_head(x)
        rotation = self.rotation_head(x)
        confidence = self.confidence_head(x)

        return coords, rotation, confidence
