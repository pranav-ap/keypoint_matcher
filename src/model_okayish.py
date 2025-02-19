import torch
import torch.nn as nn
import math
from config import config
from utils import logger 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError(f"Cannot use sin/cos positional encoding with odd dimension (got dim={d_model})")

    pe = torch.zeros(d_model, height, width, requires_grad=False)
    d_model = d_model // 2
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

    pos_w = torch.arange(width).unsqueeze(1)
    pos_h = torch.arange(height).unsqueeze(1)

    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # self.block = depthwise_separable_conv(in_channels, out_channels, stride=stride)
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), 
            
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        if self.shortcut:
            identity = self.shortcut(x)

        out = self.block(x)
        out = out + identity    

        return out


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        patch_size=config.image.patch_size

        in_channels = 1 * 2 + 2
        base_channels = 64

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
 
        self.positional_encoding = positionalencoding2d(base_channels, patch_size, patch_size).unsqueeze(0).to(device)

        out_channels = 512

        self.backbone = nn.Sequential(
            ResNetBlock(base_channels, 128, 1),
            ResNetBlock(256, 512, 1),
            ResNetBlock(512, out_channels, 1),
        )

        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, ref_patches, tgt_patches, ref_coords):
        height, width = ref_patches.shape[2], ref_patches.shape[3]
        ref_coords = ref_coords.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width) / (height - 1)

        x = torch.cat([ref_patches, tgt_patches, ref_coords], dim=1).to(device)
        x = self.feature_extractor(x)
        x = x + self.positional_encoding.clone()

        x = self.backbone(x)
        x = self.global_pool(x)
        x = self.flatten(x)

        out = self.head(x)

        target_coords, target_confidences = out[:, :2], out[:, 2].unsqueeze(-1)
        target_coords, target_confidences = torch.tanh(target_coords), torch.sigmoid(target_confidences)
        
        return target_coords, target_confidences
