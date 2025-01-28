import torch
import torch.nn as nn
import torchvision

from config import config
from utils import logger
import math

torch.set_float32_matmul_precision('medium')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model x height x width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(f"Cannot use sin/cos positional encoding with odd dimension (got dim={d_model})")
    
    pe = torch.zeros(d_model, height, width, requires_grad=False)
    
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


def depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.block = depthwise_separable_conv(in_channels, out_channels, stride=stride)

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


class BottleneckResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Bottleneck layer: 1x1 convolution to reduce dimensions
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
        )
        
        # Depthwise separable convolution
        self.block = depthwise_separable_conv(out_channels // 4, out_channels // 4, stride=stride)
        
        # Expansion layer: 1x1 convolution to restore dimensions
        self.expansion = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
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

        out = self.bottleneck(x)
        out = self.block(out)
        out = self.expansion(out)
        out = out + identity    

        return out


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = 3
        in_channels = in_channels * 2 + 2

        out_channels = 32

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.positional_encoding = positionalencoding2d(out_channels, height=32, width=32).unsqueeze(0).to(device)

        in_channels = out_channels
        out_channels = 1024 # 256 512 1024

        self.backbone = nn.Sequential(
            ResNetBlock(in_channels, 128, stride=1),
            ResNetBlock(128, 128, stride=1),
            ResNetBlock(128, 128, stride=2),

            ResNetBlock(128, 256, stride=1),
            ResNetBlock(256, 256, stride=1),
            ResNetBlock(256, 256, stride=2),

            BottleneckResNetBlock(256, 512, stride=1),
            BottleneckResNetBlock(512, out_channels, stride=1),
            BottleneckResNetBlock(out_channels, out_channels, stride=2),
        )

        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.coords_head = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 2),
            nn.Tanh(),
        )

    def forward(self, reference_patches, target_patches, reference_coords):
        height, width = reference_patches.shape[2], reference_patches.shape[3]

        reference_coords = reference_coords.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        reference_coords = reference_coords / 31.0 

        combined = torch.cat([reference_patches, target_patches, reference_coords], dim=1)
        x = self.feature_extractor(combined)
        x = x + self.positional_encoding

        x = self.backbone(x)
        x = self.global_pool(x)
        x = self.flatten(x)

        coords = self.coords_head(x)
        # confidence = self.confidence_head(x)

        return coords #, confidence

