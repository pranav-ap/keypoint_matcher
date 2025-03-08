import torch
import torch.nn as nn
import torch.nn.init as init
import math
from utils import logger
from config import config

torch.set_float32_matmul_precision('medium')

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


class BottleneckResNetBlock(nn.Module):
    expansion = 4  # ResNet bottleneck expands channels by 4x

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // self.expansion

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),  # 1x1 Conv (Reduce)
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),  # 3x3 Conv
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),  # 1x1 Conv (Expand)
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else None
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut else x
        return self.relu(self.block(x) + identity)


def depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),

        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


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

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else None
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut else x
        return self.relu(self.block(x) + identity)



class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        patch_size = config.image.patch_size

        in_channels = 6 # 3
        feature_channels = 64

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, feature_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
        )
        
        self.register_buffer(
            'positional_encoding', 
            positionalencoding2d(feature_channels, patch_size, patch_size).unsqueeze(0)
        )

        out_channels = 512  # 512  1024  2048

        self.backbone = nn.Sequential(
            ResNetBlock(feature_channels, 64, 1),

            ResNetBlock(64, 128, 1),
            ResNetBlock(128, 256, 1),

            ResNetBlock(256, 256, 2),
            ResNetBlock(256, 512, 2),
            
            ResNetBlock(512, 512, 2),
            ResNetBlock(512, 512, 2),

            nn.Conv2d(512, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(out_channels, 256), 
            nn.BatchNorm1d(256), 
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 3),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights with Kaiming normal initialization for Conv2d layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming He initialization for Conv2d layers (for ReLU activation)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Xavier (Glorot) initialization for Linear layers
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm layers
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize BatchNorm layers
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, ref_patches, tar_patches, references, estimates=None):
        batch, _, height, width = ref_patches.shape

        references = references.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width) / (height - 1)
        estimates = estimates.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width) / (height - 1)

        x = torch.cat([ref_patches, tar_patches, references, estimates], dim=1)
        x = self.feature_extractor(x) + self.positional_encoding

        x = self.backbone(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.head(x)

        coords = torch.tanh(x[:, :2])
        # coords = x[:, :2]
        confidences = torch.sigmoid(x[:, 2].unsqueeze(-1))

        return coords, confidences

