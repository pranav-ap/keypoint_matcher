import torch
import torch.nn as nn

from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()

        kernel_size = 3

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation),
        )

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        out = self.model(x)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity

        return out


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            ResidualBlock(8, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
        )

        self.global_pool = nn.MaxPool2d((32, 32))
        self.fc = nn.Linear(256, 2)
        self.sig = nn.Sigmoid()

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

    def forward(self, reference_patches, target_patches, reference_coords):
        # reference_patches & target_patches -> torch.Size([40, 6, 32, 32])
        concatenated_patches = torch.cat((reference_patches, target_patches), dim=1)

        reference_coords = reference_coords / 31.0
        # torch.Size([40, 2]) -> torch.Size([40, 2, 1, 1])
        reference_coords = reference_coords.unsqueeze(2).unsqueeze(3)
        # -> torch.Size([40, 2, 32, 32])
        reference_coords = reference_coords.expand(-1, -1, 32, 32)

        # patches & coords -> torch.Size([40, 8, 32, 32])
        x = torch.cat((concatenated_patches, reference_coords), dim=1)

        # model returns -> torch.Size([80, 256, 32, 32])
        x = self.model(x)

        # after global pooling -> torch.Size([80, 256, 1, 1])
        x = self.global_pool(x)

        # after flatten -> torch.Size([80, 256])
        x = torch.flatten(x, start_dim=1)

        # after FC -> torch.Size([80, 2])
        x = self.fc(x)
        x = self.sig(x)

        x = x * 31.0
        x = x.round()
        x = torch.clamp(x, min=0, max=31)

        return x
