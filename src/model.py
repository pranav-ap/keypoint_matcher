import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logger

torch.set_float32_matmul_precision('medium')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip_connection = nn.Conv2d(in_channels, out_channels,
                                         kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = F.leaky_relu(self.batch_norm1(self.conv1(x)), negative_slope=0.2)
        out = self.batch_norm2(self.conv2(out))

        if self.skip_connection is not None:
            identity = self.skip_connection(identity)

        out += identity  # Residual connection
        out = F.leaky_relu(out, negative_slope=0.2)

        return out


class DescriptorModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            ResidualBlock(1, 64, dilation=1),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, embedding_dim),
        )

    def forward(self, patches):
        logger.debug(f"Input shape: {patches.shape}")
        x = self.encoder[0](patches)
        logger.debug(f"After first ResidualBlock: {x.shape}")
        x = self.encoder[1](x)
        logger.debug(f"After second ResidualBlock: {x.shape}")
        x = self.encoder[2](x)
        logger.debug(f"After third ResidualBlock: {x.shape}")
        x = self.encoder[3](x)
        logger.debug(f"After fourth ResidualBlock: {x.shape}")

        # Output is a dense feature map where each pixel has an embedding vector of size `embedding_dim`
        # Shape: (batch, embedding_dim, H, W), same H, W as the input image
        return x
