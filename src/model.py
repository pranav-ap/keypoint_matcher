import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logger

torch.set_float32_matmul_precision('medium')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)

        self.skip_connection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        ) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = F.leaky_relu(self.conv1(self.batch_norm(x)), negative_slope=0.2)
        out = self.conv2(out)

        if self.skip_connection is not None:
            identity = self.skip_connection(identity)

        # Residual connection
        out += identity

        return out


class DescriptorModel(nn.Module):
    def __init__(self):
        super().__init__()

        embedding_dim = 64

        self.model = nn.Sequential(
            ResidualBlock(1, 64, dilation=2),
            ResidualBlock(64, 128),
            ResidualBlock(128, embedding_dim),
        )

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

    def forward(self, patches):
        logger.debug(f"Input shape: {patches.shape}")
        x = self.model[0](patches)
        logger.debug(f"After first ResidualBlock: {x.shape}")
        x = self.model[1](x)
        logger.debug(f"After second ResidualBlock: {x.shape}")
        x = self.model[2](x)
        logger.debug(f"After third ResidualBlock: {x.shape}")
        x = self.model[3](x)
        logger.debug(f"After fourth ResidualBlock: {x.shape}")

        # Output is a dense feature map where each pixel has an embedding vector of size `embedding_dim`
        # Shape: (batch, embedding_dim, H, W), same H, W as the input image
        return x
