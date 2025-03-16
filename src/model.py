import torch
import torch.nn as nn
import torch.nn.init as init
import math
from utils import logger
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def posemb_sincos_2d(h, w, dim, temperature: int = 10_000, dtype=torch.float32):
    assert dim % 4 == 0, "feature dimension must be a multiple of 4 for sincos emb"

    y, x = torch.meshgrid(torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype), indexing="ij")
    
    d_half = dim // 2  # Half of the embedding for X, half for Y
    omega = torch.arange(d_half // 2, dtype=dtype) / (d_half // 2 - 1)
    omega = 1.0 / (temperature ** omega)  # Frequency scaling

    x_enc = x[..., None] * omega  # (h, w, d_half/2)
    y_enc = y[..., None] * omega  # (h, w, d_half/2)

    pe_x = torch.cat((x_enc.sin(), x_enc.cos()), dim=-1)  # (h, w, d_half)
    pe_y = torch.cat((y_enc.sin(), y_enc.cos()), dim=-1)  # (h, w, d_half)

    pe = torch.cat((pe_x, pe_y), dim=-1)  # (h, w, dim)
    pe = pe.permute(2, 0, 1)  # (dim, h, w)

    return pe


class PreActBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

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
            )

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut else x
        return self.block(x) + identity


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        patch_size = config.image.patch_size

        patch_channels = 1 
        embed_dims = 512 # 64

        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(patch_channels, embed_dims, 3, 1, 1, bias=False),
        # )
        
        self.register_buffer(
            'positional_embedding', 
            posemb_sincos_2d(h=patch_size, w=patch_size, dim=embed_dims).unsqueeze(0)
        )

        out_channels = 512  # 512 1024 2048
        
        self.backbone = nn.ModuleList([
            PreActBasicBlock(patch_channels * 2, 128, 1),
            PreActBasicBlock(128, 256, 2),
            # PreActBasicBlock(256, 512, 2),
            PreActBasicBlock(256, out_channels, 2),
        ])

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(out_channels + 4, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(),

            nn.Linear(128, 3),
        )

        self._initialize_weights()

    def forward(self, ref_patches, tar_patches, references, estimates=None):
        batch, _, height, width = ref_patches.shape

        # normalize to 0 to 1
        references = references / (height - 1)
        estimates = estimates / (height - 1)
        # normalize to -1 to 1
        references = references * 2 - 1
        estimates = estimates * 2 - 1

        """
        Patch Embedding
        """

        # ref_patches = self.to_patch_embedding(ref_patches) + self.positional_embedding
        # tar_patches = self.to_patch_embedding(tar_patches) + self.positional_embedding

        """
        PreActBasicBlock ResNet Backbone
        """

        x = torch.cat([ref_patches, tar_patches], dim=1)

        logger.debug(f'before backbone {x.shape=}')

        for layer in self.backbone: 
            x = layer(x)
            logger.debug(f'{x.shape=}')
        
        logger.debug(f'backbone done!')

        x = x + self.positional_embedding
        logger.debug(f'{x.shape=}')

        x = self.global_pool(x)
        logger.debug(f'after pooling {x.shape=}')

        x = self.flatten(x)
        logger.debug(f'after flatten {x.shape=}')

        """
        Linear Layers
        """

        x = torch.cat([x, references, estimates], dim=1)

        x = self.head(x)

        """
        Final Activation
        """

        coords = torch.tanh(x[:, :2])
        confidences = torch.sigmoid(x[:, 2].unsqueeze(-1))

        return coords, 
        
    def _initialize_weights(self):
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
