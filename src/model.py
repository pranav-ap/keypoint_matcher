import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from utils import logger
from config import config

torch.set_float32_matmul_precision('medium')


def positionalencoding2d(d_model, height, width):
    assert d_model % 4 == 0, f"Cannot use sin/cos positional encoding with odd dimension (got dim={d_model})"

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

    
class PreActBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = None

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        
        identity = self.shortcut(x) if self.shortcut else x
        
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        
        return out + identity


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        patch_size = config.image.patch_size
        in_channels = 1
        embedding_length = 32
        out_channels = 512 

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embedding_length, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embedding_length),
        )

        from src import RoPENd
        self.pe = RoPENd((patch_size, patch_size, embedding_length))

        self.backbone = nn.ModuleList([
            PreActBasicBlock(embedding_length * 2, 128, 1),
            PreActBasicBlock(128, 256, 1),
            PreActBasicBlock(256, 256, 1),

            PreActBasicBlock(256, 512, 2),
            PreActBasicBlock(512, out_channels, 2),
        ])

        self.global_pool = nn.AdaptiveAvgPool2d(1)
       
        self.head = nn.Linear(out_channels + 4, 2)

        self._initialize_weights()

    def forward(self, ref_patches, tar_patches, references, estimates):
        _, _, height, width = ref_patches.shape

        # normalize to 0 to 1
        references = references / (height - 1)
        estimates = estimates / (height - 1)
        # normalize to -1 to 1
        references = references * 2 - 1
        estimates = estimates * 2 - 1

        """
        Patch Embedding
        """
        
        ref_patches = self.to_patch_embedding(ref_patches)
        ref_patches = self.pe(ref_patches)
        tar_patches = self.to_patch_embedding(tar_patches)
        tar_patches = self.pe(tar_patches)

        logger.debug(f'1 {ref_patches.shape=}')

        """
        PreActBasicBlock ResNet 
        """

        x = torch.cat([ref_patches, tar_patches], dim=1)
        logger.debug(f'2 {x.shape=}')
        
        for layer in self.backbone: 
            x = layer(x)
            
        logger.debug(f'3 {x.shape=}')
        
        x = self.global_pool(x)
        
        logger.debug(f'3.5 {x.shape=}')
        x = torch.flatten(x, start_dim=1)  
        logger.debug(f'4 {x.shape=}')
        
        """
        Linear Layer
        """
        
        # logger.debug(f'2 {ref_patches.shape=}')
        
        x = torch.cat([x, references, estimates], dim=1)  
        logger.debug(f'5 {x.shape=}')
        x = self.head(x)
        logger.debug(f'6 {x.shape=}')

        """
        Final Activations
        """

        coords = torch.tanh(x[:, :2])

        return coords

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
                
