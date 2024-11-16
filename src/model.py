import torch
import torch.nn as nn

from utils import logger

torch.set_float32_matmul_precision('medium')


class ResidualBlock(nn.Module):
    """
    Pre-activation Residual Block
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()

        kernel_size = 3

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation),

            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
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

    def forward_shapes(self, x):
        hooks = []

        def shape_hook(module, x, output):
            layer_name = module.__class__.__name__
            logger.debug(f"After {layer_name} : {output.shape}")

        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                hooks.append(layer.register_forward_hook(shape_hook))

        if self.shortcut is not None:
            hooks.append(self.shortcut.register_forward_hook(shape_hook))

        out = self.forward(x)

        for h in hooks:
            h.remove()

        return out


class DescriptorModel(nn.Module):
    def __init__(self):
        super().__init__()

        embedding_dim = 64

        self.model = nn.Sequential(
            ResidualBlock(3, 64, dilation=2),
            ResidualBlock(64, 128),
            ResidualBlock(128, embedding_dim),
        )

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

    def forward(self, patches):
        x = self.model(patches)
        return x

    def forward_shapes(self, x):
        hooks = []

        def shape_hook(block_num):
            def hook(module, x, output):
                layer_name = module.__class__.__name__
                logger.debug(f"After {layer_name} {block_num} : {output.shape}")

            return hook

        for i, layer in enumerate(self.model, start=1):
            hook = layer.register_forward_hook(shape_hook(i))
            hooks.append(hook)

        logger.debug(f"Input Shape : {x.shape}")

        for idx, block in enumerate(self.model, start=1):
            logger.debug(f"Processing ResidualBlock {idx}")
            block.forward_shapes(x)
            x = block(x)

        logger.debug(f"Output Shape : {x.shape}")

        for h in hooks:
            h.remove()

        return x
