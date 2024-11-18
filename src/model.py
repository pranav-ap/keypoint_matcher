import torch
import torch.nn as nn

from config import config
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
            ResidualBlock(3, 32, dilation=2),
            ResidualBlock(32, 64),
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


class MatcherModel:
    @staticmethod
    def get_descriptors_at_coords(patch_descriptors, patch_level_coords):
        # Shape [80, 2] -> [80] and [80]
        x_coords = patch_level_coords[:, 0]
        y_coords = patch_level_coords[:, 1]

        # Pick the descriptors at the given coordinates
        batch_size, descriptor_length, _, _ = patch_descriptors.shape
        # Shape [80, 64, 32, 32] -> [80, 64]
        descriptors_at_coords = patch_descriptors[torch.arange(batch_size), :, y_coords, x_coords]

        return descriptors_at_coords

    def get_best_target_coords(self, reference_patch_descriptors, target_patch_descriptors, reference_patch_level_coords):
        # Shape [80, 64]
        reference_descriptors_at_coords = self.get_descriptors_at_coords(
            reference_patch_descriptors,
            reference_patch_level_coords
        )

        # Prepare for broadcasting
        # Shape [80, 64, 1, 1]
        reference_descriptors_at_coords = reference_descriptors_at_coords.unsqueeze(-1).unsqueeze(-1)

        # MSE
        # Subtract -> Square -> Mean
        # Shape [80, 64, 32, 32] = [80, 64, 1, 1] - [80, 64, 32, 32]
        squared_diff = (reference_descriptors_at_coords - target_patch_descriptors) ** 2
        # Shape [80, 32, 32]
        mse = squared_diff.mean(dim=1)

        # Shape [80, 1024]
        flat_mse = mse.view(mse.shape[0], -1)
        # Shape [80]
        best_indices = torch.argmin(flat_mse, dim=1)

        # Flattened Index i = y * 32 + x = y * 32 + x
        best_y = best_indices // config.image.patch_size  # Row index (y-coordinate)
        best_x = best_indices % config.image.patch_size  # Column index (x-coordinate)

        # Shape [80, 2]
        best_target_coords = torch.stack([best_y, best_x], dim=1)

        return best_target_coords
