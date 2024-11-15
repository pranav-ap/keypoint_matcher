import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logger

torch.set_float32_matmul_precision('medium')


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
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
            ResidualBlock(3, 64, dilation=2),
            ResidualBlock(64, 128, dilation=1),
            ResidualBlock(128, embedding_dim, dilation=1),
        )

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

    def forward(self, patches):
        x = self.model(patches)
        return x

    def forward_shapes(self, patches):
        hooks = []

        def print_shape_hook(block_num):
            def shape_hook(module, x, output):
                layer_name = module.__class__.__name__
                logger.debug(f"After {layer_name} {block_num} : {output.shape}")

            return shape_hook

        for i, layer in enumerate(self.model, start=1):
            hook = layer.register_forward_hook(print_shape_hook(i))
            hooks.append(hook)

        logger.debug(f"Input Shape : {patches.shape}")
        x = self.model(patches)
        logger.debug(f"Output Shape : {x.shape}")

        for hook in hooks:
            hook.remove()

        return x

    def forward_intermediates(self, patches):
        save_output = SaveOutput()

        hooks = []

        for i, layer in enumerate(self.model, start=1):
            hook = layer.register_forward_hook(save_output)
            hooks.append(hook)

        x = self.model(patches)

        for hook in hooks:
            hook.remove()

        return x, save_output
