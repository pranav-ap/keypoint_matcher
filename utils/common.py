import os
import shutil

import torch


def get_best_device(verbose=False):
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    if verbose:
        print(f"Fastest device found is: {device}")

    return device


def make_clear_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)

    os.makedirs(directory_path, exist_ok=True)


def count_params(m):
    total_trainable_params = sum(
        p.numel() for p in m.parameters() if p.requires_grad
    )

    return total_trainable_params



def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
    """
    Perform Min-Max Normalization on a tensor.
    Args:
        tensor (torch.Tensor): Input tensor with pixel values.
        min_val (float): Minimum value for normalization (default: 0.0).
        max_val (float): Maximum value for normalization (default: 1.0).
    Returns:
        torch.Tensor: Min-Max normalized tensor.
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    # Scale the tensor to the desired range
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    normalized_tensor = normalized_tensor * (max_val - min_val) + min_val
    return normalized_tensor


