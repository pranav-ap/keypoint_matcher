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



