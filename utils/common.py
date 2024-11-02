import os
import zipfile
import shutil
import torch
from PIL import Image
from itertools import islice


def get_best_device(verbose=False):
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    if verbose:
        print(f"Fastest device found is: {device}")

    return device


def list_files_in_folder(folder_path):
    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)

    return file_paths


def make_clear_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Remove all contents of the directory
        shutil.rmtree(directory_path)

    # Recreate the directory (optional, if you want to keep the directory itself)
    os.makedirs(directory_path, exist_ok=True)

