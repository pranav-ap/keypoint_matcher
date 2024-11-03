import torch
import torchvision.transforms as T
from PIL import Image

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
import seaborn as sns
sns.set_theme(style="darkgrid")


def plot_first_image_from_batch(reference_patches, target_patches):
    to_pil = T.ToPILImage()
    
    num_patches = reference_patches.size(1)  # Should be 10 patches
    patch_size = 128  # Resized to 128x128
    
    combined_width = patch_size * 2  # Each pair takes up width of 2 patches
    combined_height = num_patches * patch_size  # Height for all pairs
    
    combined_image = Image.new('RGB', (combined_width, combined_height))
    
    # Loop through each patch pair and paste into the combined image
    for j in range(num_patches):  # 0 to 9 for 10 patches
        # Convert and resize patches
        img1 = to_pil(reference_patches[0, j]).resize((patch_size, patch_size))  # Resize to 128x128
        img2 = to_pil(target_patches[0, j]).resize((patch_size, patch_size))  # Resize to 128x128
    
        # Calculate positions
        y = j * patch_size  # Position for the current row
        x1 = 0               # Position for img1 (reference patch)
        x2 = patch_size      # Position for img2 (target patch)
    
        # Paste images into the combined image
        combined_image.paste(img1, (x1, y))  # Paste reference patch
        combined_image.paste(img2, (x2, y))  # Paste target patch
    
    return combined_image
    