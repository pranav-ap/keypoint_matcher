import torchvision.transforms as T
from PIL import Image

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')


def plot_first_image_from_batch(reference_patches, target_patches):
    to_pil = T.ToPILImage()

    num_patches = reference_patches.size(0)
    patch_size = 128

    combined_width = patch_size * 2
    combined_height = num_patches * patch_size

    combined_image = Image.new('RGB', (combined_width, combined_height))

    for j in range(num_patches):
        img1 = reference_patches[j]
        img2 = target_patches[j]

        img1 = to_pil(img1).resize((patch_size, patch_size))
        img2 = to_pil(img2).resize((patch_size, patch_size))

        y = j * patch_size   # Position for the current row
        x1 = 0               # Position for img1 (reference patch)
        x2 = patch_size      # Position for img2 (target patch)

        combined_image.paste(img1, (x1, y))
        combined_image.paste(img2, (x2, y))

    return combined_image
