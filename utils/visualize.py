import torchvision.transforms as T
from PIL import Image, ImageOps

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')


def show_batch(reference_patches, target_patches, limit_count=None, border_size=2, border_color="white"):
    to_pil = T.ToPILImage()

    denormalize = T.Compose([
        T.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5]),
    ])

    num_patches = reference_patches.size(0)
    if limit_count is not None:
        num_patches = min(limit_count, num_patches)

    patch_size = 128

    combined_width = patch_size * 2 + border_size * 4
    combined_height = num_patches * (patch_size + border_size * 2)

    combined_image = Image.new('RGB', (combined_width, combined_height))

    for j in range(num_patches):
        img1 = reference_patches[j]
        img2 = target_patches[j]

        img1 = denormalize(img1)
        img2 = denormalize(img2)

        img1 = to_pil(img1).resize((patch_size, patch_size))
        img2 = to_pil(img2).resize((patch_size, patch_size))

        img1 = ImageOps.expand(img1, border=border_size, fill=border_color)
        img2 = ImageOps.expand(img2, border=border_size, fill=border_color)

        y = j * (patch_size + 2 * border_size)
        x1 = border_size
        x2 = patch_size + 2 * border_size

        combined_image.paste(img1, (x1, y))
        combined_image.paste(img2, (x2, y))

    return combined_image
