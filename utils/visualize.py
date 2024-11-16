import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageDraw

plt.style.use('seaborn-v0_8-whitegrid')

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

denormalize = T.Compose([
    T.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5]),
])


def get_tensor_grid(pil_image):
    return to_tensor(pil_image).unsqueeze(0)


def show_batch(reference_patches, target_patches, patch_level_reference_coords, patch_level_target_coords, limit_count=None, border_size=2, border_color="white", n_columns=2):
    assert limit_count is None or limit_count > 0

    num_patches = reference_patches.size(0)
    if limit_count is not None:
        num_patches = min(limit_count, num_patches)

    patch_size = 128
    extra_col_gap = 0
    radius = 1

    num_rows = (num_patches + n_columns - 1) // n_columns

    combined_width = n_columns * (patch_size * 2 + border_size * 4) + (n_columns - 1) * extra_col_gap
    combined_height = num_rows * (patch_size + border_size * 2)

    combined_image = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

    def prepare_patch(patch, x, y, color):
        patch = denormalize(patch)
        patch = to_pil(patch)

        if patch.mode != "RGB":
            patch = patch.convert("RGB")

        draw_im = ImageDraw.Draw(patch)
        draw_im.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color)
        patch = patch.resize((patch_size, patch_size))
        patch = ImageOps.expand(patch, border=border_size, fill=border_color)
        return patch

    for i in range(num_patches):
        reference_patch = reference_patches[i]
        x, y = patch_level_reference_coords[i]
        reference_patch = prepare_patch(reference_patch, x, y, color='red')

        target_patch = target_patches[i]
        x, y = patch_level_target_coords[i]
        target_patch = prepare_patch(target_patch, x, y, color='green')

        row = i // n_columns
        col = i % n_columns

        y = row * (patch_size + 2 * border_size)
        x1 = col * (patch_size * 2 + border_size * 4) + col * extra_col_gap
        x2 = x1 + patch_size + 2 * border_size

        combined_image.paste(reference_patch, (x1, y))
        combined_image.paste(target_patch, (x2, y))

    return combined_image
