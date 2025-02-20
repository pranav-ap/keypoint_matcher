from config import config 
import math
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw, ImageFont

plt.style.use('seaborn-v0_8-whitegrid')

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

# denormalize = T.Compose([
#     T.Normalize(
#         mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
#     )
# ])

denormalize = T.Compose([
    T.Normalize(mean=[-0.5 / 0.5], std=[1 / 0.5])
])


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


def get_tensor_grid(pil_image):
    return to_tensor(pil_image).unsqueeze(0)


def show_batch(
    reference_patches, target_patches, 
    patch_level_reference_coords, 

    patch_level_target_coords, patch_level_target_coords_true, 
    
    rotations_true=None, rotations=None, 
    confidences_true=None, confidence_pred=None,

    limit_count=None, border_size=2, border_color="white", n_columns=2,
    just_gt=False,
    ):
    assert limit_count is None or limit_count > 0

    num_patches = reference_patches.size(0)
    if limit_count is not None:
        num_patches = min(limit_count, num_patches)
    
    if rotations_true is not None:
        rotations_true = rotations_true.clone().detach()
        # Convert radians to degrees
        rotations_true = rotations_true * (180 / torch.pi) 

    # print(rotations_true[:num_patches])

    if rotations is not None:
        rotations = rotations.clone().detach()
        
        if not just_gt:
            # Convert [-1, 1] to [-pi, pi]
            rotations = rotations * torch.pi 

        # Convert radians to degrees
        rotations = rotations * (180 / torch.pi) 

    try:
        font = ImageFont.truetype("arial.ttf", 14) #  20
    except IOError:
        font = ImageFont.load_default() 

    patch_size = config.image.patch_size # 128  82
    extra_col_gap = 0
    radius = 4

    gap_for_text = 25

    num_rows = (num_patches + n_columns - 1) // n_columns

    combined_width = n_columns * (patch_size * 2 + border_size * 4) + (n_columns - 1) * extra_col_gap
    combined_height = num_rows * (patch_size + border_size * 2 + gap_for_text)

    combined_image = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

    confs = None
    if confidence_pred is not None:
        # confs = F.sigmoid(confidence_pred)
        confs = confidence_pred

    def prepare_patch(patch, x, y, color):
        # patch = denormalize(patch)
        patch = min_max_normalize(patch, min_val=0.0, max_val=1.0) 
        patch = to_pil(patch)

        if patch.mode != "RGB":
            patch = patch.convert("RGB")

        draw_im = ImageDraw.Draw(patch)
        draw_im.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color)
        
        # height = 4
        # draw_im.line([(x, height + 10), (x, height)], fill="red", width=1)

        patch = patch.resize((patch_size, patch_size))
        patch = ImageOps.expand(patch, border=border_size, fill=border_color)
        return patch

    def prepare_target_patch(patch, x, y, a, b, rot=None):
        # patch = denormalize(patch)
        patch = min_max_normalize(patch, min_val=0.0, max_val=1.0)  
        patch = to_pil(patch)

        if patch.mode != "RGB":
            patch = patch.convert("RGB")

        draw_im = ImageDraw.Draw(patch)
        
        if not just_gt:
            draw_im.ellipse((x - radius, y - radius, x + radius, y + radius), outline='yellow')

        draw_im.rectangle((a - radius - 1, b - radius - 1, a + radius + 1, b + radius + 1), outline='green')

        patch = patch.resize((patch_size, patch_size))
        patch = ImageOps.expand(patch, border=border_size, fill=border_color)
        return patch

    for i in range(num_patches):
        reference_patch = reference_patches[i]
        x, y = patch_level_reference_coords[i]
        reference_patch = prepare_patch(reference_patch, x, y, color='red')

        rotation_true = f"{rotations_true[i].item():.2f}°" if rotations_true is not None else '-'
        rotation = f"{rotations[i].item():.2f}°" if rotations is not None else '-'
        conf_true = f"{confidences_true[i].item():.2f}" if confidences_true is not None else '-'
        conf = f"{confs[i].item():.2f}" if confidence_pred is not None else '-'
        
        # if confidence_pred is not None and float(conf) > 0.2:
        #     conf = f'{conf} $$'

        target_patch = target_patches[i]
        x, y = patch_level_target_coords[i]
        a, b = patch_level_target_coords_true[i]
        target_patch = prepare_target_patch(target_patch, x, y, a, b) #, rotations[i].item())

        row = i // n_columns
        col = i % n_columns

        y = row * (patch_size + 2 * border_size + gap_for_text)
        x1 = col * (patch_size * 2 + border_size * 4) + col * extra_col_gap
        x2 = x1 + patch_size + 2 * border_size

        combined_image.paste(reference_patch, (x1, y))
        combined_image.paste(target_patch, (x2, y))

        # Draw rotation value below the patches
        draw = ImageDraw.Draw(combined_image)

        text_x = x1 + patch_size // 2
        text_y = y + patch_size + 2 * border_size + 10  
        # draw.text((text_x, text_y), f"{rotation_true}", fill="black", anchor="mm", font=font)

        text_x = x2 - 5 + patch_size // 2
        text_y = y + patch_size + 2 * border_size + 10  
        
        # draw.text((text_x, text_y), f"{rotation}, {conf}, {conf_true}", fill="black", anchor="mm", font=font)
        draw.text((text_x, text_y), f"P {conf}, T {conf_true}", fill="black", anchor="mm", font=font)
        
    return combined_image
