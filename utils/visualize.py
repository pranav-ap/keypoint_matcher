import torch
import torchvision.transforms as T
from PIL import Image

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
import seaborn as sns
sns.set_theme(style="darkgrid")


def visualize_X_samples_grid(dataset, labels, n_samples=12, n_cols=4):
    n_rows = n_samples // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for i, ax in enumerate(axes.flat):
        label = labels[i]
        img = dataset[i]

        if isinstance(img, torch.Tensor):  # Check if it's a tensor
            img = img.detach().numpy()  # Detach from graph if it requires grad

        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_images_from_batch(batch, n_rows=5, col_titles=("Noisy", "Clean")):
    batch_images_1, batch_images_2 = batch
    batch_size = len(batch_images_1)
    n_rows = min(n_rows, batch_size)  # Limit n_rows to batch size if needed

    fig, axes = plt.subplots(n_rows, 2, figsize=(9, n_rows * 3))

    for j in range(n_rows):
        img1 = batch_images_1[j]
        img2 = batch_images_2[j]

        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().numpy()

        # Display Image 1
        axes[j, 0].imshow(img1.squeeze(), cmap='gray')
        axes[j, 0].set_title(f"{col_titles[0]} {j}")
        axes[j, 0].axis('off')

        # Display Image 2
        axes[j, 1].imshow(img2.squeeze(), cmap='gray')
        axes[j, 1].set_title(f"{col_titles[1]} {j}")
        axes[j, 1].axis('off')

    plt.tight_layout()
    plt.show()


def tensor_to_pil_image(tensor):
    tensor = tensor.clone().detach()  # Detach and clone to create a safe copy

    # If the tensor is 4D (batch), remove the batch dimension
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    # Convert the tensor to a format PIL can handle
    if tensor.ndim == 3 and tensor.shape[0] == 1:  # Grayscale image
        tensor = tensor.squeeze(0)  # Remove channel dimension for grayscale

    # Scale the values to [0, 255] for uint8 images
    tensor = tensor.clone().detach()  # Detach from computation graph
    if tensor.min() < 0 or tensor.max() > 1:  # Normalize to [0, 1] if needed
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    tensor = (tensor * 255).byte()  # Convert to uint8

    # Convert to PIL image
    return T.ToPILImage()(tensor)


def create_side_by_side_image(tensor1, tensor2, padding=10):
    image1 = tensor_to_pil_image(tensor1)
    image2 = tensor_to_pil_image(tensor2)

    new_width = image1.width + image2.width + padding
    new_height = max(image1.height, image2.height)
    side_by_side_image = Image.new('L', (new_width, new_height))

    side_by_side_image.paste(image1, (0, 0))  
    side_by_side_image.paste(image2, (image1.width + padding, 0))

    return side_by_side_image

def create_three_image_row(tensor1, tensor2, tensor3, padding=10):
    image1 = tensor_to_pil_image(tensor1)
    image2 = tensor_to_pil_image(tensor2)
    image3 = tensor_to_pil_image(tensor3)

    new_width = image1.width + image2.width + image3.width + 2 * padding
    new_height = max(image1.height, image2.height, image3.height)
    row_image = Image.new('L', (new_width, new_height))

    row_image.paste(image1, (0, 0))
    row_image.paste(image2, (image1.width + padding, 0))
    row_image.paste(image3, (image1.width + image2.width + 2 * padding, 0))

    return row_image

