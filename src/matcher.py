import torch
from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


class MatcherModel:
    @staticmethod
    def get_descriptors_at_coords(patch_descriptors, patch_level_coords):
        # Shape [80, 2] -> [80] and [80]
        x_coords = patch_level_coords[:, 0]
        y_coords = patch_level_coords[:, 1]

        # Pick the descriptors at the given coordinates
        batch_size, descriptor_length, _, _ = patch_descriptors.shape
        # Shape [80, 64, 32, 32] -> [80, 64]
        descriptors_at_coords = patch_descriptors[torch.arange(batch_size), :, y_coords, x_coords]

        return descriptors_at_coords

    def get_best_target_coords(self, reference_patch_descriptors, target_patch_descriptors, reference_patch_level_coords):
        # Shape [80, 64]
        reference_descriptors_at_coords = self.get_descriptors_at_coords(
            reference_patch_descriptors,
            reference_patch_level_coords
        )

        # Prepare for broadcasting
        # Shape [80, 64, 1, 1]
        reference_descriptors_at_coords = reference_descriptors_at_coords.unsqueeze(-1).unsqueeze(-1)

        # MSE
        # Subtract -> Square -> Mean
        # Shape [80, 64, 32, 32] = [80, 64, 1, 1] - [80, 64, 32, 32]
        squared_diff = (reference_descriptors_at_coords - target_patch_descriptors) ** 2
        # Shape [80, 32, 32]
        mse = squared_diff.mean(dim=1)

        # Shape [80, 1024]
        flat_mse = mse.view(mse.shape[0], -1)
        # Shape [80]
        best_indices = torch.argmin(flat_mse, dim=1)

        # Flattened Index i = y * 32 + x = y * 32 + x
        best_y = best_indices // config.image.patch_size  # Row index (y-coordinate)
        best_x = best_indices % config.image.patch_size  # Column index (x-coordinate)

        # Shape [80, 2]
        best_target_coords = torch.stack([best_y, best_x], dim=1)

        return best_target_coords
