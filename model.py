from utils import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('medium')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = F.leaky_relu(self.batch_norm1(self.conv1(x)), negative_slope=0.2)
        out = self.batch_norm2(self.conv2(out))

        if self.skip_connection is not None:
            identity = self.skip_connection(identity)

        out += identity  # Residual connection
        out = F.leaky_relu(out, negative_slope=0.2)

        return out


class KeypointDescriptorModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            ResidualBlock(10, 32, dilation=1),
            ResidualBlock(32, 64, dilation=2),
            ResidualBlock(64, 128, dilation=4),
            ResidualBlock(128, embedding_dim, dilation=8),
        )

    def forward(self, patches):
        logger.debug(f"Input shape: {patches.shape}")
        x = self.encoder[0](patches)
        logger.debug(f"After ResidualBlock 1 : {x.shape}")
        x = self.encoder[1](x)
        logger.debug(f"After ResidualBlock 2 : {x.shape}")
        x = self.encoder[2](x)
        logger.debug(f"After ResidualBlock 3 : {x.shape}")
        x = self.encoder[3](x)
        logger.debug(f"After ResidualBlock 4 : {x.shape}")
        
        # Output is a dense feature map where each pixel has an embedding vector of size `embedding_dim`
        # Shape: (batch, embedding_dim, H, W), same H, W as the input image
        return x



class KeypointMatcherModel:
    @staticmethod
    def match_keypoint_by_distance(reference_embeddings, target_embeddings, keypoint_coords):
        # Extract the reference keypoint embedding
        ref_keypoint = reference_embeddings[:, keypoint_coords[0], keypoint_coords[1]].view(1, -1)  # Shape (1, C)

        # Shape (C, H*W)
        target_flat = target_embeddings.view(target_embeddings.shape[0], -1)  
    
        # Compute distances between the reference keypoint and all points in the target patch
        distances = F.pairwise_distance(ref_keypoint, target_flat.T)
        min_idx = torch.argmin(distances)
    
        # Convert the flattened index back to 2D coordinates
        height, width = target_embeddings.shape[1:]
        closest_row = min_idx // width
        closest_col = min_idx % width
    
        return (closest_row.item(), closest_col.item())
    
    def match_keypoints_all_batches(reference_embeddings, target_embeddings, keypoint_coords):
        """
        Match a specific keypoint in the reference embeddings to the closest keypoint in the target embeddings
        for all batches, based on distance in the embedding space.
    
        Args:
            reference_embeddings (torch.Tensor): Tensor of shape (N, C, H, W) representing the reference embeddings.
            target_embeddings (torch.Tensor): Tensor of shape (N, C, H, W) representing the target embeddings.
            keypoint_coords (tuple): Coordinates of the keypoint in the reference embeddings (row, col).
    
        Returns:
            list of tuples: Coordinates of the closest matching keypoint in the target embeddings for each batch (row, col).
        """
        batch_size, channels, height, width = reference_embeddings.shape
        closest_matches = []
    
        for i in range(batch_size):
            # Extract the reference keypoint embedding for the current batch
            ref_keypoint = reference_embeddings[i, :, keypoint_coords[0], keypoint_coords[1]].view(1, -1)  # Shape (1, C)
    
            # Flatten the target patch spatial dimensions to compute distance to each location
            target_flat = target_embeddings[i].view(channels, -1)  # Shape (C, H*W)
    
            # Compute distances between the reference keypoint and all points in the target patch
            distances = F.pairwise_distance(ref_keypoint, target_flat.T)
            min_idx = torch.argmin(distances)
    
            # Convert the flattened index back to 2D coordinates
            closest_row = min_idx // width
            closest_col = min_idx % width
    
            closest_matches.append((closest_row.item(), closest_col.item()))
    
        return closest_matches

