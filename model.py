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
            ResidualBlock(1, 32, dilation=1),
            ResidualBlock(32, 64, dilation=2),
            ResidualBlock(64, 128, dilation=4),
            ResidualBlock(128, embedding_dim, dilation=8),
        )

    def forward(self, patches):
        logger.debug(f"Input shape: {patches.shape}")
        
        batch_size, num_patches, channels, height, width = patches.shape
        # Shape: [2 * 5, 1, 32, 32]
        patches = patches.view(batch_size * num_patches, channels, height, width) 
        logger.debug(f"After Reshape : {patches.shape}")
        
        x = self.encoder[0](patches)
        logger.debug(f"After ResidualBlock 1 : {x.shape}")
        x = self.encoder[1](x)
        logger.debug(f"After ResidualBlock 2 : {x.shape}")
        x = self.encoder[2](x)
        logger.debug(f"After ResidualBlock 3 : {x.shape}")
        x = self.encoder[3](x)
        logger.debug(f"After ResidualBlock 4 : {x.shape}")
        
        # Reshape back to separate batch and patch dimensions
        # Shape: [2, 5, 128, 32, 32]
        # [batch_size, num_patches, embedding_dim, height, width]
        x = x.view(batch_size, num_patches, -1, height, width)  
        logger.debug(f"After Reshape : {x.shape}")
        
        return x


class KeypointMatcherModel:
    @staticmethod
    def flatten_embeddings(embeddings):
        im_count, pat_count, e, h, w = embeddings.shape
        target_flat = embeddings.view(im_count, pat_count, e, -1)
        return target_flat
    
    @staticmethod
    def get_corresponding_descriptors(embeddings, coords):
        im_count, pat_count, _, _, _ = embeddings.shape
        
        rows = coords[..., 0]  # Shape [2, 5]
        cols = coords[..., 1]  # Shape [2, 5]
        
        descriptors = embeddings[
            torch.arange(im_count).view(-1, 1, 1),
            torch.arange(pat_count).view(1, -1, 1),
            :,  # Channel dimension, selects all channels
            rows.unsqueeze(-1),  # Row indices Shape [2, 5, 1]
            cols.unsqueeze(-1)   # Column indices Shape [2, 5, 1]
        ]
        
        descriptors = descriptors.squeeze(2)
    
        return descriptors
        
    def match_keypoint(self, reference_embeddings, target_embeddings, left_coords):
        reference_descriptors = self.get_corresponding_descriptors(reference_embeddings, left_coords)

        ref_expanded = reference_descriptors.unsqueeze(-1)      
        target_flat = self.flatten_embeddings(target_embeddings)
        
        distances = torch.sqrt(torch.sum((ref_expanded - target_flat) ** 2, dim=2)) 
        best_match_indices = torch.argmin(distances, dim=-1)  

        # torch.Size([2, 5, 128, 1024])
        a, b, c, _ = target_flat.size
        batch_indices = torch.arange(a).view(-1, 1, 1) 
        patch_indices = torch.arange(b).view(1, -1, 1)  
        channel_indices = torch.arange(c).view(1, 1, -1)
        
        best_matching_descriptors = target_flat[
            batch_indices,                              
            patch_indices,                              
            channel_indices,                            
            best_match_indices.unsqueeze(-1)            
        ] 

        return best_match_indices, best_matching_descriptors

        # # Calculate the closest row and column from the flattened index
        # height, width = target_embeddings.shape[-2], target_embeddings.shape[-1]
        # closest_row = best_match_indices // width
        # closest_col = best_match_indices % width

        # return (closest_row.item(), closest_col.item())

