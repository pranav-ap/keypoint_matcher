import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, ModelSummary
from neptune.types import File
from config import config
from utils import show_batch, logger

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils import logger
from config import config
from .positional_encoding import RoPENd, positionalencoding2d


def dual_conv(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


def get_peak_coords(heatmap):
    """Finds the (x, y) coordinates of the peak in the heatmap for each sample in the batch."""
    batch_size = heatmap.shape[0]
    peak_coords = []

    for i in range(batch_size):
        flat_idx = torch.argmax(heatmap[i])  # Get index of max value for the i-th sample
        y, x = divmod(flat_idx.item(), heatmap.shape[-1])  # Convert to 2D coordinates
        peak_coords.append((x, y))

    return torch.tensor(peak_coords)


def generate_gaussian_heatmap1(size, center, sigma=2):
    x = torch.arange(0, size, 1, dtype=torch.float32).view(-1, 1).expand(size, size)
    y = torch.arange(0, size, 1, dtype=torch.float32).view(1, -1).expand(size, size)

    x0, y0 = center
    heatmap = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    return heatmap


def generate_gaussian_heatmap(size, centers, sigma=2):
    batch_size = centers.size(0)
    heatmaps = torch.zeros(batch_size, size, size, dtype=torch.float32)

    # Create a grid of coordinates for the heatmap
    x = torch.arange(0, size, 1, dtype=torch.float32).view(-1, 1).expand(size, size)
    y = torch.arange(0, size, 1, dtype=torch.float32).view(1, -1).expand(size, size)

    for i in range(batch_size):
        x0, y0 = centers[i]

        # Calculate the Gaussian distribution for this center
        heatmap = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Store the heatmap for the current center
        heatmaps[i] = heatmap

    return heatmaps


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.heatmap_version = True #  True  False

        patch_size = config.image.patch_size
        in_channels = 1
        embedding_length = 32

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embedding_length, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embedding_length),
        )

        self.positional_encoding = RoPENd((patch_size, patch_size, embedding_length))

        extra = 0 if self.heatmap_version else 2

        self.dwn_conv2 = dual_conv(64 + 2 + extra, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = dual_conv(1024, 512)
        self.trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = dual_conv(512, 256)
        self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = dual_conv(256, 128)

        self.out = nn.Conv2d(128, 1, kernel_size=1)

        self._initialize_weights()

    def forward(self, ref_patches, tar_patches, references, estimates):
        if self.heatmap_version:
            references = generate_gaussian_heatmap(
                config.image.patch_size,
                centers=references,
                sigma=2,
            )

            estimates = generate_gaussian_heatmap(
                config.image.patch_size,
                centers=estimates,
                sigma=2,
            )

            references = references.unsqueeze(1)
            estimates = estimates.unsqueeze(1)

        else:
            _, _, height, width = ref_patches.shape

            references = references.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
            estimates = estimates.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

            # normalize to 0 to 1
            references = references / (height - 1)
            estimates = estimates / (height - 1)
            # normalize to -1 to 1
            references = references * 2 - 1
            estimates = estimates * 2 - 1

        """
        Patch Embedding
        """

        ref_patches = self.patch_embedding(ref_patches)
        # (b, c, h, w) -> (b, h, w, c)
        ref_patches = ref_patches.permute(0, 2, 3, 1).contiguous()
        ref_patches = self.positional_encoding(ref_patches)
        # (b, h, w, c) -> (b, c, h, w)
        ref_patches = ref_patches.permute(0, 3, 1, 2).contiguous()

        tar_patches = self.patch_embedding(tar_patches)
        # (b, c, h, w) -> (b, h, w, c)
        tar_patches = tar_patches.permute(0, 2, 3, 1).contiguous()
        tar_patches = self.positional_encoding(tar_patches)
        # (b, h, w, c) -> (b, c, h, w)
        tar_patches = tar_patches.permute(0, 3, 1, 2).contiguous()

        """
        PreActBasicBlock ResNet 
        """

        print(f'{ref_patches.shape=}')
        print(f'{references.shape=}')

        x = torch.cat([ref_patches, tar_patches, references, estimates], dim=1)
        print(f'{x.shape=}')

        x3 = self.dwn_conv2(x)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)

        x = self.trans1(x9)
        x = self.up_conv1(torch.cat([x, x7], 1))

        x = self.trans2(x)
        x = self.up_conv2(torch.cat([x, x5], 1))

        x = self.trans3(x)
        x = self.up_conv3(torch.cat([x, x3], 1))

        x = self.out(x)  # Raw heatmap logits
        x = torch.sigmoid(x)  # Convert to [0,1] range for probability

        peak_coords = get_peak_coords(x)

        return x, peak_coords

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming He initialization for Conv2d layers (for ReLU activation)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Xavier (Glorot) initialization for Linear layers
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm layers
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize BatchNorm layers
                init.ones_(m.weight)
                init.zeros_(m.bias)


class Light_UNET(pl.LightningModule):
    def __init__(self, neptune_logger=None, tensorboard_logger=None):
        super().__init__()

        self.neptune_logger = neptune_logger
        self.tensorboard_logger = tensorboard_logger

        self.model = MatcherModel() 

        self.learning_rate = config.train.learning_rate
        self.mae = torchmetrics.MeanAbsoluteError()

        self.save_hyperparameters({
            'learning_rate': self.learning_rate,
        },
            ignore=[
                'model',
                'neptune_logger',
                'tensorboard_logger'
            ]
        )

    def forward(self, reference_patches, target_patches, reference_coords, estimates=None):    
        pred = self.model(
            reference_patches,
            target_patches,
            reference_coords,
            estimates,
        )

        return pred

    @staticmethod
    def _compute_coords_accuracy_percentage(coords_pred, target, pixels=1):
        difference = torch.abs(coords_pred - target)

        # Check which differences exceed N pixels
        exceeds = (difference > pixels).any(dim=1)
        # Calculate the percentage
        percentage = (exceeds.sum().item() / coords_pred.shape[0]) * 100

        return percentage
    
    @staticmethod
    def _compute_coords_loss(coords_pred, references, targets, estimates):
        targets_norm = Light_UNET._normalize_coords(targets.float())
        coords_pred_norm = Light_UNET._normalize_coords(coords_pred.float())

        loss = F.l1_loss(coords_pred_norm, targets_norm)

        return loss

    def _shared_step(self, batch, stage='train'):
        ref_patches, references, tar_patches, targets, estimates, confidences = batch

        predicted_heatmap, coords_pred = self.model(
            ref_patches,
            tar_patches,
            references,
            estimates,
        ) 

        conf_pred = torch.ones(len(coords_pred), device=coords_pred.device)  # fake

        # Calculate Loss

        coords_loss = self._compute_coords_loss(
            coords_pred,
            references,
            targets, 
            estimates,
        )

        heatmap_loss = F.mse_loss(predicted_heatmap, targets)

        loss = coords_loss + heatmap_loss

        # Calculate metrics

        coords_percent_2_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=2)
        coords_percent_15_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.5)
        coords_percent_125_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.25)
        coords_percent_1_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1)

        coords_mae = self.mae(coords_pred, targets.squeeze(1))

        # Collect

        # clamp for visualization and further usage
        coords_pred = torch.clamp(coords_pred, min=0.0, max=float(config.image.patch_size))

        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/coords_loss": coords_loss,
            f"{stage}/conf_loss": coords_loss,  # fake

            f"{stage}/coords_mae": coords_mae,

            f"{stage}/coords_percent_2_pixel": coords_percent_2_pixel,
            f"{stage}/coords_percent_15_pixel": coords_percent_15_pixel,
            f"{stage}/coords_percent_125_pixel": coords_percent_125_pixel,
            f"{stage}/coords_percent_1_pixel": coords_percent_1_pixel,
        }

        return metrics, coords_pred, conf_pred

    def training_step(self, batch, batch_idx):
        metrics, coords_pred, conf_pred = self._shared_step(batch, stage='train')

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='train')

        loss = metrics['train/loss']

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        metrics, coords_pred, conf_pred = self._shared_step(batch, stage='val')

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='val')
        
        return metrics

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # ref_patches, references, tar_patches, targets, estimates, confidences = batch
        metrics, coords_pred, conf_pred = self._shared_step(batch, stage='test')

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, limit_count=limit_count, stage='val')
        
        return metrics

    def _log_images(self, batch, target_coords, confidence_pred=None, limit_count=None, stage=None):
        ref_patches, references, tar_patches, targets, estimates, confidences = batch
        
        image_grid = show_batch(
            ref_patches, tar_patches,
            references, 

            patch_level_target_coords=target_coords, patch_level_target_coords_true=targets,

            confidences_true=confidences,
            confidence_pred=confidence_pred,

            estimates=estimates,

            limit_count=limit_count,
            n_columns=8,
        )

        name = f'{stage}_epoch_{self.current_epoch}.png'

        if self.neptune_logger is not None:
            self.neptune_logger.experiment[f"{stage}/images"].append(
                File.as_image(image_grid),
                step=self.global_step,
                name=name,
            )

    def configure_optimizers(self):        
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.6)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # weight_decay=0.1

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "train/loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.train.patience,
            mode="min",
            verbose=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val/loss',
            mode='min',
            dirpath=config.paths.output.checkpoints,
            filename="best_checkpoint",
            save_top_k=1,
            save_last=True,
        )

        progress_bar_callback = TQDMProgressBar(refresh_rate=5) 
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

        summary_callback = ModelSummary(max_depth=1)

        return [
            early_stop_callback,
            checkpoint_callback,
            progress_bar_callback,
            lr_monitor_callback,
            summary_callback,
        ]

    @staticmethod
    def _normalize_coords(coords):
        """Normalizes coordinates from [0, 81] to [-1, 1] in a numerically stable way."""
        return (coords - (config.image.patch_size - 1) / 2) / ((config.image.patch_size - 1) / 2)

    @staticmethod
    def _denormalize_coords(coords):
        """Denormalizes coordinates from [-1, 1] back to [0, 81]."""
        return coords * ((config.image.patch_size - 1) / 2) + (config.image.patch_size - 1) / 2
