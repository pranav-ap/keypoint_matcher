import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchmetrics
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar
)
from neptune.types import File
from config import config
from utils import show_batch, logger
from .positional_encoding import RoPENd


class PreActBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = None

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut

        return out


class MatcherModel(nn.Module):
    def __init__(self):
        super().__init__()

        patch_size = config.image.patch_size
        in_channels = 1
        embedding_length = 32
        out_channels = 512

        logger.debug(f'{patch_size=}')
        logger.debug(f'{in_channels=}')
        logger.debug(f'{embedding_length=}')
        logger.debug(f'{out_channels=}')

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embedding_length, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embedding_length),
        )

        self.positional_encoding = RoPENd((patch_size, patch_size, embedding_length))

        self.backbone = nn.ModuleList([
            PreActBasicBlock(embedding_length * 2, 128, 1),
            PreActBasicBlock(128, 256, 1),
            PreActBasicBlock(256, out_channels, 2),
        ])

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Linear(out_channels + 4, 2)

        self._initialize_weights()

    def forward(self, ref_patches, tar_patches, references, estimates):
        _, _, height, width = ref_patches.shape

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

        logger.debug(f'1 {ref_patches.shape=}')

        tar_patches = self.patch_embedding(tar_patches)
        # (b, c, h, w) -> (b, h, w, c)
        tar_patches = tar_patches.permute(0, 2, 3, 1).contiguous()
        tar_patches = self.positional_encoding(tar_patches)
        # (b, h, w, c) -> (b, c, h, w)
        tar_patches = tar_patches.permute(0, 3, 1, 2).contiguous()

        """
        PreActBasicBlock ResNet 
        """

        x = torch.cat([ref_patches, tar_patches], dim=1)
        logger.debug(f'2 {x.shape=}')

        for layer in self.backbone:
            x = layer(x)

        logger.debug(f'3 {x.shape=}')

        x = self.global_pool(x)

        logger.debug(f'4 {x.shape=}')
        x = torch.flatten(x, start_dim=1)
        logger.debug(f'5 {x.shape=}')

        """
        Linear Layer
        """

        x = torch.cat([x, references, estimates], dim=1)
        logger.debug(f'6 {x.shape=}')
        x = self.head(x)
        logger.debug(f'7 {x.shape=}')

        """
        Final Activations
        """

        x = torch.tanh(x[:, :2])
        logger.debug(f'8 {x.shape=}')

        return x

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


class Light_A(pl.LightningModule):
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
    def _compute_coords_accuracy_percentage(coords_pred, target, pixels=1.0):
        difference = torch.abs(coords_pred - target)

        # Check which differences exceed N pixels
        exceeds = (difference > pixels).any(dim=1)
        # Calculate the percentage
        percentage = (exceeds.sum().item() / coords_pred.shape[0]) * 100

        return percentage
    
    @staticmethod
    def _compute_coords_loss(coords_delta_norm_pred, references, targets, estimates):
        targets_norm = Light_A._normalize_coords(targets.float())
        estimates_norm = Light_A._normalize_coords(estimates.float())

        coords_delta_norm_true = targets_norm - estimates_norm
        loss = F.l1_loss(coords_delta_norm_pred, coords_delta_norm_true)

        return loss

    def _shared_step(self, batch, stage='train'):
        ref_patches, references, tar_patches, targets, estimates, confidences = batch

        coords_delta_norm_pred = self.model(
            ref_patches,
            tar_patches,
            references,
            estimates,
        ) 

        # Calculate Loss

        coords_delta_pred = coords_delta_norm_pred.float() * ((config.image.patch_size - 1) / 2)
        coords_pred = estimates + coords_delta_pred  

        conf_pred = torch.ones(len(coords_pred), device=coords_pred.device)  # fake

        coords_loss = self._compute_coords_loss(
            coords_delta_norm_pred, 
            references,
            targets, 
            estimates,
        )

        loss = coords_loss

        # Calculate metrics

        coords_percent_2_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=2.0)
        coords_percent_15_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.5)
        coords_percent_125_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.25)
        coords_percent_1_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.0)

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
