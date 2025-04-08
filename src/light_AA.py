import numpy as np 
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


torch.set_float32_matmul_precision('medium')


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
        embedding_length = 64
        out_channels = 512

        logger.info(f'{patch_size=}')
        logger.info(f'{in_channels=}')
        logger.info(f'{embedding_length=}')
        logger.info(f'{out_channels=}')

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embedding_length, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embedding_length),
        )

        self.positional_encoding = RoPENd((patch_size, patch_size, embedding_length))

        layers = [
            (embedding_length * 2, 256, 1),
            (256, 512, 1),
            (512, out_channels, 2),
        ]

        self.backbone = nn.ModuleList([
            PreActBasicBlock(in_c, out_c, stride) for in_c, out_c, stride in layers
        ])

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(out_channels + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
     
        self._initialize_weights()

    def forward(self, ref_patches, tar_patches, estimates):
        _, _, height, width = ref_patches.shape

        # normalize to 0 to 1
        estimates = estimates / (height - 1)
        # normalize to -1 to 1
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

        x = torch.cat([ref_patches, tar_patches], dim=1)

        for layer in self.backbone:
            x = layer(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        """
        Linear Layers for Separate Heads
        """
        
        x = torch.cat([x, estimates], dim=1)
        x = self.head(x)
        
        coords = F.tanh(x[:, :2]) 
        confs = F.sigmoid(x[:, 2]) 
        
        return coords, confs 

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


class Light(pl.LightningModule):
    def __init__(self, neptune_logger=None, tensorboard_logger=None):
        super().__init__()

        self.neptune_logger = neptune_logger
        self.tensorboard_logger = tensorboard_logger

        self.model = MatcherModel() 

        self.learning_rate = config.train.learning_rate

        self.mae = torchmetrics.MeanAbsoluteError()
        self.conf_accuracy = torchmetrics.Accuracy(task="binary")
        self.conf_precision = torchmetrics.Precision(task="binary")
        self.conf_recall = torchmetrics.Recall(task="binary")
        self.conf_f1 = torchmetrics.F1Score(task="binary")

        self.save_hyperparameters({
            'learning_rate': self.learning_rate,
        },
            ignore=[
                'model',
                'neptune_logger',
                'tensorboard_logger'
            ]
        )

    def forward(self, reference_patches, target_patches, estimates):    
        pred = self.model(
            reference_patches,
            target_patches,
            estimates,
        )

        return pred

    @staticmethod
    def _compute_coords_accuracy_percentage(coords_pred, target, pixels=1.0, mask=None):
        if mask is not None:
            coords_pred = coords_pred[mask]
            target = target[mask]
            
        if coords_pred.numel() == 0:
            return 0.0
        
        difference = torch.abs(coords_pred - target)
        exceeds = (difference > pixels).any(dim=1)
        percentage = (exceeds.sum().item() / coords_pred.shape[0]) * 100
        return percentage

    
    def _shared_step(self, batch, stage='train'):
        ref_patches, references, tar_patches, targets, estimates, confs_true = batch
        # assert torch.isfinite(ref_patches).all(), "Invalid values in ref_patches"
        # assert torch.isfinite(tar_patches).all(), "Invalid values in tar_patches"
        # assert torch.isfinite(references).all(), "Invalid values in references"
        # assert torch.isfinite(estimates).all(), "Invalid values in estimates"
        # assert torch.isfinite(targets).all(), "Invalid values in targets"
        # assert (confs_true >= 0).all() and (confs_true <= 1).all(), "Invalid values in confs_true"

        coords_delta_norm_pred, confs_pred = self.model(
            ref_patches,
            tar_patches,
            estimates,
        ) 

        # assert torch.isfinite(confs_pred).any(), "NaN detected in confs_pred"
        # assert torch.isfinite(coords_delta_norm_pred).any(), "NaN detected in coords_delta_norm_pred"

        coords_delta_pred = coords_delta_norm_pred.float() * ((config.image.patch_size - 1) / 2)
        coords_pred = estimates + coords_delta_pred  
        
        confs_pred_binary = (confs_pred > 0.5).float()
        confs_true_binary = confs_true.float() #.unsqueeze(1)
        
        # Compute losses
        
        targets_norm = Light._normalize_coords(targets.float())
        estimates_norm = Light._normalize_coords(estimates.float())
        coords_delta_norm_true = targets_norm - estimates_norm

        # assert torch.isfinite(targets_norm).any(), "NaN detected in targets_norm"
        # assert torch.isfinite(estimates_norm).any(), "NaN detected in estimates_norm"
        # assert torch.isfinite(coords_delta_norm_true).any(), "NaN detected in coords_delta_norm_true"

        # coords_loss = F.l1_loss(coords_delta_norm_pred, coords_delta_norm_true)
        coords_loss = F.l1_loss(coords_delta_norm_pred, coords_delta_norm_true, reduction='none')
        coords_loss = (coords_loss.sum(dim=1) * confs_true).mean()

        confs_loss = F.binary_cross_entropy(confs_pred, confs_true_binary) 

        # Weighted loss
        alpha = 0.05
        alpha_confs_loss = alpha * confs_loss
        loss = coords_loss + alpha_confs_loss

        # Calculate Coords metrics

        coords_percent_3_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=3.0)
        coords_percent_2_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=2.0)
        coords_percent_15_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.5)
        coords_percent_125_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.25)
        coords_percent_1_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.0)
        coords_percent_05_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=0.5)

        coords_mae = self.mae(coords_pred, targets)

        true_mask = confs_true_binary.squeeze() == 1
        coords_percent_3_pixel_true = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=3.0, mask=true_mask)
        coords_percent_2_pixel_true = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=2.0, mask=true_mask)
        coords_percent_15_pixel_true = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.5, mask=true_mask)
        coords_percent_125_pixel_true = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.25, mask=true_mask)
        coords_percent_1_pixel_true = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.0, mask=true_mask)
        coords_percent_05_pixel_true = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=0.5, mask=true_mask)

        coords_mae_true = None 
        
        if true_mask.any():
            coords_mae_true = self.mae(coords_pred[true_mask], targets[true_mask])
        else:
            coords_mae_true = torch.tensor(0.0, device=coords_pred.device)

        # Calculate Confs metrics

        conf_accuracy = self.conf_accuracy(confs_pred_binary, confs_true_binary)
        conf_precision = self.conf_precision(confs_pred_binary, confs_true_binary)
        conf_recall = self.conf_recall(confs_pred_binary, confs_true_binary)
        conf_f1 = self.conf_f1(confs_pred_binary, confs_true_binary)

        # Collect

        # clamp for visualization and further usage
        coords_pred = torch.clamp(coords_pred, min=0.0, max=float(config.image.patch_size))

        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/coords_loss": coords_loss,
            f"{stage}/conf_loss": confs_loss, 
            f"{stage}/alpha_confs_loss": alpha_confs_loss, 

            f"{stage}/coords_mae": coords_mae,
            f"{stage}/coords_percent_3_pixel": coords_percent_3_pixel,
            f"{stage}/coords_percent_2_pixel": coords_percent_2_pixel,
            f"{stage}/coords_percent_15_pixel": coords_percent_15_pixel,
            f"{stage}/coords_percent_125_pixel": coords_percent_125_pixel,
            f"{stage}/coords_percent_1_pixel": coords_percent_1_pixel,
            f"{stage}/coords_percent_05_pixel": coords_percent_05_pixel,
            
            f"{stage}/coords_mae_true": coords_mae_true,
            f"{stage}/coords_percent_3_pixel_true": coords_percent_3_pixel_true,
            f"{stage}/coords_percent_2_pixel_true": coords_percent_2_pixel_true,
            f"{stage}/coords_percent_15_pixel_true": coords_percent_15_pixel_true,
            f"{stage}/coords_percent_125_pixel_true": coords_percent_125_pixel_true,
            f"{stage}/coords_percent_1_pixel_true": coords_percent_1_pixel_true,
            f"{stage}/coords_percent_05_pixel_true": coords_percent_05_pixel_true,
            
            f"{stage}/conf_accuracy": conf_accuracy,
            f"{stage}/conf_precision": conf_precision,
            f"{stage}/conf_recall": conf_recall,
            f"{stage}/conf_f1": conf_f1,
        }
                        
        return metrics, coords_pred, confs_pred

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
                
        if limit_count is not None:
            mae_per_patch = torch.abs(target_coords - targets.squeeze(1)).mean(dim=1)
            worst_indices = torch.argsort(mae_per_patch, descending=True)[:limit_count]
            
            ref_patches = ref_patches[worst_indices]
            tar_patches = tar_patches[worst_indices]
            references = references[worst_indices]
            targets = targets[worst_indices]
            estimates = estimates[worst_indices]
            target_coords = target_coords[worst_indices]
            confidence_pred = confidence_pred[worst_indices] if confidence_pred is not None else None
            confidences = confidences[worst_indices]
        
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
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.2)
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

        progress_bar_callback = TQDMProgressBar(refresh_rate=50, leave=False) 
        summary_callback = ModelSummary(max_depth=1)

        callbacks = [
            early_stop_callback,
            checkpoint_callback,
            progress_bar_callback,
            summary_callback,
        ]
        
        if len(config.loggers):
            lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
            callbacks.append(lr_monitor_callback)
        
        return callbacks 

    @staticmethod
    def _normalize_coords(coords):
        """Normalizes coordinates from [0, 81] to [-1, 1] in a numerically stable way."""
        return (coords - (config.image.patch_size - 1) / 2) / ((config.image.patch_size - 1) / 2)

    @staticmethod
    def _denormalize_coords(coords):
        """Denormalizes coordinates from [-1, 1] back to [0, 81]."""
        return coords * ((config.image.patch_size - 1) / 2) + (config.image.patch_size - 1) / 2
