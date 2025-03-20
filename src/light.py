import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, ModelSummary
from neptune.types import File
from torch.distributions.multivariate_normal import MultivariateNormal

from config import config
from utils import show_batch, get_tensor_grid, logger
from .model import MatcherModel


class Light(pl.LightningModule):
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
    
    def _compute_coords_loss(self, coords_delta_norm_pred, references, targets, estimates=None):
        targets_norm = Light._normalize_coords(targets.float())
        estimates_norm = Light._normalize_coords(estimates.float())

        coords_delta_norm_true = targets_norm - estimates_norm
        loss = F.l1_loss(coords_delta_norm_pred, coords_delta_norm_true)

        return loss

    @staticmethod
    def _compute_confidence_loss(conf_pred, confidences):
        # loss = F.binary_cross_entropy(conf_pred.squeeze(1), confidences)
        loss = F.mse_loss(conf_pred.squeeze(1), confidences)
        return loss

    def _shared_step(self, batch, stage='train'):
        ref_patches, references, tar_patches, targets, confidences, cert, estimates = batch
    
        coords_delta_norm_pred, conf_pred = self.model(
            ref_patches,
            tar_patches,
            references,
            estimates
        ) 

        # Calculate Loss

        coords_delta_pred = coords_delta_norm_pred.float() * ((config.image.train_patch_size - 1) / 2)
        coords_pred = estimates + coords_delta_pred  
        
        coords_loss = self._compute_coords_loss(
            coords_delta_norm_pred, 
            references,
            targets, 
            estimates,
        )

        conf_loss = 0 # 0.2 * self._compute_confidence_loss(conf_pred, confidences)

        loss = coords_loss # + conf_loss

        # Calculate other metrics

        coords_percent_2_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=2)
        coords_percent_15_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.5)
        coords_percent_125_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1.25)
        coords_percent_1_pixel = self._compute_coords_accuracy_percentage(coords_pred, targets, pixels=1)

        coords_mae = self.mae(coords_pred, targets.squeeze(1))

        # Collect

        # clamp for visualization and further usage
        coords_pred = torch.clamp(coords_pred, min=0.0, max=81.0) 
    
        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/coords_loss": coords_loss,
            f"{stage}/conf_loss": conf_loss,

            f"{stage}/coords_mae": coords_mae,

            f"{stage}/coords_percent_2_pixel": coords_percent_2_pixel,
            f"{stage}/coords_percent_15_pixel": coords_percent_15_pixel,
            f"{stage}/coords_percent_125_pixel": coords_percent_125_pixel,
            f"{stage}/coords_percent_1_pixel": coords_percent_1_pixel,
        }

        return metrics, coords_pred, conf_pred

    def training_step(self, batch, batch_idx):
        ref_patches, references, tar_patches, targets, confidences, cert, estimates = batch
        # ref_patches, references, tar_patches, targets, confidences, cert = batch

        metrics, coords_pred, conf_pred = self._shared_step(batch, stage='train')

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, estimates=estimates, limit_count=limit_count, stage='train')

        loss = metrics['train/loss']

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        ref_patches, references, tar_patches, targets, confidences, cert, estimates = batch
        # ref_patches, references, tar_patches, targets, confidences, cert = batch

        metrics, coords_pred, conf_pred = self._shared_step(batch, stage='val')

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, estimates=estimates, limit_count=limit_count, stage='val')
        
        return metrics

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        ref_patches, references, tar_patches, targets, confidences, cert, estimates = batch
        # ref_patches, references, tar_patches, targets, confidences, cert = batch

        metrics, coords_pred, conf_pred = self._shared_step(batch, stage='test')

        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(batch, coords_pred, confidence_pred=conf_pred, estimates=estimates, limit_count=limit_count, stage='val')
        
        return metrics

    def _log_images(self, batch, target_coords, rotation_pred=None, confidence_pred=None, estimates=None, limit_count=None, stage=None):
        ref_patches, references, tar_patches, targets, confidences, cert, estimates = batch
        # ref_patches, references, tar_patches, targets, confidences, cert = batch
        
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) # weight_decay=0.1

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
        return (coords - (config.image.train_patch_size - 1) / 2) / ((config.image.train_patch_size - 1) / 2)

    @staticmethod
    def _denormalize_coords(coords):
        """Denormalizes coordinates from [-1, 1] back to [0, 81]."""
        return coords * ((config.image.train_patch_size - 1) / 2) + (config.image.train_patch_size - 1) / 2

