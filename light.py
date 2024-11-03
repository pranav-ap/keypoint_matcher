from utils import logger, make_clear_directory
from config import config
import numpy as np
import os
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor


torch.set_float32_matmul_precision('medium')


class KeypointDescriptorLightning(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

        self.learning_rate = config.train.learning_rate
        self.save_hyperparameters(ignore=['model'])

        make_clear_directory(config.dirs.output_val_images)
        make_clear_directory(config.dirs.output_test_images)
        
    def forward(self, patches):
        return self.model(patches)

    def compute_loss(self, reference_embeddings, target_embeddings):
        patch_loss = F.mse_loss(reference_embeddings, target_embeddings)
        return patch_loss

    def training_step(self, batch, batch_idx):
        left_coords, right_coords, reference_patches, target_patches = batch

        reference_embeddings = self.model(reference_patches)
        target_embeddings = self.model(target_patches)
        
        loss = self.compute_loss(reference_embeddings, target_embeddings)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        left_coords, right_coords, reference_patches, target_patches = batch

        reference_embeddings = self.model(reference_patches)
        target_embeddings = self.model(target_patches)

        loss = self.compute_loss(reference_embeddings, target_embeddings)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        left_coords, right_coords, reference_patches, target_patches = batch

        reference_embeddings = self.model(reference_patches)
        target_embeddings = self.model(target_patches)

        loss = self.compute_loss(reference_embeddings, target_embeddings)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5),
            "monitor": "train_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=4,
            mode="min",
            verbose=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=f'{config.dirs.output}/checkpoints/',
            filename="best-checkpoint",
            save_top_k=1,
            save_last=True,
        )

        progress_bar_callback = TQDMProgressBar(refresh_rate=10)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

        return [checkpoint_callback, early_stop_callback, progress_bar_callback, lr_monitor_callback]
