import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from config import config
from utils import make_clear_directory, show_batch, get_tensor_grid, logger
from .dataset import Match
from .model import DescriptorModel

torch.set_float32_matmul_precision('medium')


class Light(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = DescriptorModel()

        self.learning_rate = config.train.learning_rate
        self.save_hyperparameters(ignore=['model'])

    def forward(self, patches):
        return self.model(patches)

    @staticmethod
    def compute_loss(reference_descriptors, target_descriptors):
        patch_loss = F.mse_loss(reference_descriptors, target_descriptors)
        return patch_loss

    def training_step(self, batch: Match, batch_idx):
        reference_descriptors = self.model(batch.reference_patches)
        target_descriptors = self.model(batch.target_patches)

        loss = self.compute_loss(reference_descriptors, target_descriptors)
        self.log(f"train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def log_images(self, batch, patch_level_target_coords, limit_count=None):
        image_grid = show_batch(
            batch.reference_patches, batch.target_patches,
            batch.patch_level_reference_coords, patch_level_target_coords,
            limit_count=limit_count,
            n_columns=4,
        )

        image_grid = get_tensor_grid(image_grid)

        # noinspection PyUnresolvedReferences
        self.logger.experiment.add_images(
            "val_image_grid",
            image_grid,
            global_step=self.global_step
        )

    @torch.no_grad()
    def validation_step(self, batch: Match, batch_idx):
        reference_descriptors_pred = self.model(batch.reference_patches)
        target_descriptors_pred = self.model(batch.target_patches)

        loss = self.compute_loss(reference_descriptors_pred, target_descriptors_pred)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        patch_level_target_coords = None

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self.log_images(batch, patch_level_target_coords, limit_count)

        return loss

    @torch.no_grad()
    def test_step(self, batch: Match, batch_idx):
        reference_descriptors = self.model(batch.reference_patches)
        target_descriptors = self.model(batch.target_patches)

        loss = self.compute_loss(reference_descriptors, target_descriptors)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        patch_level_target_coords = None

        if batch_idx == 0:
            self.log_images(batch, patch_level_target_coords)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate
        )

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=2,
                factor=0.5
            ),
            "monitor": "train_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=config.train.patience,
            mode="min",
            verbose=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=f'{config.paths.output}/checkpoints/',
            filename="best_checkpoint",
            save_top_k=1,
            save_last=True,
        )

        progress_bar_callback = TQDMProgressBar(refresh_rate=10)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

        return [checkpoint_callback, early_stop_callback, progress_bar_callback, lr_monitor_callback]


    def compute_match_loss(self, reference_embeddings, target_embeddings, left_coords_pa, right_coords_pa):
        target_coords_pa_pred = self.matcher_model.match_keypoints(reference_embeddings, target_embeddings, left_coords_pa)
        match_loss = F.mse_loss(right_coords_pa.float(), target_coords_pa_pred.float())
        return match_loss


    def training_step(self, batch, batch_idx):
        left_coords_im, right_coords_im, left_coords_pa, right_coords_pa, reference_patches, target_patches = batch

        reference_embeddings = self.descriptor_model(reference_patches)
        target_embeddings = self.descriptor_model(target_patches)

        reference_embeddings = F.normalize(reference_embeddings, p=2, dim=2)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=2)

        loss = self.compute_loss(reference_embeddings, target_embeddings, left_coords_pa, right_coords_pa)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        left_coords_im, right_coords_im, left_coords_pa, right_coords_pa, reference_patches, target_patches = batch

        reference_embeddings = self.descriptor_model(reference_patches)
        target_embeddings = self.descriptor_model(target_patches)

        reference_embeddings = F.normalize(reference_embeddings, p=2, dim=2)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=2)

        loss = self.compute_loss(reference_embeddings, target_embeddings, left_coords_pa, right_coords_pa)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        match_loss = self.compute_match_loss(reference_embeddings, target_embeddings, left_coords_pa, right_coords_pa)
        self.log("val_match_loss", match_loss, prog_bar=True, on_epoch=True, on_step=False)

        return {"val_loss": loss, "val_match_loss": match_loss}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        left_coords_im, right_coords_im, left_coords_pa, right_coords_pa, reference_patches, target_patches = batch

        reference_embeddings = self.descriptor_model(reference_patches)
        target_embeddings = self.descriptor_model(target_patches)

        reference_embeddings = F.normalize(reference_embeddings, p=2, dim=2)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=2)

        loss = self.compute_loss(reference_embeddings, target_embeddings, left_coords_pa, right_coords_pa)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        match_loss = self.compute_match_loss(reference_embeddings, target_embeddings, left_coords_pa, right_coords_pa)
        self.log("test_match_loss", match_loss, prog_bar=True, on_epoch=True, on_step=False)

        return {"test_loss": loss, "test_match_loss": match_loss}

