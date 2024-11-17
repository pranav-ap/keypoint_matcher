import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from config import config
from utils import show_batch, get_tensor_grid, logger
from .dataset import Match
from .model import DescriptorModel, MatcherModel

torch.set_float32_matmul_precision('medium')


class Light(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.descriptor_model = DescriptorModel()
        self.matcher_model = MatcherModel()

        self.learning_rate = config.train.learning_rate
        self.save_hyperparameters(ignore=['model'])

    def forward(self, reference_patches, target_patches, patch_level_reference_coords):
        reference_patch_descriptors_pred = self.descriptor_model(reference_patches)
        target_patch_descriptors_pred = self.descriptor_model(target_patches)

        target_patch_level_coords_pred = self.matcher_model.get_best_target_coords(
            reference_patch_descriptors_pred,
            target_patch_descriptors_pred,
            patch_level_reference_coords
        )

        return target_patch_level_coords_pred

    @staticmethod
    def _compute_match_loss(target_patch_level_coords, target_patch_level_coords_pred):
        match_loss = F.mse_loss(
            target_patch_level_coords.float(),
            target_patch_level_coords_pred.float()
        )

        return match_loss

    @staticmethod
    def _compute_descriptor_loss(reference_patch_descriptors, target_patch_descriptors):
        patch_loss = F.mse_loss(
            reference_patch_descriptors,
            target_patch_descriptors
        )

        return patch_loss

    def training_step(self, batch: Match, batch_idx):
        reference_patch_descriptors_pred = self.descriptor_model(batch.reference_patches)
        target_patch_descriptors_pred = self.descriptor_model(batch.target_patches)

        descriptor_loss = self._compute_descriptor_loss(
            reference_patch_descriptors_pred,
            target_patch_descriptors_pred
        )

        self.log(f"train_loss", descriptor_loss, prog_bar=True, on_epoch=True, on_step=False)

        return descriptor_loss

    @torch.no_grad()
    def validation_step(self, batch: Match, batch_idx):
        reference_patch_descriptors_pred = self.descriptor_model(batch.reference_patches)
        target_patch_descriptors_pred = self.descriptor_model(batch.target_patches)

        descriptor_loss = self._compute_descriptor_loss(
            reference_patch_descriptors_pred,
            target_patch_descriptors_pred
        )

        target_patch_level_coords_pred = self.matcher_model.get_best_target_coords(
            reference_patch_descriptors_pred,
            target_patch_descriptors_pred,
            batch.patch_level_reference_coords
        )

        match_loss = self._compute_match_loss(
            batch.patch_level_target_coords,
            target_patch_level_coords_pred
        )

        self.log_dict({
            "val_loss": descriptor_loss,
            "val_match_loss": match_loss,
        },
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )

        if batch_idx == 0:
            limit_count = config.val.num_patch_pairs_to_save
            self._log_images(
                batch,
                target_patch_level_coords_pred,
                limit_count=limit_count,
                stage='val'
            )

        return {"val_loss": descriptor_loss, "val_match_loss": match_loss}

    @torch.no_grad()
    def test_step(self, batch: Match, batch_idx):
        reference_patch_descriptors_pred = self.descriptor_model(batch.reference_patches)
        target_patch_descriptors_pred = self.descriptor_model(batch.target_patches)

        descriptor_loss = self._compute_descriptor_loss(
            reference_patch_descriptors_pred,
            target_patch_descriptors_pred
        )

        target_patch_level_coords_pred = self.matcher_model.get_best_target_coords(
            reference_patch_descriptors_pred,
            target_patch_descriptors_pred,
            batch.patch_level_reference_coords
        )

        match_loss = self._compute_match_loss(
            batch.patch_level_target_coords,
            target_patch_level_coords_pred
        )

        self.log_dict({
            "test_loss": descriptor_loss,
            "test_match_loss": match_loss,
        },
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )

        if batch_idx == 0:
            self._log_images(
                batch,
                target_patch_level_coords_pred,
                stage='test'
            )

        return {"test_loss": descriptor_loss, "test_match_loss": match_loss}

    def _log_images(self, batch, patch_level_target_coords, limit_count=None, stage=None):
        image_grid = show_batch(
            batch.reference_patches, batch.target_patches,
            batch.patch_level_reference_coords, patch_level_target_coords,
            limit_count=limit_count,
            n_columns=4,
        )

        image_grid = get_tensor_grid(image_grid)

        name = f'{stage}_image_grid' if stage is not None else 'image_grid'

        # noinspection PyUnresolvedReferences
        self.logger.experiment.add_images(
            name,
            image_grid,
            global_step=self.global_step
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.descriptor_model.parameters(),
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
