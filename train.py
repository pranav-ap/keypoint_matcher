import lightning as L
import lightning.pytorch as pl
import torch

from config import config
from src import Light, MatchesDataModule
from utils import logger

torch.set_float32_matmul_precision('medium')


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    light = Light()
    dm = MatchesDataModule()

    # checkpoint_path = ''
    # light = Light.load_from_checkpoint(checkpoint_path)

    trainer = pl.Trainer(
        default_root_dir=config.paths.output,
        logger=L.pytorch.loggers.TensorBoardLogger(save_dir=config.paths.output),
        devices='auto',
        accelerator="auto",
        max_epochs=config.train.max_epochs,
        log_every_n_steps=config.train.log_every_n_steps,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        num_sanity_val_steps=config.train.num_sanity_val_steps,
        enable_model_summary=False,
        fast_dev_run=config.train.fast_dev_run,
        overfit_batches=config.train.overfit_batches,
    )

    trainer.fit(light, datamodule=dm)

    if trainer.checkpoint_callback.best_model_path:
        logger.info(f"Best model path : {trainer.checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
