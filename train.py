from config import config
from utils import logger
import torch
import lightning as L
import lightning.pytorch as pl
from light import KeypointDescriptorLightning
from dataset import MatchesDataModule

torch.set_float32_matmul_precision('medium')


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    lightning_model = KeypointDescriptorLightning()
    dm = MatchesDataModule()

    trainer = pl.Trainer(
        default_root_dir=config.dirs.output,
        logger=L.pytorch.loggers.CSVLogger(save_dir=config.dirs.output),
        devices='auto',
        accelerator="auto",
        max_epochs=config.train.max_epochs,
        log_every_n_steps=config.train.log_every_n_steps,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        num_sanity_val_steps=config.train.num_sanity_val_steps,
        enable_model_summary=False,
        # fast_dev_run=True,
        # overfit_batches=2,
    )
    
    trainer.fit(lightning_model, datamodule=dm)

    if trainer.checkpoint_callback.best_model_path:
        logger.info(f"Best model path : {trainer.checkpoint_callback.best_model_path}")

    trainer.test(lightning_model, datamodule=dm)


if __name__ == '__main__':
    main()
