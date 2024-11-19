import lightning.pytorch as pl
import torch

from config import config
from src import Light, MatchesDataModule
from utils import logger, make_clear_directory, MyLogger

torch.set_float32_matmul_precision('medium')


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    neptune_logger, tensorboard_logger = MyLogger.get_loggers()

    dm = MatchesDataModule()

    checkpoint_path = ''
    light = Light.load_from_checkpoint(
        checkpoint_path,
        neptune_logger=neptune_logger,
        tensorboard_logger=tensorboard_logger,
    )

    neptune_logger.log_model_summary(model=light, max_depth=-1)

    trainer = pl.Trainer(
        default_root_dir=config.paths.roots.output,
        logger=[neptune_logger, tensorboard_logger],
        devices='auto',
        accelerator="auto",
        log_every_n_steps=config.train.log_every_n_steps,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    trainer.test(light, datamodule=dm)


def prep_directories():
    logger.info("Clearing Directories")
    make_clear_directory(config.paths.output.val_images)


def main():
    torch.cuda.empty_cache()
    MyLogger.init_loggers()
    prep_directories()

    train()


if __name__ == '__main__':
    main()
