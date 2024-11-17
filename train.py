import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import click
# import mlflow

from config import config
from src import Light, MatchesDataModule
from utils import logger, make_clear_directory

torch.set_float32_matmul_precision('medium')


def handle_clear_logs(clear_opt):
    if clear_opt == 'exp':
        make_clear_directory(f'{config.paths.output.logs}/{config.experiment.name}')
    elif clear_opt == 'all':
        make_clear_directory(config.paths.output.logs)

    make_clear_directory(config.paths.output.val_images)
    make_clear_directory(config.paths.output.test_images)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    light = Light()
    dm = MatchesDataModule()

    tensorboard_logger = TensorBoardLogger(
        save_dir=config.paths.output.logs,
        name=config.experiment.name,
    )

    trainer = pl.Trainer(
        default_root_dir=config.paths.roots.output,
        logger=tensorboard_logger,
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


@click.command()
@click.option(
    '--clear',
    type=click.Choice(['none', 'exp', 'all'], case_sensitive=False),
    default='none',
    help='Clear current experiment logs')
def main(clear):
    handle_clear_logs(clear)
    torch.cuda.empty_cache()

    # mlflow.set_tracking_uri(config.experiment.url)
    # mlflow.pytorch.autolog()

    train()


if __name__ == '__main__':
    main()
