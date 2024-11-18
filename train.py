import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger, NeptuneLogger
import torch
import neptune
import mlflow

from config import config
from src import Light, MatchesDataModule
from utils import logger, make_clear_directory

torch.set_float32_matmul_precision('medium')


def train(ml_logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    light = Light()
    dm = MatchesDataModule()

    trainer = pl.Trainer(
        default_root_dir=config.paths.roots.output,
        logger=ml_logger,
        devices='auto',
        accelerator="auto",
        max_epochs=config.train.max_epochs,
        log_every_n_steps=config.train.log_every_n_steps,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        num_sanity_val_steps=config.train.num_sanity_val_steps,
        fast_dev_run=config.train.fast_dev_run,
        overfit_batches=config.train.overfit_batches,
        enable_model_summary=False,
        enable_checkpointing=True,
    )

    trainer.fit(light, datamodule=dm)

    if trainer.checkpoint_callback.best_model_path:
        logger.info(f"Best model path : {trainer.checkpoint_callback.best_model_path}")


def prep_directories():
    logger.info("Clearing Directories")
    make_clear_directory(config.paths.output.val_images)
    make_clear_directory(config.paths.output.test_images)
    make_clear_directory(config.paths.output.checkpoints)


def get_mlflow_logger(run_id):
    mlflow_logger = MLFlowLogger(
        experiment_name=config.experiment.name,
        tracking_uri=config.experiment.url,
        artifact_location=config.paths.roots.output,
        log_model=False,
        run_id=run_id
    )

    return mlflow_logger


def get_neptune_logger():
    run = neptune.init_run(
        project='neptune-ws/keypoint-matcher',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZjQ4MDM4Yi1kZGIwLTQwMjYtODVhNi0yMjQzNmY1N2M5MGYifQ==',
    )

    neptune_logger = NeptuneLogger(
        run=run,
        # log_model_checkpoints=False,
        # dependencies='environment.yml',
    )

    return neptune_logger


def start_mlflow_version():
    torch.cuda.empty_cache()
    prep_directories()

    mlflow.set_experiment(config.experiment.name)

    with mlflow.start_run() as run:
        ml_logger = get_mlflow_logger(run_id=run.info.run_id)
        train(ml_logger)


def start_tensorboard_version():
    torch.cuda.empty_cache()
    prep_directories()

    tensorboard_logger = TensorBoardLogger(
        save_dir=config.paths.output.logs,
        name=config.experiment.name,
    )

    train(tensorboard_logger)


def start_neptune_version():
    torch.cuda.empty_cache()
    prep_directories()

    neptune_logger = get_neptune_logger()

    train(neptune_logger)


if __name__ == '__main__':
    start_neptune_version()
