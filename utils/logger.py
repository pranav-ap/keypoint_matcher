import sys
import os
import warnings
import neptune
from loguru import logger
# noinspection PyProtectedMember
from lightning.pytorch.loggers import TensorBoardLogger, NeptuneLogger

from config import config


class MyLogger:
    neptune_logger = None
    tensorboard_logger = None

    @staticmethod
    def init_loggers():
        if MyLogger.neptune_logger is None:
            MyLogger.neptune_logger = MyLogger.get_neptune_logger()

        if MyLogger.tensorboard_logger is None:
            MyLogger.tensorboard_logger = MyLogger.get_tensorboard_logger()

    @staticmethod
    def get_loggers():
        return [MyLogger.neptune_logger, MyLogger.tensorboard_logger]

    @staticmethod
    def init_loguru():
        logger.remove()  # Remove the default handler
        logger.add(
            sys.stdout,
            format="<level>{level: <8}</level> | "
                   "<cyan>{function}</cyan> | "
                   "<level>{message}</level>",
            level=config.log_level,
        )

    @staticmethod
    def init_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="tensorboard")
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    @staticmethod
    def get_neptune_logger():
        logger.info('Setup Neptune Logger')

        # api_token = os.environ.get('NEPTUNE_API_TOKEN')
        project = os.environ.get('NEPTUNE_PROJECT')

        run = neptune.init_run(
            project=project,
            source_files=['src'],
            dependencies=f'{config.paths.roots.project}/environment.yaml',
        )

        neptune_logger = NeptuneLogger(
            run=run,
            log_model_checkpoints=True,
        )

        return neptune_logger

    @staticmethod
    def get_tensorboard_logger():
        logger.info('Setup Tensorboard Logger')

        tensorboard_logger = TensorBoardLogger(
            save_dir=config.paths.output.logs,
        )

        return tensorboard_logger


def setup_logging():
    MyLogger.init_loguru()
    MyLogger.init_warnings()


setup_logging()
