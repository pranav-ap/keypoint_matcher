import hydra
from omegaconf import DictConfig, OmegaConf


config: DictConfig | None = None


def setup_config():
    hydra.initialize(version_base=None, config_path=".")
    cfg = hydra.compose("config")

    # print(OmegaConf.to_yaml(cfg))

    global config
    config = cfg


setup_config()
