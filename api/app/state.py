from .schemas.config import AppConfig

_config = AppConfig()

def get_config() -> AppConfig:
    return _config

def set_config(cfg: AppConfig) -> None:
    global _config
    _config = cfg
