import os
import sys
import yaml
import copy
import logging
import dataclasses
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar, Union, List

LOG_LEVELS = {
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'CRITICAL': logging.CRITICAL
}

LOG_PRED = True # Enable logging of model predictions during inference

CHOICE_MAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

LLM_API_BASE = 'https://api.openai.com/v1'
LLM_API_MODEL = 'gpt-4o'

INVALID_ANS = "[invalid]"
IGNORE_INDEX = -100

PROJECT_ROOT = Path(__file__).parent.parent
CONF_DIR = PROJECT_ROOT / "conf"

sys.path.append(str(CONF_DIR))
from tokens import *

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / "checkpoint"
DEFAULT_EVAL_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "eval"
DEFAULT_TRAIN_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "train"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

MD_PATH = "targets/md"
MD_SRC = PROJECT_ROOT / MD_PATH / "src"

RESERVED_MODELS = [MD_PATH]

T = TypeVar('T')

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = CONF_DIR / "settings.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_datasets_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = CONF_DIR / "datasets.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_rl_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    from config import CONF_DIR
    
    if config_path is None:
        config_path = CONF_DIR / "rl.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"RL config file not found at {config_path}, using defaults")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}

def update_dataclass_from_config(
    dataclass_obj: T,
    config: Dict[str, Any],
    modes: Union[str, List[str]] = 'common'
) -> T:
    if not dataclasses.is_dataclass(dataclass_obj):
        raise ValueError(f"Expected a dataclass, got {type(dataclass_obj)}")
    
    if isinstance(modes, str):
        modes = [modes]
    
    protected_fields = {'dataset', 'model', 'batch_size', 'num_train', 'num_test'}
    
    for mode in modes:
        if mode in config:
            mode_config = config[mode]
            for key, value in mode_config.items():
                if key == 'lora_config' and isinstance(value, dict):
                    for lora_key, lora_value in value.items():
                        model_arg_key = f"lora_{lora_key}"
                        if hasattr(dataclass_obj, model_arg_key):
                            setattr(dataclass_obj, model_arg_key, lora_value)
                elif hasattr(dataclass_obj, key) and key not in protected_fields:
                    setattr(dataclass_obj, key, value)
    
    return dataclass_obj

def setup_directories(config: Dict[str, Any]):
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)

def get_config_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None
) -> Any:
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

def save_config(config: Dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

config = load_config()
datasets_config = load_datasets_config()
dataset_names = list(datasets_config.keys())
log_level = config.get('logging', {}).get('log_level', 'INFO').upper()
level = LOG_LEVELS.get(log_level, logging.INFO)
logging.basicConfig(
    level=level,
    format='%(message)s'
)

class ExitOnErrorHandler(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            sys.exit(1)

logger = logging.getLogger()
logger.addHandler(ExitOnErrorHandler())
