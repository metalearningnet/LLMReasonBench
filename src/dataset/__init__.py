from .aqua import AquaGenerator, Aqua
from .gsm8k import GSM8KGenerator, GSM8K
from .aime24 import AIME24Generator, AIME24
from .aime25 import AIME25Generator, AIME25
from .mmlupro import MMLUProGenerator, MMLUPro
from .strategyqa import StrategyQAGenerator, StrategyQA
from .truthfulqa import TruthfulQAGenerator, TruthfulQA
from .metamathqa import MetaMathQAGenerator, MetaMathQA
from .commonsenseqa import CommonsenseQAGenerator, CommonsenseQA

__all__ = [
    'AquaGenerator',
    'GSM8KGenerator',
    'AIME24Generator',
    'AIME25Generator',
    'MMLUProGenerator',
    'TruthfulQAGenerator',
    'MetaMathQAGenerator',
    'StrategyQAGenerator',
    'CommonsenseQAGenerator',
    'Aqua',
    'GSM8K',
    'AIME24',
    'AIME25',
    'MMLUPro',
    'TruthfulQA',
    'MetaMathQA',
    'StrategyQA',
    'CommonsenseQA'
]

GENERATOR_MAP = {
    'aqua': AquaGenerator,
    'gsm8k': GSM8KGenerator,
    'aime24': AIME24Generator,
    'aime25': AIME25Generator,
    'mmlupro': MMLUProGenerator,
    'strategy_qa': StrategyQAGenerator,
    'truthful_qa': TruthfulQAGenerator,
    'metamath_qa': MetaMathQAGenerator,
    'commonsense_qa': CommonsenseQAGenerator
}

DATASET_MAP = {
    'aqua': Aqua,
    'gsm8k': GSM8K,
    'aime24': AIME24,
    'aime25': AIME25,
    'mmlupro': MMLUPro,
    'strategy_qa': StrategyQA,
    'truthful_qa': TruthfulQA,
    'metamath_qa': MetaMathQA,
    'commonsense_qa': CommonsenseQA
}
