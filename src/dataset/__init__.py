from .arc3 import ARC3Generator, ARC3
from .aqua import AquaGenerator, Aqua
from .gsm8k import GSM8KGenerator, GSM8K
from .aime24 import AIME24Generator, AIME24
from .aime25 import AIME25Generator, AIME25
from .mmlupro import MMLUProGenerator, MMLUPro
from .alfworld import AlfworldGenerator, Alfworld
from .strategyqa import StrategyQAGenerator, StrategyQA
from .truthfulqa import TruthfulQAGenerator, TruthfulQA
from .metamathqa import MetaMathQAGenerator, MetaMathQA
from .commonsenseqa import CommonsenseQAGenerator, CommonsenseQA

GENERATOR_MAP = {
    'arc3': ARC3Generator,
    'aqua': AquaGenerator,
    'gsm8k': GSM8KGenerator,
    'aime24': AIME24Generator,
    'aime25': AIME25Generator,
    'mmlupro': MMLUProGenerator,
    'alfworld': AlfworldGenerator,
    'strategyqa': StrategyQAGenerator,
    'truthfulqa': TruthfulQAGenerator,
    'metamathqa': MetaMathQAGenerator,
    'commonsenseqa': CommonsenseQAGenerator
}

DATASET_MAP = {
    'arc3': ARC3,
    'aqua': Aqua,
    'gsm8k': GSM8K,
    'aime24': AIME24,
    'aime25': AIME25,
    'mmlupro': MMLUPro,
    'alfworld': Alfworld,
    'strategyqa': StrategyQA,
    'truthfulqa': TruthfulQA,
    'metamathqa': MetaMathQA,
    'commonsenseqa': CommonsenseQA
}
