from preprocess import JsonDataset
from generator import DatasetGenerator

class StrategyQAGenerator(DatasetGenerator):
    pass

class StrategyQA(JsonDataset):
    INSTRUCTION = "Answer the following yes/no question. Reason step by step, then provide your final answer as True or False."
