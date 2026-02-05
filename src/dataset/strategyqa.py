from preprocess import JsonBasedData
from generator import DatasetGenerator

class StrategyQAGenerator(DatasetGenerator):
    pass

class StrategyQA(JsonBasedData):
    INSTRUCTION = "Answer the following yes/no question. Reason step by step, then provide your final answer as True or False."
