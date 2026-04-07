from preprocess import JsonDataset
from generator import DatasetGenerator

class MetaMathQAGenerator(DatasetGenerator):
    pass

class MetaMathQA(JsonDataset):
    INSTRUCTION = "Solve the following math word problem. Show your reasoning step by step, then provide the final numerical answer."
