from preprocess import JsonDataset
from generator import DatasetGenerator

class GSM8KGenerator(DatasetGenerator):
    pass

class GSM8K(JsonDataset):
    INSTRUCTION = "Solve the following math word problem. Show your reasoning step by step, then provide the final numerical answer."
