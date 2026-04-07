from preprocess import JsonDataset
from generator import DatasetGenerator

class AIME24Generator(DatasetGenerator):
    pass

class AIME24(JsonDataset):
    INSTRUCTION = "Solve the following math word problem. Show your reasoning step by step, then provide the final numerical answer."
