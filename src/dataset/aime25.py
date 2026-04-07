from preprocess import JsonDataset
from generator import DatasetGenerator

class AIME25Generator(DatasetGenerator):
    pass

class AIME25(JsonDataset):
    INSTRUCTION = "Solve the following math word problem. Show your reasoning step by step, then provide the final numerical answer."
