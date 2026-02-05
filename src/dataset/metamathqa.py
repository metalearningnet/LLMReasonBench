from preprocess import JsonBasedData
from generator import DatasetGenerator

class MetaMathQAGenerator(DatasetGenerator):
    pass

class MetaMathQA(JsonBasedData):
    INSTRUCTION = "Solve the following math word problem. Show your reasoning step by step, then provide the final numerical answer."
