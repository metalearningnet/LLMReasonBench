from preprocess import JsonBasedData
from generator import DatasetGenerator

class AquaGenerator(DatasetGenerator):
    pass

class Aqua(JsonBasedData):
    INSTRUCTION = "Answer the following math question. Choose the most appropriate answer from options A through E."
