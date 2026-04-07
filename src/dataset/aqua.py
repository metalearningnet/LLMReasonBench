from preprocess import JsonDataset
from generator import DatasetGenerator

class AquaGenerator(DatasetGenerator):
    pass

class Aqua(JsonDataset):
    INSTRUCTION = "Answer the following math question. Choose the most appropriate answer from options A through E."
