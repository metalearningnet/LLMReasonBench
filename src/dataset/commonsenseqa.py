from preprocess import JsonBasedData
from generator import DatasetGenerator

class CommonsenseQAGenerator(DatasetGenerator):
    pass

class CommonsenseQA(JsonBasedData):
    INSTRUCTION = "Answer the following commonsense question. Choose the most appropriate answer from options A through E."
