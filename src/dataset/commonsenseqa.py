from preprocess import JsonDataset
from generator import DatasetGenerator

class CommonsenseQAGenerator(DatasetGenerator):
    pass

class CommonsenseQA(JsonDataset):
    INSTRUCTION = "Answer the following commonsense question. Choose the most appropriate answer from options A through E."
