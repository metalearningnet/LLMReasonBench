from preprocess import JsonDataset
from generator import DatasetGenerator

class TruthfulQAGenerator(DatasetGenerator):
    pass

class TruthfulQA(JsonDataset):
    INSTRUCTION = "You are given a multiple-choice question. Analyze carefully, then choose the correct answer from the provided options."
