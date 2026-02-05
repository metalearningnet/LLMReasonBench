from preprocess import JsonBasedData
from generator import DatasetGenerator

class MMLUProGenerator(DatasetGenerator):
    pass

class MMLUPro(JsonBasedData):
    INSTRUCTION = "You are given a multiple-choice question. Analyze carefully, then choose the correct answer from the provided options."
