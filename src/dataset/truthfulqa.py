from preprocess import JsonBasedData
from generator import DatasetGenerator

class TruthfulQAGenerator(DatasetGenerator):
    pass

class TruthfulQA(JsonBasedData):
    INSTRUCTION = "You are given a multiple-choice question. Analyze carefully, then choose the correct answer from the provided options."
    
    def get_instruction(self) -> str:
        return self.INSTRUCTION
