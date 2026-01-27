from preprocess import JsonBasedData
from generator import DatasetGenerator

class GSM8KGenerator(DatasetGenerator):
    pass

class GSM8K(JsonBasedData):
    INSTRUCTION = "Solve the following math word problem. Show your reasoning step by step, then provide the final numerical answer."
    
    def get_instruction(self) -> str:
        return self.INSTRUCTION
