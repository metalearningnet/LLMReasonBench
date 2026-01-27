# ReasonBench

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Integrate reasoning benchmarks in minutes—not days**

## Overview

ReasonBench makes it easy to evaluate how well Large Language Models separate **memory recall** from **logical reasoning**. Based on the method from *"[Disentangling Memory and Reasoning Ability in Large Language Models](https://github.com/MingyuJ666/Disentangling-Memory-and-Reasoning)"*, it uses customizable special tokens to explicitly isolate these cognitive processes during inference.

## Quick Start

### Installation & Setup

Get started in minutes:

```bash
# Install dependencies
./install.sh

# Generate Chain-of-Thought data (TruthfulQA example)
./run.sh --generate --dataset truthful_qa --mode train

# Train your model
./run.sh --train --dataset truthful_qa

# Evaluate performance
./run.sh --eval --model-path <your_model_path> --dataset truthful_qa
```

## Configuration

### Special Token Configuration

Customize your special tokens and their behavior in `conf/tokens.py`:

```python
# Token Names Configuration
MEMORY_TOKEN_NAME = 'memory'
REASON_TOKEN_NAME = 'reason'
...

# Step Definitions with Detailed Guidelines
STEPS = {
    MEMORY_TOKEN_NAME: {
        'description': "Extract and state ONLY the given facts, numbers, or formulas from the problem statement.",
        'guidelines': [
            "Restate exactly what is provided in the question",
            "Never include calculations, reasoning, or inferences",
            "Separate different facts into individual steps when possible"
        ]
    },
    REASON_TOKEN_NAME: {
        'description': "Perform calculations and logical operations using the facts from memory steps.",
        'guidelines': [
            "Reference facts from memory steps (e.g., 'Using that...')",
            "Show each calculation or logical step clearly",
            "Explain the reasoning process, not just the operation"
        ]
    }
}

...
```

### Dataset Configuration

Configure datasets in `conf/datasets.yaml`. Here are examples for each answer type:

```yaml
# Boolean answer type example:
strategy_qa:
  source: "ChilleD/StrategyQA"
  split_mapping:
    train: "train"
    test: "test"
  answer_type: "boolean"
  field_mapping:
    question: "question"
    answer: "answer"
  clean_latex: false

# Multiple choice answer type example:
commonsense_qa:
  source: "tau/commonsense_qa"
  split_mapping:
    train: "train"
    test: "validation"
  answer_type: "multiple_choice"
  valid_answers: ["A", "B", "C", "D", "E"]
  field_mapping:
    question: "question"
    answer: "answerKey"
    choices: "choices"
  clean_latex: false

# Numeric answer type example:
gsm8k:
  source: "openai/gsm8k"
  config_name: "main"
  split_mapping:
    train: "train"
    test: "test"
  answer_type: "numeric"
  field_mapping:
    question: "question"
    answer: "answer"
  clean_latex: true
```

### Configuration Fields Explained

Each dataset configuration supports the following fields:

- **`source`**: Hugging Face dataset identifier or local path
- **`config_name`**: (Optional) Specific configuration name for datasets with multiple configurations
- **`split_mapping`**: Maps ReasonBench splits (train/test) to dataset splits
- **`answer_type`**: Type of answer expected:
  - `"boolean"`: True/False answers
  - `"numeric"`: Numerical answers
  - `"multiple_choice"`: Multiple choice answers (A, B, C, etc.)
- **`valid_answers`**: (Optional) List of valid answer choices for multiple-choice questions
- **`field_mapping`**: Maps dataset field names to ReasonBench field names:
  - `question`: Question text field
  - `answer`: Answer field
  - `choices`/`options`: (Optional) Multiple choice options field
- **`clean_latex`**: Whether to clean LaTeX formatting (default: `false`)

### Special Token Behavior

- **Memory Steps** (`MEMORY_TOKEN_NAME`): Extract only given facts without any reasoning
- **Reason Steps** (`REASON_TOKEN_NAME`): Perform calculations using facts from memory steps
- **Chain-of-Thought**: Combines both memory and reason tokens for structured reasoning

### Additional Settings

Edit `conf/settings.yaml` to customize:

- **LLM API** endpoints and keys
- **Training** parameters
- **Dataset** selection

### Supported Datasets (Ready to Use)

| Dataset | Config Key | Answer Type | HF Source |
|---------|------------|-------------|-----------|
| GSM8K | `gsm8k` | Numeric | `openai/gsm8k` |
| AIME24 | `aime24` | Numeric | `HuggingFaceH4/aime_2024` |
| AIME25 | `aime25` | Numeric | `math-ai/aime25` |
| AQUA-RAT | `aqua` | Multiple Choice | `deepmind/aqua_rat` |
| MMLU-Pro | `mmlupro` | Multiple Choice | `TIGER-Lab/MMLU-Pro` |
| TruthfulQA | `truthful_qa` | Multiple Choice | `truthfulqa/truthful_qa` |
| StrategyQA | `strategy_qa` | Boolean | `ChilleD/StrategyQA` |
| MetaMathQA | `metamath_qa` | Numeric | `meta-math/MetaMathQA` |
| CommonsenseQA | `commonsense_qa` | Multiple Choice | `tau/commonsense_qa` |

## Adding Custom Datasets

To add your own dataset, you need to complete two steps:

### 1. Add Configuration to `conf/datasets.yaml`

Add a new entry with your dataset configuration:

```yaml
your_dataset:
  source: "your_hf_dataset"
  split_mapping:
    train: "train"
    test: "test"
  answer_type: "multiple_choice" # or "numeric", "boolean"
  field_mapping:
    question: "question_field"
    answer: "answer_field"
```

### 2. Create Dataset Class in `src/dataset/`

Create a Python file for your dataset. In the current implementation, your dataset class must inherit from `JsonBasedData`. For example:

```python
# src/dataset/your_dataset.py
from preprocess import JsonBasedData
from generator import DatasetGenerator

class YourDatasetGenerator(DatasetGenerator):
    pass

class YourDataset(JsonBasedData):
    INSTRUCTION = "Your custom instruction for this dataset."
    
    def get_instruction(self) -> str:
        return self.INSTRUCTION
```

The dataset class should implement the `get_instruction()` method to provide dataset-specific instructions. The `JsonBasedData` base class handles JSON-formatted data loading and preprocessing.

## License

MIT License - see [LICENSE](LICENSE) for details.
