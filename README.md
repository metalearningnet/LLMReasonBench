# ReasonBench

Benchmark and enhance reasoning capabilities in large language models (LLMs)

## Overview

ReasonBench is an evaluation and training framework designed to measure and enhance how well LLMs separate **memory recall** from **logical reasoning**. Based on the methodology from *"[Disentangling Memory and Reasoning Ability in Large Language Models](https://github.com/MingyuJ666/Disentangling-Memory-and-Reasoning)"*, this tool uses customizable special tokens to explicitly isolate these cognitive processes during inference.

---

## 🚀 Quick Start

### Installation & Basic Usage

Follow these steps to set up the environment, generate data, and run your first training and evaluation pipeline:

```bash
# 1. Install dependencies
./install.sh

# 2. Generate Chain-of-Thought data (TruthfulQA example)
./run.sh --generate --dataset truthfulqa --mode train

# 3. Train your model with standard fine-tuning
./run.sh --train --dataset truthfulqa

# 4. Evaluate performance
./run.sh --eval --model <your_model_path> --dataset truthfulqa

```

### Reinforcement Learning (RL) Training

ReasonBench supports advanced RL fine-tuning methods, including DPO, CPO, KTO, and ORPO. Configuration for DPO training is located in `conf/dpo.yaml`.

```bash
# Train with DPO using a specific model
./run.sh --train --rl dpo --dataset truthfulqa --model Qwen/Qwen3.5-9B

```

---

## 📊 Model Evaluation

The evaluation script is designed to automatically detect and handle various model types—including standard fine-tuned models, RL-trained models, and custom PEFT (Parameter-Efficient Fine-Tuning) adapters.

```bash
# Evaluate any model with one command
./run.sh --eval --model <model_path_or_checkpoint> --dataset <dataset_name>

```

---

## 🗂️ Dataset Management

### Supported Datasets

The following datasets are pre-configured and ready for immediate use:

| Dataset | Config Key | Answer Type | Hugging Face Source |
| --- | --- | --- | --- |
| **GSM8K** | `gsm8k` | Numeric | `openai/gsm8k` |
| **AIME24** | `aime24` | Numeric | `HuggingFaceH4/aime_2024` |
| **AIME25** | `aime25` | Numeric | `math-ai/aime25` |
| **AQUA-RAT** | `aqua` | Multiple Choice | `deepmind/aqua_rat` |
| **MMLU-Pro** | `mmlupro` | Multiple Choice | `TIGER-Lab/MMLU-Pro` |
| **TruthfulQA** | `truthfulqa` | Multiple Choice | `truthfulqa/truthful_qa` |
| **StrategyQA** | `strategyqa` | Boolean | `ChilleD/StrategyQA` |
| **MetaMathQA** | `metamathqa` | Numeric | `meta-math/MetaMathQA` |
| **CommonsenseQA** | `commonsenseqa` | Multiple Choice | `tau/commonsense_qa` |

### Direct Dataset Download

You can quickly download and prepare unconfigured Hugging Face datasets using the `install.sh` script. This bypasses the standard generation step and saves the data directly in the required ReasonBench JSON format.

```bash
# Syntax
./install.sh --dataset <huggingface_dataset_identifier> --name <output_name>

# Example: Downloading MetaMathQA
./install.sh --dataset metalearningnet/qwen3.5-metamathqa-cot --name metamathqa

```

> **Important Notes for Direct Downloads:**
> * Files are saved directly to `data/<output_name>_train.json`.
> * They are immediately ready for training and evaluation (skipping `./run.sh --generate`).
> * You must still register the dataset in `conf/datasets.yaml` and create a Python dataset class (see **Adding Custom Datasets** below).
>
>

### Adding Custom Datasets

Integrating a custom or newly downloaded dataset requires three simple steps:

**1. Update `conf/datasets.yaml**` Add your dataset's configuration mapping:

```yaml
your_dataset:
  source: "your_hf_dataset"
  split_mapping:
    train: "train"
    test: "test"
  answer_type: "multiple_choice" # Options: "multiple_choice", "numeric", "boolean"
  field_mapping:
    question: "question_field"
    answer: "answer_field"

```

**2. Create a Dataset Class** Create a Python file (`src/dataset/your_dataset.py`) that inherits from `JsonBasedData` and implements your specific instructions.

```python
# src/dataset/your_dataset.py
from preprocess import JsonBasedData
from generator import DatasetGenerator

class YourDatasetGenerator(DatasetGenerator):
    pass

class YourDataset(JsonBasedData):
    INSTRUCTION = "Your custom instruction for this dataset."

```

**3. Register the Classes in `__init__.py`** Update `src/dataset/__init__.py` to expose your new classes and add them to the maps.

```python
# src/dataset/__init__.py
from .your_dataset import YourDatasetGenerator, YourDataset

GENERATOR_MAP = {
    # ... existing generators ...
    'your_dataset': YourDatasetGenerator
}

DATASET_MAP = {
    # ... existing datasets ...
    'your_dataset': YourDataset
}

```

---

## ⚙️ Configuration

### Global Settings

Modify `conf/settings.yaml` to customize your environment. This includes configuring LLM API endpoints, API keys, training hyper-parameters, and default dataset selections.

### Dataset Configuration Fields

When adjusting `conf/datasets.yaml`, use the following fields to dictate dataset behavior:

* **`source`**: Hugging Face dataset identifier or local file path.
* **`config_name`**: *(Optional)* Specific configuration name for datasets with multiple variants.
* **`split_mapping`**: Maps internal ReasonBench splits (`train`/`test`) to native dataset splits.
* **`answer_type`**: Expected answer format (`boolean`, `numeric`, `multiple_choice`).
* **`valid_answers`**: *(Optional)* Allowed choices for multiple-choice questions (e.g., `["A", "B", "C"]`).
* **`field_mapping`**: Maps native dataset fields to ReasonBench fields (`question`, `answer`, `choices`/`options`).
* **`clean_latex`**: Boolean flag to toggle LaTeX formatting cleanup (default: `false`).

### Special Token Configuration

Customize special tokens and their expected behavior during inference in `conf/tokens.py`. By default, **Memory Steps** extract facts without reasoning, while **Reason Steps** execute calculations using those extracted facts. Together, they form a highly structured Chain-of-Thought pipeline.

```python
# Token Names Configuration
MEMORY_TOKEN_NAME = 'memory'
REASON_TOKEN_NAME = 'reason'

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

```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/) file for details.
