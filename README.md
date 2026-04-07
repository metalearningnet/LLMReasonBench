# ReasonBench

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Benchmark and enhance reasoning capabilities in large language models (LLMs)

## 📖 Overview

ReasonBench is an evaluation and training framework that measures and improves how well LLMs separate **memory recall** from **logical reasoning**. Based on the methodology from *[Disentangling Memory and Reasoning Ability in Large Language Models](https://github.com/MingyuJ666/Disentangling-Memory-and-Reasoning)*, this tool uses customizable special tokens to explicitly isolate these cognitive processes during inference.

## 🚀 Quick Start

### Installation

Clone the repository and run the installation script:

```bash
git clone https://github.com/metalearningnet/ReasonBench.git
cd ReasonBench
./install.sh
```

### Generate Chain‑of‑Thought (CoT) Data

Before training, you must generate structured CoT steps for your dataset. The generator uses an LLM (vLLM for local inference or OpenAI API) to create steps annotated with the special tokens defined in `COT_TOKENS` (by default, `<memory>` for factual extraction and `<reason>` for logical operations).

```bash
# Generate CoT data for TruthfulQA training set (using vLLM backend by default)
./run.sh --generate --dataset truthfulqa --mode train
```

> **Note:** Generation requires a valid LLM backend. The default is `vllm` with teacher model `Qwen/Qwen3.5-27B`. Configure paths or API keys in `conf/settings.yaml`.

### Train a Model

After generating CoT data, you can fine‑tune a base model (configured via `common.model` in `conf/settings.yaml`):

```bash
# Standard supervised fine‑tuning (using LoRA by default)
./run.sh --train --dataset truthfulqa

# Reinforcement learning (DPO example)
./run.sh --train --rl dpo --dataset truthfulqa
```

Supported RL methods: `dpo`, `cpo`, `kto`, `orpo`.  
RL configuration files (e.g., `conf/dpo.yaml`) control hyperparameters.

### Evaluate a Model

Evaluate any fine‑tuned model, adapter, or base model:

```bash
./run.sh --eval --model /path/to/checkpoint --dataset truthfulqa
```

The evaluation script automatically detects model type (full fine‑tune, PEFT adapter, or base) and loads the appropriate tokenizer.

## 📊 Model Evaluation

The evaluation script supports:
- Standard Hugging Face models
- PEFT adapters
- vLLM accelerated inference
- OpenAI API models (with `--backend api`)

```bash
# Evaluate with vLLM
./run.sh --eval --model meta-llama/Llama-3.2-3B --dataset gsm8k --backend vllm

# Evaluate using OpenAI GPT-4o
./run.sh --eval --model gpt-4o --dataset mmlupro --backend api
```

## 🗂️ Dataset Management

### Supported Datasets

The following datasets are pre‑configured and ready for immediate use:

| Dataset | Config Key | Answer Type | Hugging Face Source |
|---------|------------|-------------|----------------------|
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

You can quickly download and prepare **any** Hugging Face dataset without running the full generation pipeline. This is useful for datasets that already contain CoT‑style reasoning or for test sets.

```bash
./install.sh --dataset <hf_dataset_id> --name <output_name>
```

**Example:**
```bash
./install.sh --dataset metalearningnet/qwen3-metamathqa-cot --name metamathqa
```

By default, this downloads the **train** split and saves it as `data/<output_name>_train.json`. To download a different split, use `--split`. The dataset will be saved to `data/<output_name>_<split>.json`.  
> **Important:** You still need to register the dataset in `conf/datasets.yaml` and create a Python dataset class (see below).

### Adding Custom Datasets

To add a new dataset, follow these three steps:

#### 1. Update `conf/datasets.yaml`

```yaml
my_dataset:
  source: "my-org/my-dataset"
  split_mapping:
    train: "train"
    test: "test"
  answer_type: "multiple_choice"   # one of: multiple_choice, numeric, boolean
  field_mapping:
    question: "question_field"
    answer: "answer_field"
    options: "choices_field"       # optional, for multiple choice
  clean_latex: false               # optional, strip LaTeX formatting
```

#### 2. Create a Dataset Class

Create `src/dataset/my_dataset.py`:

```python
from preprocess import JsonDataset
from generator import DatasetGenerator

class MyDatasetGenerator(DatasetGenerator):
    """Optional custom generator logic."""
    pass

class MyDataset(JsonDataset):
    INSTRUCTION = "Solve the following problem step by step."
```

#### 3. Register in `src/dataset/__init__.py`

```python
from .my_dataset import MyDatasetGenerator, MyDataset

GENERATOR_MAP['my_dataset'] = MyDatasetGenerator
DATASET_MAP['my_dataset'] = MyDataset
```

Now you can use `--dataset my_dataset` with all ReasonBench commands.

## ⚙️ Configuration

All configuration files reside in the `conf/` directory.

### Global Settings

`conf/settings.yaml` controls:
- Base model (`common.model`)
- LoRA configuration (`common.lora_config`)
- Training hyperparameters (`train` section)
- Generation backend and teacher model (`generator` section)

Example snippet (showing key defaults):

```yaml
common:
  model: "Qwen/Qwen3.5-4B"
  parameter_efficient_mode: "lora-cog-tuned"
  lora_config:
    r: 16
    alpha: 16
    use_rslora: true

train:
  learning_rate: 0.00001
  num_train_epochs: 1
  max_prompt_length: 512
  max_completion_length: 2048

generator:
  backend: "vllm"                     # 'vllm' or 'api'
  temperature: 0.3
  vllm:
    model: "Qwen/Qwen3.5-27B"         # Teacher model for CoT generation
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.8
```

### Dataset Configuration

`conf/datasets.yaml` defines each dataset’s properties. The main fields are:

| Field | Description |
|-------|-------------|
| `source` | Hugging Face dataset identifier or local path |
| `config_name` | (Optional) Dataset configuration name |
| `split_mapping` | Maps `train`/`test` to actual split names |
| `answer_type` | `boolean`, `numeric`, or `multiple_choice` |
| `valid_answers` | (Multiple choice) Allowed letters, e.g., `["A","B","C"]` |
| `field_mapping` | Maps dataset fields to `question`, `answer`, `options` |
| `clean_latex` | Remove LaTeX formatting (e.g., `\frac{1}{2}` → `1/2`) |

### Special Token Configuration

`conf/tokens.py` defines the CoT tokens and their expected behavior. The following configuration is used throughout ReasonBench:

```python
COT_TOKENS = {
    'memory': {
        'description': 'Extract and state ONLY the given facts, numbers, or formulas from the problem statement.',
        'prerequisite': True # Generate these steps first and use as context for the rest
    },
    'reason': {
        'description': 'Perform calculations and logical operations using the facts from the prerequisite memory steps to derive conclusions or solve the problem.'
    }
}

END_MARK = True # True: <token> content </token> | False: <token>: content
```

- The **`prerequisite`** flag indicates that steps of that token type are generated **before** the main reasoning and then used as context (like memory) for the remaining steps.
- All token types without `prerequisite: True` are generated in the second stage, after the prerequisite steps are available.

### Chain‑of‑Thought (CoT) Format

ReasonBench supports two output formats controlled by the `END_MARK` flag in `conf/tokens.py`:

- **Colon format** (`END_MARK = False`):
  ```
  <memory>: The problem states that x = 5.
  <reason>: Using x = 5, calculate 2*x = 10.
  ```

- **Closing‑tag format** (`END_MARK = True`):
  ```
  <memory> The problem states that x = 5. </memory>
  <reason> Using x = 5, calculate 2*x = 10. </reason>
  ```

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](https://github.com/metalearningnet/ReasonBench/blob/main/LICENSE) file for details.
