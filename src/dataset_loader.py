import json
import argparse
from pathlib import Path
from typing import Optional, Union
from config import DEFAULT_DATA_DIR, logger
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset

def download_dataset(
    dataset_name: str,
    split: str = "train",
    config_name: Optional[str] = None,
    streaming: bool = False,
    **load_kwargs
) -> Union[Dataset, DatasetDict, IterableDataset]:
    try:
        logger.info(f"Loading dataset: {dataset_name}")
        if config_name:
            logger.info(f"  Configuration: {config_name}")
        
        try:
            logger.info(f"  Attempting to load split: {split}")
            dataset = load_dataset(
                dataset_name,
                name=config_name,
                split=split,
                streaming=streaming,
                **load_kwargs
            )
            logger.info(f"  Successfully loaded split: {split}")
        except ValueError as e:
            logger.warning(f"  Split '{split}' not found: {e}")
            logger.info("  Loading all splits to examine available options...")
            
            all_splits = load_dataset(
                dataset_name,
                name=config_name,
                streaming=streaming,
                **load_kwargs
            )
            
            if isinstance(all_splits, DatasetDict):
                available_splits = list(all_splits.keys())
                logger.info(f"  Available splits: {available_splits}")
                
                if 'train' in available_splits:
                    logger.info(f"  Using 'train' split as fallback")
                    dataset = all_splits['train']
                else:
                    first_split = available_splits[0]
                    logger.info(f"  No 'train' split found. Using first available: {first_split}")
                    dataset = all_splits[first_split]
            else:
                logger.info("  Dataset has only one split. Using entire dataset.")
                dataset = all_splits
        
        if isinstance(dataset, (Dataset, IterableDataset)):
            logger.info(f"  Adding 'split' column with value: {split}")
            
            def add_split_column(batch):
                batch['split'] = [split] * len(batch[list(batch.keys())[0]])
                return batch
            
            if isinstance(dataset, IterableDataset):
                dataset = dataset.map(add_split_column)
            else:
                dataset = dataset.map(add_split_column, batched=True)
        
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def filter_dataset(
    dataset: Union[Dataset, IterableDataset],
    max_length: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42
) -> Dataset:
    if isinstance(dataset, IterableDataset):
        logger.info("Converting streaming dataset to regular dataset...")
        take_amount = max_length * 2 if max_length else 10000
        dataset = Dataset.from_list(list(dataset.take(take_amount)))
    
    original_length = len(dataset)
    logger.info(f"Original dataset length: {original_length}")
    
    if max_length is not None and max_length < original_length:
        if shuffle:
            logger.info(f"Shuffling dataset with seed {seed}...")
            dataset = dataset.shuffle(seed=seed)
        
        logger.info(f"Truncating to {max_length} examples...")
        dataset = dataset.select(range(max_length))
    
    logger.info(f"Final dataset length: {len(dataset)}")
    return dataset

def save_as_json(
    dataset: Dataset,
    output_path: Path,
    indent: int = 4
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = dataset.to_dict()
    examples = []
    num_examples = len(data[list(data.keys())[0]])
    
    for i in range(num_examples):
        example = {}
        for key in data.keys():
            example[key] = data[key][i]
        examples.append(example)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=indent, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face datasets and convert to JSON"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path of the dataset on Hugging Face Hub (e.g., 'metalearningnet/qwen3.5-metamathqa-cot')"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the dataset to be saved"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory to save the downloaded dataset"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download (e.g., 'train', 'validation', 'test'). "
             "Default: 'train'. If not found, will use 'train' if available, "
             "otherwise the only available split."
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum number of examples to keep. If not specified, keeps all examples."
    )
    
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Configuration name for datasets with multiple configurations"
    )
    
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset before truncating"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets"
    )
    
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indentation level (default: 4)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = download_dataset(
        dataset_name=args.dataset,
        split=args.split,
        config_name=args.config_name,
        streaming=args.streaming
    )
    
    if args.max_length is not None:
        dataset = filter_dataset(
            dataset,
            max_length=args.max_length,
            shuffle=args.shuffle,
            seed=args.seed
        )
    
    filename = f"{args.name}_{args.split}.json"

    save_as_json(dataset, output_dir / filename, args.indent)
    
    logger.info(f"Dataset successfully downloaded and saved to: {output_dir / filename}")

if __name__ == "__main__":
    main()
