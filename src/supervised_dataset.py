import torch
import random
import transformers
from config import IGNORE_INDEX
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional, Sequence

def prepare_data(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict[str, List[torch.Tensor]]:
    input_ids_list = []
    labels_list = []
    
    for source, target in zip(sources, targets):
        source_ids = tokenizer.encode(source, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        
        input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
        
        labels = [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]
        
        input_ids_list.append(torch.tensor(input_ids))
        labels_list.append(torch.tensor(labels))
    
    return {
        "input_ids": input_ids_list,
        "labels": labels_list
    }

class SupervisedDataset(Dataset):
    def __init__(self, dataset, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        
        data_dict = prepare_data(self.dataset.x, self.dataset.y, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        input_ids_padded = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        labels_padded = pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        
        attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id)
        
        return {
            "labels": labels_padded,
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask
        }

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    dataset,
    eval_dataset: Optional[Dataset],
    max_num_eval: Optional[int],
    seed: int = 42
) -> Dict:
    random.seed(seed)
    
    # Subsample evaluation dataset if needed
    if eval_dataset is not None and max_num_eval is not None and len(eval_dataset) > max_num_eval:
        indices = random.sample(range(len(eval_dataset)), max_num_eval)
        
        # Create a simple dataset-like object with x and y attributes
        class SubsampledDataset:
            def __init__(self, original_dataset, indices):
                self.x = [original_dataset[i]['x'] for i in indices]
                self.y = [original_dataset[i]['y'] for i in indices]
            def __len__(self):
                return len(self.x)
        
        eval_dataset = SubsampledDataset(eval_dataset, indices)
    
    train_dataset = SupervisedDataset(dataset, tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator
    }
