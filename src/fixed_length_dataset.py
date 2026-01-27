import torch
import random
import transformers
from typing import Dict
from config import logger
from torch.utils.data import IterableDataset

class FixedLengthDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6
    ):
        super(FixedLengthDataset, self).__init__()
        self.data = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.epoch = 0
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.num_of_sequences = num_of_sequences
        
        logger.info(
            f"Initialized FixedLengthDataset with: "
            f"seq_length={seq_length}, "
            f"num_of_sequences={num_of_sequences}, "
            f"max_buffer_size={self.max_buffer_size:.0f}, "
            f"dataset_size={len(dataset)}"
        )
    
    def set_epoch(self, epoch):
        random.seed(epoch)
        logger.debug(f"Set epoch to: {epoch}")

    def iter_fun(self):
        ids = list(range(len(self.data)))
        random.shuffle(ids)
        logger.debug(f"Created iterator with {len(ids)} shuffled examples")
        for i in ids:
            sent = self.data[i]['x'] + self.data[i]['y']
            yield sent

    def __iter__(self):
        more_examples = True
        iterator = self.iter_fun()
        buffer_count = 0
        batch_count = 0
        
        logger.debug("Starting dataset iteration")
        
        while more_examples:
            buffer, buffer_len = [], 0
            logger.debug("Filling buffer...")
            
            while True:
                if buffer_len >= self.max_buffer_size:
                    logger.debug(f"Buffer filled with {len(buffer)} sequences, total length: {buffer_len}")
                    break
                try:
                    next_sent = next(iterator)
                    buffer.append(next_sent)
                    buffer_len += len(next_sent)
                    buffer_count += 1
                except StopIteration:
                    logger.info(f"Dataset epoch {self.epoch} completed")
                    iterator = self.iter_fun()
                    self.epoch += 1
                    logger.info(f"Starting dataset epoch: {self.epoch}")
            
            logger.debug(f"Shuffling buffer of {len(buffer)} sequences")
            random.shuffle(buffer)
            
            logger.debug("Tokenizing buffer...")
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.tokenizer.eos_token_id])
            
            total_tokens = len(all_token_ids)
            logger.debug(f"Tokenization complete: {total_tokens} total tokens")
            
            for i in range(0, total_tokens, self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    batch_count += 1
                    
                    if batch_count % 1000 == 0:
                        logger.debug(f"Generated {batch_count} batches so far")
                    
                    yield dict(
                        input_ids=torch.tensor(input_ids),
                        labels=torch.tensor(input_ids)
                    )
            
            logger.debug(f"Processed buffer: generated {total_tokens // self.seq_length} batches")

def make_fixed_length_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    dataset,
    eval_dataset,
    seq_length
) -> Dict:
    logger.info("Creating fixed length data module")
    logger.info(f"Sequence length: {seq_length}")
    logger.info(f"Training dataset size: {len(dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset) if eval_dataset else 0}")
    
    random.seed(42)
    
    if eval_dataset is not None and len(eval_dataset) > 1000:
        logger.info(f"Downsampling evaluation dataset from {len(eval_dataset)} to 1000 examples")
        idx = random.choices(list(range(len(eval_dataset))), k=1000)
        new_x = []
        new_y = []
        
        for i in idx:
            new_x.append(eval_dataset[i]['x'])
            new_y.append(eval_dataset[i]['y'])
        
        eval_dataset.x = new_x
        eval_dataset.y = new_y
        logger.debug("Evaluation dataset downsampled successfully")
    
    if eval_dataset is not None:
        if len(eval_dataset) > 1000:
            error_msg = f"Evaluation dataset size ({len(eval_dataset)}) exceeds 1000"
            logger.error(error_msg)
            raise AssertionError(error_msg)
        logger.info(f"Final evaluation dataset size: {len(eval_dataset)}")
    
    logger.info("Creating training dataset")
    train_dataset = FixedLengthDataset(dataset, tokenizer, seq_length)
    
    result = dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    logger.info("Data module created successfully")
    return result
