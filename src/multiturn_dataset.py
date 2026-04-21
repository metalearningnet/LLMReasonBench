import torch
import random
from config import IGNORE_INDEX, logger
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, List, Optional, Sequence, Tuple, Union

class MultiTurnDataset(TorchDataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        label_pad_token_id: int = IGNORE_INDEX
    ):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id
        self._warned_native_masking = False

        logger.info("Preprocessing multi‑turn conversations...")
        self.input_ids, self.labels = self._preprocess_all()

    def _get_messages(self, idx: int) -> List[Dict[str, str]]:
        item = self.raw_dataset[idx]
        if hasattr(item, "messages"):
            return item["messages"] if isinstance(item, dict) else item.messages
        elif isinstance(item, dict) and "messages" in item:
            return item["messages"]
        else:
            raise ValueError(
                "Dataset item must have a 'messages' field containing a list of "
                "role/content dicts."
            )

    def _preprocess_all(self) -> Tuple[List[List[int]], List[List[int]]]:
        all_input_ids = []
        all_labels = []
        for idx in range(len(self.raw_dataset)):
            messages = self._get_messages(idx)
            clean_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]
            input_ids, labels = self._build_sft_data(clean_messages)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        logger.info(f"Preprocessed {len(all_input_ids)} multi‑turn examples.")
        return all_input_ids, all_labels

    def _safe_truncate(
        self, input_ids: List[int], labels: List[int]
    ) -> Tuple[List[int], List[int]]:
        if self.max_length is None or len(input_ids) <= self.max_length:
            return input_ids, labels

        truncated_labels = labels[: self.max_length]
        max_seq_len = self.max_length

        if truncated_labels[-1] == self.label_pad_token_id:
            for j in range(self.max_length - 1, -1, -1):
                if truncated_labels[j] != self.label_pad_token_id:
                    max_seq_len = j + 1
                    break

        return input_ids[:max_seq_len], labels[:max_seq_len]

    def _build_sft_data(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[int], List[int]]:
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is required. Please load with `use_fast=True`")

        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                max_length=None,
                truncation=False,
                add_generation_prompt=False
            )
            input_ids = encoded["input_ids"]
            assistant_masks = encoded.get("assistant_masks")

            if not assistant_masks or not any(assistant_masks):
                raise ValueError("Missing assistant_masks in tokenizer output")

            labels = [
                tid if mask == 1 else self.label_pad_token_id
                for tid, mask in zip(input_ids, assistant_masks)
            ]
            return self._safe_truncate(input_ids, labels)

        except (TypeError, KeyError, ValueError) as e:
            if not self._warned_native_masking:
                logger.warning(
                    f"Native assistant masking unavailable ({e}). "
                    "Falling back to offset mapping."
                )
                self._warned_native_masking = True

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoding = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
        labels = [self.label_pad_token_id] * len(input_ids)

        assistant_regions = []
        search_cursor = 0
        for i, msg in enumerate(messages):
            raw_content = msg["content"]
            content = raw_content
            content_start = full_text.find(raw_content, search_cursor)
            if content_start == -1:
                content = raw_content.strip()
                content_start = full_text.find(content, search_cursor)
            if content_start == -1:
                content_start = full_text.find(content[:50], search_cursor)
            if content_start == -1:
                logger.debug(f"Could not locate content for role '{msg['role']}'")
                content_start = search_cursor
                content_end = search_cursor
            else:
                content_end = content_start + len(content)

            next_start = len(full_text)
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                next_raw = next_msg["content"]
                found_next = full_text.find(next_raw, content_end)
                if found_next == -1:
                    found_next = full_text.find(next_raw.strip()[:50], content_end)
                if found_next != -1 and found_next >= content_end:
                    next_start = found_next

            if msg["role"] == "assistant":
                assistant_regions.append((content_start, next_start))

            search_cursor = next_start

        for i, (c_start, c_end) in enumerate(offsets):
            if c_start == 0 and c_end == 0:
                continue
            for r_start, r_end in assistant_regions:
                if c_start >= r_start and c_end <= r_end:
                    labels[i] = input_ids[i]
                    break

        return self._safe_truncate(input_ids, labels)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def is_correct(self, prediction: str, ground_truth: str) -> bool:
        if hasattr(self.raw_dataset, "is_correct"):
            return self.raw_dataset.is_correct(prediction, ground_truth)

        return prediction.strip() == ground_truth.strip()


class DataCollatorForMultiTurnDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        label_pad_token_id: int = IGNORE_INDEX
    ):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(
        self, instances: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]

        input_ids_padded = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels_padded = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.label_pad_token_id
        )
        attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask
        }

def make_multiturn_data_module(
    tokenizer: PreTrainedTokenizer,
    train_dataset: Union[Sequence, TorchDataset],
    eval_dataset: Optional[Union[Sequence, TorchDataset]] = None,
    max_length: Optional[int] = None,
    seed: int = 42
) -> Dict:
    random.seed(seed)

    train_ds = MultiTurnDataset(
        train_dataset,
        tokenizer,
        max_length=max_length
    )

    eval_ds = None
    if eval_dataset is not None:
        eval_ds = MultiTurnDataset(
            eval_dataset,
            tokenizer,
            max_length=max_length
        )

    data_collator = DataCollatorForMultiTurnDataset(tokenizer=tokenizer)

    return {
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": data_collator
    }
