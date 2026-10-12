import torch
import random
from config import IGNORE_INDEX
from fixed_length_dataset import FixedLengthDataset
from multiturn_dataset import DataCollatorForMultiTurnDataset, MultiTurnDataset
from supervised_dataset import DataCollatorForSupervisedDataset, prepare_data


def test_prepare_data_masks_source(fake_tokenizer):
    out = prepare_data(["hello world"], ["answer"], fake_tokenizer)
    input_ids = out["input_ids"][0].tolist()
    labels = out["labels"][0].tolist()
    assert input_ids == [5, 5, 6, fake_tokenizer.eos_token_id]
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 6, fake_tokenizer.eos_token_id]


def test_supervised_collator_pads(fake_tokenizer):
    collator = DataCollatorForSupervisedDataset(tokenizer=fake_tokenizer)
    instances = [
        {"input_ids": torch.tensor([5, 6]), "labels": torch.tensor([IGNORE_INDEX, 6])},
        {"input_ids": torch.tensor([7]), "labels": torch.tensor([8])},
    ]
    out = collator(instances)
    assert out["input_ids"].shape == (2, 2)
    assert out["labels"].shape == (2, 2)
    assert out["attention_mask"].tolist() == [[True, True], [True, False]]


def test_safe_truncate_within_limit():
    ds = MultiTurnDataset.__new__(MultiTurnDataset)
    ds.max_length = 10
    ds.label_pad_token_id = IGNORE_INDEX
    ids = [1, 2, 3]
    labels = [4, 5, 6]
    assert ds._safe_truncate(ids, labels) == (ids, labels)


def test_safe_truncate_trims_trailing_padding():
    ds = MultiTurnDataset.__new__(MultiTurnDataset)
    ds.max_length = 5
    ds.label_pad_token_id = IGNORE_INDEX
    input_ids = [1, 2, 3, 4, 5, 6, 7]
    labels = [10, 11, 12, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
    ids, labs = ds._safe_truncate(input_ids, labels)
    assert ids == [1, 2, 3]
    assert labs == [10, 11, 12]


def test_get_messages():
    ds = MultiTurnDataset.__new__(MultiTurnDataset)
    ds.raw_dataset = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    ]
    msgs = ds._get_messages(0)
    assert msgs[0] == {"role": "user", "content": "hi"}
    assert msgs[1] == {"role": "assistant", "content": "hello"}


def test_multiturn_collator_pads(fake_tokenizer):
    collator = DataCollatorForMultiTurnDataset(
        tokenizer=fake_tokenizer, label_pad_token_id=IGNORE_INDEX
    )
    instances = [
        {"input_ids": torch.tensor([5, 6]), "labels": torch.tensor([6, 7])},
        {"input_ids": torch.tensor([7]), "labels": torch.tensor([8])},
    ]
    out = collator(instances)
    assert out["input_ids"].shape == (2, 2)
    assert out["labels"].shape == (2, 2)


def test_iter_fun_concatenates_x_and_y():
    random.seed(0)
    ds = FixedLengthDataset.__new__(FixedLengthDataset)
    ds.data = [{"x": "hello ", "y": "world"}, {"x": "a ", "y": "b"}]
    out = list(ds.iter_fun())
    assert sorted(out) == ["a b", "hello world"]
