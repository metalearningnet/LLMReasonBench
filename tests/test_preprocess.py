import json
import pytest
from preprocess import (
    INVALID_ANS,
    BaseData,
    BooleanAnswerExtractor,
    BooleanAnswerNormalizer,
    DataConfig,
    EnhancedNumericExtractor,
    JsonDataset,
    MultipleChoiceExtractor,
    MultipleChoiceNormalizer,
    NumericAnswerNormalizer
)

VALID = {"a", "b", "c", "d", "e"}


class _MinimalData(BaseData):
    """Concrete BaseData used to exercise its helper methods directly."""

    def get_instruction(self) -> str:
        return ""

    def load_data(self, split):
        return []


def _make_base() -> BaseData:
    obj = _MinimalData.__new__(_MinimalData)
    obj.config = DataConfig()
    return obj


def _make_data(answer_type: str, extractor) -> BaseData:
    obj = _MinimalData.__new__(_MinimalData)
    obj.config = DataConfig()
    obj._answer_type = answer_type
    obj._extractor = extractor
    return obj


def test_multiple_choice_normalizer():
    norm = MultipleChoiceNormalizer(VALID)
    assert norm.normalize("A") == "a"
    assert norm.normalize("b") == "b"
    assert norm.normalize("Option C") == "c"
    assert norm.normalize("Answer: D") == "d"
    assert norm.normalize("3") == "c"


def test_boolean_normalizer():
    norm = BooleanAnswerNormalizer()
    assert norm.normalize(True) == "true"
    assert norm.normalize(False) == "false"
    assert norm.normalize("Yes") == "true"
    assert norm.normalize("No") == "false"
    assert norm.normalize("correct") == "true"
    assert norm.normalize("wrong") == "false"


def test_numeric_normalizer():
    norm = NumericAnswerNormalizer()
    assert norm.normalize("#### 42") == "42"
    assert norm.normalize("The answer is: 3.14") == "3.14"


def test_multiple_choice_extractor():
    ext = MultipleChoiceExtractor(VALID)
    assert ext.extract("The answer is A") == "a"
    assert ext.extract("B") == "b"
    assert ext.extract("gibberish without answer") == INVALID_ANS


def test_boolean_extractor():
    ext = BooleanAnswerExtractor()
    assert ext.extract("True") == "true"
    assert ext.extract("The answer is: no") == "false"
    assert ext.extract("hello") == INVALID_ANS


def test_numeric_extractor_boxed_and_gsm8k():
    ext = EnhancedNumericExtractor()
    assert ext.extract(r"\boxed{42}") == "42"
    assert ext.extract("#### 7") == "7"
    assert ext.extract("The final answer is 7") == "7"


def test_numeric_extractor_keeps_decimals():
    ext = EnhancedNumericExtractor()
    assert ext.extract("The answer is 0.5") == "0.5"
    assert ext.extract("The answer is 3.14") == "3.14"


def test_normalize_numeric_string_fraction():
    assert _make_base()._normalize_numeric_string("1/2") == "0.5"
    assert _make_base()._normalize_numeric_string("3/4") == "0.75"


def test_normalize_numeric_string_percent():
    assert _make_base()._normalize_numeric_string("50%") == "0.5"


def test_normalize_numeric_string_plain():
    assert _make_base()._normalize_numeric_string("5") == "5"


def test_compare_numeric_answers_fraction_matches_decimal():
    assert _make_base()._compare_numeric_answers("1/2", "0.5") is True


def test_compare_numeric_answers_no_rounding_false_positive():
    assert _make_base()._compare_numeric_answers("2.5", "2.4") is False


def test_compare_numeric_answers_equivalent_floats():
    assert _make_base()._compare_numeric_answers("5", "5.0") is True


def test_is_correct_multiple_choice():
    obj = _make_data("multiple_choice", MultipleChoiceExtractor(VALID))
    assert obj.is_correct("The answer is B", "The answer is: b") is True
    assert obj.is_correct("The answer is B", "The answer is: c") is False


def test_is_correct_numeric():
    obj = _make_data("numeric", EnhancedNumericExtractor())
    assert obj.is_correct("The answer is 0.5", "1/2") is True
    assert obj.is_correct("The answer is 3", "4") is False


def test_is_correct_boolean():
    obj = _make_data("boolean", BooleanAnswerExtractor())
    assert obj.is_correct("Yes", "true") is True
    assert obj.is_correct("No", "true") is False


def test_json_dataset_load_data(tmp_path):
    (tmp_path / "dummy_train.json").write_text(
        json.dumps([{"question": "q", "answer": "a"}])
    )
    ds = JsonDataset.__new__(JsonDataset)
    ds.json_path_pattern = str(tmp_path / "dummy_{split}.json")
    assert ds.load_data("train") == [{"question": "q", "answer": "a"}]


def test_json_dataset_load_data_missing_raises(tmp_path):
    ds = JsonDataset.__new__(JsonDataset)
    ds.json_path_pattern = str(tmp_path / "missing_{split}.json")
    with pytest.raises(FileNotFoundError):
        ds.load_data("train")


def test_json_dataset_load_data_invalid_json_raises(tmp_path):
    (tmp_path / "bad_train.json").write_text("{not valid json")
    ds = JsonDataset.__new__(JsonDataset)
    ds.json_path_pattern = str(tmp_path / "bad_{split}.json")
    with pytest.raises(json.JSONDecodeError):
        ds.load_data("train")
