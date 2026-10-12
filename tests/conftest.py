import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for directory in (ROOT / "src", ROOT / "conf"):
    path = str(directory)
    if path not in sys.path:
        sys.path.insert(0, path)


class _FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.bos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [len(word) for word in text.split()]

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [self.encode(text) for text in texts]}


@pytest.fixture(scope="session", autouse=True)
def _disable_exit_on_error_handler():
    import logging

    from config import ExitOnErrorHandler

    root = logging.getLogger()
    handlers = [h for h in root.handlers if isinstance(h, ExitOnErrorHandler)]
    for handler in handlers:
        root.removeHandler(handler)

    yield

    for handler in handlers:
        root.addHandler(handler)


@pytest.fixture
def fake_tokenizer():
    return _FakeTokenizer()
