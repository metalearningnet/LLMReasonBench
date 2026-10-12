from config import END_MARK
from generator import (
    API_PROVIDERS,
    CoTGenerator,
    DatasetGenerator,
    detect_api_provider,
    format_trajectory,
    resolve_api_key,
)


def _make_cot_generator(**kwargs) -> CoTGenerator:
    obj = CoTGenerator.__new__(CoTGenerator)
    obj.end_mark = kwargs.get("end_mark", END_MARK)
    obj.debug = kwargs.get("debug", False)
    obj.should_validate_cot_steps = kwargs.get("should_validate_cot_steps", True)
    obj.min_cot_steps = kwargs.get("min_cot_steps", 1)
    obj.max_cot_steps = kwargs.get("max_cot_steps", 200)
    return obj


def _make_dataset_generator() -> DatasetGenerator:
    obj = DatasetGenerator.__new__(DatasetGenerator)
    obj.dataset_config = {
        "answer_type": "multiple_choice",
        "valid_answers": ["A", "B", "C", "D", "E"],
    }
    obj._answer_type = "multiple_choice"
    obj._valid_answers = {"A", "B", "C", "D", "E"}
    obj._choice_labels = ["A", "B", "C", "D", "E"]
    obj._clean_latex_enabled = False
    return obj


def test_api_providers_registry_has_common_providers():
    assert "deepseek" in API_PROVIDERS
    assert "openai" in API_PROVIDERS
    assert API_PROVIDERS["deepseek"]["api_key_env"] == ["DEEPSEEK_API_KEY"]
    assert API_PROVIDERS["openai"]["api_key_env"] == ["OPENAI_API_KEY"]


def test_detect_api_provider_from_base_url():
    assert detect_api_provider("https://api.deepseek.com/v1", "") == "deepseek"
    assert detect_api_provider("https://api.openai.com/v1", "gpt-4o") == "openai"


def test_detect_api_provider_from_model_name():
    assert detect_api_provider("", "deepseek-chat") == "deepseek"
    assert detect_api_provider("", "gpt-4o") is None


def test_resolve_api_key_precedence(monkeypatch):
    for var in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "API_KEY", "LLM_API_KEY", "MY_KEY"):
        monkeypatch.delenv(var, raising=False)

    assert resolve_api_key("sk-explicit", "deepseek", None) == ("sk-explicit", "config")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
    assert resolve_api_key("", "deepseek", None) == ("sk-ds", "env:DEEPSEEK_API_KEY")

    monkeypatch.setenv("MY_KEY", "sk-custom")
    assert resolve_api_key("", "deepseek", "MY_KEY") == ("sk-custom", "env:MY_KEY")

    assert resolve_api_key("", None, None, "https://api.deepseek.com/v1", "deepseek-chat") == (
        "sk-ds",
        "env:DEEPSEEK_API_KEY",
    )

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MY_KEY", raising=False)
    assert resolve_api_key("", None, None) == ("", None)


def test_format_trajectory():
    steps = [
        {
            "observation": "You see a table",
            "action": "go to table",
            "admissible_commands": ["go to table"],
        }
    ]
    out = format_trajectory(steps, "put apple on table")
    assert "Goal: put apple on table" in out
    assert "Action: go to table" in out


def test_parse_step():
    cg = _make_cot_generator()
    assert cg._parse_step("<memory> x = 5 </memory>") == ("memory", "x = 5")
    assert cg._parse_step("<reason>: 2*x = 10") == ("reason", "2*x = 10")
    assert cg._parse_step("plain text") == (None, None)


def test_extract_labeled_content():
    cg = _make_cot_generator()
    text = (
        "<memory> x = 5 </memory>\n"
        "unlabeled reasoning here\n"
        "<reason> 2*x = 10 </reason>"
    )
    steps = cg.extract_labeled_content(text)
    assert "<memory> x = 5 </memory>" in steps
    assert "<reason> 2*x = 10 </reason>" in steps
    assert any(s.startswith("Unlabeled reasoning here") for s in steps)


def test_extract_labeled_content_colon_mode():
    cg = _make_cot_generator(end_mark=False)
    text = "<memory>: x = 5\nplain reasoning\n<reason>: 2*x = 10"
    steps = cg.extract_labeled_content(text)
    assert "<memory>: x = 5" in steps
    assert "<reason>: 2*x = 10" in steps
    assert "Plain reasoning" in steps


def test_validate_cot_steps_empty_is_invalid():
    assert _make_cot_generator().validate_cot_steps([]) is False


def test_validate_cot_steps_valid():
    cg = _make_cot_generator()
    steps = [
        "<memory> The problem states x = 5 </memory>",
        "<reason> compute 2*x = 10 </reason>",
    ]
    assert cg.validate_cot_steps(steps) is True


def test_validate_cot_steps_rejects_garbage():
    cg = _make_cot_generator()
    assert cg.validate_cot_steps(["<memory> x </memory>"]) is False


def test_is_garbage_content():
    cg = _make_cot_generator()
    assert cg._is_garbage_content("") is True
    assert cg._is_garbage_content("ok") is True
    assert cg._is_garbage_content("Wait, let me think") is True
    assert cg._is_garbage_content("first step") is True
    assert cg._is_garbage_content("The problem states x = 5") is False


def test_clean_latex_fraction_and_sqrt():
    dg = _make_dataset_generator()
    assert dg._clean_latex(r"\frac{1}{2}") == "(1)/(2)"
    assert dg._clean_latex(r"\sqrt{4}") == "sqrt(4)"


def test_clean_latex_dollar_and_text():
    dg = _make_dataset_generator()
    assert dg._clean_latex(r"$x$") == "x"
    assert dg._clean_latex(r"\text{hello}") == "hello"


def test_clean_latex_greek_and_symbols():
    dg = _make_dataset_generator()
    assert dg._clean_latex(r"\alpha") == "α"
    assert dg._clean_latex(r"\times") == "×"


def test_clean_latex_collapses_whitespace():
    assert _make_dataset_generator()._clean_latex("a   b") == "a b"


def test_get_valid_answers_and_choice_labels():
    dg = _make_dataset_generator()
    assert dg.get_valid_answers() == {"A", "B", "C", "D", "E"}
    assert dg.get_choice_labels() == ["A", "B", "C", "D", "E"]


def test_normalize_multiple_choice_answer():
    dg = _make_dataset_generator()
    assert dg._normalize_multiple_choice_answer("B", None) == "B"
    assert dg._normalize_multiple_choice_answer("option c", None) == "C"
    assert dg._normalize_multiple_choice_answer("Answer: D", None) == "D"
    assert dg._normalize_multiple_choice_answer("3", None) == "C"


def test_normalize_boolean_answer():
    dg = _make_dataset_generator()
    assert dg._normalize_boolean_answer("yes") == "true"
    assert dg._normalize_boolean_answer(False) == "false"
    assert dg._normalize_boolean_answer("wrong") == "false"


def test_normalize_numeric_answer():
    dg = _make_dataset_generator()
    assert dg._normalize_numeric_answer("#### 42") == "42"
    assert dg._normalize_numeric_answer("The answer is: 3.14") == "3.14"


def test_parse_options_internal():
    dg = _make_dataset_generator()
    assert dg._parse_options_internal(["a", "b", "c"]) == ["a", "b", "c"]
    assert dg._parse_options_internal("a, b, c") == ["a", "b", "c"]
