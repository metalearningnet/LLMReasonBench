from config import get_config_value, load_config, merge_configs


def test_load_config_has_expected_sections():
    config = load_config()
    assert isinstance(config, dict)
    for section in ("common", "train", "generator", "logging"):
        assert section in config, f"missing section: {section}"


def test_merge_configs_deep_merges_nested_dicts():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20}, "c": 4}
    merged = merge_configs(base, override)
    assert merged["a"] == {"x": 1, "y": 20}
    assert merged["b"] == 3
    assert merged["c"] == 4
    assert base["a"] == {"x": 1, "y": 2}


def test_get_config_value_dotted_path():
    config = {"a": {"b": {"c": 42}}}
    assert get_config_value(config, "a.b.c") == 42
    assert get_config_value(config, "a.b.missing", "default") == "default"
    assert get_config_value(config, "missing.path") is None
