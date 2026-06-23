import argparse
import pytest
from cache_dit._utils.utils import parse_extra_input_kwargs
from cache_dit._utils.registers import ExampleInputData


class TestParseExtraInputKwargs:

  def test_none_returns_none(self):
    assert parse_extra_input_kwargs(None) is None

  def test_empty_string_returns_none(self):
    assert parse_extra_input_kwargs("") is None
    assert parse_extra_input_kwargs("   ") is None

  def test_simple_int_values(self):
    result = parse_extra_input_kwargs("{'a': 1, 'b': 42}")
    assert result == {"a": 1, "b": 42}

  def test_float_values(self):
    result = parse_extra_input_kwargs("{'f': 3.14, 'g': -0.5}")
    assert result == {"f": 3.14, "g": -0.5}

  def test_bool_values(self):
    result = parse_extra_input_kwargs("{'t': True, 'f': False}")
    assert result == {"t": True, "f": False}

  def test_none_value(self):
    result = parse_extra_input_kwargs("{'n': None}")
    assert result == {"n": None}

  def test_string_values(self):
    result = parse_extra_input_kwargs("{'s': 'hello', 't': 'world'}")
    assert result == {"s": "hello", "t": "world"}

  def test_list_values(self):
    result = parse_extra_input_kwargs("{'arr': [1, 2, 3]}")
    assert result == {"arr": [1, 2, 3]}

  def test_nested_list(self):
    result = parse_extra_input_kwargs("{'m': [[1, 2], [3, 4]]}")
    assert result == {"m": [[1, 2], [3, 4]]}

  def test_nested_dict(self):
    result = parse_extra_input_kwargs("{'d': {'inner': 42}}")
    assert result == {"d": {"inner": 42}}

  def test_mixed_types(self):
    result = parse_extra_input_kwargs(
      "{'a': 1, 'b': 2.5, 'c': True, 'd': None, 'e': [1, 2, 3], 'f': 'str'}")
    assert result == {"a": 1, "b": 2.5, "c": True, "d": None, "e": [1, 2, 3], "f": "str"}

  def test_double_quote_keys(self):
    result = parse_extra_input_kwargs('{"key": "value"}')
    assert result == {"key": "value"}

  def test_invalid_syntax_raises(self):
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid Python dict literal"):
      parse_extra_input_kwargs("not a dict")

  def test_non_dict_raises(self):
    with pytest.raises(argparse.ArgumentTypeError, match="expects a dict literal"):
      parse_extra_input_kwargs("[1, 2, 3]")

  def test_invalid_python_raises(self):
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid Python dict literal"):
      parse_extra_input_kwargs("{key: value}")


class TestExampleInputDataExtraKwargs:

  def test_cli_extra_kwargs_override_programmatic(self):
    """CLI --extra-input-kwargs should override programmatic extra_input_kwargs."""
    programmatic_extra = {"max_sequence_length": 1536, "frame_rate": 24.0}
    cli_extra = {"max_sequence_length": 2048}

    input_data = ExampleInputData(
      prompt="test",
      height=512,
      width=512,
      extra_input_kwargs=programmatic_extra,
    )

    args = argparse.Namespace(
      prompt=None,
      negative_prompt=None,
      skip_negative_prompt=False,
      height=None,
      width=None,
      num_inference_steps=None,
      num_frames=None,
      image_path=None,
      mask_image_path=None,
      input_height=None,
      input_width=None,
      seed=None,
      generator_device=None,
      extra_input_kwargs=cli_extra,
    )

    result = input_data.data(args)
    assert result["max_sequence_length"] == 2048  # CLI overrides
    assert result["frame_rate"] == 24.0  # Programmatic preserved

  def test_cli_extra_kwargs_adds_new_keys(self):
    """CLI --extra-input-kwargs can add new keys not in programmatic extras."""
    input_data = ExampleInputData(
      prompt="test",
      height=512,
      width=512,
    )

    args = argparse.Namespace(
      prompt=None,
      negative_prompt=None,
      skip_negative_prompt=False,
      height=None,
      width=None,
      num_inference_steps=None,
      num_frames=None,
      image_path=None,
      mask_image_path=None,
      input_height=None,
      input_width=None,
      seed=None,
      generator_device=None,
      extra_input_kwargs={"cfg_normalization": False},
    )

    result = input_data.data(args)
    assert result["cfg_normalization"] is False

  def test_no_cli_extra_kwargs_no_effect(self):
    """When --extra-input-kwargs is not provided, programmatic extras work as before."""
    programmatic_extra = {"max_sequence_length": 1536}
    input_data = ExampleInputData(
      prompt="test",
      height=512,
      width=512,
      extra_input_kwargs=programmatic_extra,
    )

    args = argparse.Namespace(
      prompt=None,
      negative_prompt=None,
      skip_negative_prompt=False,
      height=None,
      width=None,
      num_inference_steps=None,
      num_frames=None,
      image_path=None,
      mask_image_path=None,
      input_height=None,
      input_width=None,
      seed=None,
      generator_device=None,
      extra_input_kwargs=None,
    )

    result = input_data.data(args)
    assert result["max_sequence_length"] == 1536

  def test_cli_extra_does_not_override_standard_fields(self):
    """CLI --extra-input-kwargs should not interfere with explicit CLI arg overrides."""
    input_data = ExampleInputData(
      prompt="default_prompt",
      height=512,
      width=512,
    )

    args = argparse.Namespace(
      prompt="cli_prompt",
      negative_prompt=None,
      skip_negative_prompt=False,
      height=None,
      width=None,
      num_inference_steps=None,
      num_frames=None,
      image_path=None,
      mask_image_path=None,
      input_height=None,
      input_width=None,
      seed=None,
      generator_device=None,
      extra_input_kwargs={"prompt": "extra_prompt"},
    )

    result = input_data.data(args)
    # CLI --prompt should win since it's applied after extra_input_kwargs merge
    assert result["prompt"] == "cli_prompt"
