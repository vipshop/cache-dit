import warnings

import pytest

from cache_dit.quantization import QuantizeConfig


@pytest.mark.parametrize("quant_type", ["svdq_int4_r128_dq", "svdq_nvfp4_r128_dq"])
def test_quantize_config_strify_keeps_identity_svdq_dq_name(quant_type: str) -> None:
  config = QuantizeConfig(quant_type=quant_type)

  assert config.strify() == quant_type
  assert config.get_svdq_rank() == 128


def test_quantize_config_accepts_svdq_nvfp4_ptq() -> None:
  config = QuantizeConfig(
    quant_type="svdq_nvfp4_r64",
    calibrate_fn=lambda **_: None,
    serialize_to=".tmp/tests/test_quantize_config_nvfp4_ptq",
  )

  assert config.is_svdq_ptq()
  assert config.get_svdq_precision() == "nvfp4"
  assert config.serialize_to.endswith("svdq_nvfp4_r64.safetensors")


def test_quantize_config_rejects_svdq_nvfp4_v2_runtime_kernel() -> None:
  with pytest.raises(ValueError, match="NVFP4 currently only supports"):
    QuantizeConfig(
      quant_type="svdq_nvfp4_r128_dq",
      svdq_kwargs={"runtime_kernel": "v2"},
    )


def test_quantize_config_strify_appends_weight_svdq_dq_name() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r128_dq",
    svdq_kwargs={"smooth_strategy": "weight"},
  )

  assert config.strify() == "svdq_int4_r128_dq_weight"


def test_quantize_config_strify_appends_weight_inv_svdq_dq_name() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r128_dq",
    svdq_kwargs={"smooth_strategy": "weight_inv"},
  )

  assert config.strify() == "svdq_int4_r128_dq_weight_inv"


def test_quantize_config_strify_appends_few_shot_svdq_dq_name() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r128_dq",
    svdq_kwargs={"smooth_strategy": "few_shot"},
  )

  assert config.strify() == "svdq_int4_r128_dq_few_shot_auto"


def test_quantize_config_strify_appends_few_shot_relax_strategy_name() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r128_dq",
    svdq_kwargs={
      "smooth_strategy": "few_shot",
      "few_shot_relax_strategy": "auto",
    },
  )

  assert config.strify() == "svdq_int4_r128_dq_few_shot_auto"


def test_quantize_config_strify_appends_few_shot_stable_auto_relax_strategy_name() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r128_dq",
    svdq_kwargs={
      "smooth_strategy": "few_shot",
      "few_shot_relax_strategy": "stable_auto",
    },
  )

  assert config.strify() == "svdq_int4_r128_dq_few_shot_stable_auto"


def test_quantize_config_svdq_dq_few_shot_defaults_are_resolved() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r128_dq",
    svdq_kwargs={"smooth_strategy": "few_shot"},
  )

  assert config.is_svdq_dq_few_shot()
  assert config.get_svdq_kwargs()["few_shot_steps"] == 1
  assert config.get_svdq_kwargs()["few_shot_relax_factor"] == 1.5
  assert config.get_svdq_kwargs()["few_shot_relax_top_ratio"] == 0.25
  assert config.get_svdq_kwargs()["few_shot_relax_strategy"] == "auto"
  assert config.get_svdq_kwargs()["few_shot_auto_compile"] is False


def test_quantize_config_svdq_dq_few_shot_warns_for_large_relax_factor() -> None:
  with pytest.warns(RuntimeWarning, match="oversmooth or blur outputs"):
    QuantizeConfig(
      quant_type="svdq_int4_r128_dq",
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_relax_factor": 4.0,
      },
    )


def test_quantize_config_svdq_dq_few_shot_fixed_skips_large_relax_warning() -> None:
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    config = QuantizeConfig(
      quant_type="svdq_int4_r128_dq",
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_relax_factor": 4.0,
        "few_shot_relax_strategy": "fixed",
      },
    )

  assert config.get_svdq_kwargs()["few_shot_relax_strategy"] == "fixed"
  assert caught == []


def test_quantize_config_svdq_dq_few_shot_rejects_invalid_relax_values() -> None:
  try:
    QuantizeConfig(
      quant_type="svdq_int4_r128_dq",
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_relax_factor": 0.0,
      },
    )
  except ValueError as exc:
    assert "few_shot_relax_factor" in str(exc)
  else:
    raise AssertionError("Expected invalid few_shot_relax_factor to raise ValueError.")

  try:
    QuantizeConfig(
      quant_type="svdq_int4_r128_dq",
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_relax_factor": 0.5,
      },
    )
  except ValueError as exc:
    assert "few_shot_relax_factor" in str(exc)
    assert ">= 1.0" in str(exc)
  else:
    raise AssertionError("Expected few_shot_relax_factor < 1.0 to raise ValueError.")

  try:
    QuantizeConfig(
      quant_type="svdq_int4_r128_dq",
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_relax_top_ratio": 1.5,
      },
    )
  except ValueError as exc:
    assert "few_shot_relax_top_ratio" in str(exc)
  else:
    raise AssertionError("Expected invalid few_shot_relax_top_ratio to raise ValueError.")

  try:
    QuantizeConfig(
      quant_type="svdq_int4_r128_dq",
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_relax_strategy": "unknown",
      },
    )
  except ValueError as exc:
    assert "few_shot_relax_strategy" in str(exc)
  else:
    raise AssertionError("Expected invalid few_shot_relax_strategy to raise ValueError.")


def test_quantize_config_svdq_dq_few_shot_normalizes_top_q4_alias() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r128_dq",
    svdq_kwargs={
      "smooth_strategy": "few_shot",
      "few_shot_relax_strategy": "top_q4",
    },
  )

  assert config.get_svdq_kwargs()["few_shot_relax_strategy"] == "top"


@pytest.mark.parametrize("strategy", ["fixed", "auto", "stable_auto", "power", "log", "rank"])
def test_quantize_config_svdq_dq_few_shot_accepts_relax_strategies(strategy: str) -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r128_dq",
    svdq_kwargs={
      "smooth_strategy": "few_shot",
      "few_shot_relax_strategy": strategy,
    },
  )

  assert config.get_svdq_kwargs()["few_shot_relax_strategy"] == strategy
