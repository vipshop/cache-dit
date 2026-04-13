from cache_dit.quantization import QuantizeConfig


def test_quantize_config_strify_keeps_identity_svdq_dq_name() -> None:
  config = QuantizeConfig(quant_type="svdq_int4_r128_dq")

  assert config.strify() == "svdq_int4_r128_dq"


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
