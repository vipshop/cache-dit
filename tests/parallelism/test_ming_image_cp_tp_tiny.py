from types import SimpleNamespace

import pytest

from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from cache_dit.distributed.transformers.zimage import (
  MingImageContextParallelismPlanner,
  ZImageTensorParallelismPlanner,
)
from cache_dit.models.pipelines.ming_image.transformer import MingImageTransformer2DModel


def _transformer(alignment_padding_mode="zero_masked"):
  return MingImageTransformer2DModel(
    in_channels=4,
    dim=64,
    n_layers=2,
    n_refiner_layers=1,
    n_heads=2,
    n_kv_heads=2,
    cap_feat_dim=8,
    axes_dims=(8, 12, 12),
    axes_lens=(256, 64, 64),
    alignment_padding_mode=alignment_padding_mode,
  )


def test_ming_uses_zimage_cp_plan():
  transformer = _transformer()
  planner_cls = ContextParallelismPlannerRegister.get_planner(transformer)
  assert planner_cls is MingImageContextParallelismPlanner

  plan = MingImageContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=SimpleNamespace(
      ulysses_async=False,
      ring_size=None,
      ulysses_anything=False,
      _has_controlnet=False,
    ),
  )
  assert "noise_refiner.0" in plan
  assert "context_refiner.0" in plan
  assert "layers.0" in plan
  assert "layers.*" in plan
  assert "all_final_layer" in plan


@pytest.mark.parametrize("ring_size,ulysses_anything", [(2, False), (None, True)])
def test_ming_rejects_zero_masked_ring_and_uaa(ring_size, ulysses_anything):
  with pytest.raises(NotImplementedError, match="zero-masked|full-sequence"):
    MingImageContextParallelismPlanner()._apply(
      transformer=_transformer("zero_masked"),
      parallelism_config=SimpleNamespace(
        ulysses_async=False,
        ring_size=ring_size,
        ulysses_anything=ulysses_anything,
        _has_controlnet=False,
      ),
    )


def test_ming_learned_padding_keeps_zimage_cp_modes():
  plan = MingImageContextParallelismPlanner()._apply(
    transformer=_transformer("learned"),
    parallelism_config=SimpleNamespace(
      ulysses_async=False,
      ring_size=2,
      ulysses_anything=True,
      _has_controlnet=False,
    ),
  )
  assert "layers.0" in plan
  assert "all_final_layer" in plan


def test_ming_text_encoder_plan_shards_qkv_and_gate_up_only(monkeypatch):
  from cache_dit.distributed.text_encoders.ming_image import (
    MingImageMLLMTensorParallelismPlanner, )

  dense_block = SimpleNamespace(is_moe=False)
  moe_block = SimpleNamespace(
    is_moe=True,
    mlp=SimpleNamespace(experts=[SimpleNamespace(), SimpleNamespace()], shared_experts=object()),
  )
  text_encoder = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(
    layers=[dense_block, moe_block])))
  mesh = SimpleNamespace(get_group=lambda: SimpleNamespace(size=lambda: 2))
  calls = []

  def fake_parallelize_module(*, module, device_mesh, parallelize_plan):
    calls.append((module, parallelize_plan))

  monkeypatch.setattr("cache_dit.distributed.text_encoders.ming_image.parallelize_module",
                      fake_parallelize_module)
  monkeypatch.setattr("cache_dit.distributed.text_encoders.ming_image.maybe_empty_cache",
                      lambda: None)

  _, plans = MingImageMLLMTensorParallelismPlanner().parallelize_text_encoder(text_encoder, mesh)

  assert len(calls) == len(plans) == 2
  dense_plan, moe_plan = plans
  assert "attention.query_key_value" in dense_plan
  assert "mlp.gate_proj" in dense_plan
  assert "mlp.up_proj" in dense_plan
  assert "mlp.down_proj" not in dense_plan
  assert "mlp.experts.0.gate_proj" in moe_plan
  assert "mlp.experts.1.up_proj" in moe_plan
  assert "mlp.shared_experts.gate_proj" in moe_plan
  assert not any("down_proj" in key for key in moe_plan)


def test_ming_tp_uses_real_zimage_attribute_names(monkeypatch):
  transformer = _transformer()
  planner_cls = TensorParallelismPlannerRegister.get_planner(transformer)
  assert planner_cls is ZImageTensorParallelismPlanner

  calls = []

  def fake_parallelize_module(*, module, device_mesh, parallelize_plan):
    calls.append((module, parallelize_plan))

  monkeypatch.setattr("cache_dit.distributed.transformers.zimage.parallelize_module",
                      fake_parallelize_module)
  monkeypatch.setattr("cache_dit.distributed.transformers.zimage.shard_div_attr",
                      lambda *args: None)
  mesh = SimpleNamespace(get_group=lambda: SimpleNamespace(size=lambda: 2))

  _, plans = ZImageTensorParallelismPlanner().parallelize_transformer(transformer, mesh)

  assert len(calls) == len(plans) == 4
  for module, plan in calls:
    assert hasattr(module, "attention")
    assert hasattr(module.feed_forward, "w1")
    assert "attention.to_q" in plan
    assert "feed_forward.w1" in plan
    assert "feed_forward.w2" in plan
    assert "feed_forward.w3" in plan
