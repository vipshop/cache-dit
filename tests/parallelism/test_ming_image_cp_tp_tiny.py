from types import SimpleNamespace

from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from cache_dit.distributed.transformers.zimage import (
  ZImageContextParallelismPlanner,
  ZImageTensorParallelismPlanner,
)
from cache_dit.models.pipelines.ming_image.transformer import MingImageTransformer2DModel


def _transformer():
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
  )


def test_ming_uses_zimage_cp_plan():
  transformer = _transformer()
  planner_cls = ContextParallelismPlannerRegister.get_planner(transformer)
  assert planner_cls is ZImageContextParallelismPlanner

  plan = ZImageContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=SimpleNamespace(ulysses_async=False, _has_controlnet=False),
  )
  assert "noise_refiner.0" in plan
  assert "context_refiner.0" in plan
  assert "layers.0" in plan
  assert "layers.*" in plan
  assert "all_final_layer" in plan


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
