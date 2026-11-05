"""CPU checks for the Qwen-Image 1.x TP layer plan.

`QwenDoubleStreamAttnProcessor2_0` reshapes the column-sharded QKV through the attention module's
metadata, and the two diffusers generations read that metadata differently:

- `< 0.40`: `query.unflatten(-1, (attn.heads, -1))` — explicit head count, implied head dim.
- `>= 0.40`: `head_dim = attn.inner_dim // attn.heads` then `unflatten(-1, (-1, head_dim))` —
  explicit head dim, implied head count.

Dividing only `heads` therefore leaves the local head dim at `2 * head_dim` on `>= 0.40`, which
breaks the QK-norm and the 3-axis RoPE tables; keeping `heads` whole breaks the `< 0.40` path. The
plan must divide both, which keeps `head_dim` invariant. The assertions below encode that invariant
so that either of the two wrong fixes fails.

Usage:
  python -m pytest tests/parallelism/test_qwen_image_tp_plan.py -q
"""

import pytest

from cache_dit.distributed.transformers.planners import _activate_tp_planners
from cache_dit.distributed.transformers.qwen_image import \
  QwenImageTensorParallelismPlanner
from cache_dit.distributed.transformers.register import \
  TensorParallelismPlannerRegister

_NUM_BLOCKS = 2
_HEADS = 4
_HEAD_DIM = 16


class _FakeGroup:

  def __init__(self, size: int):
    self._size = size

  def size(self) -> int:
    return self._size

  def rank(self) -> int:
    return 0


class _FakeMesh:
  """Minimal device-mesh stand-in: the planner only reads its size before sharding."""

  def __init__(self, size: int):
    self._size = size

  def size(self) -> int:
    return self._size

  def get_group(self) -> _FakeGroup:
    return _FakeGroup(self._size)


def _tiny_transformer():
  from diffusers import QwenImageTransformer2DModel

  return QwenImageTransformer2DModel(
    patch_size=2,
    in_channels=64,
    out_channels=16,
    num_layers=_NUM_BLOCKS,
    attention_head_dim=_HEAD_DIM,
    num_attention_heads=_HEADS,
    joint_attention_dim=32,
    axes_dims_rope=(4, 6, 6),
  )


def _locals(attn) -> tuple[int, int, int]:
  """(to_q local out_features, heads, head_dim inferred by the processor)."""
  return attn.to_q.out_features, attn.heads, attn.inner_dim // attn.heads


def test_tp_layer_plan_divides_heads_and_inner_dim(monkeypatch):
  import cache_dit.distributed.transformers.qwen_image as planner_module

  captured = []
  monkeypatch.setattr(
    planner_module,
    "parallelize_module",
    lambda module, device_mesh, parallelize_plan: captured.append(parallelize_plan),
  )

  transformer = _tiny_transformer()
  full_inner_dim = transformer.transformer_blocks[0].attn.inner_dim
  _, layer_plans = QwenImageTensorParallelismPlanner().parallelize_transformer(
    transformer, _FakeMesh(2))

  assert len(layer_plans) == _NUM_BLOCKS
  assert len(captured) == _NUM_BLOCKS
  assert {
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "img_mod.1",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mod.1",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
  } <= set(layer_plans[0])

  for block in transformer.transformer_blocks:
    attn = block.attn
    local_out, heads, head_dim = _locals(attn)
    # Both metadata attributes follow the shard.
    assert attn.heads == _HEADS // 2
    assert attn.inner_dim == full_inner_dim // 2
    # The invariant: the column shard equals heads * head_dim of the local view, and the processor
    # still derives the real head dim. Either wrong fix (heads only / heads kept) breaks this.
    assert local_out // 2 == heads * head_dim
    assert head_dim == _HEAD_DIM


def test_tp_invariant_rejects_the_two_wrong_fixes():
  t = _tiny_transformer()
  attn = t.transformer_blocks[0].attn
  full_inner_dim, full_heads = attn.inner_dim, attn.heads

  def _invariant(local_out, heads, inner_dim):
    return local_out // 2 == heads * (inner_dim // heads)

  # heads-only (the pre-fix planner): head dim doubles on diffusers >= 0.40.
  assert not _invariant(full_inner_dim, full_heads // 2, full_inner_dim)
  # heads-kept (the "keep heads full" variant): breaks the diffusers < 0.40 path.
  assert not _invariant(full_inner_dim, full_heads, full_inner_dim)
  # both divided (shipped): head dim preserved.
  assert _invariant(full_inner_dim, full_heads // 2, full_inner_dim // 2)


def test_tp_rejects_an_indivisible_head_count():
  with pytest.raises(ValueError, match="divisible by tp_size"):
    QwenImageTensorParallelismPlanner().parallelize_transformer(_tiny_transformer(), _FakeMesh(3))


def test_planner_is_registered_under_the_model_name():
  from diffusers import QwenImageTransformer2DModel

  _activate_tp_planners()
  assert TensorParallelismPlannerRegister.get_planner(
    QwenImageTransformer2DModel.__name__) is QwenImageTensorParallelismPlanner
