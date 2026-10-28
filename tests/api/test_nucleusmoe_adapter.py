"""Regression tests for the NucleusMoE-Image cache integration.

NucleusMoE-Image is a regular diffusers pipeline whose transformer
(`NucleusMoEImageTransformer2DModel`) keeps a single `transformer_blocks` stack of blocks that
return `hidden_states` only, i.e. `ForwardPattern.Pattern_2`. The blocks mix dense and
Mixture-of-Experts layers, so the tests also pin the property that a cached run is bit-exact when no
step is skipped: a skipped block must not matter, and a *computed* block must produce exactly what
the unpatched transformer produces.

No weights are required: every model is built from a tiny config.
"""
import inspect

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "NucleusMoEImageTransformer2DModel"):
  pytest.skip("NucleusMoE-Image requires a newer diffusers", allow_module_level=True)

# Captured before any test installs the workaround, so a test can always start from upstream code.
from diffusers.models.transformers.transformer_nucleusmoe_image import (  # noqa: E402
  SwiGLUExperts as _SwiGLUExperts, )

_UPSTREAM_EXPERTS_GROUPED_MM = _SwiGLUExperts._run_experts_grouped_mm
assert not getattr(_UPSTREAM_EXPERTS_GROUPED_MM, "_cache_dit_patched", False), (
  "another module installed the grouped-GEMM offsets patch at import time; the workaround tests "
  "below need the upstream method to start from")

_TEXT_LEN = 6
_FRAMES = 1
_LATENT_H = 4
_LATENT_W = 4
_IN_CHANNELS = 4
_DIM = 128


def _tiny_transformer(num_layers: int = 4):
  from diffusers import NucleusMoEImageTransformer2DModel

  torch.manual_seed(0)
  transformer = NucleusMoEImageTransformer2DModel(
    patch_size=2,
    in_channels=_IN_CHANNELS,
    out_channels=_IN_CHANNELS,
    num_layers=num_layers,
    attention_head_dim=8,
    num_attention_heads=_DIM // 8,
    num_key_value_heads=2,
    joint_attention_dim=32,
    # `sum(axes_dims_rope)` must equal `head_dim` (8) and every axis must be even.
    axes_dims_rope=(2, 2, 4),
    moe_enabled=True,
    # Keep exactly one dense block so both block kinds are exercised.
    dense_moe_strategy="leave_first_block_dense",
    num_experts=4,
    moe_intermediate_dim=32,
    capacity_factors=2.0,
    use_grouped_mm=False,
  )
  # The routed expert tables are `torch.empty` on purpose upstream: they always come from a
  # checkpoint. Fill them so an uninitialized buffer cannot leak NaNs into the comparison.
  with torch.no_grad():
    for block in transformer.transformer_blocks:
      if block.moe_enabled:
        block.img_mlp.experts.gate_up_proj.normal_(0.0, 0.02)
        block.img_mlp.experts.down_proj.normal_(0.0, 0.02)
  return transformer.eval()


class _NucleusLikePipe:
  """Minimal stand-in for `NucleusMoEImagePipeline`: a transformer plus a `__call__` default."""

  def __init__(self, transformer, guidance_scale: float = 4.0):
    self.transformer = transformer

    def __call__(self, guidance_scale: float = guidance_scale):
      return None

    self.__call__ = __call__


def _inputs(batch_size: int = 1, seed: int = 1):
  generator = torch.Generator().manual_seed(seed)
  image_seq_len = _FRAMES * _LATENT_H * _LATENT_W
  return dict(
    hidden_states=torch.randn(batch_size, image_seq_len, _IN_CHANNELS, generator=generator),
    img_shapes=[(_FRAMES, _LATENT_H, _LATENT_W)] * batch_size,
    encoder_hidden_states=torch.randn(batch_size, _TEXT_LEN, 32, generator=generator),
    encoder_hidden_states_mask=torch.ones(batch_size, _TEXT_LEN, dtype=torch.bool),
    timestep=torch.tensor([0.5] * batch_size),
  )


def test_nucleus_prefix_dispatches_to_the_nucleus_adapter():
  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "NucleusMoE" in prefixes
  matches = [name for name in prefixes if "NucleusMoEImagePipeline".startswith(name)]
  assert matches and matches[0] == "NucleusMoE", f"wrong adapter for NucleusMoE: {matches}"


@pytest.mark.parametrize(("guidance_scale", "expected"), [(4.0, True), (1.0, False)])
def test_has_separate_cfg_follows_pipeline_default(guidance_scale, expected):
  from cache_dit.caching.block_adapters import nucleus_moe_adapter

  pipe = _NucleusLikePipe(_tiny_transformer(), guidance_scale=guidance_scale)
  adapter = nucleus_moe_adapter(pipe)
  assert adapter.has_separate_cfg is expected
  assert adapter.blocks is pipe.transformer.transformer_blocks
  assert adapter.forward_pattern is ForwardPattern.Pattern_2


def test_cache_run_is_bit_exact_when_no_step_is_cached():
  from cache_dit.caching.block_adapters import nucleus_moe_adapter

  reference = _tiny_transformer()
  inputs = _inputs()
  with torch.no_grad():
    expected = reference(**inputs, return_dict=False)[0]

  transformer = _tiny_transformer()
  adapter = nucleus_moe_adapter(_NucleusLikePipe(transformer))
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=0,
      num_inference_steps=4,
      enable_separate_cfg=True,
      residual_diff_threshold=0.0,
    ),
  )
  signature = str(inspect.signature(transformer.forward))
  with torch.no_grad():
    output = transformer(**inputs, return_dict=False)[0]

  assert str(inspect.signature(transformer.forward)) == signature
  assert torch.equal(output, expected), (
    "the cached (never-skipping) run must match the unpatched transformer, including the dense and "
    "the MoE blocks")


def test_only_the_tensor_parallel_planner_is_registered():
  from cache_dit.distributed.transformers.planners import (
    _activate_cp_planners,
    _activate_tp_planners,
  )
  from cache_dit.distributed.transformers.register import (
    ContextParallelismPlannerRegister,
    TensorParallelismPlannerRegister,
  )

  _activate_cp_planners()
  _activate_tp_planners()
  cp_planners = ContextParallelismPlannerRegister.supported_planners()[1]
  tp_planners = TensorParallelismPlannerRegister.supported_planners()[1]
  # Context parallelism is intentionally not offered: the expert-choice routing derives its
  # capacity from the sequence length, so a sharded sequence selects a different token set.
  assert "NucleusMoEImageTransformer2DModel" not in cp_planners
  assert "NucleusMoEImageTransformer2DModel" in tp_planners


def test_grouped_mm_offsets_patch_is_int32_and_matches_the_loop_path(monkeypatch):
  from cache_dit import patch_nucleus_moe_grouped_mm_offsets

  monkeypatch.setattr(_SwiGLUExperts, "_run_experts_grouped_mm", _UPSTREAM_EXPERTS_GROUPED_MM)
  seen = []

  def _fake_grouped_mm(x, weight, *, offs=None, out_dtype=None):
    # Same signature and contract as `aten._grouped_mm`, which the upstream code violates under
    # compile; the math is the real grouped GEMM so the copy can be checked against the loop path.
    assert offs is not None and offs.dtype is torch.int32
    seen.append(offs.dtype)
    outs, start = [], 0
    for expert_idx, end in enumerate(offs.tolist()):
      outs.append(torch.matmul(x[start:end], weight[expert_idx]))
      start = end
    return torch.cat(outs, dim=0)

  monkeypatch.setattr(torch.nn.functional, "grouped_mm", _fake_grouped_mm, raising=False)

  assert patch_nucleus_moe_grouped_mm_offsets() is True
  assert patch_nucleus_moe_grouped_mm_offsets() is False, "the patch must not wrap twice"

  torch.manual_seed(0)
  experts = _SwiGLUExperts(hidden_size=8, moe_intermediate_dim=16, num_experts=2)
  # The expert tables are `torch.empty` upstream (they always come from a checkpoint).
  with torch.no_grad():
    experts.gate_up_proj.normal_(0.0, 0.05)
    experts.down_proj.normal_(0.0, 0.05)
  x = torch.randn(4, 8)
  num_tokens = torch.full((2, ), 2, dtype=torch.long)
  fused = experts._run_experts_grouped_mm(x, num_tokens)
  assert len(seen) == 2, "both grouped GEMMs of the SwiGLU pair must receive int32 offsets"
  torch.testing.assert_close(fused, experts._run_experts_for_loop(x, num_tokens))


def test_adapter_installs_the_patch_only_when_the_env_switch_is_on(monkeypatch):
  from cache_dit.caching.block_adapters import nucleus_moe_adapter
  from cache_dit.envs import ENV

  monkeypatch.setattr(_SwiGLUExperts, "_run_experts_grouped_mm", _UPSTREAM_EXPERTS_GROUPED_MM)
  monkeypatch.setattr(ENV, "CACHE_DIT_NUCLEUS_MOE_GROUPED_MM_OFFSETS_FIX", False)
  nucleus_moe_adapter(_NucleusLikePipe(_tiny_transformer()))
  assert _SwiGLUExperts._run_experts_grouped_mm is _UPSTREAM_EXPERTS_GROUPED_MM

  monkeypatch.setattr(ENV, "CACHE_DIT_NUCLEUS_MOE_GROUPED_MM_OFFSETS_FIX", True)
  nucleus_moe_adapter(_NucleusLikePipe(_tiny_transformer()))
  assert _SwiGLUExperts._run_experts_grouped_mm is not _UPSTREAM_EXPERTS_GROUPED_MM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="`F.grouped_mm` needs CUDA")
@pytest.mark.skipif(not hasattr(torch.nn.functional, "grouped_mm"),
                    reason="torch without `F.grouped_mm`")
def test_grouped_mm_offsets_patch_unblocks_torch_compile(monkeypatch):
  from cache_dit import patch_nucleus_moe_grouped_mm_offsets

  monkeypatch.setattr(_SwiGLUExperts, "_run_experts_grouped_mm", _UPSTREAM_EXPERTS_GROUPED_MM)

  transformer = _tiny_transformer().to("cuda", torch.bfloat16)
  block = next(b for b in transformer.transformer_blocks if b.moe_enabled)
  # The released checkpoint runs the fused kernel; the CPU tests above use the for-loop path.
  block.img_mlp.experts.use_grouped_mm = True
  dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
  bf16 = dict(device="cuda", dtype=torch.bfloat16)
  inputs = dict(
    hidden_states=torch.randn(1, 16, dim, **bf16),
    encoder_hidden_states=torch.randn(1, _TEXT_LEN, transformer.config.joint_attention_dim, **bf16),
    temb=torch.randn(1, dim, **bf16),
    image_rotary_emb=None,
  )

  # Control: without the workaround the fused kernel must be what breaks, otherwise this test only
  # pins "something compiles" and would stay green after upstream or torch fixes the bug.
  try:
    with torch.no_grad():
      torch.compile(block, fullgraph=True)(**inputs)
  except Exception as exc:  # noqa: BLE001
    assert "int32" in str(exc) or "grouped_mm" in str(exc), exc
  else:
    pytest.skip("upstream/torch no longer need the offsets workaround")

  patch_nucleus_moe_grouped_mm_offsets()
  torch._dynamo.reset()  # drop the cached failure so the patched body is traced again
  with torch.no_grad():
    torch.compile(block, fullgraph=True)(**inputs)
