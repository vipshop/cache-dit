"""Multi-rank parity test for the Ideogram-4 CP and TP plumbing (tiny model, no weights).

Ideogram-4 packs `[left-pad][text][image]` into one sequence and always passes a
`(B, 1, L, L)` block-diagonal mask built from `segment_ids`. Context parallelism splits the
packed sequence at the transformer entry and keeps `segment_ids` whole, relying on the
Ulysses all-to-all to concatenate the per-rank contiguous chunks back into the original
order. A wrong mask layout or a wrong chunk order still produces a smooth image, so both
paths are compared against a cp-free / tp-free reference on rank 0.

Text and image tokens straddle the CP chunk boundary on purpose (rank 0 holds text plus a
couple of image tokens, rank 1 holds the rest), which is the layout that exposes a mask
that was rebuilt instead of kept whole.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_ideogram4_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

try:
  from diffusers.models.transformers.transformer_ideogram4 import (  # noqa: F401
    Ideogram4Transformer2DModel, )
except ImportError:
  pytest.skip("diffusers without Ideogram-4 support", allow_module_level=True)

_TEXT_TOKENS = 8
_PAD_TOKENS = 2
_GRID = 4
_IMAGE_TOKENS = _GRID * _GRID
_SEQ = _PAD_TOKENS + _TEXT_TOKENS + _IMAGE_TOKENS
_IN_CHANNELS = 8
_LLM_FEATURES = 16
_HEADS = 2
_HEAD_DIM = 16
_LAYERS = 2
_IMAGE_POSITION_OFFSET = 65536
_LLM_TOKEN_INDICATOR = 3
_OUTPUT_IMAGE_INDICATOR = 2
_SEQUENCE_PADDING_INDICATOR = -1
_REL_TOL = 2e-2  # bf16 all-reduce order noise on a 2-layer random model
_PORT = 29591


def _build(device):
  torch.manual_seed(0)
  transformer = Ideogram4Transformer2DModel(
    in_channels=_IN_CHANNELS,
    num_layers=_LAYERS,
    attention_head_dim=_HEAD_DIM,
    num_attention_heads=_HEADS,
    intermediate_size=2 * _HEAD_DIM * _HEADS,
    adaln_dim=8,
    llm_features_dim=_LLM_FEATURES,
    mrope_section=(2, 2, 2),
  )
  return transformer.to(device=device, dtype=torch.bfloat16).eval()


def _layout():
  """Build the packed `[pad][text][image]` layout used by the pipeline."""
  position_ids = torch.zeros(1, _SEQ, 3, dtype=torch.long)
  segment_ids = torch.full((1, _SEQ), _SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
  indicator = torch.zeros(1, _SEQ, dtype=torch.long)

  text_start = _PAD_TOKENS
  image_start = _PAD_TOKENS + _TEXT_TOKENS

  for token in range(_TEXT_TOKENS):
    position_ids[0, text_start + token] = token
  h_idx = torch.arange(_GRID).view(-1, 1).expand(_GRID, _GRID).reshape(-1)
  w_idx = torch.arange(_GRID).view(1, -1).expand(_GRID, _GRID).reshape(-1)
  image_pos = torch.stack([torch.zeros_like(h_idx), h_idx, w_idx], dim=1) + _IMAGE_POSITION_OFFSET
  position_ids[0, image_start:] = image_pos

  indicator[0, text_start:image_start] = _LLM_TOKEN_INDICATOR
  indicator[0, image_start:] = _OUTPUT_IMAGE_INDICATOR
  segment_ids[0, text_start:image_start] = 1
  segment_ids[0, image_start:] = 2
  return position_ids, segment_ids, indicator


def _inputs():
  generator = torch.Generator().manual_seed(3)
  position_ids, segment_ids, indicator = _layout()
  return {
    "hidden_states":
    torch.randn(1, _SEQ, _IN_CHANNELS, generator=generator).to(torch.bfloat16),
    "timestep":
    torch.tensor([0.5], dtype=torch.bfloat16),
    "encoder_hidden_states":
    torch.randn(1, _SEQ, _LLM_FEATURES, generator=generator).to(torch.bfloat16),
    "position_ids":
    position_ids,
    "segment_ids":
    segment_ids,
    "indicator":
    indicator,
  }


def _to_device(inputs, device):
  return {
    key: (value.to(device) if torch.is_tensor(value) else value)
    for key, value in inputs.items()
  }


def _run_rank(rank: int, world: int, mode: str) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(_PORT)
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"

  try:
    from cache_dit.distributed import ParallelismConfig, enable_parallelism

    inputs = _inputs()
    transformer = _build(device)

    reference = None
    if rank == 0:
      with torch.no_grad():
        reference = [
          t.float().cpu() for t in transformer(**_to_device(inputs, device), return_dict=False)
        ]
    torch.distributed.barrier()

    if mode == "cp":
      parallelism_config = ParallelismConfig(ulysses_size=world)
    else:
      parallelism_config = ParallelismConfig(tp_size=world)
    enable_parallelism(transformer, parallelism_config)

    # Guard against a silently inert plan: CP must hand `layers.0` a local chunk, otherwise
    # the parity below would pass on an unsharded forward that ran twice the work.
    seen_seq = {"value": None}
    handle = None
    if mode == "cp":
      handle = transformer.layers[0].register_forward_pre_hook(
        lambda module, args: seen_seq.__setitem__("value", args[0].shape[1]))
    try:
      with torch.no_grad():
        out = transformer(**_to_device(inputs, device), return_dict=False)[0].float()
    finally:
      if handle is not None:
        handle.remove()
    if mode == "cp":
      assert seen_seq["value"] == _SEQ // world, (
        f"cp did not split the packed sequence: layers.0 saw {seen_seq['value']} tokens, "
        f"expected {_SEQ // world}")

    # Every rank must agree on the gathered output, i.e. the split/gather is symmetric.
    gathered = [torch.empty_like(out) for _ in range(world)]
    torch.distributed.all_gather(gathered, out)

    if rank == 0:
      got, ref = out.cpu(), reference[0]
      relative = (got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)
      print(f"[ideogram4-{mode}] rel={relative.item():.6f} shape={tuple(got.shape)}")
      assert got.shape == ref.shape, f"{mode} shape changed: {got.shape} vs {ref.shape}"
      assert relative.item() < _REL_TOL, f"{mode} diverged (rel={relative.item():.6f})"
      for other in gathered[1:]:
        relative = (other - gathered[0]).abs().mean() / gathered[0].abs().mean().clamp_min(1e-6)
        print(f"[ideogram4-{mode}] rank delta={relative.item():.6f}")
        assert relative.item() < _REL_TOL, f"{mode} ranks diverged"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_ideogram4_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode), nprocs=2, join=True)
