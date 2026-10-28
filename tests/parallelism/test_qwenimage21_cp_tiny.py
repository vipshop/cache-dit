"""Two-rank Qwen-Image 2.1 full-sequence Ulysses parity tests (tiny model, no weights).

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_qwenimage21_cp_tiny.py -q
"""
import os

import pytest
import torch
import torch.multiprocessing as mp

_TEXT_LEN = 5
_GRID_H = _GRID_W = 4
_TARGET_H = _TARGET_W = 8
_CONDITION_TOKENS = _GRID_H * _GRID_W
_TARGET_TOKENS = _TARGET_H * _TARGET_W
_PREFIX_LEN = _TEXT_LEN + 2 * _CONDITION_TOKENS
_PORT = 29632
_REL_TOL = 2e-3


def _build(device, world):
  from diffusers import QwenImage21Transformer2DModel

  torch.manual_seed(0)
  return QwenImage21Transformer2DModel(
    in_channels=4,
    context_in_dim=16,
    num_layers=2,
    num_attention_heads=2 * world,
    attention_head_dim=8,
    axes_dims_rope=(2, 2, 4),
    mlp_ratio=2,
  ).to(device=device, dtype=torch.bfloat16).eval()


def _inputs(use_mask):
  generator = torch.Generator().manual_seed(3)
  image_tokens = 2 * _CONDITION_TOKENS + _TARGET_TOKENS
  img_mask = torch.zeros(1, _TEXT_LEN + image_tokens // 4, dtype=torch.bool)
  img_mask[:, _TEXT_LEN:] = True
  encoder_len = _TEXT_LEN + 2 * _CONDITION_TOKENS // 4
  encoder_mask = None
  if use_mask:
    encoder_mask = torch.ones(1, encoder_len, dtype=torch.bool)
    encoder_mask[:, _TEXT_LEN - 1] = False
  return dict(
    hidden_states=torch.randn(1, image_tokens, 4, generator=generator),
    encoder_hidden_states=torch.randn(1, encoder_len, 16, generator=generator),
    img_shapes=[[(1, _GRID_H, _GRID_W), (1, _GRID_H, _GRID_W), (1, _TARGET_H, _TARGET_W)]],
    img_mask=img_mask,
    encoder_hidden_states_mask=encoder_mask,
  )


def _run_steps(model, inputs, device, use_kv_cache):
  from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

  kv_cache = QwenImage21KVCache(len(model.transformer_blocks)) if use_kv_cache else None
  outputs = []
  with torch.no_grad():
    for step in range(3):
      mode = None
      if kv_cache is not None:
        mode = "extract" if step == 0 else "cached"
      model_inputs = {
        key: (value.to(device=device, dtype=torch.bfloat16) if torch.is_floating_point(value) else
              value.to(device)) if torch.is_tensor(value) else value
        for key, value in inputs.items()
      }
      outputs.append(
        model(
          **model_inputs,
          timestep=torch.tensor([1.0 - step / 3], device=device, dtype=torch.bfloat16),
          kv_cache=kv_cache,
          kv_cache_mode=mode,
          return_dict=False,
        )[0].float().cpu())
  return outputs, kv_cache


def _relative(actual, expected):
  return ((actual - expected).abs().mean() / expected.abs().mean().clamp_min(1e-6)).item()


def _run_rank(rank, world):
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(_PORT)
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"
  try:
    import cache_dit
    from cache_dit.caching.block_adapters import BlockAdapterRegister
    from cache_dit.distributed.config import ParallelismConfig

    for use_kv_cache, use_mask in ((True, True), (True, False), (False, True), (False, False)):
      transformer = _build(device, world)
      inputs = _inputs(use_mask)
      reference, reference_cache = _run_steps(transformer, inputs, device, use_kv_cache)
      torch.distributed.barrier()

      cache_dit.enable_cache(
        BlockAdapterRegister.get_adapter(transformer),
        parallelism_config=ParallelismConfig(ulysses_size=world),
      )
      actual, actual_cache = _run_steps(transformer, inputs, device, use_kv_cache)

      for step, (got, expected) in enumerate(zip(actual, reference)):
        relative = _relative(got, expected)
        assert got.shape == expected.shape
        assert relative < _REL_TOL, (
          f"rank={rank} cache={use_kv_cache} mask={use_mask} step={step} rel={relative:.6f}")
        torch.testing.assert_close(got, expected, atol=1e-2, rtol=2e-3)

      if use_kv_cache:
        for layer in range(len(transformer.transformer_blocks)):
          got_k, got_v = actual_cache.get_layer(layer).get()
          expected_k, expected_v = reference_cache.get_layer(layer).get()
          head_start = rank * (expected_k.shape[2] // world)
          head_end = head_start + expected_k.shape[2] // world
          assert got_k.shape == (1, _PREFIX_LEN, expected_k.shape[2] // world, expected_k.shape[3])
          assert _relative(got_k.float().cpu(),
                           expected_k[:, :, head_start:head_end].float().cpu()) < _REL_TOL
          assert _relative(got_v.float().cpu(),
                           expected_v[:, :, head_start:head_end].float().cpu()) < _REL_TOL
      if rank == 0:
        print(f"qwen21-cp cache={use_kv_cache} mask={use_mask}: PASS")
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_qwenimage21_full_sequence_cp2_matches_single_rank():
  mp.spawn(_run_rank, args=(2, ), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs four GPUs")
def test_qwenimage21_full_sequence_cp4_matches_single_rank():
  mp.spawn(_run_rank, args=(4, ), nprocs=4, join=True)
