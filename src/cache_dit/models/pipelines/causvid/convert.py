"""Checkpoint conversion for CausVid-Wan.

`tianweiy/CausVid` ships a single `autoregressive_checkpoint/model.pt` whose `generator` entry is a
native Wan state dict (`model.blocks.N.self_attn.q.weight`, ...). The vendored transformer uses the
diffusers Wan module tree, so the keys are renamed here. Nothing is re-packed: every projection is a
separate `Linear` on both sides, and `patch_embedding` is a `Conv3d` on both sides.

The rename that is easiest to get wrong is `norm2`/`norm3`: they are swapped between the two trees.
Native `norm3` is the cross-attention norm and `norm2` the FFN norm, diffusers names them the other
way round.

The released student runs with `cross_attn_norm=True` (same as
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/config.json`), so the cross-attention norm is the one
that carries weights and the FFN norm is parameter-free. That is what makes a wrong swap visible: the
keys would land on a module without parameters. :func:`validate_causal_structure` asserts the shape
of the two norms so the direction can never silently flip if a future revision turns the flag off
(with `cross_attn_norm=False` both norms are parameter-free and `strict=True` alone cannot tell).
"""

import os
import re
import time
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from ....logger import init_logger

logger = init_logger(__name__)

_BLOCK_RENAMES = (
  ("self_attn.q.", "attn1.to_q."),
  ("self_attn.k.", "attn1.to_k."),
  ("self_attn.v.", "attn1.to_v."),
  ("self_attn.o.", "attn1.to_out.0."),
  ("self_attn.norm_q.", "attn1.norm_q."),
  ("self_attn.norm_k.", "attn1.norm_k."),
  ("cross_attn.q.", "attn2.to_q."),
  ("cross_attn.k.", "attn2.to_k."),
  ("cross_attn.v.", "attn2.to_v."),
  ("cross_attn.o.", "attn2.to_out.0."),
  ("cross_attn.norm_q.", "attn2.norm_q."),
  ("cross_attn.norm_k.", "attn2.norm_k."),
  ("cross_attn.k_img.", "attn2.add_k_proj."),
  ("cross_attn.v_img.", "attn2.add_v_proj."),
  ("cross_attn.norm_k_img.", "attn2.norm_added_k."),
  ("ffn.0.", "ffn.net.0.proj."),
  ("ffn.2.", "ffn.net.2."),
)

_TOP_RENAMES = (
  ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
  ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
  ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
  ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
  ("time_projection.1.", "condition_embedder.time_proj."),
  ("head.head.", "proj_out."),
  ("head.modulation", "scale_shift_table"),
)

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


def convert_causvid_state_dict(
    state_dict: Dict[str, torch.Tensor],
    strip_prefixes: Optional[tuple] = ("generator.", "model."),
) -> Dict[str, torch.Tensor]:
  """Rename a native Wan state dict to the vendored diffusers-style keys.

  :param state_dict: The raw checkpoint dict, or its `generator` entry.
  :param strip_prefixes: Leading wrapper prefixes removed when present.
  :returns: A new dict keyed like :class:`CausVidTransformer3DModel`.
  """
  converted: Dict[str, torch.Tensor] = {}
  for key, value in state_dict.items():
    for prefix in strip_prefixes:
      if key.startswith(prefix):
        key = key[len(prefix):]
        break

    if key in ("freqs", "model.freqs"):
      continue

    match = _BLOCK_RE.match(key)
    if match is not None:
      index, rest = match.group(1), match.group(2)
      # Swap before the value renames so `norm2`/`norm3` cannot collide with an `attn` rule.
      if rest.startswith("norm2."):
        rest = "norm3." + rest[len("norm2."):]
      elif rest.startswith("norm3."):
        rest = "norm2." + rest[len("norm3."):]
      elif rest == "modulation":
        rest = "scale_shift_table"
      else:
        for src, dst in _BLOCK_RENAMES:
          if rest.startswith(src):
            rest = dst + rest[len(src):]
            break
      converted[f"blocks.{index}.{rest}"] = value
      continue

    renamed = False
    for src, dst in _TOP_RENAMES:
      if key.startswith(src):
        converted[dst + key[len(src):]] = value
        renamed = True
        break
    if not renamed:
      converted[key] = value

  return converted


def load_causvid_state_dict(
  transformer: nn.Module,
  state_dict: Dict[str, torch.Tensor],
  dtype: Optional[torch.dtype] = None,
) -> None:
  """Load a CausVid checkpoint into the vendored transformer, strictly.

  :param transformer: A :class:`CausVidTransformer3DModel` instance.
  :param state_dict: The raw checkpoint dict, or its `generator` entry.
  :param dtype: Optional cast applied while loading (the released weights are bf16 already).
  :raises RuntimeError: When any key is missing or unexpected after the conversion.
  """
  converted = convert_causvid_state_dict(state_dict)
  model_keys = set(transformer.state_dict().keys())
  converted_keys = set(converted.keys())
  missing = sorted(model_keys - converted_keys)
  unexpected = sorted(converted_keys - model_keys)
  if missing or unexpected:
    raise RuntimeError(
      f"CausVid checkpoint does not match the vendored transformer: {len(missing)} missing "
      f"(e.g. {missing[:5]}), {len(unexpected)} unexpected (e.g. {unexpected[:5]}).")

  if dtype is not None:
    converted = {
      key: (value.to(dtype) if value.is_floating_point() else value)
      for key, value in converted.items()
    }
  transformer.load_state_dict(converted, strict=True)


# Architecture of `tianweiy/CausVid/autoregressive_checkpoint/model.pt` (Wan2.1-T2V-1.3B student).
# `cross_attn_norm=True` matches both the native Wan2.1 1.3B config and
# `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/config.json`; it is also what makes the
# `norm2`/`norm3` swap observable, since only the cross-attention norm then carries weights.
CAUSVID_1_3B_CONFIG = dict(
  patch_size=(1, 2, 2),
  num_attention_heads=12,
  attention_head_dim=128,
  in_channels=16,
  out_channels=16,
  text_dim=4096,
  freq_dim=256,
  ffn_dim=8960,
  num_layers=30,
  cross_attn_norm=True,
  qk_norm="rms_norm_across_heads",
  eps=1e-6,
  rope_max_seq_len=1024,
)


def build_causvid_transformer(
  checkpoint_path: str,
  config_path: Optional[str] = None,
  dtype: torch.dtype = torch.bfloat16,
  device: Union[str, torch.device] = "cpu",
):
  """Build the vendored transformer and load the released student checkpoint.

  :param checkpoint_path: Local `.../autoregressive_checkpoint/model.pt`.
  :param config_path: Optional `transformer/config.json` of the Wan2.1 base checkpoint; the built-in
    1.3B config is used when omitted.
  :param dtype: Parameter dtype of the returned model.
  :param device: Device the model is created on before loading.
  :returns: The loaded transformer in eval mode.
  """
  from diffusers import WanTransformer3DModel

  from .transformer import CausVidTransformer3DModel

  config = dict(CAUSVID_1_3B_CONFIG)
  if config_path is not None and os.path.exists(config_path):
    base_config = WanTransformer3DModel.load_config(config_path)
    config.update({
      key: value
      for key, value in base_config.items() if key in CAUSVID_1_3B_CONFIG and value is not None
    })
    config["patch_size"] = tuple(config["patch_size"])

  t0 = time.perf_counter()
  transformer = CausVidTransformer3DModel(**config).to(device=device, dtype=dtype)
  logger.info(f"Instantiated CausVidTransformer3DModel in {time.perf_counter() - t0:.1f}s.")

  t0 = time.perf_counter()
  raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  if isinstance(raw, dict) and "generator" in raw:
    raw = raw["generator"]
  load_causvid_state_dict(transformer, raw, dtype=dtype)
  del raw
  logger.info(f"Loaded CausVid weights from {checkpoint_path} in {time.perf_counter() - t0:.1f}s.")

  validate_causal_structure(transformer)
  return transformer.eval().requires_grad_(False)


def validate_causal_structure(transformer: nn.Module) -> None:
  """Assert the structural facts the checkpoint cannot prove.

  `strict=True` proves the *keys* match, not that `norm2` and `norm3` were mapped the right way
  round: with `cross_attn_norm=False` both norms are parameter-free, so a flipped swap leaves no
  trace in the state dict. The released student runs with `cross_attn_norm=True`, where only the
  cross-attention norm carries weights, so asserting the two modules turns a silent numeric drift
  into a load-time failure for either setting of the flag.

  :param transformer: A :class:`CausVidTransformer3DModel` instance.
  """
  blocks = list(transformer.blocks)
  if not blocks:
    raise ValueError("CausVid transformer has no blocks.")
  first = blocks[0]
  if first.__class__.__name__ != "CausVidTransformerBlock":
    raise ValueError(f"Expected CausVidTransformerBlock blocks, got {first.__class__.__name__}.")

  # `norm2` is the cross-attention norm and `norm3` the FFN norm in the diffusers tree (the native
  # tree names them the other way round). With `cross_attn_norm=True` only `norm2` carries weights,
  # which pins the direction the conversion must use.
  if transformer.config.cross_attn_norm:
    if isinstance(first.norm2, nn.Identity) or getattr(first.norm2, "elementwise_affine",
                                                       False) is not True:
      raise ValueError(
        "cross_attn_norm=True but block.norm2 is not the affine cross-attention norm; "
        "the `norm2`/`norm3` conversion direction is wrong.")
  elif not isinstance(first.norm2, nn.Identity):
    raise ValueError("cross_attn_norm=False but block.norm2 is not Identity; the `norm2`/`norm3` "
                     "conversion direction is wrong.")
  if isinstance(first.norm3, nn.Identity):
    raise ValueError("block.norm3 must be the (parameter-free) FFN norm, not Identity.")
  if getattr(first.norm3, "elementwise_affine", True) is not False:
    raise ValueError("block.norm3 must be parameter-free, matching the Wan FFN norm.")
  if tuple(first.scale_shift_table.shape) != (1, 6, transformer.config.attention_head_dim *
                                              transformer.config.num_attention_heads):
    raise ValueError("block.scale_shift_table must be shaped (1, 6, dim).")
  if tuple(transformer.scale_shift_table.shape) != (1, 2, transformer.config.attention_head_dim *
                                                    transformer.config.num_attention_heads):
    raise ValueError("transformer.scale_shift_table must be shaped (1, 2, dim).")
  logger.debug("CausVid transformer structure validated.")
