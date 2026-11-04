"""CPU checks for the LongLive 2.0 checkpoint conversion.

The released generator is a native Wan state dict while the vendored transformer keeps the diffusers
module tree, so the mapping has to be an exact bijection. The `norm2`/`norm3` direction is the
dangerous part: the two trees use the names in **opposite** order, and only the diffusers `norm2`
carries a weight (`cross_attn_norm=True`), so mapping the native `norm3` onto the diffusers `norm3`
would leave the affine norm unloaded - correct shapes, wrong modulation.

Usage:
  python -m pytest tests/api/test_longlive2_convert.py -q
"""

import pytest
import torch

from cache_dit.models.pipelines.longlive2.convert import (
  convert_state_dict,
  derive_config,
  validate_causal_structure,
)
from cache_dit.models.pipelines.longlive2.transformer import LongLive2Transformer3DModel

TINY_CONFIG = dict(
  patch_size=(1, 2, 2),
  num_attention_heads=2,
  attention_head_dim=8,
  in_channels=16,
  out_channels=16,
  text_dim=32,
  freq_dim=32,
  ffn_dim=64,
  num_layers=2,
  cross_attn_norm=True,
  qk_norm="rms_norm_across_heads",
  eps=1e-6,
  rope_max_seq_len=64,
)

# Independent, literal description of the native LongLive module tree.
_DIFFUSERS_TO_NATIVE = (
  ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
  ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
  ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
  ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
  ("condition_embedder.time_proj.", "time_projection.1."),
  ("patch_embedding.", "patch_embedding."),
  ("proj_out.", "head.head."),
)

_DIFFUSERS_TO_NATIVE_BLOCK = (
  ("attn1.to_q.", "self_attn.q."),
  ("attn1.to_k.", "self_attn.k."),
  ("attn1.to_v.", "self_attn.v."),
  ("attn1.to_out.0.", "self_attn.o."),
  ("attn1.norm_q.", "self_attn.norm_q."),
  ("attn1.norm_k.", "self_attn.norm_k."),
  ("attn2.to_q.", "cross_attn.q."),
  ("attn2.to_k.", "cross_attn.k."),
  ("attn2.to_v.", "cross_attn.v."),
  ("attn2.to_out.0.", "cross_attn.o."),
  ("attn2.norm_q.", "cross_attn.norm_q."),
  ("attn2.norm_k.", "cross_attn.norm_k."),
  ("ffn.net.0.proj.", "ffn.0."),
  ("ffn.net.2.", "ffn.2."),
  # swapped on purpose: diffusers `norm2` is the affine cross-attention norm
  ("norm2.", "norm3."),
)


def _tiny_transformer() -> LongLive2Transformer3DModel:
  torch.manual_seed(0)
  return LongLive2Transformer3DModel(**TINY_CONFIG).eval()


def _to_native(diffusers_sd):
  native = {}
  for key, value in diffusers_sd.items():
    renamed = None
    if key.startswith("blocks."):
      _, index, rest = key.split(".", 2)
      for src, dst in _DIFFUSERS_TO_NATIVE_BLOCK:
        if rest.startswith(src):
          renamed = f"model.blocks.{index}.{dst}{rest[len(src):]}"
          break
      if renamed is None and rest == "scale_shift_table":
        renamed = f"model.blocks.{index}.modulation"
    elif key == "scale_shift_table":
      renamed = "model.head.modulation"
    else:
      for src, dst in _DIFFUSERS_TO_NATIVE:
        if key.startswith(src):
          renamed = f"model.{dst}{key[len(src):]}"
          break
    native[renamed if renamed is not None else f"model.{key}"] = value
  return native


def test_conversion_is_a_bijection_on_the_real_module_tree():
  transformer = _tiny_transformer()
  diffusers_sd = transformer.state_dict()
  native_sd = _to_native(diffusers_sd)

  # The fake checkpoint must really look like the release: native names only, `model.` prefix, and
  # exactly one norm weight per block (the diffusers `norm3` has none, so it cannot appear).
  assert all(key.startswith("model.") for key in native_sd)
  assert any("self_attn.q." in key for key in native_sd)
  assert any("norm3.weight" in key for key in native_sd)
  assert not any("norm2." in key for key in native_sd)

  converted = convert_state_dict(native_sd)
  assert set(converted) == set(diffusers_sd), (
    f"missing={sorted(set(diffusers_sd) - set(converted))[:5]} "
    f"unexpected={sorted(set(converted) - set(diffusers_sd))[:5]}")
  for key, value in diffusers_sd.items():
    assert torch.equal(converted[key], value), key


def test_convert_rejects_an_unknown_key():
  transformer = _tiny_transformer()
  native_sd = _to_native(transformer.state_dict())
  broken = {
    key.replace("model.blocks.0.self_attn.q.weight", "model.blocks.0.self_attn.q_typo.weight"):
    value
    for key, value in native_sd.items()
  }
  assert broken.keys() != native_sd.keys(), "the corruption did not apply to any key"

  with pytest.raises(KeyError, match="no diffusers mapping"):
    convert_state_dict(broken)


def test_derive_config_reads_the_released_geometry():
  # `derive_config` cross-checks against the released numbers, so it must reject a tiny checkpoint.
  native_sd = _to_native(_tiny_transformer().state_dict())
  with pytest.raises(ValueError, match="disagrees with the released value"):
    derive_config(native_sd)


def test_structure_validation_catches_a_swapped_norm():
  transformer = _tiny_transformer()
  validate_causal_structure(transformer)

  # Pretend the mapping landed the weight on the parameter-free `norm3`.
  transformer.blocks[0].norm2 = torch.nn.Identity()
  with pytest.raises(ValueError, match="norm2"):
    validate_causal_structure(transformer)
