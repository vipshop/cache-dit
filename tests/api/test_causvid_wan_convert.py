"""CPU checks for the CausVid checkpoint conversion.

The released student is a native Wan state dict while the vendored transformer keeps the diffusers
module tree, so the mapping has to be an exact bijection. The `norm2`/`norm3` swap is the dangerous
part: the two trees use the names in **opposite** order, and on the T2V 1.3B checkpoint
(`cross_attn_norm=False`) both norms are parameter-free, so a wrong swap still loads with
`strict=True`. The test therefore builds a model with `cross_attn_norm=True`, where only one of the
two norms carries weights, which makes the direction observable.

Usage:
  python -m pytest tests/api/test_causvid_wan_convert.py -q
"""

import pytest
import torch

from cache_dit.models.pipelines.causvid.convert import (
  convert_causvid_state_dict,
  load_causvid_state_dict,
  validate_causal_structure,
)
from cache_dit.models.pipelines.causvid.transformer import CausVidTransformer3DModel

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

# Independent, literal description of the native Wan module tree, used to build a fake checkpoint.
_DIFFUSERS_TO_NATIVE = (
  ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
  ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
  ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
  ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
  ("condition_embedder.time_proj.", "time_projection.1."),
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
  # swapped on purpose: diffusers `norm2` is the cross-attention norm, native `norm3` is
  ("norm2.", "norm3."),
  ("norm3.", "norm2."),
)


def _tiny_transformer() -> CausVidTransformer3DModel:
  torch.manual_seed(0)
  return CausVidTransformer3DModel(**TINY_CONFIG).eval()


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
  from cache_dit.models.pipelines.causvid.convert import CAUSVID_1_3B_CONFIG  # noqa: F401

  transformer = _tiny_transformer()
  diffusers_sd = transformer.state_dict()
  native_sd = _to_native(diffusers_sd)

  # The fake checkpoint must really look like the release: native names only, `model.` prefix.
  assert all(key.startswith("model.") for key in native_sd)
  assert any("self_attn.q." in key for key in native_sd)
  assert any("norm3.weight" in key for key in native_sd)

  converted = convert_causvid_state_dict(native_sd)
  assert set(converted) == set(diffusers_sd), (
    f"missing={sorted(set(diffusers_sd) - set(converted))[:5]} "
    f"unexpected={sorted(set(converted) - set(diffusers_sd))[:5]}")
  for key, value in diffusers_sd.items():
    assert torch.equal(converted[key], value), key


def test_load_is_strict_and_reproduces_the_weights():
  transformer = _tiny_transformer()
  native_sd = _to_native(transformer.state_dict())

  target = CausVidTransformer3DModel(**TINY_CONFIG).eval()
  with torch.no_grad():
    for param in target.parameters():
      param.zero_()
  load_causvid_state_dict(target, native_sd)

  for (name, expected), (_, actual) in zip(transformer.state_dict().items(),
                                           target.state_dict().items()):
    assert torch.equal(expected, actual), name


def test_load_rejects_a_renamed_projection():
  transformer = _tiny_transformer()
  native_sd = _to_native(transformer.state_dict())
  broken = {
    key.replace("model.blocks.0.self_attn.q.weight", "model.blocks.0.self_attn.q_typo.weight"):
    value
    for key, value in native_sd.items()
  }
  assert broken.keys() != native_sd.keys(), "the corruption did not apply to any key"

  with pytest.raises(RuntimeError, match="does not match"):
    load_causvid_state_dict(transformer, broken)


def test_structure_validation_catches_a_swapped_norm():
  transformer = _tiny_transformer()
  validate_causal_structure(transformer)  # cross_attn_norm=True path

  # Pretend the two norms were swapped: the affine cross-attention norm must not sit on `norm3`.
  transformer.blocks[0].norm2 = torch.nn.Identity()
  with pytest.raises(ValueError, match="norm3"):
    validate_causal_structure(transformer)
