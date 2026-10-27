"""Loader for the official FP8 Ideogram 4 checkpoint.

`ideogram-ai/ideogram-4-fp8` stores both denoisers and the text encoder as FP8 weights with one
scale per output row and a fused attention `qkv`, none of which `diffusers` maps onto the module
parameters: `from_pretrained` leaves the unmatched parameters on the meta device — a `device_map`
included — and `pipe.to` then fails on them. The helpers below rebuild the three affected
components as bfloat16 modules, which `from_pretrained` accepts as ready-made objects, and leave
the rest of the pipeline alone.
"""

import json
import os
from typing import Any, Dict, List, Optional

import torch

from .base import BaseLoader

__all__ = ["Ideogram4Loader"]

_IDEOGRAM4_FP8_MARKER = "text_encoder/config.json"
_IDEOGRAM4_FP8_MARKER_KEY = "ideogram_fp8_weight_only"
# What the checkpoint appends to a weight's own name to name the scale of that weight's rows.
_IDEOGRAM4_SCALE_SUFFIX = "_scale"
# The keyword arguments that select a revision of a Hub repository. They are forwarded to every
# lookup the conversion makes, so the marker and the shards come from the same snapshot.
_HUB_KWARGS = (
  "revision",
  "token",
  "local_files_only",
  "cache_dir",
  "proxies",
  "force_download",
)


def _ideogram4_hub_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
  """Pick the hub selectors out of one `from_pretrained` call.

  :param kwargs: Keyword arguments of the `from_pretrained` call.
  :returns: The selectors the caller set, to forward to every file lookup.
  """
  return {key: kwargs[key] for key in _HUB_KWARGS if kwargs.get(key) is not None}


def _ideogram4_checkpoint_file(
  model_path: str,
  filename: str,
  **hub_kwargs: Any,
) -> Optional[str]:
  """Resolve one checkpoint file to a local path.

  A local directory is served as-is, without a network round trip; a repository id falls back to
  `hf_hub_download`. The caller decides whether the file has to exist. A path that is absolute or
  dotted is never a repository id, so a mistyped local path keeps failing with the loader's own
  error instead of a hub one.

  :param model_path: Local checkpoint directory or HuggingFace Hub repository id.
  :param filename: File path relative to the checkpoint root.
  :param hub_kwargs: `revision`, `token`, `local_files_only` and `cache_dir` of the call.
  :returns: The local path, or `None` when a repository does not ship the file.
  """
  if (os.path.isdir(model_path) or os.path.isabs(model_path) or model_path.startswith(os.curdir)):
    return os.path.join(model_path, filename)
  from huggingface_hub import hf_hub_download
  from huggingface_hub.errors import EntryNotFoundError

  try:
    return hf_hub_download(repo_id=model_path, filename=filename, **hub_kwargs)
  except EntryNotFoundError:
    return None


def _ideogram4_optional_checkpoint_file(
  model_path: str,
  filename: str,
  **hub_kwargs: Any,
) -> Optional[str]:
  """Resolve one checkpoint file that the checkpoint is allowed not to ship.

  :param model_path: Local checkpoint directory or HuggingFace Hub repository id.
  :param filename: File path relative to the checkpoint root.
  :param hub_kwargs: `revision`, `token`, `local_files_only` and `cache_dir` of the call.
  :returns: The local path, or `None` when the file is absent.
  """
  path = _ideogram4_checkpoint_file(model_path, filename, **hub_kwargs)
  if path is None or not os.path.exists(path):
    return None
  return path


def _ideogram4_shard_paths(
  model_path: str,
  subfolder: str,
  basename: str,
  **hub_kwargs: Any,
) -> List[str]:
  """Shards of one component, in a stable order.

  The official FP8 checkpoint ships a single `{basename}.safetensors` per component, while a
  re-uploaded checkpoint may ship the sharded form listed by
  `{basename}.safetensors.index.json`. Both are supported, and the index wins when both are
  present, because it is the file that says which shards hold the weights.

  :param model_path: Local checkpoint directory or HuggingFace Hub repository id.
  :param subfolder: Component directory inside the checkpoint.
  :param basename: Weight file basename; `model` for a `transformers` component.
  :param hub_kwargs: `revision`, `token`, `local_files_only` and `cache_dir` of the call.
  :returns: Local paths of every shard, sorted by file name.
  """
  index_path = _ideogram4_optional_checkpoint_file(
    model_path, f"{subfolder}/{basename}.safetensors.index.json", **hub_kwargs)
  if index_path is None:
    return [
      _ideogram4_checkpoint_file(model_path, f"{subfolder}/{basename}.safetensors", **hub_kwargs)
    ]
  with open(index_path) as index_file:
    shard_names = json.load(index_file)["weight_map"].values()
  return [
    _ideogram4_checkpoint_file(model_path, f"{subfolder}/{shard}", **hub_kwargs)
    for shard in sorted(set(shard_names))
  ]


def _ideogram4_is_fp8(model_path: str, **hub_kwargs: Any) -> bool:
  """Whether a checkpoint is an FP8 release, according to its own text-encoder config.

  The exporter writes `ideogram_fp8_weight_only` into `text_encoder/config.json`, which is a flag a
  bfloat16 re-upload does not carry: the absence of the file or of the key means the checkpoint
  loads through the plain `from_pretrained` path.

  Only a missing file is answered with `False`. Any other hub failure stays an error: a probe that
  swallows a network problem would report every unreachable checkpoint as a bfloat16 one.

  :param model_path: Local checkpoint directory or HuggingFace Hub repository id.
  :param hub_kwargs: `revision`, `token`, `local_files_only` and `cache_dir` of the call.
  :returns: Whether the checkpoint has to be converted before it can be loaded.
  """
  from huggingface_hub.errors import EntryNotFoundError

  try:
    marker_path = _ideogram4_optional_checkpoint_file(model_path, _IDEOGRAM4_FP8_MARKER,
                                                      **hub_kwargs)
  except EntryNotFoundError:
    return False
  if marker_path is None:
    return False
  with open(marker_path) as marker_file:
    return bool(json.load(marker_file).get(_IDEOGRAM4_FP8_MARKER_KEY, False))


def _ideogram4_component_dtype(kwargs: Dict[str, Any], name: str) -> torch.dtype:
  """The dtype one component is materialized in, resolved the way `diffusers` resolves it.

  `from_pretrained` takes `dtype` and, under its deprecated spelling, `torch_dtype`; either may be
  one dtype for every component or a mapping of component name to dtype, with a `default` entry for
  the components it does not name. The rebuilt components bypass that resolution entirely, so the
  loader has to reach the same answer on its own.

  :param kwargs: Keyword arguments of the `from_pretrained` call.
  :param name: Component name, e.g. `transformer`.
  :returns: The dtype to convert the component to, bfloat16 when the call names none.
  """
  dtype = kwargs.get("dtype")
  if dtype is None:
    dtype = kwargs.get("torch_dtype")
  if dtype is None:
    return torch.bfloat16
  if isinstance(dtype, dict):
    return dtype.get(name, dtype.get("default", torch.bfloat16))
  return dtype


def _ideogram4_dequantize(weight: torch.Tensor, scale: torch.Tensor,
                          dtype: torch.dtype) -> torch.Tensor:
  """Read one FP8 weight through its per-row scale.

  The checkpoint keeps one scale per output row, so the scale multiplies along dim 0 only, and the
  product is taken in float32 to keep the FP8 values exact before the cast to the target dtype.

  :param weight: FP8 weight of shape `(out_features, in_features)`.
  :param scale: One scale per output row, of shape `(out_features,)`.
  :param dtype: Dtype the dequantized weight is materialized in.
  :returns: The weight in `dtype`.
  """
  return (weight.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)).to(dtype)


def _ideogram4_convert_tensor(
  name: str,
  weight: torch.Tensor,
  scale: Optional[torch.Tensor],
  dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
  """Map one stored tensor onto the parameter names and values the module expects.

  Two renames come on top of the dequantization: the fused `attention.qkv` weight is the `to_q`,
  `to_k` and `to_v` weights concatenated along dim 0 (their per-row scales likewise), and the
  module calls the attention output projection `to_out.0` where the checkpoint says `o`. The scale
  tensors have no counterpart in the module, so they are consumed here and never forwarded.

  :param name: The tensor's name in the checkpoint.
  :param weight: The stored weight, FP8 or already in the target dtype (biases, norms).
  :param scale: The scale stored beside the weight, or `None` for an unquantized weight.
  :param dtype: Dtype the weights are materialized in.
  :returns: One entry per module parameter the stored tensor provides.
  """
  if name.endswith(".attention.qkv.weight"):
    base = name.removesuffix(".attention.qkv.weight")
    weights = weight.chunk(3, dim=0)
    projections = ("to_q", "to_k", "to_v")
    if scale is None:
      return {
        f"{base}.attention.{projection}.weight": chunk.to(dtype)
        for projection, chunk in zip(projections, weights)
      }
    return {
      f"{base}.attention.{projection}.weight": _ideogram4_dequantize(chunk, part, dtype)
      for projection, chunk, part in zip(projections, weights, scale.chunk(3, dim=0))
    }
  if name.endswith(".attention.o.weight"):
    name = name.replace(".attention.o.weight", ".attention.to_out.0.weight")
  if scale is None:
    return {name: weight.to(dtype)}
  return {name: _ideogram4_dequantize(weight, scale, dtype)}


def _ideogram4_component_state_dict(
  model_path: str,
  subfolder: str,
  basename: str,
  dtype: torch.dtype,
  **hub_kwargs: Any,
) -> Dict[str, torch.Tensor]:
  """One component's converted weights, built one stored tensor at a time.

  A denoiser is ~9 GiB of FP8 on disk and ~18 GiB once dequantized, so the shards are never read as
  a whole: each stored tensor is converted, its FP8 source released, and only the converted copy
  kept. The converted copy still reaches `load_state_dict` in one piece, which is why the caller
  converts one component at a time and drops the result before starting the next.

  :param model_path: Local checkpoint directory or HuggingFace Hub repository id.
  :param subfolder: Component directory inside the checkpoint.
  :param basename: Weight file basename; `model` for a `transformers` component.
  :param dtype: Dtype the weights are dequantized to.
  :param hub_kwargs: `revision`, `token`, `local_files_only` and `cache_dir` of the call.
  :returns: Converted state dict, keyed by the module's own parameter names.
  """
  from safetensors import safe_open

  shard_paths = _ideogram4_shard_paths(model_path, subfolder, basename, **hub_kwargs)
  # Scales are looked up by name, and a scale is not guaranteed to sit in the same shard as the
  # weight it belongs to.
  scale_shards: Dict[str, str] = {}
  for shard_path in shard_paths:
    with safe_open(shard_path, framework="pt") as handle:
      for name in handle.keys():
        if name.endswith(_IDEOGRAM4_SCALE_SUFFIX):
          scale_shards[name] = shard_path

  state_dict: Dict[str, torch.Tensor] = {}
  for shard_path in shard_paths:
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
      stored = set(handle.keys())
      for name in sorted(stored):
        if name.endswith(_IDEOGRAM4_SCALE_SUFFIX):
          continue
        scale_name = f"{name}{_IDEOGRAM4_SCALE_SUFFIX}"
        if scale_name in stored:
          scale = handle.get_tensor(scale_name)
        elif scale_name in scale_shards:
          with safe_open(scale_shards[scale_name], framework="pt", device="cpu") as scale_handle:
            scale = scale_handle.get_tensor(scale_name)
        else:
          scale = None
        state_dict.update(_ideogram4_convert_tensor(name, handle.get_tensor(name), scale, dtype))
  return state_dict


def _ideogram4_from_config_with_dtype(module_class: Any, config: Any, dtype: torch.dtype) -> Any:
  """Instantiate a `diffusers` module whose temporary parameters already have `dtype`.

  `from_config` has no dtype argument, and the freshly initialized parameters are replaced by the
  converted weights right afterwards, so initializing them in float32 would cost a second copy of
  the model for nothing. A global default dtype is the only handle `diffusers` offers here; it is
  restored before this function returns.

  :param module_class: The `ModelMixin` subclass to instantiate.
  :param config: Its config, as returned by `load_config`.
  :param dtype: Dtype the module is initialized in.
  :returns: The instantiated module, with its parameters not yet loaded.
  """
  previous_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    return module_class.from_config(config)
  finally:
    torch.set_default_dtype(previous_dtype)


def _ideogram4_bf16_transformer(
  model_path: str,
  subfolder: str,
  dtype: torch.dtype,
  **hub_kwargs: Any,
) -> Any:
  """Materialize one of the two denoisers from an FP8 checkpoint.

  The module is built from the checkpoint's own `config.json` first, because the converted state
  dict has to be assigned into an existing module. `assign=True` adopts the converted tensors
  instead of copying into the initialized ones, and `strict=True` proves the mapping covered every
  parameter, so neither an unhandled rename nor a leftover FP8 weight can pass silently.

  :param model_path: Local checkpoint directory or HuggingFace Hub repository id.
  :param subfolder: `transformer` or `unconditional_transformer`.
  :param dtype: Dtype the weights are dequantized to.
  :param hub_kwargs: `revision`, `token`, `local_files_only` and `cache_dir` of the call.
  :returns: The denoiser, in eval mode.
  """
  from diffusers import Ideogram4Transformer2DModel

  transformer = _ideogram4_from_config_with_dtype(
    Ideogram4Transformer2DModel,
    Ideogram4Transformer2DModel.load_config(model_path, subfolder=subfolder, **hub_kwargs),
    dtype,
  )
  transformer.load_state_dict(
    _ideogram4_component_state_dict(model_path, subfolder, "diffusion_pytorch_model", dtype,
                                    **hub_kwargs),
    assign=True,
    strict=True,
  )
  return transformer.eval()


def _ideogram4_bf16_text_encoder(model_path: str, dtype: torch.dtype, **hub_kwargs: Any) -> Any:
  """Materialize the text encoder from an FP8 checkpoint.

  The encoder is a plain `transformers` model, so the same two steps apply: build it from its
  config, then assign the dequantized weights. Its FP8 weights carry per-row scales exactly like the
  denoisers, but none of the renames, since `Qwen3VLModel` already stores `q_proj`/`k_proj`/`v_proj`
  separately.

  :param model_path: Local checkpoint directory or HuggingFace Hub repository id.
  :param dtype: Dtype the weights are dequantized to.
  :param hub_kwargs: `revision`, `token`, `local_files_only` and `cache_dir` of the call.
  :returns: The text encoder, in eval mode.
  """
  from transformers import AutoConfig, AutoModel

  # `trust_remote_code` because the official checkpoint ships the encoder through an `auto_map`.
  config = AutoConfig.from_pretrained(
    model_path,
    subfolder="text_encoder",
    trust_remote_code=True,
    **hub_kwargs,
  )
  text_encoder = AutoModel.from_config(config, trust_remote_code=True, dtype=dtype)
  text_encoder.load_state_dict(
    _ideogram4_component_state_dict(model_path, "text_encoder", "model", dtype, **hub_kwargs),
    assign=True,
    strict=True,
  )
  return text_encoder.eval()


class Ideogram4Loader(BaseLoader):
  """Convert the official FP8 Ideogram 4 checkpoint while `from_pretrained` loads it.

  `apply` patches `Ideogram4Pipeline.from_pretrained`, which covers the two ways the pipeline is
  built in practice: the direct call, and `AutoPipeline*.from_pretrained`, which resolves
  `Ideogram4Pipeline` from the checkpoint's `model_index.json` and then delegates to it. A bare
  `DiffusionPipeline.from_pretrained` is *not* covered: it resolves its class the same way but
  instantiates it through its own loader, so the patched subclass is never called.

  A checkpoint without the `ideogram_fp8_weight_only` marker in `text_encoder/config.json` takes the
  original path untouched: no component is read, converted or injected, and the call is forwarded
  exactly as it was made.
  """

  def _load(self, path: str, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rebuild the three FP8 components, or decline to.

    The dtype of each component is resolved separately, and a component the caller passed itself is
    left alone — an explicit `None` means "do not load this one" and stays `None`.

    :param path: The checkpoint path or repository id of the call.
    :param kwargs: Keyword arguments of the `from_pretrained` call.
    :returns: The three rebuilt components, keyed like the pipeline's own parameters, or `None` for
      a checkpoint that is not an FP8 release.
    """
    hub_kwargs = _ideogram4_hub_kwargs(kwargs)
    if not _ideogram4_is_fp8(path, **hub_kwargs):
      return None
    if kwargs.get("dtype") is not None and kwargs.get("torch_dtype") is not None:
      raise ValueError(
        "Pass either `dtype` or `torch_dtype`, not both: diffusers rejects the combination for "
        "every pipeline, and rejecting it here avoids converting the FP8 weights first.")
    # Keep the whole pipeline in one dtype: without this, the converted components would be bf16
    # while diffusers loaded the remaining components in their checkpoint dtype.
    if kwargs.get("dtype") is None and kwargs.get("torch_dtype") is None:
      kwargs["dtype"] = torch.bfloat16
    components: Dict[str, Any] = {}
    missing = [
      name for name in ("transformer", "unconditional_transformer", "text_encoder")
      if name not in kwargs
    ]
    # Refused before a single weight is read: `from_pretrained` only places the components it loads
    # itself, so the rebuilt ones would stay on the host and fail at the first forward pass, long
    # after the conversion was paid for.
    if missing and kwargs.get("device_map") is not None:
      raise ValueError(
        "device_map is not supported for an FP8 Ideogram 4 checkpoint: the converted components "
        "are built on the host and have to be placed by `pipeline.to(device)` afterwards, which a "
        "device_map replaces. Load the pipeline without `device_map` instead.")
    # One component at a time: the converted state dict of the previous one is released before the
    # next one is read, so the peak stays at one dequantized component instead of three.
    for name in missing:
      dtype = _ideogram4_component_dtype(kwargs, name)
      if name == "text_encoder":
        components[name] = _ideogram4_bf16_text_encoder(path, dtype, **hub_kwargs)
      else:
        components[name] = _ideogram4_bf16_transformer(path, name, dtype, **hub_kwargs)
    return components
