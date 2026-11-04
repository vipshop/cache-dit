"""Convert the released LongLive 2.0 generator to a diffusers Wan checkpoint.

The release (`Efficient-Large-Model/LongLive-2.0-5B`) ships a single native Wan state dict,
`model_bf16.pt`, whose keys follow the training code (`model.blocks.N.self_attn.q...`). The diffusers
`WanTransformer3DModel` parameter tree is the same, only the names differ, so this module is a pure
key mapping plus an assembly step that drops the Wan2.2-TI2V-5B text encoder / tokenizer / VAE /
scheduler next to the converted transformer.

Two details are worth writing down, because a silent rename here produces a plausible-looking but
wrong video rather than an error:

* the native `blocks.N.norm3` is the **affine** pre-cross-attention norm, i.e. the diffusers
  `blocks.N.norm2` (`cross_attn_norm=True`). The diffusers `blocks.N.norm3` has
  `elementwise_affine=False` and therefore owns no parameters at all, which is why no native key maps
  to it;
* `ffn.0` / `ffn.2` are the two linears of diffusers' `FeedForward`, i.e. `ffn.net.0.proj` (inside the
  `GELU` module) and `ffn.net.2`.

Usage:

```bash
python -m cache_dit.models.pipelines.longlive2.convert \
    --ckpt /path/LongLive-2.0-5B-native/model_bf16.pt \
    --base /path/Wan2.2-TI2V-5B-Diffusers \
    --out  /path/LongLive-2.0-5B
```
"""

import argparse
import json
import os
import re
import shutil
from typing import Dict, Optional, Tuple

import torch

from ....logger import init_logger

logger = init_logger(__name__)

# The released configuration, cross-checked against the state dict by `derive_config`.
RELEASED_CONFIG: Dict[str, object] = {
  "patch_size": [1, 2, 2],
  "num_attention_heads": 24,
  "attention_head_dim": 128,
  "in_channels": 48,
  "out_channels": 48,
  "ffn_dim": 14336,
  "num_layers": 30,
  "cross_attn_norm": True,
  "qk_norm": "rms_norm_across_heads",
  "eps": 1e-6,
  "freq_dim": 256,
  "text_dim": 4096,
  "rope_max_seq_len": 1024,
}

# Components taken from the Wan2.2-TI2V-5B diffusers checkpoint: the generator only replaces the
# transformer, and keeping the stock modules is what keeps TE-P / VAE-P dispatching. The second item
# is the file that proves a component directory is complete - a half-downloaded directory holds the
# configs but none of the weights, and would otherwise be linked in silently.
BASE_COMPONENTS = (
  ("text_encoder", "model.safetensors.index.json"),
  ("tokenizer", "tokenizer_config.json"),
  ("vae", "diffusion_pytorch_model.safetensors"),
  ("scheduler", "scheduler_config.json"),
)

_QKV_OUT = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}

_KEY_PATTERNS: Tuple[Tuple[str, str], ...] = (
  (r"^model\.patch_embedding\.(weight|bias)$", r"patch_embedding.\1"),
  (r"^model\.text_embedding\.0\.(weight|bias)$", r"condition_embedder.text_embedder.linear_1.\1"),
  (r"^model\.text_embedding\.2\.(weight|bias)$", r"condition_embedder.text_embedder.linear_2.\1"),
  (r"^model\.time_embedding\.0\.(weight|bias)$", r"condition_embedder.time_embedder.linear_1.\1"),
  (r"^model\.time_embedding\.2\.(weight|bias)$", r"condition_embedder.time_embedder.linear_2.\1"),
  (r"^model\.time_projection\.1\.(weight|bias)$", r"condition_embedder.time_proj.\1"),
  (r"^model\.head\.modulation$", "scale_shift_table"),
  (r"^model\.head\.head\.(weight|bias)$", r"proj_out.\1"),
)


def _map_block_key(name: str) -> Optional[str]:
  """Map one `model.blocks.<N>.*` key to its diffusers name.

  :param name: Native key.
  :returns: The diffusers key, or `None` when the name is not a block key.
  """
  match = re.match(r"^model\.blocks\.(\d+)\.(.*)$", name)
  if match is None:
    return None
  index, rest = match.group(1), match.group(2)
  if rest == "modulation":
    return f"blocks.{index}.scale_shift_table"
  sub = re.match(r"^(self_attn|cross_attn)\.(q|k|v|o|norm_q|norm_k)\.(weight|bias)$", rest)
  if sub is not None:
    stream = "attn1" if sub.group(1) == "self_attn" else "attn2"
    leaf = _QKV_OUT.get(sub.group(2), sub.group(2))
    return f"blocks.{index}.{stream}.{leaf}.{sub.group(3)}"
  sub = re.match(r"^norm3\.(weight|bias)$", rest)
  if sub is not None:
    # Affine pre-cross-attention norm; the diffusers non-affine `norm3` owns no parameters.
    return f"blocks.{index}.norm2.{sub.group(1)}"
  sub = re.match(r"^ffn\.0\.(weight|bias)$", rest)
  if sub is not None:
    return f"blocks.{index}.ffn.net.0.proj.{sub.group(1)}"
  sub = re.match(r"^ffn\.2\.(weight|bias)$", rest)
  if sub is not None:
    return f"blocks.{index}.ffn.net.2.{sub.group(1)}"
  return None


def map_key(name: str) -> Optional[str]:
  """Map one native key to its diffusers name.

  :param name: Native state-dict key.
  :returns: The diffusers key, or `None` when the key is unknown.
  """
  for pattern, replacement in _KEY_PATTERNS:
    if re.match(pattern, name):
      return re.sub(pattern, replacement, name)
  return _map_block_key(name)


def find_state_dict(obj, depth: int = 0):
  """Locate the transformer state dict inside a loaded checkpoint.

  :param obj: A loaded `torch.load` object.
  :param depth: Recursion depth, capped to keep the search bounded.
  :returns: A mapping of tensor names to tensors.
  """
  if isinstance(obj, dict):
    if any(isinstance(k, str) and k.startswith(("model.blocks.0", "blocks.0")) for k in obj):
      return obj
    if depth >= 3:
      return None
    for value in obj.values():
      found = find_state_dict(value, depth + 1)
      if found is not None:
        return found
  return None


def convert_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
  """Rename a native generator state dict to the diffusers Wan names.

  :param state: Native state dict.
  :returns: The converted state dict.
  :raises KeyError: If a native key has no mapping or two keys collide.
  """
  converted: Dict[str, torch.Tensor] = {}
  unmapped = []
  for name, tensor in state.items():
    target = map_key(name)
    if target is None:
      unmapped.append(name)
      continue
    if target in converted:
      raise KeyError(f"Two native keys map to `{target}`.")
    converted[target] = tensor
  if unmapped:
    raise KeyError(
      f"{len(unmapped)} native keys have no diffusers mapping, e.g. {sorted(unmapped)[:8]}. Add "
      f"them to `_map_block_key` / `_KEY_PATTERNS` before converting.")
  return converted


def derive_config(state: Dict[str, torch.Tensor]) -> Dict[str, object]:
  """Read the released hyper-parameters from the state dict and check them against the release.

  :param state: Native state dict.
  :returns: A `WanTransformer3DModel` configuration dict.
  :raises ValueError: If a derived value disagrees with `RELEASED_CONFIG`.
  """
  config = dict(RELEASED_CONFIG)
  q_weight = state["model.blocks.0.self_attn.q.weight"]
  heads = int(config["num_attention_heads"])
  config["attention_head_dim"] = int(q_weight.shape[0] // heads)
  config["in_channels"] = int(state["model.patch_embedding.weight"].shape[1])
  config["out_channels"] = int(
    state["model.head.head.weight"].shape[0] //
    (config["patch_size"][0] * config["patch_size"][1] * config["patch_size"][2]))
  config["ffn_dim"] = int(state["model.blocks.0.ffn.0.weight"].shape[0])
  config["text_dim"] = int(state["model.text_embedding.0.weight"].shape[1])
  layers = {int(m.group(1)) for m in (re.match(r"^model\.blocks\.(\d+)\.", k) for k in state) if m}
  config["num_layers"] = len(layers)
  if layers != set(range(len(layers))):
    raise ValueError(f"Block indices are not contiguous: {sorted(layers)[:5]}...")
  for key in ("num_attention_heads", "ffn_dim", "in_channels", "out_channels", "num_layers",
              "text_dim", "attention_head_dim"):
    if config[key] != RELEASED_CONFIG[key]:
      raise ValueError(
        f"Derived `{key}`={config[key]} disagrees with the released value "
        f"{RELEASED_CONFIG[key]}; the checkpoint is not the released LongLive-2.0-5B generator.")
  return config


def validate_causal_structure(transformer) -> None:
  """Assert the block tree still matches the checkpoint's norm/attention layout.

  The native checkpoint carries exactly one norm weight per block. If the mapping ever put it on the
  diffusers non-affine `norm3`, a strict load would fail - but a `strict=False` load would silently
  drop it and every frame would be modulated by an un-normalised residual. This check makes the
  direction explicit instead of relying on the loader.

  :param transformer: A `LongLive2Transformer3DModel` instance.
  :raises ValueError: If the norms or the attention processors do not match the release.
  """
  from .transformer import LongLive2CausalAttnProcessor

  for index, block in enumerate(transformer.blocks):
    if getattr(block.norm2, "weight", None) is None:
      raise ValueError(
        f"block {index}: `norm2` is not affine. The released LongLive 2.0 sets `cross_attn_norm=True` "
        f"and the native `blocks.N.norm3` maps onto this module; a parameter-free `norm2` means the "
        f"norm mapping was inverted.")
    if getattr(block.norm3, "weight", None) is not None:
      raise ValueError(f"block {index}: `norm3` is affine, but the release keeps it parameter-free "
                       f"(`elementwise_affine=False`). Check the norm mapping.")
    if not isinstance(block.attn1.processor, LongLive2CausalAttnProcessor):
      raise ValueError(f"block {index}: self-attention does not use the causal KV-cache processor.")
    if block.index != index:
      raise ValueError(
        f"block {index}: `index` is {block.index}; the KV cache entry would be wrong.")


def build_longlive2_transformer(
  ckpt_path: str,
  dtype: torch.dtype = torch.bfloat16,
):
  """Build the transformer and load the native checkpoint with `strict=True`.

  :param ckpt_path: Path of the released `model_bf16.pt`, or of an already converted directory.
  :param dtype: Weight dtype.
  :returns: The loaded `LongLive2Transformer3DModel`.
  """
  from .transformer import LongLive2Transformer3DModel

  if os.path.isdir(ckpt_path):
    return LongLive2Transformer3DModel.from_pretrained(ckpt_path, torch_dtype=dtype)

  obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  state = find_state_dict(obj)
  if state is None:
    raise ValueError(f"No transformer state dict found inside {ckpt_path}.")
  config = derive_config(state)
  converted = convert_state_dict(state)
  with torch.device("meta"):
    transformer = LongLive2Transformer3DModel(**config)
  missing, unexpected = transformer.load_state_dict(converted, strict=False, assign=True)
  if missing or unexpected:
    raise RuntimeError(
      f"Converted checkpoint does not match the module tree: missing={list(missing)[:8]} "
      f"unexpected={list(unexpected)[:8]}.")
  return transformer.to(dtype)


def assemble_pipeline_dir(
  ckpt_path: str,
  out_dir: str,
  base_dir: str,
  dtype: torch.dtype = torch.bfloat16,
  copy_components: bool = False,
) -> str:
  """Write a complete diffusers-style directory for `LongLive2Pipeline.from_pretrained`.

  :param ckpt_path: Path of the released `model_bf16.pt`.
  :param out_dir: Destination directory.
  :param base_dir: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` directory holding the non-transformer parts.
  :param dtype: Weight dtype of the saved transformer.
  :param copy_components: Copy the base components instead of symlinking them.
  :returns: `out_dir`.
  """
  transformer = build_longlive2_transformer(ckpt_path, dtype=dtype)
  os.makedirs(out_dir, exist_ok=True)
  transformer.save_pretrained(os.path.join(out_dir, "transformer"))
  for component, marker in BASE_COMPONENTS:
    source = os.path.join(base_dir, component)
    if not os.path.isfile(os.path.join(source, marker)):
      raise FileNotFoundError(
        f"`{base_dir}/{component}` is missing `{marker}`; point `--base` at a complete "
        f"Wan-AI/Wan2.2-TI2V-5B-Diffusers checkout.")
    target = os.path.join(out_dir, component)
    if os.path.exists(target) or os.path.islink(target):
      if os.path.islink(target):
        continue
      if os.path.isfile(os.path.join(target, marker)):
        continue
      raise FileExistsError(
        f"`{target}` exists but is incomplete (no `{marker}`), probably a half-finished download. "
        f"Remove it and re-run so it can be linked to the Wan2.2 components.")
    if copy_components:
      shutil.copytree(source, target)
    else:
      os.symlink(os.path.abspath(source), target)
  model_index = {
    "_class_name": "LongLive2Pipeline",
    "_diffusers_version": "0.42.0.dev0",
    "_name_or_path": "",
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["cache_dit.models.pipelines.longlive2", "LongLive2Transformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
    "expand_timesteps": True,
  }
  with open(os.path.join(out_dir, "model_index.json"), "w") as handle:
    json.dump(model_index, handle, indent=2)
  logger.info(f"Assembled the LongLive 2.0 pipeline directory at {out_dir}.")
  return out_dir


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ckpt", required=True, help="Released `model_bf16.pt`.")
  parser.add_argument("--base", required=True, help="Wan2.2-TI2V-5B-Diffusers directory.")
  parser.add_argument("--out", required=True, help="Destination pipeline directory.")
  parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
  parser.add_argument("--copy-components", action="store_true")
  args = parser.parse_args()
  dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
  assemble_pipeline_dir(args.ckpt,
                        args.out,
                        args.base,
                        dtype=dtype,
                        copy_components=args.copy_components)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
