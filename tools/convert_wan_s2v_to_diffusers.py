"""Convert the native Wan2.2-S2V checkpoint into the layout cache-dit loads.

The official release (`Wan-AI/Wan2.2-S2V-14B`) ships Wan's original module tree plus a VAE `.pth`
and an umT5 `.pth`. cache-dit owns a diffusers-style implementation of the transformer
(:class:`cache_dit.models.pipelines.wan_s2v.transformer.WanS2VTransformer3DModel`), so the
transformer weights have to be renamed once; VAE / umT5 / tokenizer / scheduler are taken from an
already converted diffusers repo (Wan2.2-T2V-A14B is enough: same 16-channel VAE, same umT5-XXL).

Usage::

  python tools/convert_wan_s2v_to_diffusers.py \
    --native-dir ~/hf_models/Wan2.2-S2V-14B \
    --components-dir ~/hf_models/Wan2.2-T2V-A14B-Diffusers \
    --output-dir ~/hf_models/Wan2.2-S2V-14B-Diffusers

The script is a one-off utility, not a runtime dependency: the produced directory is what
`WanSpeechToVideoPipeline.from_pretrained` consumes.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from cache_dit.models.pipelines.wan_s2v.transformer import WanS2VTransformer3DModel

# Wan's original key -> the vendored module tree. Ordered: the first matching rule wins, so the
# attention sub-layers come before their `self_attn.` / `cross_attn.` prefix rule.
_RENAME_RULES = [
  (r"^time_embedding\.0\.", "condition_embedder.time_embedder.linear_1."),
  (r"^time_embedding\.2\.", "condition_embedder.time_embedder.linear_2."),
  (r"^text_embedding\.0\.", "condition_embedder.text_embedder.linear_1."),
  (r"^text_embedding\.2\.", "condition_embedder.text_embedder.linear_2."),
  (r"^time_projection\.1\.", "condition_embedder.time_proj."),
  (r"^head\.modulation$", "scale_shift_table"),
  (r"^head\.head\.", "proj_out."),
  (r"^cond_encoder\.", "condition_embedder.pose_embedder."),
  (r"^trainable_cond_mask\.", "trainable_condition_mask."),
  # The audio bridge keeps Wan's typo (`casual`), and conv2/conv3 nest one `conv` deeper because
  # they are wrapped in a causal-conv layer.
  (r"^casual_audio_encoder\.weights$",
   "condition_embedder.causal_audio_encoder.weighted_avg.weights"),
  (r"^casual_audio_encoder\.encoder\.conv([23])\.conv\.",
   r"condition_embedder.causal_audio_encoder.encoder.conv\1.conv.conv."),
  (r"^casual_audio_encoder\.encoder\.", "condition_embedder.causal_audio_encoder.encoder."),
  (r"^audio_injector\.injector\.(\d+)\.q\.", r"audio_injector.injector.\1.to_q."),
  (r"^audio_injector\.injector\.(\d+)\.k\.", r"audio_injector.injector.\1.to_k."),
  (r"^audio_injector\.injector\.(\d+)\.v\.", r"audio_injector.injector.\1.to_v."),
  (r"^audio_injector\.injector\.(\d+)\.o\.", r"audio_injector.injector.\1.to_out.0."),
  (r"^blocks\.(\d+)\.modulation$", r"blocks.\1.scale_shift_table"),
  # Wan's `norm3` is the affine cross-attention norm, which is `norm2` in the diffusers layout.
  (r"^blocks\.(\d+)\.norm3\.", r"blocks.\1.norm2."),
  (r"^blocks\.(\d+)\.self_attn\.q\.", r"blocks.\1.attn1.to_q."),
  (r"^blocks\.(\d+)\.self_attn\.k\.", r"blocks.\1.attn1.to_k."),
  (r"^blocks\.(\d+)\.self_attn\.v\.", r"blocks.\1.attn1.to_v."),
  (r"^blocks\.(\d+)\.self_attn\.o\.", r"blocks.\1.attn1.to_out.0."),
  (r"^blocks\.(\d+)\.self_attn\.", r"blocks.\1.attn1."),
  (r"^blocks\.(\d+)\.cross_attn\.q\.", r"blocks.\1.attn2.to_q."),
  (r"^blocks\.(\d+)\.cross_attn\.k\.", r"blocks.\1.attn2.to_k."),
  (r"^blocks\.(\d+)\.cross_attn\.v\.", r"blocks.\1.attn2.to_v."),
  (r"^blocks\.(\d+)\.cross_attn\.o\.", r"blocks.\1.attn2.to_out.0."),
  (r"^blocks\.(\d+)\.cross_attn\.", r"blocks.\1.attn2."),
  (r"^blocks\.(\d+)\.ffn\.0\.", r"blocks.\1.ffn.net.0.proj."),
  (r"^blocks\.(\d+)\.ffn\.2\.", r"blocks.\1.ffn.net.2."),
]

_TRANSFORMER_CONFIG = {
  "patch_size": [1, 2, 2],
  "num_attention_heads": 40,
  "attention_head_dim": 128,
  "in_channels": 16,
  "out_channels": 16,
  "text_dim": 4096,
  "freq_dim": 256,
  "audio_dim": 1024,
  "audio_inject_layers": [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
  "enable_adain": True,
  "adain_mode": "attn_norm",
  "pose_dim": 16,
  "ffn_dim": 13824,
  "num_layers": 40,
  "num_weighted_avg_layers": 25,
  "cross_attn_norm": True,
  "qk_norm": "rms_norm_across_heads",
  "eps": 1e-6,
  "added_kv_proj_dim": None,
  "rope_max_seq_len": 1024,
  "enable_framepack": True,
  "framepack_drop_mode": "padd",
  "add_last_motion": True,
  "zero_timestep": True,
}


def rename_key(key: str) -> str:
  """Map one native Wan key onto the vendored module tree.

  :param key: key from the released checkpoint.
  :returns: the key used by :class:`WanS2VTransformer3DModel`.
  """
  for pattern, replacement in _RENAME_RULES:
    key = re.sub(pattern, replacement, key)
  return key


def convert_transformer(native_dir: Path, output_dir: Path, dtype: torch.dtype) -> None:
  """Rename and save the transformer weights.

  :param native_dir: directory holding `diffusion_pytorch_model*.safetensors` and the index.
  :param output_dir: directory the converted `transformer/` is written to.
  :param dtype: dtype of the saved weights.
  """
  index_path = native_dir / "diffusion_pytorch_model.safetensors.index.json"
  index = json.loads(index_path.read_text())
  shards = sorted(set(index["weight_map"].values()))

  state_dict = {}
  for shard in shards:
    state_dict.update(load_file(str(native_dir / shard)))

  renamed = {}
  for key, value in state_dict.items():
    renamed[rename_key(key)] = value.to(dtype)

  model = WanS2VTransformer3DModel(**_TRANSFORMER_CONFIG)
  model = model.to(dtype)
  missing, unexpected = model.load_state_dict(renamed, strict=False)
  if missing or unexpected:
    raise RuntimeError(f"state-dict mismatch, missing={missing[:8]} unexpected={unexpected[:8]}")

  transformer_dir = output_dir / "transformer"
  transformer_dir.mkdir(parents=True, exist_ok=True)
  model.save_pretrained(transformer_dir, safe_serialization=True)
  del model, state_dict, renamed
  torch.cuda.empty_cache()
  print(f"transformer: {len(index['weight_map'])} keys -> {transformer_dir}")


def copy_components(components_dir: Path, output_dir: Path) -> None:
  """Copy VAE / umT5 / tokenizer / scheduler from an already converted diffusers repo.

  :param components_dir: e.g. `Wan2.2-T2V-A14B-Diffusers`.
  :param output_dir: target directory.
  """
  for sub in ("vae", "text_encoder", "tokenizer", "scheduler"):
    src = components_dir / sub
    dst = output_dir / sub
    if dst.exists():
      shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"{sub}: {src} -> {dst}")

  # Wan2.2-S2V uses the Wan2.1 flow-matching shift of 5.0 (the T2V repo ships a different one).
  # `use_flow_sigmas` / `prediction_type` are rewritten together with the shift on purpose: without
  # them UniPC runs epsilon prediction on a flow-matching model and produces noise instead of
  # failing, so the manifest is asserted rather than trusted.
  scheduler_config = output_dir / "scheduler" / "scheduler_config.json"
  config = json.loads(scheduler_config.read_text())
  config.update({
    "_class_name": "UniPCMultistepScheduler",
    "flow_shift": 5.0,
    "use_flow_sigmas": True,
    "prediction_type": "flow_prediction",
    "final_sigmas_type": "zero",
  })
  scheduler_config.write_text(json.dumps(config, indent=2))

  from diffusers import AutoencoderKLWan

  vae_config = json.loads((output_dir / "vae" / "config.json").read_text())
  # 16 channels is the Wan2.1 VAE; the Wan2.2 TI2V line ships a 48-channel one and the S2V
  # transformer's patch embedding would silently reshape against the wrong channel count.
  if vae_config.get("z_dim") != 16:
    raise ValueError(f"expected the 16-channel Wan2.1 VAE, got z_dim="
                     f"{vae_config.get('z_dim')} from {components_dir / 'vae'}")
  vae = AutoencoderKLWan.from_pretrained(output_dir / "vae")
  if vae.config.latents_mean is None or len(vae.config.latents_mean) != 16:
    raise ValueError("the VAE must carry the 16-channel `latents_mean` / `latents_std`")
  print("scheduler: flow_shift=5.0 use_flow_sigmas=True; vae: z_dim=16 (checked)")


def convert_audio_encoder(native_dir: Path, output_dir: Path) -> None:
  """Split the CTC checkpoint into a `Wav2Vec2Model` backbone and its feature extractor.

  :param native_dir: directory holding `wav2vec2-large-xlsr-53-english/`.
  :param output_dir: target directory.
  """
  source = native_dir / "wav2vec2-large-xlsr-53-english"
  ctc = Wav2Vec2ForCTC.from_pretrained(source, torch_dtype=torch.float32)
  ctc.wav2vec2.save_pretrained(output_dir / "audio_encoder")

  feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(source)
  feature_extractor.save_pretrained(output_dir / "audio_processor")
  print(f"audio_encoder / audio_processor: {source}")


def write_model_index(output_dir: Path) -> None:
  """Write the component manifest (`WanSpeechToVideoPipeline` loads it via its own from_pretrained).

  :param output_dir: target directory.
  """
  model_index = {
    "_class_name": "WanSpeechToVideoPipeline",
    "_diffusers_version": "0.42.0.dev0",
    "audio_encoder": ["transformers", "Wav2Vec2Model"],
    "audio_processor": ["transformers", "Wav2Vec2FeatureExtractor"],
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["cache_dit", "WanS2VTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
  }
  (output_dir / "model_index.json").write_text(json.dumps(model_index, indent=2))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--native-dir", required=True, type=Path, help="Wan2.2-S2V-14B directory.")
  parser.add_argument("--components-dir",
                      required=True,
                      type=Path,
                      help="converted Wan2.2-T2V-A14B-Diffusers.")
  parser.add_argument("--output-dir", required=True, type=Path, help="output directory.")
  parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
  args = parser.parse_args()

  dtype = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32
  }[args.dtype]
  args.output_dir.mkdir(parents=True, exist_ok=True)

  convert_transformer(args.native_dir, args.output_dir, dtype)
  copy_components(args.components_dir, args.output_dir)
  convert_audio_encoder(args.native_dir, args.output_dir)
  write_model_index(args.output_dir)
  print(f"done: {args.output_dir}")


if __name__ == "__main__":
  main()
