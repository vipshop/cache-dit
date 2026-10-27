"""Built-in MOSS-Image text-to-image pipeline.

Vendored from the MOSS-Image release (`pipeline_moss_image.py`, Apache-2.0, Copyright OpenMOSS
Team) because diffusers does not ship it: the checkpoint's `model_index.json` points `transformer`
and `scheduler` at its own `pipeline_moss_image` module, so a plain
`DiffusionPipeline.from_pretrained` cannot import them. Keeping one copy here lets cache-dit run
`MOSS-Image-SFT` / `MOSS-Image-Base` without `trust_remote_code` or a remote `custom_pipeline`.

Upstream defaults: bf16 DiT/text-encoder weights, fp32 RoPE frequencies and phase/sin/cos
calculation (RoPE is then cast to the activation dtype), initial Gaussian noise sampled in fp32,
CFG and Euler latent updates in fp32, fp32 VAE decoding, scheduler shift 5.0. Default guidance uses
CFG 3 with `uncond_skip_layers=(22,)`, i.e. two DiT forwards per sampling step; optional SLG adds a
third one.

Note:
  Keep the constraints below intact, cache-dit's dispatch depends on them:

  * :class:`MOSSImagePipeline` must inherit `DiffusionPipeline` and
    :class:`MOSSImageTransformer` must inherit `ModelMixin` + `ConfigMixin`. cache-dit resolves
    pipelines/transformers through `is_pipeline()`, `ModelMixin` assertions and
    `_is_auto_encoder()` / `_is_text_encoder()` module-name checks, so dropping an inheritance
    breaks the caching and tensor-parallelism wiring.
  * `vae` must stay a diffusers `AutoencoderKLFlux2`, never a local copy, for the same reason.
  * The block loop must call one block per iteration and nothing else, so cache-dit's cached-block
    wrapper (which replaces `layers` with a single wrapper module) reproduces the exact computation.
    Layer skipping therefore lives inside the block and is passed as `skip_layers`.
  * `rope_split` / `sequence_gather` are identity markers used by context parallelism; they hold no
    parameters and do not affect the state dict.
  * `_repeated_blocks` makes regional quantization and `compile_repeated_blocks()` find the
    denoising stack instead of silently degrading to whole-module work.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from diffusers import AutoencoderKLFlux2, DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoModel, AutoTokenizer

from ...logger import init_logger

logger = init_logger(__name__)

_INFERENCE_DTYPE = torch.bfloat16
_VAE_DTYPE = torch.float32


class _RMSNorm(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
  return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _validate_skip_layers(layers, depth, name='skip_layers'):
  layers = () if layers is None else tuple(layers)
  if any(type(index) is not int or not 0 <= index < depth for index in layers):
    raise ValueError(f'{name} must contain integer block indices in [0, {depth}).')
  return frozenset(layers)


def _rotate(x, rope):
  sin, cos = rope.chunk(2, -1)
  sin = torch.cat((sin, sin), -1).unsqueeze(-2)
  cos = torch.cat((cos, cos), -1).unsqueeze(-2)
  a, b = x.chunk(2, -1)
  return x * cos + torch.cat((-b, a), -1) * sin


def _context_parallel_attention(q, k, v, cp_config):
  """Run joint attention over a sequence already split across context-parallel ranks.

  MOSS-Image has no diffusers `Attention` module, so the templated CP attention is dispatched
  directly: the backend picks Ulysses/Ring from `cp_config` and all-to-all exchanges the local
  sequence chunks. RoPE is applied before this call because rotation is per token.

  :param q: Query states `[B, S_local, heads, head_dim]`, RoPE applied.
  :param k: Key states `[B, S_local, heads, head_dim]`, RoPE applied.
  :param v: Value states `[B, S_local, heads, head_dim]`.
  :param cp_config: cache-dit context-parallel config of the running transformer.
  :returns: Attention output `[B, S_local, heads, head_dim]`.
  """
  from ...attention import _dispatch_attention_fn

  return _dispatch_attention_fn(
    q,
    k,
    v,
    dropout_p=0.0,
    is_causal=False,
    cp_config=cp_config,
  )


def _attention(q, k, v, rope, heads, cp_config=None):
  shape = (*q.shape[:2], heads, q.shape[-1] // heads)
  # Keep the `[B, S, heads, head_dim]` layout: cache-dit's CP dispatcher takes it as-is, while
  # `scaled_dot_product_attention` wants `[B, heads, S, head_dim]`.
  q = _rotate(q.reshape(shape), rope)
  k = _rotate(k.reshape(shape), rope)
  v = v.reshape(shape)
  q, k = q.to(v.dtype), k.to(v.dtype)
  if cp_config is None:
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2)).transpose(1, 2)
  else:
    out = _context_parallel_attention(q, k, v, cp_config)
  return out.flatten(2)


class _TextBlock(nn.Module):

  def __init__(self, dim, ffn_dim, heads):
    super().__init__()
    self.heads = heads
    self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
    self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
    self.qkv = nn.Linear(dim, 3 * dim)
    self.proj = nn.Linear(dim, dim)
    self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
    self.ffn_down = nn.Linear(ffn_dim, dim)

  def forward(self, x, rope):
    q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
    q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
    k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
    x = x + self.proj(_attention(q, k, v, rope, self.heads))
    y = self.norm2(x)
    return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):

  def __init__(self, text_dim, dim, ffn_dim, heads):
    super().__init__()
    self.proj = nn.Linear(text_dim, dim)
    self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

  def forward(self, x, rope):
    x = self.proj(x.to(self.proj.weight.dtype))
    for block in self.blocks:
      x = block(x, rope)
    return x


class _Block(nn.Module):

  def __init__(self, dim, ffn_dim, heads, index=0):
    super().__init__()
    self.heads = heads
    # Zero-based position in the stack; used for block-level skipping.
    self.index = index
    self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
    self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
    self.proj = nn.ModuleDict(
      {name: nn.Linear(dim, dim, bias=False)
       for name in ('q', 'k', 'v', 'o')})
    self.proj['ffn_up_gate'] = nn.Linear(dim, 2 * ffn_dim, bias=False)
    self.proj['ffn_down'] = nn.Linear(ffn_dim, dim, bias=False)

  def forward(self, hidden_states, rope, modulation, skip_layers=(), cp_config=None):
    """One MOSS-Image DiT block over the joint `[image, text]` sequence.

    :param hidden_states: Joint hidden states `[B, S, dim]`.
    :param rope: RoPE table `[1, S, head_dim]` shared by every block.
    :param modulation: Per-sample modulation `[B, 6, dim]`.
    :param skip_layers: Block indices skipped in this pass (used by CFG/SLG); a skipped block
      returns its input unchanged so that caching wrappers keep an identical graph.
    :param cp_config: cache-dit context-parallel config, `None` for plain attention.
    :returns: Joint hidden states `[B, S, dim]`.
    """
    if self.index in skip_layers:
      return hidden_states

    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
    x = hidden_states
    y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
    dtype = self.proj['q'].weight.dtype
    y = y.to(dtype)
    q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
    k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
    v = self.proj['v'](y)
    attention = _attention(q, k, v, rope, self.heads, cp_config)
    x = x + self.proj['o'](attention) * gate_a[:, None]
    y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
    hidden, gate = self.proj['ffn_up_gate'](y.to(dtype)).chunk(2, -1)
    return x + self.proj['ffn_down'](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):

  def __init__(self, head_dim, base):
    super().__init__()
    self.num_bands = head_dim // 8
    self.base = base
    self.register_buffer('bands', self._default_bands(torch.device('cpu')))

  def _default_bands(self, device):
    # Match training's CPU fp32 construction, including under meta init.
    exponent = torch.arange(self.num_bands, device='cpu', dtype=torch.float32) / self.num_bands
    return (1 / (self.base ** exponent)).to(device)

  def _apply(self, fn, recurse=True):
    super()._apply(fn, recurse=recurse)
    # .float() cannot recover frequencies already rounded by .to(bf16).
    self.bands = self._default_bands(self.bands.device)
    return self

  def _load_from_state_dict(self, *args, **kwargs):
    super()._load_from_state_dict(*args, **kwargs)
    # Frequencies are derived constants; retain the key for checkpoint compatibility.
    self.bands = self._default_bands(self.bands.device)

  def forward(self, height, width, text_length, device):
    # Diffusers/Accelerate can install tensors directly, bypassing load hooks.
    if self.bands.dtype != torch.float32:
      self.bands = self._default_bands(device)
    # Coordinates: (r, f, h, w); sequence order is image followed by text.
    h, w = torch.meshgrid(
      torch.arange(height, device=device),
      torch.arange(width, device=device),
      indexing='ij',
    )
    image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w),
                        -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(ModelMixin, ConfigMixin):
  """MOSS-Image DiT with training-compatible state-dict names and SDPA attention.

  Each sample is evaluated at its true text length, including text refinement; padded tokens cannot
  influence either text or joint attention.

  Note:
    `layers` is the denoising stack cache-dit caches, and `layers[i].index` carries the block
    position so that a single call per block (all cache-dit's cached-block wrapper can replay)
    reproduces `skip_layers` exactly.
  """

  # Denoising stack discovered by regional quantization and `compile_repeated_blocks()`.
  _repeated_blocks = ["_Block"]

  @register_to_config
  def __init__(self,
               dim=3072,
               ffn_dim=8192,
               num_heads=24,
               num_layers=42,
               text_dim=4096,
               freq_dim=256,
               latent_channels=128,
               rope_base=10000.0):
    super().__init__()
    if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
      raise ValueError(
        'Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.')
    self.video_embedder = nn.Linear(latent_channels, dim)
    self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
    self.rope = _RoPE(dim // num_heads, rope_base)
    # Context-parallel markers: identity modules that give CP hooks a module boundary for the RoPE
    # table and for the joint sequence right before it is sliced down to image tokens.
    self.rope_split = nn.Identity()
    self.sequence_gather = nn.Identity()
    self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    self.shared_block_modulation = nn.Linear(dim, 6 * dim)
    self.layers = nn.ModuleList(
      [_Block(dim, ffn_dim, num_heads, index=index) for index in range(num_layers)])
    self.final_norm_video = _RMSNorm(dim)
    self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
    self.video_head_modulation = nn.Linear(dim, 2 * dim)

  def load_training_state_dict(self, state_dict, *, prefix=''):
    """Load a consolidated raw/EMA model mapping, strictly; no silent key drops.

    :param state_dict: Consolidated training state dict.
    :param prefix: Required key prefix in `state_dict`, stripped before loading.
    :returns: The result of `load_state_dict`.
    """
    if prefix:
      if any(not key.startswith(prefix) for key in state_dict):
        raise ValueError('All keys must start with the explicit prefix.')
      state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
    return self.load_state_dict(state_dict, strict=True)

  def forward(self,
              hidden_states,
              timestep,
              encoder_hidden_states,
              encoder_attention_mask=None,
              return_dict=True,
              skip_layers=None):
    """Predict the flow-matching velocity for one batch of noisy latents.

    :param hidden_states: Noisy latents `[B, latent_channels, H, W]`.
    :param timestep: Current sigma / timestep, broadcast over the batch.
    :param encoder_hidden_states: Text features `[B, S, text_dim]`.
    :param encoder_attention_mask: Binary `[B, S]` mask; `None` means all-ones.
    :param return_dict: Return a `Transformer2DModelOutput` instead of a tuple.
    :param skip_layers: Block indices to skip, used by CFG (`uncond_skip_layers`) and SLG.
    :returns: Predicted velocity with the shape of `hidden_states`.
    """
    # Validate against the configured depth: cache-dit swaps `layers` for a single wrapper module
    # while this forward runs, so `len(self.layers)` is not the real stack size there.
    skip_layers = _validate_skip_layers(skip_layers, self.config.num_layers)
    if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
      raise ValueError('hidden_states must be [B, latent_channels, H, W].')
    batch, _, height, width = hidden_states.shape
    if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
      raise ValueError('encoder_hidden_states must be [B, S, text_dim].')
    mask = encoder_attention_mask
    if mask is None:
      mask = torch.ones(encoder_hidden_states.shape[:2],
                        dtype=torch.bool,
                        device=hidden_states.device)
    if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
      raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
    times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
    times = times.expand(batch)
    # Training computes the sinusoid in float64 and feeds float32 into its MLP.
    half = self.config.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
    mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(
      batch, 6, -1)
    head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(
      batch, 2, -1)
    cp_config = getattr(self, "_cp_config", None)
    outputs = []
    for index in range(batch):
      context = encoder_hidden_states[index:index + 1, mask[index].bool()]
      if context.shape[1] == 0:
        raise ValueError('Each sample needs at least one valid text token.')
      video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
      video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
      # Match the reference activation dtype under fp32 weights + autocast.
      rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
      text = self.text_embedder(context, rope[:, height * width:])
      x = torch.cat((video, text), 1)
      # Split the RoPE table only after text refinement consumed the full sequence.
      rope = self.rope_split(rope)
      for layer in self.layers:
        x = layer(x, rope, mod[index:index + 1], skip_layers, cp_config)
      x = self.sequence_gather(x)
      x = self.final_norm_video(x[:, :height * width].to(self.final_norm_video.weight.dtype))
      x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
      x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
      outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
    sample = torch.cat(outputs)
    return Transformer2DModelOutput(sample=sample) if return_dict else (sample, )


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
  """Euler integration using the training project's V2, nearest-grid schedule."""

  @register_to_config
  def __init__(self, num_train_timesteps=1000, shift=5.0, training_shift=1.0):
    super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
    self.register_to_config(shift=shift, training_shift=training_shift)

  def set_timesteps(self, num_inference_steps=50, device=None):
    """Build the snapped, pre-shifted sigma schedule.

    :param num_inference_steps: Number of denoising steps.
    :param device: Device the timesteps are placed on.
    """
    if num_inference_steps < 1 or self.config.shift <= 0 or self.config.training_shift <= 0:
      raise ValueError('Steps and shifts must be positive.')
    total = self.config.num_train_timesteps

    def grid(count, shift):
      lower = shift / (total + shift - 1)
      values = torch.linspace(1, lower, count)
      return shift * values / (1 + (shift - 1) * values)

    sigmas = grid(num_inference_steps, self.config.shift)
    training = grid(total, self.config.training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
                          lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    # Our sigmas have already been shifted and snapped. Avoid a second shift.
    self._shift = 1.0
    super().set_timesteps(device=device, sigmas=sigmas.tolist())


def _resolve_dtypes(dtype_spec):
  """Split a pipeline dtype spec into the inference dtype and the VAE dtype.

  :param dtype_spec: `None`, a single `torch.dtype`, or a per-component mapping such as
    `{"default": torch.bfloat16, "vae": torch.float32}`.
  :returns: `(inference_dtype, vae_dtype)`.
  """
  if dtype_spec is None:
    return _INFERENCE_DTYPE, _VAE_DTYPE
  if isinstance(dtype_spec, dict):
    inference_dtype = dtype_spec.get("default") or dtype_spec.get("transformer") or _INFERENCE_DTYPE
    vae_dtype = dtype_spec.get("vae") or _VAE_DTYPE
    return inference_dtype, vae_dtype
  return dtype_spec, _VAE_DTYPE


def _ensure_chat_template(tokenizer, tokenizer_dir: str) -> None:
  """Attach `chat_template.jinja` when the tokenizer did not read it itself.

  The MOSS-Image `tokenizer_config.json` carries no inline template and the template ships as a
  sibling `chat_template.jinja` file. transformers 5.x picks that file up, older releases do not, so
  this stays as a fallback that is a no-op on current versions.

  :param tokenizer: Loaded tokenizer, modified in place.
  :param tokenizer_dir: Directory the tokenizer was loaded from.
  """
  if getattr(tokenizer, "chat_template", None) is not None:
    return
  template_path = os.path.join(tokenizer_dir, "chat_template.jinja")
  if not os.path.isfile(template_path):
    return
  with open(template_path, encoding="utf-8") as file:
    tokenizer.chat_template = file.read()
  logger.debug(f"Attached chat template from {template_path}.")


class MOSSImagePipeline(DiffusionPipeline):
  """Text-to-image pipeline for MOSS-Image-SFT / MOSS-Image-Base.

  Components live in their own subdirectories (`transformer/`, `text_encoder/`, `tokenizer/`,
  `vae/`, `scheduler/`), so `from_pretrained(<root>)` assembles them directly and no remote pipeline
  code is required.

  Note:
    cache-dit caching assumes one `transformer.forward()` per CFG branch, i.e. batch size 1 and
    `guidance_scale != 1`. The optional SLG forward is a third branch and must stay disabled when
    caching is enabled.
  """

  model_cpu_offload_seq = 'text_encoder->transformer->vae'
  _optional_components = ['text_encoder', 'tokenizer', 'vae']

  def __init__(self,
               transformer: MOSSImageTransformer,
               scheduler: MOSSImageScheduler,
               text_encoder=None,
               tokenizer=None,
               vae=None):
    super().__init__()
    self.register_modules(transformer=transformer,
                          scheduler=scheduler,
                          text_encoder=text_encoder,
                          tokenizer=tokenizer,
                          vae=vae)
    self.image_processor = VaeImageProcessor(vae_scale_factor=16)

  @classmethod
  def from_pretrained(cls,
                      pretrained_model_name_or_path,
                      *args,
                      torch_dtype=None,
                      dtype=None,
                      quantization_config=None,
                      device_map=None,
                      trust_remote_code=False,
                      scheduler=None,
                      transformer=None,
                      vae=None,
                      text_encoder=None,
                      tokenizer=None,
                      controlnet=None,
                      local_files_only=False,
                      **kwargs) -> "MOSSImagePipeline":
    """Load MOSS-Image components from a checkpoint root.

    :param pretrained_model_name_or_path: Checkpoint root holding the component subdirectories.
    :param torch_dtype: dtype for the DiT and text encoder, or a per-component mapping such as
      `{"default": torch.bfloat16, "vae": torch.float32}`. `dtype` is accepted as an alias.
    :param quantization_config: Ignored; quantize through cache-dit's own flags instead.
    :param device_map: Ignored; call `.to(device)` (or cache-dit offloading) explicitly.
    :param trust_remote_code: Ignored; all components here are loaded locally.
    :param scheduler: Pre-built scheduler, replaces the checkpoint's.
    :param transformer: Pre-built transformer, replaces the checkpoint's.
    :param vae: Pre-built VAE, replaces the checkpoint's.
    :param text_encoder: Pre-built text encoder, replaces the checkpoint's.
    :param tokenizer: Pre-built tokenizer, replaces the checkpoint's.
    :param controlnet: Unsupported, must stay `None`.
    :param local_files_only: Whether to forbid network access while resolving files.
    :param kwargs: Additional diffusers arguments, ignored.
    :returns: The assembled pipeline.
    """
    if args:
      raise ValueError("MOSSImagePipeline.from_pretrained takes keyword arguments only.")
    if controlnet is not None:
      raise ValueError("MOSS-Image does not support ControlNet.")

    inference_dtype, vae_dtype = _resolve_dtypes(torch_dtype if torch_dtype is not None else dtype)
    if quantization_config is not None:
      logger.warning("Ignoring quantization_config: quantize MOSS-Image through cache-dit's "
                     "`--quantize-type` flags so the DiT blocks are discovered.")
    if device_map is not None:
      raise NotImplementedError(
        "MOSS-Image does not support `--device-map-balance`: the CLI skips `pipe.to(device)` for "
        "that flag, which would leave the whole pipeline on CPU. Drop the flag or use cache-dit "
        "offloading.")
    if trust_remote_code:
      logger.debug("trust_remote_code is not needed: MOSS-Image code is built into cache-dit.")
    if kwargs:
      logger.warning(f"Ignoring unsupported from_pretrained kwargs: {sorted(kwargs)}.")

    root = str(pretrained_model_name_or_path)
    local_only = dict(local_files_only=local_files_only)

    if transformer is None:
      transformer = MOSSImageTransformer.from_pretrained(os.path.join(root, "transformer"),
                                                         torch_dtype=inference_dtype,
                                                         **local_only)
    if text_encoder is None:
      text_encoder = AutoModel.from_pretrained(os.path.join(root, "text_encoder"),
                                               dtype=inference_dtype,
                                               **local_only)
    tokenizer_dir = os.path.join(root, "tokenizer")
    if tokenizer is None:
      tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, **local_only)
    _ensure_chat_template(tokenizer, tokenizer_dir)
    if scheduler is None:
      scheduler = MOSSImageScheduler.from_pretrained(os.path.join(root, "scheduler"), **local_only)
    if vae is None:
      # Must stay a diffusers module: `_is_auto_encoder()` dispatches VAE parallelism by module
      # path.
      vae = AutoencoderKLFlux2.from_pretrained(os.path.join(root, "vae"),
                                               torch_dtype=vae_dtype,
                                               **local_only)

    for name, module in (("transformer", transformer), ("text_encoder", text_encoder), ("vae",
                                                                                        vae)):
      if module is not None:
        module.eval()

    return cls(
      transformer=transformer,
      scheduler=scheduler,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      vae=vae,
    )

  @classmethod
  def from_components_pretrained(cls,
                                 transformer_path,
                                 text_encoder_path,
                                 vae_path,
                                 torch_dtype=torch.bfloat16,
                                 vae_dtype=torch.float32,
                                 local_files_only=False,
                                 tokenizer_path=None):
    """Load separately exported components (the upstream component-level entry point).

    :param transformer_path: DiT `save_pretrained` directory.
    :param text_encoder_path: Text encoder directory.
    :param vae_path: FLUX.2 VAE directory, or a root holding it under `vae/`.
    :param torch_dtype: DiT and text encoder dtype.
    :param vae_dtype: VAE dtype.
    :param local_files_only: Whether to forbid network access while resolving files.
    :param tokenizer_path: Tokenizer directory; defaults to `text_encoder_path`, which only works
      when the tokenizer files were exported next to the text encoder weights.
    :returns: The assembled pipeline.
    """
    common = dict(local_files_only=local_files_only)
    transformer = MOSSImageTransformer.from_pretrained(transformer_path,
                                                       torch_dtype=torch_dtype,
                                                       **common)
    tokenizer_dir = text_encoder_path if tokenizer_path is None else tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, **common)
    _ensure_chat_template(tokenizer, tokenizer_dir)
    text_encoder = AutoModel.from_pretrained(text_encoder_path, dtype=torch_dtype, **common).eval()
    subfolder = None if (Path(vae_path) / 'config.json').is_file() else 'vae'
    vae = AutoencoderKLFlux2.from_pretrained(vae_path,
                                             subfolder=subfolder,
                                             torch_dtype=vae_dtype,
                                             **common).eval()
    return cls(transformer.eval(), MOSSImageScheduler(), text_encoder, tokenizer, vae)

  @torch.no_grad()
  def encode_prompt(self, prompt, max_sequence_length=4096):
    """Encode prompts into text features and their attention mask.

    :param prompt: Prompt string or list of strings.
    :param max_sequence_length: Token limit per prompt.
    :returns:`(prompt_embeds, prompt_attention_mask)`.
    """
    if self.text_encoder is None or self.tokenizer is None:
      raise ValueError('Text encoder/tokenizer required for string prompts.')
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    if not prompts or not all(isinstance(p, str) for p in prompts):
      raise ValueError('prompt must be a string or a nonempty list of strings.')
    rows = []
    # Unpadded individual forwards also avoid padding effects in Qwen GDN.
    for value in prompts:
      formatted = self.tokenizer.apply_chat_template([{
        'role': 'user',
        'content': value
      }],
                                                     tokenize=False,
                                                     add_generation_prompt=True,
                                                     enable_thinking=True)
      inputs = self.tokenizer(formatted,
                              return_tensors='pt',
                              truncation=True,
                              max_length=max_sequence_length).to(self._execution_device)
      output = self.text_encoder(**inputs, use_cache=False, return_dict=True)
      rows.append(output.last_hidden_state[0])
    length = max(row.shape[0] for row in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
    for index, row in enumerate(rows):
      embeds[index, :row.shape[0]] = row
      mask[index, :row.shape[0]] = True
    return embeds, mask

  @torch.no_grad()
  def __call__(self,
               prompt=None,
               *,
               height=1024,
               width=1024,
               num_inference_steps=50,
               guidance_scale=3.0,
               negative_prompt='',
               num_images_per_prompt=1,
               generator=None,
               latents=None,
               prompt_embeds=None,
               prompt_attention_mask=None,
               negative_prompt_embeds=None,
               negative_prompt_attention_mask=None,
               max_sequence_length=4096,
               output_type='pil',
               return_dict=True,
               uncond_skip_layers=(22, ),
               slg_layers=None,
               slg_scale=0.0,
               slg_start=0.0,
               slg_stop=1.0):
    """Generate images with optional full-block skipping (zero-based indices).

    `uncond_skip_layers` defaults to `(22,)` and replaces CFG's negative-prompt prediction with a
    skipped-block prediction, retaining two forwards per step. Pass None or an empty sequence to
    disable skipping. `slg_layers` adds a skipped positive-prompt forward and the fp32 correction
    `slg_scale * (cond - cond_skip)` to CFG. With `guidance_scale=1`, active SLG uses
    `cond_skip + slg_scale * (cond - cond_skip)`, matching Diffusers' default SkipLayerGuidance
    formulation, with two forwards and no negative branch. Both options can combine. `slg_scale=1`
    then gives normal cond. Both uncond skipping and SLG are active where
    `slg_start <= step_index / num_steps < slg_stop`; the interval follows denoising order, from
    initial noise to final image. Outside it, inference uses normal CFG without skipping.
    `slg_scale=0` or empty `slg_layers` disables the additional forward.

    Note:
      cache-dit caching requires `num_images_per_prompt=1` and one forward per CFG branch, so
      `guidance_scale != 1` (or `enable_separate_cfg=False`) and `slg_layers=None`.

    :param prompt: Prompt string or list of strings, mutually exclusive with `prompt_embeds`.
    :param height: Output height, a positive multiple of 16.
    :param width: Output width, a positive multiple of 16.
    :param num_inference_steps: Number of Euler steps.
    :param guidance_scale: CFG scale; `1.0` disables the negative branch.
    :param negative_prompt: Negative prompt used when CFG is active.
    :param num_images_per_prompt: Images per prompt.
    :param generator: Torch RNG used for the initial noise.
    :param latents: Pre-sampled latents `[B, latent_channels, H/16, W/16]`.
    :param prompt_embeds: Pre-computed positive text features.
    :param prompt_attention_mask: Mask matching `prompt_embeds`.
    :param negative_prompt_embeds: Pre-computed negative text features.
    :param negative_prompt_attention_mask: Mask matching `negative_prompt_embeds`.
    :param max_sequence_length: Token limit used when encoding prompts here.
    :param output_type: `pil`, `np`, `pt` or `latent`.
    :param return_dict: Return an `ImagePipelineOutput` instead of a tuple.
    :param uncond_skip_layers: Block indices skipped in the negative branch.
    :param slg_layers: Block indices skipped in the additional SLG forward.
    :param slg_scale: SLG correction scale.
    :param slg_start: SLG interval start, in denoising progress.
    :param slg_stop: SLG interval end, in denoising progress.
    :returns: Generated images.
    """
    depth = self.transformer.config.num_layers
    uncond_skip_layers = _validate_skip_layers(uncond_skip_layers, depth, 'uncond_skip_layers')
    slg_layers = _validate_skip_layers(slg_layers, depth, 'slg_layers')
    if not math.isfinite(slg_scale) or slg_scale < 0:
      raise ValueError('slg_scale must be finite and nonnegative.')
    if not 0 <= slg_start <= slg_stop <= 1:
      raise ValueError('Require 0 <= slg_start <= slg_stop <= 1.')
    if height <= 0 or width <= 0 or height % 16 or width % 16:
      raise ValueError('height and width must be positive multiples of 16.')
    if output_type not in {'pil', 'np', 'pt', 'latent'}:
      raise ValueError('output_type must be pil, np, pt or latent.')
    if num_images_per_prompt < 1 or not math.isfinite(guidance_scale) or guidance_scale < 0:
      raise ValueError('Invalid image count or guidance scale.')
    if not 1 <= max_sequence_length <= 4096:
      raise ValueError('max_sequence_length must be between 1 and 4096.')
    if (prompt is None) == (prompt_embeds is None):
      raise ValueError('Provide exactly one of prompt and prompt_embeds.')
    device = self._execution_device
    if prompt_embeds is None:
      prompt_embeds, prompt_attention_mask = self.encode_prompt(prompt, max_sequence_length)
    batch = prompt_embeds.shape[0]

    def prepare(embeds, mask):
      if embeds.ndim != 3 or embeds.shape[0] != batch:
        raise ValueError('Prompt embeddings must be [B, S, C] with matching batch size.')
      if mask is None:
        mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device=embeds.device)
      return (embeds.to(device).repeat_interleave(num_images_per_prompt, 0),
              mask.to(device).repeat_interleave(num_images_per_prompt, 0))

    positive, positive_mask = prepare(prompt_embeds, prompt_attention_mask)
    do_cfg = guidance_scale != 1.0
    if getattr(self.transformer, "_is_cached", False):
      # cache-dit's separate-CFG bookkeeping pairs exactly two forward passes per denoising step.
      if not do_cfg:
        raise ValueError(
          "cache-dit caching assumes one conditional and one unconditional forward per step; "
          "`guidance_scale=1.0` runs a single forward and would desynchronize the step counters. "
          "Use a guidance scale above 1, or disable separate CFG in the cache config.")
      if slg_layers and slg_scale > 0:
        raise ValueError(
          "cache-dit caching assumes two forward passes per step; skip-layer guidance adds a "
          "third one. Leave `slg_layers`/`slg_scale` unset when caching is enabled.")
    if do_cfg:
      if negative_prompt_embeds is None:
        negatives = ([negative_prompt] *
                     batch if isinstance(negative_prompt, str) else negative_prompt)
        if len(negatives) != batch:
          raise ValueError('negative_prompt must match prompt batch size.')
        negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
          negatives, max_sequence_length)
      negative, negative_mask = prepare(negative_prompt_embeds, negative_prompt_attention_mask)
    shape = (batch * num_images_per_prompt, self.transformer.config.latent_channels, height // 16,
             width // 16)
    if latents is None:
      # Sample in fp32: promoting bf16 noise afterward cannot restore its precision.
      latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
    elif tuple(latents.shape) != shape:
      raise ValueError(f'latents must have shape {shape}.')
    else:
      latents = latents.to(device=device, dtype=torch.float32)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    num_steps = len(self.scheduler.timesteps)
    for step_index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
      skip_active = slg_start <= step_index / num_steps < slg_stop
      # CFG must combine fp32 predictions, even when the DiT runs in bf16.
      # Converting only the combined velocity would preserve rounding errors.
      prediction = self.transformer(latents, timestep, positive, positive_mask).sample.float()
      conditional = prediction
      if do_cfg:
        skip_kwargs = ({
          'skip_layers': uncond_skip_layers
        } if uncond_skip_layers and skip_active else {})
        uncond = self.transformer(latents, timestep, negative, negative_mask,
                                  **skip_kwargs).sample.float()
        prediction = uncond + guidance_scale * (prediction - uncond)
      if slg_layers and slg_scale > 0 and skip_active:
        skipped = self.transformer(latents,
                                   timestep,
                                   positive,
                                   positive_mask,
                                   skip_layers=slg_layers).sample.float()
        if not do_cfg:
          prediction = skipped
        prediction = prediction + slg_scale * (conditional - skipped)
      latents = self.scheduler.step(prediction, timestep, latents, return_dict=False)[0]
    if output_type == 'latent':
      images = latents
    else:
      if self.vae is None:
        raise ValueError('A FLUX.2 VAE is required for decoded output.')
      z = latents.to(dtype=self.vae.dtype)
      bn = self.vae.bn
      eps = self.vae.config.batch_norm_eps
      z = (z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) +
           bn.running_mean.to(z).view(1, -1, 1, 1))
      b, c, h, w = z.shape
      z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
      z = z.reshape(b, c // 4, 2 * h, 2 * w)
      images = self.vae.decode(z, return_dict=False)[0].clamp(-1, 1)
      images = self.image_processor.postprocess(images, output_type=output_type)
    self.maybe_free_model_hooks()
    return ImagePipelineOutput(images=images) if return_dict else (images, )
