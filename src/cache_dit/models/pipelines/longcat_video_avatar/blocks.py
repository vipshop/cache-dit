"""Building blocks shared by the LongCat-Video-Avatar transformer.

Copied from `meituan-longcat/LongCat-Video` (main, 2026-11-03),
`longcat_video/modules/blocks.py` and `longcat_video/modules/avatar/blocks.py`.

Intentional differences from the upstream copy:

* repository style (2-space indent, Sphinx docstrings), no trailing-whitespace churn;
* `AudioProjModel` drops the `flops` counter: it mutates Python state inside `forward`, which is an
  unwanted graph side effect under `torch.compile`;
* `PatchEmbed3D` drops the unused `norm_layer` branch's shape juggling on the flatten path.

Every `state_dict` key of the upstream classes is preserved.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_fp32(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
  """Run a linear layer in float32 regardless of the parameter dtype.

  Upstream runs these calls under `torch.autocast(device_type="cuda", dtype=torch.float32)`, which
  casts the weights on the fly; the same region is used here instead of casting the parameters
  directly, because casting bypasses the module call and therefore the layerwise-offload hooks,
  which leaves the weight on the host and turns the matmul into a device mismatch. On CPU there is
  no autocast, so the layer simply runs in the dtype it was built with (the tests build float32).

  :param linear: a linear module whose parameters are still in the checkpoint dtype.
  :param x: float32 input.
  :returns: float32 output on CUDA, the native dtype on CPU.
  """

  with torch.autocast(device_type=x.device.type, dtype=torch.float32, enabled=x.is_cuda):
    return linear(x.float())


def silu_linear_fp32(module: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
  """Run `SiLU -> Linear` in float32, the upstream modulation head.

  :param module: the two-element `nn.Sequential(SiLU, Linear)` of an adaLN modulation.
  :param x: float32 input.
  :returns: float32 modulation parameters.
  """

  return linear_fp32(module[1], F.silu(x.float()))


class FeedForwardSwiGLU(nn.Module):
  """SwiGLU feed-forward used by every avatar block.

  :param dim: model width.
  :param hidden_dim: pre-rounding hidden width; the real width is `2 / 3` of it, rounded up to
    `multiple_of`.
  :param multiple_of: rounding granularity of the hidden width.
  :param ffn_dim_multiplier: optional extra multiplier applied before rounding.
  """

  def __init__(
    self,
    dim: int,
    hidden_dim: int,
    multiple_of: int = 256,
    ffn_dim_multiplier: Optional[float] = None,
  ):
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    self.dim = dim
    self.hidden_dim = hidden_dim
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the gated feed-forward.

    :param x: `[..., dim]` activations.
    :returns: `[..., dim]` activations.
    """

    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm_FP32(nn.Module):
  """RMSNorm that accumulates in float32, used for the per-head Q/K norms.

  :param dim: normalized width (the head dim for the attention norms).
  :param eps: epsilon of the reciprocal square root.
  """

  def __init__(self, dim: int, eps: float):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Normalize the last dimension.

    :param x: `[..., dim]` activations.
    :returns: `[..., dim]` normalized activations in the input dtype.
    """

    output = self._norm(x.float()).type_as(x)
    return output * self.weight


class LayerNorm_FP32(nn.LayerNorm):
  """LayerNorm whose normalization runs in float32 regardless of the input dtype.

  :param dim: normalized width.
  :param eps: epsilon of the normalization.
  :param elementwise_affine: whether the affine weight (and bias) exist.
  """

  def __init__(self, dim, eps, elementwise_affine):
    super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Normalize the last dimension in float32.

    :param inputs: `[..., dim]` activations.
    :returns: `[..., dim]` normalized activations in the input dtype.
    """

    origin_dtype = inputs.dtype
    out = F.layer_norm(
      inputs.float(),
      self.normalized_shape,
      None if self.weight is None else self.weight.float(),
      None if self.bias is None else self.bias.float(),
      self.eps,
    ).to(origin_dtype)
    return out


class PatchEmbed3D(nn.Module):
  """Video to patch embedding (a strided 3-D convolution).

  :param patch_size:`(t, h, w)` patch size.
  :param in_chans: latent channels of the VAE.
  :param embed_dim: transformer width.
  :param norm_layer: unused, kept for signature compatibility with the upstream constructor.
  :param flatten: whether to flatten the `(t, h, w)` grid into a token axis.
  """

  def __init__(
      self,
      patch_size=(2, 4, 4),
      in_chans=3,
      embed_dim=96,
      norm_layer=None,
      flatten=True,
  ):
    super().__init__()
    self.patch_size = patch_size
    self.flatten = flatten

    self.in_chans = in_chans
    self.embed_dim = embed_dim

    self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    self.norm = norm_layer(embed_dim) if norm_layer is not None else None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Patchify a latent video.

    :param x:`[B, C, T, H, W]` latents.
    :returns:`[B, T' * H' * W', embed_dim]` tokens, ordered frame major.
    """

    _, _, depth, height, width = x.size()
    if width % self.patch_size[2] != 0:
      x = F.pad(x, (0, self.patch_size[2] - width % self.patch_size[2]))
    if height % self.patch_size[1] != 0:
      x = F.pad(x, (0, 0, 0, self.patch_size[1] - height % self.patch_size[1]))
    if depth % self.patch_size[0] != 0:
      x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - depth % self.patch_size[0]))

    x = self.proj(x)
    if self.norm is not None:
      d, wh, ww = x.size(2), x.size(3), x.size(4)
      x = self.norm(x.flatten(2).transpose(1, 2)).transpose(1,
                                                            2).view(-1, self.embed_dim, d, wh, ww)
    if self.flatten:
      x = x.flatten(2).transpose(1, 2)
    return x


def modulate_fp32(norm_func, x: torch.Tensor, shift: torch.Tensor,
                  scale: torch.Tensor) -> torch.Tensor:
  """Apply `norm(x) * (1 + scale) + shift` with float32 modulation parameters.

  :param norm_func: normalization module, applied in float32.
  :param x: `[B, N, D]` activations.
  :param shift: `[B, T, 1, D]` broadcast shift.
  :param scale: `[B, T, 1, D]` broadcast scale.
  :returns: `[B, N, D]` modulated activations in the dtype of `x`.
  """

  dtype = x.dtype
  x = norm_func(x.to(torch.float32))
  x = x * (scale + 1) + shift
  return x.to(dtype)


class FinalLayer_FP32(nn.Module):
  """Output head: final norm + modulation + patch-unflattening projection.

  :param hidden_size: transformer width.
  :param num_patch: `prod(patch_size)`.
  :param out_channels: latent channels of the VAE.
  :param adaln_tembed_dim: width of the timestep embedding.
  """

  def __init__(self, hidden_size, num_patch, out_channels, adaln_tembed_dim):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_patch = num_patch
    self.out_channels = out_channels
    self.adaln_tembed_dim = adaln_tembed_dim

    self.norm_final = LayerNorm_FP32(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
    self.adaLN_modulation = nn.Sequential(nn.SiLU(),
                                          nn.Linear(adaln_tembed_dim, 2 * hidden_size, bias=True))

  def forward(self, x: torch.Tensor, t: torch.Tensor, latent_shape) -> torch.Tensor:
    """Project the residual stream to patch features.

    :param x: `[B, N, hidden_size]` residual stream, `N = T * H * W`.
    :param t: `[B, T, adaln_tembed_dim]` float32 timestep embedding.
    :param latent_shape: `(T, H, W)` of the token grid `x` describes; under context parallelism this
      is the **local** grid.
    :returns: `[B, N, num_patch * out_channels]` patch features.
    """

    assert t.dtype == torch.float32, "the final layer expects a float32 timestep embedding"
    b, n, c = x.shape
    frames = latent_shape[0]

    shift, scale = silu_linear_fp32(self.adaLN_modulation, t).unsqueeze(2).chunk(2, dim=-1)
    x = modulate_fp32(self.norm_final, x.view(b, frames, -1, c), shift, scale).view(b, n, c)
    return linear_fp32(self.linear, x)


class TimestepEmbedder(nn.Module):
  """Sinusoidal timestep embedding followed by a two-layer MLP.

  :param t_embed_dim: output width.
  :param frequency_embedding_size: width of the sinusoidal features.
  """

  def __init__(self, t_embed_dim, frequency_embedding_size=256):
    super().__init__()
    self.t_embed_dim = t_embed_dim
    self.frequency_embedding_size = frequency_embedding_size
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, t_embed_dim, bias=True),
      nn.SiLU(),
      nn.Linear(t_embed_dim, t_embed_dim, bias=True),
    )

  @staticmethod
  def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Build sinusoidal features for a 1-D batch of timesteps.

    :param t: 1-D tensor of (possibly fractional) timesteps.
    :param dim: feature width.
    :param max_period: largest period of the sinusoid bank.
    :returns:`[len(t), dim]` features.
    """

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) /
                      half)
    freqs = freqs.to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Embed timesteps.

    :param t: 1-D tensor of timesteps.
    :param dtype: dtype of the returned features.
    :returns:`[len(t), t_embed_dim]` embeddings.
    """

    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    if t_freq.dtype != dtype:
      t_freq = t_freq.to(dtype)
    # Upstream runs this MLP under `autocast(float32)`, i.e. float32 math with on-the-fly weight
    # casts; the timestep embedding is the float32 carrier of the whole modulation path.
    t_emb = linear_fp32(self.mlp[0], t_freq)
    t_emb = self.mlp[1](t_emb)
    return linear_fp32(self.mlp[2], t_emb)


class CaptionEmbedder(nn.Module):
  """Text encoder output to transformer width.

  :param in_channels: text encoder hidden size (4096 for UMT5-XXL).
  :param hidden_size: transformer width.
  """

  def __init__(self, in_channels, hidden_size):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_size = hidden_size
    self.y_proj = nn.Sequential(
      nn.Linear(in_channels, hidden_size, bias=True),
      nn.GELU(approximate="tanh"),
      nn.Linear(hidden_size, hidden_size, bias=True),
    )

  def forward(self, caption: torch.Tensor) -> torch.Tensor:
    """Project the text embeddings.

    :param caption:`[B, 1, N_text, in_channels]` text embeddings.
    :returns:`[B, 1, N_text, hidden_size]` projected embeddings.
    """

    return self.y_proj(caption)


class AudioProjModel(nn.Module):
  """Per-frame audio window projection used outside the transformer stack.

  The Whisper features of one latent frame are `audio_window` vectors of `blocks * channels`
  values (the encoder layer groups); this module flattens and projects them into
  `context_tokens` tokens of `output_dim`, once per frame.

  :param seq_len: audio window length of the first frame.
  :param seq_len_vf: audio window length of the later frames.
  :param blocks: number of encoder feature groups per audio vector.
  :param channels: width of the audio encoder.
  :param intermediate_dim: hidden width of the projection MLP.
  :param output_dim: width of one audio token.
  :param context_tokens: number of audio tokens produced per frame.
  :param norm_output_audio: whether to LayerNorm the produced tokens.
  """

  def __init__(
    self,
    seq_len=5,
    seq_len_vf=12,
    blocks=12,
    channels=768,
    intermediate_dim=512,
    output_dim=768,
    context_tokens=32,
    norm_output_audio=True,
  ):
    super().__init__()

    self.seq_len = seq_len
    self.blocks = blocks
    self.channels = channels
    self.input_dim = seq_len * blocks * channels
    self.input_dim_vf = seq_len_vf * blocks * channels
    self.intermediate_dim = intermediate_dim
    self.context_tokens = context_tokens
    self.output_dim = output_dim

    self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
    self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
    self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
    self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
    self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

  def forward(self, audio_embeds: torch.Tensor, audio_embeds_vf: torch.Tensor) -> torch.Tensor:
    """Project the per-frame audio windows.

    :param audio_embeds:`[B, 1, W, blocks, channels]` audio of the first frame.
    :param audio_embeds_vf:`[B, T-1, W_vf, blocks, channels]` audio of the later frames.
    :returns:`[B, T, context_tokens, output_dim]` audio tokens, one group per latent frame.
    """

    video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
    b = audio_embeds.shape[0]

    audio_embeds = audio_embeds.reshape(-1, *audio_embeds.shape[2:])
    batch_size, window_size, blocks, channels = audio_embeds.shape
    audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

    audio_embeds_vf = audio_embeds_vf.reshape(-1, *audio_embeds_vf.shape[2:])
    batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
    audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

    audio_embeds = torch.relu(self.proj1(audio_embeds))
    audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))

    audio_embeds = audio_embeds.view(b, -1, self.intermediate_dim)
    audio_embeds_vf = audio_embeds_vf.view(b, -1, self.intermediate_dim)
    audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
    batch_size_c, n_t, c_a = audio_embeds_c.shape
    audio_embeds_c = audio_embeds_c.reshape(batch_size_c * n_t, c_a)

    audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

    context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c * n_t, self.context_tokens,
                                                        self.output_dim)
    context_tokens = self.norm(context_tokens)
    return context_tokens.view(b, video_length, self.context_tokens, self.output_dim)
