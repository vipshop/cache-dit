"""MAGI-2 Preview classifier-free-guidance sampling.

Adapted from ``vllm-omni @ 7f217a1`` ``vllm_omni/diffusion/models/magi2/sampler_magi2.py``
(Apache-2.0, Copyright (c) 2026 SandAI). Intentional differences:

* the CFG-parallel branch is gone: cache-dit caches inside one process, and MAGI-2 already packs the
  conditional and unconditional branches into one varlen sequence, so `has_separate_cfg=False` and
  both branches are always evaluated in one transformer call;
* the scheduler is cache-dit's already-vendored flow UniPC class
  (``cache_dit.models.pipelines.lingbot_video.scheduler.FlowUniPCMultistepScheduler``), verified
  bit-exact against the released one (sigma schedule and per-step latents) - see the integration
  plan's verification note;
* no ``vllm_omni`` imports; cache-dit style (2-space indent, Sphinx docstrings).
"""

import dataclasses
import math

import torch
import torch.nn.functional as F

from ..lingbot_video.scheduler import FlowUniPCMultistepScheduler
from .data_proxy import Magi2DataProxy, ModelInput

__all__ = [
  "CFGConfig",
  "Magi2PreviewSampler",
  "Magi2SamplerInput",
  "build_magi2_preview_schedulers",
]


def _pad_or_trim(tensor: torch.Tensor, target_size: int, dim: int, pad_value: float = 0.0):
  """Pad or trim one dimension to ``target_size``.

  :param tensor: Input tensor.
  :param target_size: Target size of ``dim``.
  :param dim: Dimension to resize.
  :param pad_value: Value used when padding.
  :returns: ``(resized_tensor, original_size)``.
  """

  current_size = tensor.size(dim)
  if current_size < target_size:
    padding_amount = target_size - current_size
    padding = [0] * (2 * tensor.dim())
    padding_dim_index = tensor.dim() - 1 - dim
    padding[2 * padding_dim_index + 1] = padding_amount
    return F.pad(tensor, tuple(padding), "constant", pad_value), current_size
  slices = [slice(None)] * tensor.dim()
  slices[dim] = slice(0, target_size)
  return tensor[tuple(slices)], current_size


@dataclasses.dataclass(frozen=True)
class CFGConfig:
  """Preview CFG controls, defaulted to the released MAGI-2 recipe.

  :param use_cfg_trick: Lower the video guidance on the first frames.
  :param cfg_trick_start_frame: Last latent frame the trick applies to.
  :param cfg_trick_value: Video guidance used by the trick.
  :param use_dynamic_cfg: Cut the guidance below ``dynamic_cfg_start_t``.
  :param dynamic_cfg_start_t: Timestep threshold of the dynamic rule.
  :param dynamic_cfg_cutoff_value: Guidance cap below the threshold.
  :param video_txt_guidance_scale: Video CFG scale.
  :param audio_txt_guidance_scale: Audio CFG scale.
  :param use_ref_for_uncond: Keep the reference image in both CFG branches (released I2V recipe).
  :param use_skimmed_cfg_linear: Enable the released skimming correction.
  :param skimmed_cfg_scale: Skimming fallback scale.
  :param cfg_rescale: Std-ratio rescaling strength; ``0`` disables it.
  """

  use_cfg_trick: bool = False
  cfg_trick_start_frame: int = 13
  cfg_trick_value: float = 2.0
  use_dynamic_cfg: bool = False
  dynamic_cfg_start_t: int = 500
  dynamic_cfg_cutoff_value: float = 2.0
  video_txt_guidance_scale: float = 5.0
  audio_txt_guidance_scale: float = 7.0
  use_ref_for_uncond: bool = True
  use_skimmed_cfg_linear: bool = False
  skimmed_cfg_scale: float = 3.0
  cfg_rescale: float = 0.0

  def __post_init__(self) -> None:
    finite_values = {
      "cfg_trick_value": self.cfg_trick_value,
      "dynamic_cfg_cutoff_value": self.dynamic_cfg_cutoff_value,
      "video_txt_guidance_scale": self.video_txt_guidance_scale,
      "audio_txt_guidance_scale": self.audio_txt_guidance_scale,
      "skimmed_cfg_scale": self.skimmed_cfg_scale,
      "cfg_rescale": self.cfg_rescale,
    }
    for name, value in finite_values.items():
      if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value}")
    if self.cfg_trick_start_frame < 0:
      raise ValueError("cfg_trick_start_frame must be non-negative")
    if self.skimmed_cfg_scale < 0 or self.cfg_rescale < 0:
      raise ValueError("skimmed_cfg_scale and cfg_rescale must be non-negative")


@dataclasses.dataclass
class Magi2SamplerInput:
  """One request's sampling state.

  :param video_t_list: Video timesteps of the flow schedule.
  :param audio_t_list: Audio timesteps of the flow schedule.
  :param latent: ``[B, C, T, H, W]`` video latents.
  :param audio_latent: ``[B, L, C]`` audio latents.
  :param txt_feat: ``[B, L, D]`` conditional text features.
  :param null_txt_feat: ``[B, L, D]`` unconditional text features.
  :param ref_audio_feat: Optional reference audio latents.
  :param ref_video_feat: Optional reference video latents.
  :param video_scheduler: Video flow scheduler.
  :param audio_scheduler: Audio flow scheduler.
  :param cfg_config: CFG controls.
  :param ref_image_feat: Optional reference-image latents ``[B, M, C, 1, H, W]``.
  :param ref_image_feat_len: Optional reference-image latent grids ``[B, M, 2]``.
  :param ref_image_special_token_embedding: Optional reference-image markers ``[B, M, D]``.
  """

  video_t_list: object
  audio_t_list: object
  latent: torch.Tensor
  audio_latent: torch.Tensor
  txt_feat: torch.Tensor
  null_txt_feat: torch.Tensor
  ref_audio_feat: torch.Tensor
  ref_video_feat: torch.Tensor
  video_scheduler: object
  audio_scheduler: object
  cfg_config: CFGConfig
  ref_image_feat: torch.Tensor = None
  ref_image_feat_len: torch.Tensor = None
  ref_image_special_token_embedding: torch.Tensor = None


def build_magi2_preview_schedulers(
  num_inference_steps: int,
  *,
  device,
  shift: float = 7.0,
):
  """Create independent, identically configured video and audio schedulers.

  :param num_inference_steps: Number of denoising steps.
  :param device: Device of the sigma schedule.
  :param shift: Flow-matching schedule shift.
  :returns: ``(video_scheduler, audio_scheduler)``.
  """

  if num_inference_steps <= 0:
    raise ValueError("num_inference_steps must be positive")
  if not math.isfinite(shift) or shift <= 0:
    raise ValueError(f"shift must be positive and finite, got {shift}")
  video_scheduler = FlowUniPCMultistepScheduler()
  audio_scheduler = FlowUniPCMultistepScheduler()
  video_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
  audio_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
  return video_scheduler, audio_scheduler


class Magi2PreviewSampler:
  """Run MAGI-2 Preview's joint video/audio denoising loop.

  :param model: The preview transformer.
  :param data_proxy: Packing helper; a default one is created when omitted.
  :param device: Device of the transformer.
  :param dtype: Compute dtype.
  """

  def __init__(
    self,
    model: torch.nn.Module,
    data_proxy: Magi2DataProxy = None,
    device=None,
    dtype: torch.dtype = None,
  ) -> None:
    self.model = model
    self.data_proxy = data_proxy or Magi2DataProxy()
    self.device = torch.device(device) if device is not None else None
    self.dtype = dtype

  @torch.inference_mode()
  def forward(self, model_input: ModelInput):
    """Run one transformer call and depack its output.

    :param model_input: Packed request tensors (both CFG branches).
    :returns: ``(video_velocity, audio_velocity)``.
    """

    packed_input = self.data_proxy.process_input(model_input)
    model_output = self.model(*packed_input.model_args)
    if not isinstance(model_output, torch.Tensor):
      raise TypeError(
        f"MAGI-2 preview transformer must return a tensor, got {type(model_output)!r}")
    return self.data_proxy.process_output(model_output, packed_input.output_layout)

  def predict_noise(self, model_input: ModelInput):
    """Run one conditional or unconditional branch.

    :param model_input: Request tensors.
    :returns: ``(video_velocity, audio_velocity)``.
    """

    return self.forward(model_input)

  @torch.inference_mode()
  def sample(self, sampler_input: Magi2SamplerInput):
    """Run the denoising loop.

    :param sampler_input: Request sampling state.
    :returns: ``(video_latents, audio_latents)``.
    """

    video_timesteps = list(sampler_input.video_t_list)
    audio_timesteps = list(sampler_input.audio_t_list)
    if not video_timesteps:
      raise ValueError("MAGI-2 sampling needs at least one timestep")
    if len(video_timesteps) != len(audio_timesteps):
      raise ValueError("video and audio timestep schedules must have equal lengths")

    video_cfgs, audio_cfgs = self.precalculate_cfg(
      video_timesteps,
      sampler_input.latent.shape[2],
      sampler_input.cfg_config,
      device=sampler_input.latent.device,
    )
    latent = sampler_input.latent.clone()
    audio_latent = sampler_input.audio_latent.clone()

    for timestep, video_cfg, audio_cfg in zip(video_timesteps, video_cfgs, audio_cfgs, strict=True):
      model_input = self.prepare_model_input(
        latent=latent,
        audio_latent=audio_latent,
        txt_feat=sampler_input.txt_feat,
        null_txt_feat=sampler_input.null_txt_feat,
        ref_audio_feat=sampler_input.ref_audio_feat,
        ref_video_feat=sampler_input.ref_video_feat,
        ref_image_feat=sampler_input.ref_image_feat,
        ref_image_feat_len=sampler_input.ref_image_feat_len,
        ref_image_special_token_embedding=sampler_input.ref_image_special_token_embedding,
        t=timestep,
        cfg_config=sampler_input.cfg_config,
      )
      model_pred = self.forward(model_input)
      latent, audio_latent, _, _ = self.step(
        model_pred,
        latent,
        audio_latent,
        video_cfg,
        audio_cfg,
        sampler_input.video_scheduler,
        sampler_input.audio_scheduler,
        timestep,
        cfg_config=sampler_input.cfg_config,
      )
    return latent, audio_latent

  @staticmethod
  def _validate_prepare_inputs(latent, audio_latent, txt_feat, null_txt_feat) -> None:
    if latent.ndim != 5:
      raise ValueError("latent must be [B,C,T,H,W]")
    if audio_latent.ndim != 3 or txt_feat.ndim != 3 or null_txt_feat.ndim != 3:
      raise ValueError("audio and text features must be [B,L,C]")
    batch_size = latent.shape[0]
    if not (audio_latent.shape[0] == txt_feat.shape[0] == null_txt_feat.shape[0] == batch_size):
      raise ValueError("video, audio, positive-text, and negative-text batch sizes must match")

  def prepare_model_input(
    self,
    latent: torch.Tensor,
    audio_latent: torch.Tensor,
    txt_feat: torch.Tensor,
    null_txt_feat: torch.Tensor,
    ref_audio_feat: torch.Tensor = None,
    ref_video_feat: torch.Tensor = None,
    ref_image_feat: torch.Tensor = None,
    ref_image_feat_len: torch.Tensor = None,
    ref_image_special_token_embedding: torch.Tensor = None,
    t=None,
    cfg_config: CFGConfig = None,
  ) -> ModelInput:
    """Pack the conditional and unconditional branches into one request.

    :param latent: ``[B, C, T, H, W]`` video latents.
    :param audio_latent: ``[B, L, C]`` audio latents.
    :param txt_feat: ``[B, L, D]`` conditional text features.
    :param null_txt_feat: ``[B, L, D]`` unconditional text features.
    :param ref_audio_feat: Optional reference audio latents.
    :param ref_video_feat: Optional reference video latents.
    :param ref_image_feat: Optional reference-image latents.
    :param ref_image_feat_len: Optional reference-image latent grids.
    :param ref_image_special_token_embedding: Optional reference-image markers.
    :param t: Timestep, tensor or float.
    :param cfg_config: CFG controls.
    :returns: The packed model input, batch = 2 * video batch.
    """

    self._validate_prepare_inputs(latent, audio_latent, txt_feat, null_txt_feat)
    batch_size = latent.shape[0]
    audio_latent_len = audio_latent.shape[1]
    txt_feat_len = txt_feat.shape[1]
    null_txt_feat_len = null_txt_feat.shape[1]
    target_length = max(txt_feat_len, null_txt_feat_len)
    txt_feat, _ = _pad_or_trim(txt_feat, target_length, 1)
    null_txt_feat, _ = _pad_or_trim(null_txt_feat, target_length, 1)

    if ref_audio_feat is None:
      ref_audio_feat = audio_latent.new_empty(batch_size, 0, audio_latent.shape[-1])
    if ref_audio_feat.ndim != 3 or ref_audio_feat.shape[0] != batch_size:
      raise ValueError("ref_audio_feat must be [B,L,C] with the same batch size")
    ref_audio_feat_len = ref_audio_feat.shape[1]

    if ref_video_feat is None:
      ref_video_feat = latent[:, :, :0]
      ref_video_feat_len = 0
    else:
      if ref_video_feat.ndim != 5 or ref_video_feat.shape[0] != batch_size:
        raise ValueError("ref_video_feat must be [B,C,T,H,W] with the same batch size")
      ref_video_feat_len = (ref_video_feat.shape[3] * ref_video_feat.shape[4]) // 4

    ref_image_feat_cfg, ref_image_feat_len_cfg, ref_image_special_tokens_cfg = self._prepare_ref_image_cfg(
      ref_image_feat,
      ref_image_feat_len,
      ref_image_special_token_embedding,
      cfg_config,
    )

    if t is None:
      t_value = torch.tensor(0.0, device=latent.device, dtype=latent.dtype)
    elif isinstance(t, torch.Tensor):
      t_value = t.reshape(-1)[0].to(device=latent.device, dtype=latent.dtype)
    else:
      t_value = torch.tensor(float(t), device=latent.device, dtype=latent.dtype)
    t_normalized = t_value / 1000.0
    cfg_batch = batch_size * 2
    t_batch = t_normalized.expand(cfg_batch)

    _, _, video_t, video_h, video_w = latent.shape
    audio_t = audio_latent.shape[1]
    per_token_video_t = t_batch.view(-1, 1, 1, 1, 1).expand(cfg_batch, 1, video_t, video_h, video_w)
    per_token_audio_t = t_batch.view(-1, 1, 1).expand(cfg_batch, audio_t, 1)

    return ModelInput(
      x_t=torch.cat((latent, latent), dim=0),
      audio_x_t=torch.cat((audio_latent, audio_latent), dim=0),
      audio_feat_len=[audio_latent_len] * cfg_batch,
      txt_feat=torch.cat((txt_feat, null_txt_feat), dim=0),
      txt_feat_len=[txt_feat_len] * batch_size + [null_txt_feat_len] * batch_size,
      t=t_batch,
      per_token_video_t=per_token_video_t,
      per_token_audio_t=per_token_audio_t,
      ref_audio_feat=torch.cat((ref_audio_feat, torch.zeros_like(ref_audio_feat)), dim=0),
      ref_audio_feat_len=[ref_audio_feat_len] * cfg_batch,
      ref_video_feat=torch.cat((ref_video_feat, torch.zeros_like(ref_video_feat)), dim=0),
      ref_video_feat_len=[ref_video_feat_len] * cfg_batch,
      ref_image_feat=ref_image_feat_cfg,
      ref_image_feat_len=ref_image_feat_len_cfg,
      ref_image_special_token_embedding=ref_image_special_tokens_cfg,
    )

  @staticmethod
  def _prepare_ref_image_cfg(
    ref_image_feat: torch.Tensor,
    ref_image_feat_len: torch.Tensor,
    ref_image_special_token_embedding: torch.Tensor,
    cfg_config: CFGConfig,
  ):
    if ref_image_feat is None:
      if ref_image_feat_len is not None or ref_image_special_token_embedding is not None:
        raise ValueError(
          "reference-image features, lengths, and special tokens must be provided together")
      return None, None, None
    if ref_image_feat_len is None or ref_image_special_token_embedding is None:
      raise ValueError(
        "reference-image features, lengths, and special tokens must be provided together")

    keep_ref = bool(cfg_config is not None and cfg_config.use_ref_for_uncond)
    uncond_feature = ref_image_feat if keep_ref else torch.zeros_like(ref_image_feat)
    return (
      torch.cat((ref_image_feat, uncond_feature), dim=0),
      torch.cat((ref_image_feat_len, ref_image_feat_len), dim=0),
      torch.cat((ref_image_special_token_embedding, ref_image_special_token_embedding), dim=0),
    )

  def precalculate_cfg(self, t_list, latent_length: int, cfg_config: CFGConfig, *, device=None):
    """Precompute the per-step guidance scales.

    :param t_list: Timesteps of the schedule.
    :param latent_length: Latent frame count (used by the cfg trick).
    :param cfg_config: CFG controls.
    :param device: Device of the scales.
    :returns: ``(video_cfgs, audio_cfgs)``.
    """

    if latent_length < 0:
      raise ValueError("latent_length must be non-negative")
    if device is None:
      if self.device is not None:
        device = self.device
      elif t_list:
        device = t_list[0].device
      else:
        device = torch.device("cpu")

    if cfg_config.use_cfg_trick:
      video_cfg_base = (torch.tensor(cfg_config.video_txt_guidance_scale,
                                     device=device).expand(1, 1, latent_length, 1, 1).clone())
      video_cfg_base[:, :, :cfg_config.cfg_trick_start_frame] = min(
        cfg_config.cfg_trick_value, cfg_config.video_txt_guidance_scale)
    else:
      video_cfg_base = torch.tensor([cfg_config.video_txt_guidance_scale], device=device)

    all_video_cfgs = []
    all_audio_cfgs = []
    for timestep in t_list:
      video_cfg = video_cfg_base.clone()
      if cfg_config.use_dynamic_cfg:
        timestep_value = float(timestep.detach().reshape(-1)[0].item())
        if timestep_value < cfg_config.dynamic_cfg_start_t:
          video_cfg[video_cfg >
                    cfg_config.dynamic_cfg_cutoff_value] = cfg_config.dynamic_cfg_cutoff_value
      all_video_cfgs.append(video_cfg)
      all_audio_cfgs.append(cfg_config.audio_txt_guidance_scale)
    return all_video_cfgs, all_audio_cfgs

  @staticmethod
  def _cfg_scale_tensor(guidance_scale, like: torch.Tensor) -> torch.Tensor:
    if isinstance(guidance_scale, torch.Tensor):
      return guidance_scale.to(device=like.device, dtype=like.dtype)
    return torch.tensor(float(guidance_scale), device=like.device, dtype=like.dtype)

  @classmethod
  def _get_skimming_mask(cls, x_orig, cond, uncond, guidance_scale) -> torch.Tensor:
    guidance_scale = cls._cfg_scale_tensor(guidance_scale, cond)
    x_orig = x_orig.to(device=cond.device, dtype=cond.dtype)
    denoised = x_orig - ((x_orig - uncond) + guidance_scale * ((x_orig - cond) - (x_orig - uncond)))
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = cond.sign() == (cond * guidance_scale - uncond *
                                          (guidance_scale - 1)).sign()
    deviation_influence = denoised.sign() == (denoised - x_orig).sign()
    return matching_pred_signs & matching_diff_after & deviation_influence

  @classmethod
  def _apply_skimmed_cfg_linear(cls, x_orig, cond, uncond, guidance_scale, cfg_config):
    if (cfg_config is None or not cfg_config.use_skimmed_cfg_linear or x_orig is None
        or not torch.any(uncond).item()):
      return uncond

    guidance_scale = cls._cfg_scale_tensor(guidance_scale, cond)
    valid_scale = guidance_scale > 1
    if not torch.any(valid_scale).item():
      return uncond
    scale_delta = torch.where(valid_scale, guidance_scale - 1, torch.ones_like(guidance_scale))
    fallback_weight = (cfg_config.skimmed_cfg_scale - 1) / scale_delta
    target_uncond = cond * (1 - fallback_weight) + uncond * fallback_weight
    skim_mask = cls._get_skimming_mask(x_orig, cond, uncond, guidance_scale)
    uncond = torch.where(skim_mask & valid_scale, target_uncond, uncond)

    target_uncond = cond * (1 - fallback_weight) + uncond * fallback_weight
    skim_mask = cls._get_skimming_mask(x_orig, uncond, cond, guidance_scale)
    return torch.where(skim_mask & valid_scale, target_uncond, uncond)

  @staticmethod
  def _cfg_rescale_dims(tensor: torch.Tensor):
    if tensor.ndim <= 1:
      return ()
    if tensor.ndim in (2, 3):
      return (1, )
    return tuple(range(2, tensor.ndim))

  @classmethod
  def _apply_cfg_rescale(cls, pos, cfg, cfg_config):
    if cfg_config is None or cfg_config.cfg_rescale <= 0 or cfg.numel() == 0:
      return cfg
    spatial_dims = cls._cfg_rescale_dims(cfg)
    if not spatial_dims:
      return cfg
    pos_std = pos.float().std(dim=spatial_dims, keepdim=True, unbiased=False)
    cfg_std = cfg.float().std(dim=spatial_dims, keepdim=True, unbiased=False).clamp_min(1e-6)
    factor = cfg_config.cfg_rescale * (pos_std / cfg_std) + (1 - cfg_config.cfg_rescale)
    return cfg * factor.to(device=cfg.device, dtype=cfg.dtype)

  def cfg_velocity(
    self,
    model_output,
    video_txt_guidance_scale,
    audio_txt_guidance_scale,
    cfg_config: CFGConfig = None,
    latent: torch.Tensor = None,
    audio_latent: torch.Tensor = None,
  ):
    """Combine the conditional and unconditional predictions.

    :param model_output: ``(video, audio)`` predictions with both CFG halves stacked.
    :param video_txt_guidance_scale: Video guidance scale.
    :param audio_txt_guidance_scale: Audio guidance scale.
    :param cfg_config: CFG controls.
    :param latent: Video latents of the current step (skimming / rescale).
    :param audio_latent: Audio latents of the current step.
    :returns: ``(video_velocity, audio_velocity)``.
    """

    cond_uncond_video, cond_uncond_audio = model_output
    if cond_uncond_video.shape[0] % 2 or cond_uncond_audio.shape[0] % 2:
      raise ValueError("MAGI-2 CFG outputs must contain equal conditional/unconditional halves")
    video_batch = cond_uncond_video.shape[0] // 2
    audio_batch = cond_uncond_audio.shape[0] // 2
    if video_batch != audio_batch:
      raise ValueError("video and audio CFG output batch sizes differ")

    if cond_uncond_video.numel() > 0:
      cond_video = cond_uncond_video[:video_batch]
      uncond_video = cond_uncond_video[video_batch:]
      uncond_video = self._apply_skimmed_cfg_linear(latent, cond_video, uncond_video,
                                                    video_txt_guidance_scale, cfg_config)
      cfg_video = uncond_video + video_txt_guidance_scale * (cond_video - uncond_video)
      cfg_video = self._apply_cfg_rescale(cond_video, cfg_video, cfg_config)
    else:
      cfg_video = None

    if cond_uncond_audio.numel() > 0:
      cond_audio = cond_uncond_audio[:audio_batch]
      uncond_audio = cond_uncond_audio[audio_batch:]
      uncond_audio = self._apply_skimmed_cfg_linear(audio_latent, cond_audio, uncond_audio,
                                                    audio_txt_guidance_scale, cfg_config)
      cfg_audio = uncond_audio + audio_txt_guidance_scale * (cond_audio - uncond_audio)
      cfg_audio = self._apply_cfg_rescale(cond_audio, cfg_audio, cfg_config)
    else:
      cfg_audio = None
    return cfg_video, cfg_audio

  def step(
    self,
    model_output,
    latent: torch.Tensor,
    audio_latent: torch.Tensor,
    video_txt_guidance_scale,
    audio_txt_guidance_scale,
    video_scheduler,
    audio_scheduler,
    t,
    cfg_config: CFGConfig = None,
  ):
    """Apply one guided scheduler step.

    :param model_output: ``(video, audio)`` predictions with both CFG halves stacked.
    :param latent: Video latents of the current step.
    :param audio_latent: Audio latents of the current step.
    :param video_txt_guidance_scale: Video guidance scale.
    :param audio_txt_guidance_scale: Audio guidance scale.
    :param video_scheduler: Video flow scheduler.
    :param audio_scheduler: Audio flow scheduler.
    :param t: Current timestep.
    :param cfg_config: CFG controls.
    :returns: ``(latent, audio_latent, cfg_video, cfg_audio)``.
    """

    cfg_video, cfg_audio = self.cfg_velocity(
      model_output,
      video_txt_guidance_scale,
      audio_txt_guidance_scale,
      cfg_config,
      latent,
      audio_latent,
    )
    if cfg_video is not None:
      latent = video_scheduler.step(cfg_video, t, latent, return_dict=False)[0]
    if cfg_audio is not None:
      audio_latent = audio_scheduler.step(cfg_audio, t, audio_latent, return_dict=False)[0]
    return latent, audio_latent, cfg_video, cfg_audio
