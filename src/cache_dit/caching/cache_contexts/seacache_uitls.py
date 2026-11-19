"""Model-agnostic Spectral-Evolution-Aware indicators for DBCache probes."""

import math
from numbers import Real
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist


def _finite_scalar(value, name: str) -> float:
  """Convert one scalar schedule value to a finite Python float."""

  if isinstance(value, torch.Tensor):
    if value.numel() != 1:
      raise ValueError(f"SeaCache {name} must be a scalar, got shape {tuple(value.shape)}.")
    value = value.detach().item()
  if isinstance(value, bool) or not isinstance(value, Real):
    raise ValueError(f"SeaCache {name} must be numeric, got {type(value).__name__}.")
  value = float(value)
  if not math.isfinite(value):
    raise ValueError(f"SeaCache {name} must be finite, got {value}.")
  return value


def resolve_seacache_ab(
  scheduler=None,
  index: int = 0,
  mode: str = "flow",
  ab_schedule: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[float, float]:
  """Resolve signal/noise coefficients from an explicit schedule or a generic scheduler.

  :param scheduler: Object exposing flow `sigmas` or VP `alphas_cumprod` and optionally `timesteps`.
  :param index: Zero-based inference-step index.
  :param mode: Either `flow` or `vp`.
  :param ab_schedule: Optional explicit sequence of `(a, b)` coefficient pairs.
  :returns: The finite, positive `(a, b)` pair for the requested step.
  """

  if isinstance(index, bool) or not isinstance(index, int) or index < 0:
    raise ValueError(f"SeaCache step index must be a non-negative integer, got {index!r}.")
  if mode not in ("flow", "vp"):
    raise ValueError(f"SeaCache mode must be 'flow' or 'vp', got {mode!r}.")

  if ab_schedule is not None:
    if index >= len(ab_schedule):
      raise ValueError(f"SeaCache AB schedule has {len(ab_schedule)} entries, but step {index} "
                       "was requested.")
    pair = ab_schedule[index]
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
      raise ValueError(f"SeaCache AB schedule entry {index} must contain exactly (a, b).")
    a = _finite_scalar(pair[0], "a")
    b = _finite_scalar(pair[1], "b")
    if a <= 0.0 or b <= 0.0:
      raise ValueError(f"SeaCache AB coefficients must be positive, got a={a}, b={b}.")
    return a, b

  if scheduler is None:
    raise ValueError("SeaCache requires a pipeline scheduler or an explicit seacache_ab_schedule.")

  if mode == "flow":
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or index >= len(sigmas):
      raise ValueError(f"SeaCache flow mode requires scheduler.sigmas covering step {index}.")
    sigma = min(1.0 - 1e-6, max(1e-6, _finite_scalar(sigmas[index], "sigma")))
    return 1.0 - sigma, sigma

  alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
  if alphas_cumprod is None:
    raise ValueError("SeaCache VP mode requires scheduler.alphas_cumprod.")
  train_index = index
  timesteps = getattr(scheduler, "timesteps", None)
  if timesteps is not None:
    if index >= len(timesteps):
      raise ValueError(f"SeaCache scheduler.timesteps does not cover step {index}.")
    train_index = int(_finite_scalar(timesteps[index], "timestep"))
  if train_index < 0 or train_index >= len(alphas_cumprod):
    raise ValueError(f"SeaCache alphas_cumprod does not cover training timestep {train_index}.")
  alpha_bar = min(1.0 - 1e-6, max(1e-6, _finite_scalar(alphas_cumprod[train_index], "alpha_bar")))
  return math.sqrt(alpha_bar), math.sqrt(max(1e-16, 1.0 - alpha_bar))


def _normalize_feature_shape(feature_shape) -> Tuple[int, ...]:
  """Validate and normalize a two- or three-dimensional feature geometry."""

  if not isinstance(feature_shape, (list, tuple)) or len(feature_shape) not in (2, 3):
    raise ValueError("seacache_feature_shape must be (H, W) or (F, H, W) for packed tokens.")
  if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in feature_shape):
    raise ValueError(f"SeaCache feature dimensions must be positive integers, got {feature_shape}.")
  return tuple(feature_shape)


def _all_gather_uneven_tokens(tensor: torch.Tensor, mesh) -> torch.Tensor:
  """Gather uneven token shards with padding for both Gloo and NCCL process groups."""

  from ...distributed.communicate.primitives import _gather_size

  group = mesh.get_group()
  sizes = _gather_size(tensor.shape[1], group)
  max_size = max(sizes)
  if tensor.shape[1] < max_size:
    padding_shape = list(tensor.shape)
    padding_shape[1] = max_size - tensor.shape[1]
    tensor = torch.cat((tensor, tensor.new_zeros(padding_shape)), dim=1)
  tensor = tensor.contiguous()
  gathered = [torch.empty_like(tensor) for _ in sizes]
  dist.all_gather(gathered, tensor, group=group)
  return torch.cat([value[:, :size] for value, size in zip(gathered, sizes)], dim=1)


def _gather_cp_tokens(tensor: torch.Tensor, global_tokens: int, parallelism_config) -> torch.Tensor:
  """Restore packed tokens using the existing context-parallel mesh, if necessary."""

  local_tokens = tensor.shape[1]
  if local_tokens == global_tokens:
    return tensor
  if local_tokens > global_tokens:
    raise ValueError(f"SeaCache token count {local_tokens} exceeds global feature geometry "
                     f"{global_tokens}.")
  flat_cp_mesh = getattr(parallelism_config, "_flat_cp_mesh", None)
  flattened_mesh = getattr(parallelism_config, "_flattened_mesh", None)
  mesh = flat_cp_mesh if flat_cp_mesh is not None else flattened_mesh
  if mesh is None:
    raise ValueError(f"SeaCache received {local_tokens} tokens, but feature geometry requires "
                     f"{global_tokens}; provide complete tokens or an initialized CP mesh.")

  from ...distributed.parallel.context_parallel import _EquipartitionSharder

  if getattr(parallelism_config, "ulysses_anything", False):
    return _all_gather_uneven_tokens(tensor, mesh)
  return _EquipartitionSharder.unshard(tensor.contiguous(), 1, mesh)


def apply_sea_filter(
  tensor: torch.Tensor,
  a: float,
  b: float,
  *,
  feature_shape: Optional[Sequence[int]] = None,
  power_exp: float = 2.0,
  power_const: float = 1.0,
  eps: float = 1e-16,
) -> torch.Tensor:
  """Apply the official separable SEA/Wiener filter to a complete feature tensor.

  Dense inputs use channels-first BCHW or BCTHW layout. Packed inputs use `[B, S, C]` and require an
  explicit `(H, W)` or `(F, H, W)` geometry. The returned tensor has the input shape and dtype; FFT
  and gain computation use fp32.

  :param tensor: Complete dense or packed feature tensor.
  :param a: Signal coefficient at the tensor's diffusion step.
  :param b: Noise coefficient at the tensor's diffusion step.
  :param feature_shape: Spatial or temporal-spatial geometry for packed tokens.
  :param power_exp: Exponent of the clean-signal power prior.
  :param power_const: Constant numerator of the clean-signal power prior.
  :param eps: Positive numerical stabilizer.
  :returns: The SEA-filtered tensor in the original layout.
  """

  a = _finite_scalar(a, "a")
  b = _finite_scalar(b, "b")
  power_exp = _finite_scalar(power_exp, "power_exp")
  power_const = _finite_scalar(power_const, "power_const")
  eps = _finite_scalar(eps, "eps")
  if a <= 0.0 or b <= 0.0 or power_exp <= 0.0 or power_const <= 0.0 or eps <= 0.0:
    raise ValueError("SeaCache coefficients, power values, and eps must be positive.")
  if tensor.ndim == 3:
    if feature_shape is None:
      raise ValueError("Packed [B, S, C] tensors require seacache_feature_shape.")
    shape = _normalize_feature_shape(feature_shape)
    if math.prod(shape) != tensor.shape[1]:
      raise ValueError(
        f"SeaCache feature geometry {shape} does not match {tensor.shape[1]} tokens.")
    original_shape = tensor.shape
    value = tensor.reshape(tensor.shape[0], *shape, tensor.shape[-1])
    dims = tuple(range(1, value.ndim - 1))

    def restore(filtered):
      return filtered.reshape(original_shape)

  elif tensor.ndim in (4, 5):
    shape = tuple(tensor.shape[2:])
    if feature_shape is not None and _normalize_feature_shape(feature_shape) != shape:
      raise ValueError(f"SeaCache feature geometry {feature_shape} does not match dense spatial "
                       f"shape {shape}.")
    value = tensor.movedim(1, -1)
    dims = tuple(range(1, value.ndim - 1))

    def restore(filtered):
      return filtered.movedim(-1, 1)

  else:
    raise ValueError("SeaCache expects [B, S, C], BCHW, or BCTHW input, got "
                     f"{tuple(tensor.shape)}.")

  original_dtype = tensor.dtype
  value = value.contiguous().float()
  spectrum = torch.fft.fftn(value, dim=dims)
  gain = None
  for axis in dims:
    frequencies = torch.fft.fftfreq(value.shape[axis], device=value.device, dtype=torch.float32)
    clean_power = power_const / (frequencies.abs().pow(power_exp) + eps)
    axis_gain = a * clean_power / (a * a * clean_power + b * b + eps)
    axis_shape = [1] * value.ndim
    axis_shape[axis] = axis_gain.shape[0]
    axis_gain = axis_gain.reshape(axis_shape)
    gain = axis_gain if gain is None else gain * axis_gain
  mean_gain = gain.mean()
  if not bool(torch.isfinite(mean_gain)) or float(mean_gain) <= 0.0:
    raise ValueError("SeaCache filter gain must have a finite, positive mean.")
  gain = gain / mean_gain
  filtered = torch.fft.ifftn(spectrum * gain, dim=dims).real.to(original_dtype)
  return restore(filtered)


def restore_seacache_geometry(
  tensor: torch.Tensor,
  *,
  feature_shape: Optional[Sequence[int]] = None,
  parallelism_config=None,
) -> torch.Tensor:
  """Restore a complete probe layout without applying FFT, for CFG diff reuse.

  :param tensor: Local dense tensor or packed token shard.
  :param feature_shape: Global spatial or temporal-spatial geometry.
  :param parallelism_config: Existing cache-dit parallel configuration, when available.
  :returns: A tensor with complete spatial/token geometry.
  """

  if tensor.ndim == 3:
    if feature_shape is None:
      raise ValueError("Packed [B, S, C] tensors require seacache_feature_shape.")
    shape = _normalize_feature_shape(feature_shape)
    return _gather_cp_tokens(tensor, math.prod(shape), parallelism_config)
  if tensor.ndim not in (4, 5):
    raise ValueError(f"SeaCache cannot restore geometry for shape {tuple(tensor.shape)}.")
  if feature_shape is not None and _normalize_feature_shape(feature_shape) != tuple(
      tensor.shape[2:]):
    raise ValueError("SeaCache dense spatial shape does not match seacache_feature_shape.")
  return tensor


def build_seacache_indicator(
  tensor: torch.Tensor,
  a: float,
  b: float,
  *,
  feature_shape: Optional[Sequence[int]] = None,
  power_exp: float = 2.0,
  parallelism_config=None,
) -> torch.Tensor:
  """Build one DBCache probe indicator, restoring CP token geometry before filtering.

  :param tensor: Local dense tensor or packed token shard.
  :param a: Signal coefficient for the current tensor.
  :param b: Noise coefficient for the current tensor.
  :param feature_shape: Global spatial or temporal-spatial geometry.
  :param power_exp: Exponent of the clean-signal power prior.
  :param parallelism_config: Existing cache-dit parallel configuration, when available.
  :returns: The complete filtered indicator; CP outputs intentionally have global token length.
  """

  tensor = restore_seacache_geometry(
    tensor,
    feature_shape=feature_shape,
    parallelism_config=parallelism_config,
  )
  return apply_sea_filter(
    tensor,
    a,
    b,
    feature_shape=feature_shape,
    power_exp=power_exp,
  )
