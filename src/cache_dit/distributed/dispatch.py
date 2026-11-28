from collections.abc import Callable

import torch
from diffusers.models.modeling_utils import ModelMixin
from .backend import ParallelismBackend
from .config import ParallelismConfig
from ..utils import maybe_empty_cache
from ..utils import check_text_encoder
from ..utils import check_auto_encoder
from ..utils import check_controlnet
from ..utils import check_parallelized
from ..logger import init_logger

logger = init_logger(__name__)


def enable_parallelism(
  transformer: torch.nn.Module | ModelMixin,
  parallelism_config: ParallelismConfig,
) -> torch.nn.Module:
  """Enable cache-dit parallelism on one transformer and any configured extra modules.

  :param transformer: Transformer module to parallelize.
  :param parallelism_config: Validated parallelism configuration describing the target layout.
  :returns: The parallelized transformer module.
  """

  return _enable_parallelism([transformer], parallelism_config,
                             parallelism_config.extra_parallel_modules)[0]


def _prepare_tp_planners(
  transformers: list[torch.nn.Module],
  parallelism_config: ParallelismConfig,
  extra_parallel_modules: list[torch.nn.Module],
) -> dict[torch.nn.Module, Callable[..., torch.nn.Module]]:
  pending_transformers = dict.fromkeys(
    transformer for transformer in transformers if not check_parallelized(transformer))
  if not pending_transformers:
    return {}

  from .text_encoders.dispatch import _prepare_text_encoder_tp
  from .transformers.dispatch import _prepare_transformer_tp

  tp_apply_by_module = {}
  has_local_shards = False
  if parallelism_config.tp_enabled():
    for transformer in pending_transformers:
      tp_apply_by_module[transformer], uses_local = _prepare_transformer_tp(
        transformer, parallelism_config)
      has_local_shards = has_local_shards or uses_local

  for module in extra_parallel_modules:
    if (check_text_encoder(module) and not check_parallelized(module)
        and module not in tp_apply_by_module):
      tp_apply_by_module[module], uses_local = _prepare_text_encoder_tp(module, parallelism_config)
      has_local_shards = has_local_shards or uses_local

  if not has_local_shards:
    raise ValueError("local_weight_shards requires a selected TP planner with CPU weight sharding "
                     "support.")
  return tp_apply_by_module


def _enable_parallelism(
  transformers: list[torch.nn.Module],
  parallelism_config: ParallelismConfig,
  extra_parallel_modules: list[str | torch.nn.Module] | None,
) -> list[torch.nn.Module]:
  extra_modules = [] if extra_parallel_modules is None else extra_parallel_modules
  assert isinstance(
    extra_modules,
    list), "extra_parallel_modules should be a list of module names or module instances."

  tp_apply_by_module = {}
  if parallelism_config.local_weight_shards:
    # Finish every local weight check before the first CP or TP hook is installed.
    tp_apply_by_module = _prepare_tp_planners(transformers, parallelism_config, extra_modules)

  parallelism_config.extra_parallel_modules = extra_parallel_modules
  return [
    _apply_parallelism(transformer, parallelism_config, extra_modules, tp_apply_by_module)
    for transformer in transformers
  ]


def _apply_parallelism(
  transformer: torch.nn.Module | ModelMixin,
  parallelism_config: ParallelismConfig,
  extra_parallel_modules: list[torch.nn.Module],
  tp_apply_by_module: dict[torch.nn.Module, Callable[..., torch.nn.Module]],
) -> torch.nn.Module:
  assert isinstance(
    transformer,
    (torch.nn.Module,
     ModelMixin)), ("transformer must be an instance of torch.nn.Module or ModelMixin, "
                    f"but got {type(transformer)}")
  if getattr(transformer, "_is_parallelized", False):
    logger.warning("The transformer is already parallelized. Skipping parallelism enabling.")
    return transformer

  from ..attention import _maybe_register_custom_attn_backends, set_attn_backend

  # Ensure custom attention backends are registered in cache-dit.
  _maybe_register_custom_attn_backends()

  def _set_module_attn_backend(module: torch.nn.Module | ModelMixin) -> None:
    attention_backend = parallelism_config.attention_backend
    backend = ParallelismBackend.canonicalize(parallelism_config.backend)
    if (attention_backend is None and isinstance(module, ModelMixin)
        and hasattr(module, "set_attention_backend")
        and backend in (ParallelismBackend.CACHE_DIT, ParallelismBackend.NATIVE_HYBRID)):
      attention_backend = "native"

    set_attn_backend(module, attention_backend=attention_backend)

  # Parallelize Transformer: The check of parallelism backend is only for transformer
  # here. Text Encoder and VAE does not have different parallelism backends now.
  from .transformers import parallelize_transformer

  transformer = parallelize_transformer(
    transformer=transformer,
    parallelism_config=parallelism_config,
    tp_apply=tp_apply_by_module.get(transformer),
  )

  for module in extra_parallel_modules:
    if check_parallelized(module):
      continue
    if check_text_encoder(module):
      from .text_encoders import parallelize_text_encoder

      parallelize_text_encoder(
        text_encoder=module,
        parallelism_config=parallelism_config,
        tp_apply=tp_apply_by_module.get(module),
      )
    elif check_controlnet(module):
      from .controlnets import parallelize_controlnet

      parallelize_controlnet(controlnet=module, parallelism_config=parallelism_config)
      _set_module_attn_backend(module)
    elif check_auto_encoder(module):
      from .autoencoders import parallelize_autoencoder

      parallelize_autoencoder(auto_encoder=module, parallelism_config=parallelism_config)

  # Set attention backend for both context parallelism and tensor parallelism if the
  # transformer is from diffusers and supports setting attention backend.
  _set_module_attn_backend(transformer)

  transformer._extra_parallel_modules = extra_parallel_modules  # type: ignore[attr-defined]
  # NOTE: Workaround for potential memory peak issue after parallelism
  # enabling, specially for tensor parallelism in native pytorch backend.
  maybe_empty_cache()

  return transformer


def remove_parallelism_stats(module: torch.nn.Module, ) -> torch.nn.Module:
  """Remove cache-dit parallelism bookkeeping attributes from a module tree.

  :param module: Module whose parallelism metadata should be cleared.
  :returns: The same module with cache-dit parallelism markers removed.
  """

  if not getattr(module, "_is_parallelized", False):
    return module

  def _remove_parallel_stats(module: torch.nn.Module) -> None:
    if hasattr(module, "_is_parallelized"):
      del module._is_parallelized
    if hasattr(module, "_parallelism_config"):
      del module._parallelism_config

    # Only use 1 depth for the recursion of removing parallel stats in extra sub modules,
    # since the extra parallel modules should not have nested extra parallel modules.
    extra_parallel_modules = getattr(module, "_extra_parallel_modules", [])
    for extra_module in extra_parallel_modules:
      _remove_parallel_stats(extra_module)

    if hasattr(module, "_extra_parallel_modules"):
      del module._extra_parallel_modules

  _remove_parallel_stats(module)

  return module
