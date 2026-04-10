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
from ..envs import ENV

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

  assert isinstance(
    transformer,
    (torch.nn.Module,
     ModelMixin)), ("transformer must be an instance of torch.nn.Module or ModelMixin, "
                    f"but got {type(transformer)}")
  if getattr(transformer, "_is_parallelized", False):
    logger.warning("The transformer is already parallelized. Skipping parallelism enabling.")
    return transformer

  from ..attention import _maybe_register_custom_attn_backends

  # Ensure custom attention backends are registered in cache-dit.
  _maybe_register_custom_attn_backends()

  # Parallelize Transformer: The check of parallelism backend is only for transformer
  # here. Text Encoder and VAE does not have different parallelism backends now.
  from .transformers import maybe_enable_parallelism_for_transformer

  transformer = maybe_enable_parallelism_for_transformer(
    transformer=transformer,
    parallelism_config=parallelism_config,
  )

  # Check text encoder and VAE for extra parallel modules
  extra_parallel_modules: list[torch.nn.Module] = []
  if parallelism_config.extra_parallel_modules is not None:
    extra_parallel_modules = parallelism_config.extra_parallel_modules
    assert isinstance(
      extra_parallel_modules,
      list), "extra_parallel_modules should be a list of module names or module instances."

  if extra_parallel_modules:
    for module in extra_parallel_modules:
      # Enable parallelism for text encoder
      if check_text_encoder(module) and not check_parallelized(module):
        from .text_encoders import (
          maybe_enable_parallelism_for_text_encoder, )

        maybe_enable_parallelism_for_text_encoder(
          text_encoder=module,
          parallelism_config=parallelism_config,
        )
      # Enable parallelism for ControlNet
      elif check_controlnet(module) and not check_parallelized(module):
        from .controlnets import (
          maybe_enable_parallelism_for_controlnet, )

        maybe_enable_parallelism_for_controlnet(
          controlnet=module,
          parallelism_config=parallelism_config,
        )
        _maybe_set_module_attention_backend(
          module=module,
          parallelism_config=parallelism_config,
        )
      # Enable parallelism for VAE
      elif check_auto_encoder(module) and not check_parallelized(module):
        from .autoencoders import (
          maybe_enable_parallelism_for_auto_encoder, )

        maybe_enable_parallelism_for_auto_encoder(
          auto_encoder=module,
          parallelism_config=parallelism_config,
        )

  # Set attention backend for both context parallelism and tensor parallelism if the
  # transformer is from diffusers and supports setting attention backend.
  _maybe_set_module_attention_backend(
    module=transformer,
    parallelism_config=parallelism_config,
  )

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


# Some helper functions for parallelism enabling
def _maybe_set_module_attention_backend(
  module: torch.nn.Module | ModelMixin,
  parallelism_config: ParallelismConfig,
) -> None:
  """Apply the configured attention backend to one diffusers module when supported.

  :param module: Module whose attention backend may be updated.
  :param parallelism_config: Parallelism configuration carrying the backend preference.
  """

  # Set attention backend for both context parallelism and tensor parallelism if the
  # transformer is from diffusers and supports setting attention backend.
  module_cls_name = module.__class__.__name__
  attention_backend = parallelism_config.attention_backend
  resolved_cache_dit_backend = None

  if attention_backend is not None:
    from ..attention import _AttnBackendRegistry

    try:
      resolved_cache_dit_backend = _AttnBackendRegistry.normalize_backend(attention_backend).value
    except ValueError:
      resolved_cache_dit_backend = None

  def _set_backend_locally(backend_name: str) -> bool:
    applied = False
    for current in module.modules():
      processor = getattr(current, "processor", None)
      if processor is not None:
        processor._attention_backend = backend_name
        current._attention_backend = backend_name
        applied = True
        continue
      if hasattr(current, "_attention_backend"):
        current._attention_backend = backend_name
        applied = True
    return applied

  if hasattr(module, "set_attention_backend") and isinstance(module, ModelMixin):
    if attention_backend is None:
      # Default to native for context parallelism due to:
      # - attn mask support (re-registered in cache-dit)
      # - general compatibility with various models
      # NOTE: We only set default attention backend for NATIVE_DIFFUSER backend here
      # while using context parallelism. For other backends, we do not change the
      # attention backend if it is None.
      if (parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER
          or parallelism_config.backend == ParallelismBackend.NATIVE_HYBRID):
        module.set_attention_backend("native")
        logger.warning("attention_backend is None, set default attention backend of "
                       f"{module_cls_name} to: <native>.")
    else:
      # Ensure custom attention backends are registered in cache-dit.
      if resolved_cache_dit_backend is not None and not ENV.CACHE_DIT_CUSTOM_ATTN_ALREADY_DISPATCH:
        from ..attention import _maybe_register_custom_attn_backends

        _maybe_register_custom_attn_backends()

      backend_name = resolved_cache_dit_backend or attention_backend
      module.set_attention_backend(backend_name)
      logger.info("Found attention_backend from config, set attention backend of "
                  f"{module_cls_name} to: <{backend_name}>.")
  elif attention_backend is not None:
    if resolved_cache_dit_backend is None:
      raise ValueError("Non-diffusers parallel modules only support cache-dit attention backends. "
                       f"Got unsupported backend: {attention_backend}.")
    if not _set_backend_locally(resolved_cache_dit_backend):
      return
    logger.info("Found attention_backend from config, set local cache-dit attention backend of "
                f"{module_cls_name} to: <{resolved_cache_dit_backend}>.")
