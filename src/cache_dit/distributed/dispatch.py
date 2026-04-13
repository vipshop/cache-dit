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
    if (attention_backend is None and isinstance(module, ModelMixin)
        and hasattr(module, "set_attention_backend") and parallelism_config.backend
        in (ParallelismBackend.NATIVE_DIFFUSER, ParallelismBackend.NATIVE_HYBRID)):
      attention_backend = "native"

    set_attn_backend(module, attention_backend=attention_backend)

  # Parallelize Transformer: The check of parallelism backend is only for transformer
  # here. Text Encoder and VAE does not have different parallelism backends now.
  from .transformers import parallelize_transformer

  transformer = parallelize_transformer(
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
          parallelize_text_encoder, )

        parallelize_text_encoder(
          text_encoder=module,
          parallelism_config=parallelism_config,
        )
      # Enable parallelism for ControlNet
      elif check_controlnet(module) and not check_parallelized(module):
        from .controlnets import (
          parallelize_controlnet, )

        parallelize_controlnet(
          controlnet=module,
          parallelism_config=parallelism_config,
        )
        _set_module_attn_backend(module)
      # Enable parallelism for VAE
      elif check_auto_encoder(module) and not check_parallelized(module):
        from .autoencoders import (
          parallelize_autoencoder, )

        parallelize_autoencoder(
          auto_encoder=module,
          parallelism_config=parallelism_config,
        )

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
