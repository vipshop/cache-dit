import torch
from diffusers.models.modeling_utils import ModelMixin
from .backend import ParallelismBackend
from .config import ParallelismConfig
from cache_dit.utils import maybe_empty_cache
from cache_dit.logger import init_logger
from cache_dit.envs import ENV


logger = init_logger(__name__)


def enable_parallelism(
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: ParallelismConfig,
) -> torch.nn.Module:
    assert isinstance(transformer, (torch.nn.Module, ModelMixin)), (
        "transformer must be an instance of torch.nn.Module or ModelMixin, "
        f"but got {type(transformer)}"
    )
    if getattr(transformer, "_is_parallelized", False):
        logger.warning("The transformer is already parallelized. Skipping parallelism enabling.")
        return transformer

    # Parallelize Transformer: The check of parallelism backend is only for transformer
    # here. Text Encoder and VAE does not have different parallelism backends now.
    from .transformers import maybe_enable_parallelism_for_transformer

    transformer = maybe_enable_parallelism_for_transformer(
        transformer=transformer,
        parallelism_config=parallelism_config,
    )
    # Set attention backend for both context parallelism and tensor parallelism if the
    # transformer is from diffusers and supports setting attention backend.
    _maybe_set_module_attention_backend(
        module=transformer,
        parallelism_config=parallelism_config,
    )

    # Check text encoder and VAE for extra parallel modules
    extra_parallel_modules: list[torch.nn.Module] = []
    if parallelism_config.parallel_kwargs is not None:
        extra_parallel_modules = parallelism_config.parallel_kwargs.get(
            "extra_parallel_modules", []
        )

    if extra_parallel_modules:
        for module in extra_parallel_modules:
            # Enable parallelism for text encoder
            if _is_text_encoder(module) and not _is_parallelized(module):
                from .text_encoders import (
                    maybe_enable_parallelism_for_text_encoder,
                )

                maybe_enable_parallelism_for_text_encoder(
                    text_encoder=module,
                    parallelism_config=parallelism_config,
                )
            # Enable parallelism for ControlNet
            elif _is_controlnet(module) and not _is_parallelized(module):
                from .controlnets import (
                    maybe_enable_parallelism_for_controlnet,
                )

                maybe_enable_parallelism_for_controlnet(
                    controlnet=module,
                    parallelism_config=parallelism_config,
                )
                _maybe_set_module_attention_backend(
                    module=module,
                    parallelism_config=parallelism_config,
                )
            # Enable parallelism for VAE
            elif _is_auto_encoder(module) and not _is_parallelized(module):
                from .autoencoders import (
                    maybe_enable_parallelism_for_auto_encoder,
                )

                maybe_enable_parallelism_for_auto_encoder(
                    auto_encoder=module,
                    parallelism_config=parallelism_config,
                )

    transformer._extra_parallel_modules = extra_parallel_modules  # type: ignore[attr-defined]
    # NOTE: Workaround for potential memory peak issue after parallelism
    # enabling, specially for tensor parallelism in native pytorch backend.
    maybe_empty_cache()

    return transformer


def remove_parallelism_stats(
    module: torch.nn.Module,
) -> torch.nn.Module:

    if not getattr(module, "_is_parallelized", False):
        return module

    def _remove_parallel_stats(module: torch.nn.Module) -> None:
        if hasattr(module, "_is_parallelized"):
            del module._is_parallelized
        if hasattr(module, "_parallelism_config"):
            del module._parallelism_config

    _remove_parallel_stats(module)

    # remove parallelism stats for extra parallel modules
    if not hasattr(module, "_extra_parallel_modules"):
        return module

    extra_parallel_modules = getattr(module, "_extra_parallel_modules", [])
    for extra_module in extra_parallel_modules:
        _remove_parallel_stats(extra_module)

    del module._extra_parallel_modules  # type: ignore[attr-defined]
    return module


# Some helper functions for parallelism enabling
def _maybe_set_module_attention_backend(
    module: torch.nn.Module | ModelMixin,
    parallelism_config: ParallelismConfig,
) -> None:
    # Set attention backend for both context parallelism and tensor parallelism if the
    # transformer is from diffusers and supports setting attention backend.
    module_cls_name = module.__class__.__name__
    if hasattr(module, "set_attention_backend") and isinstance(module, ModelMixin):
        attention_backend = parallelism_config.parallel_kwargs.get("attention_backend", None)
        # native, _native_cudnn, flash, etc.
        if attention_backend is None:
            # Default to native for context parallelism due to:
            # - attn mask support (re-registered in cache-dit)
            # - general compatibility with various models
            # NOTE: We only set default attention backend for NATIVE_DIFFUSER backend here
            # while using context parallelism. For other backends, we do not change the
            # attention backend if it is None.
            if (
                parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER
                or parallelism_config.backend == ParallelismBackend.NATIVE_HYBRID
            ):
                module.set_attention_backend("native")
                logger.warning(
                    "attention_backend is None, set default attention backend of "
                    f"{module_cls_name} to native for context parallelism or "
                    "hybrid parallelism."
                )
        else:
            # Ensure custom attention backends are registered in cache-dit.
            if not ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_ALREADY_DISPATCH:
                from .attention import (
                    _maybe_register_custom_attn_backends,
                )

                _maybe_register_custom_attn_backends()

            module.set_attention_backend(attention_backend)
            logger.info(
                "Found attention_backend from config, set attention backend of "
                f"{module_cls_name} to: {attention_backend}."
            )


def _is_text_encoder(module: torch.nn.Module) -> bool:
    _import_module = module.__class__.__module__
    return _import_module.startswith("transformers")


def _is_controlnet(module: torch.nn.Module) -> bool:
    _import_module = module.__class__.__module__
    return _import_module.startswith("diffusers.models.controlnet")


def _is_auto_encoder(module: torch.nn.Module) -> bool:
    _import_module = module.__class__.__module__
    return _import_module.startswith("diffusers.models.autoencoder")


def _is_parallelized(module: torch.nn.Module) -> bool:
    return getattr(module, "_is_parallelized", False)
