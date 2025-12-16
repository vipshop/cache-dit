import torch
import torch.distributed as dist
from typing import Union, Optional
from diffusers.models.modeling_utils import ModelMixin
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from .parallel_backend import ParallelismBackend
from .parallel_config import ParallelismConfig
from cache_dit.utils import maybe_empty_cache
from cache_dit.envs import ENV
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def enable_parallelism(
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: ParallelismConfig,
) -> torch.nn.Module:
    assert isinstance(
        transformer, torch.nn.Module
    ), f"transformer must be an instance of torch.nn.Module, but got {type(transformer)}"
    if getattr(transformer, "_is_parallelized", False):
        logger.warning("The transformer is already parallelized. Skipping parallelism enabling.")
        return transformer
    # Parallelize Transformer: The check of parallelism backend is only for transformer
    # here. Text Encoder and VAE does not have different parallelism backends now.
    if parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER:
        from .transformers.native_diffusers import (
            maybe_enable_parallelism_for_transformer,
        )

        transformer = maybe_enable_parallelism_for_transformer(
            transformer,
            parallelism_config,
        )
    elif parallelism_config.backend == ParallelismBackend.NATIVE_PYTORCH:
        from .transformers.native_pytorch import (
            maybe_enable_parallelism_for_transformer,
        )

        transformer = maybe_enable_parallelism_for_transformer(
            transformer,
            parallelism_config,
        )
    else:
        raise ValueError(f"Parallel backend {parallelism_config.backend} is not supported yet.")

    # Set attention backend for both context parallelism and tensor parallelism if the
    # transformer is from diffusers and supports setting attention backend.
    if hasattr(transformer, "set_attention_backend") and isinstance(transformer, ModelMixin):
        attention_backend = parallelism_config.parallel_kwargs.get("attention_backend", None)
        # native, _native_cudnn, flash, etc.
        if attention_backend is None:
            # Default to native for context parallelism due to:
            # - attn mask support (re-registered in cache-dit)
            # - general compatibility with various models
            # NOTE: We only set default attention backend for NATIVE_DIFFUSER backend here
            # while using context parallelism. For other backends, we do not change the
            # attention backend if it is None.
            if parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER:
                transformer.set_attention_backend("native")
                logger.warning(
                    "attention_backend is None, set default attention backend "
                    "to native for context parallelism in NATIVE_DIFFUSER backend."
                )
        else:
            # Ensure custom attention backends are registered in cache-dit.
            if not ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_ALREADY_DISPATCH:
                from .transformers.native_diffusers import (
                    _maybe_register_custom_attn_backends,
                )

                _maybe_register_custom_attn_backends()

            transformer.set_attention_backend(attention_backend)
            logger.info(
                "Found attention_backend from config, set attention "
                f"backend to: {attention_backend}"
            )

    # Check text encoder and VAE for extra parallel modules
    extra_parallel_modules: list[torch.nn.Module] = []
    if parallelism_config.parallel_kwargs is not None:
        extra_parallel_modules = parallelism_config.parallel_kwargs.get(
            "extra_parallel_modules", []
        )

    if extra_parallel_modules:
        from .text_encoders.native_pytorch import (
            maybe_enable_parallelism_for_text_encoder,
        )

        for module in extra_parallel_modules:
            # Enable parallelism for text encoder
            maybe_enable_parallelism_for_text_encoder(
                text_encoder=module,
                parallelism_config=parallelism_config,
            )
            if getattr(module, "_is_parallelized", False):
                continue
            # TODO: Enable parallelism for VAE if needed.

    # NOTE: Workaround for potential memory peak issue after parallelism
    # enabling, specially for tensor parallelism in native pytorch backend.
    maybe_empty_cache()

    return transformer


def remove_parallelism_stats(
    transformer: torch.nn.Module,
) -> torch.nn.Module:
    if not getattr(transformer, "_is_parallelized", False):
        logger.warning("The transformer is not parallelized. Skipping removing parallelism.")
        return transformer

    if hasattr(transformer, "_is_parallelized"):
        del transformer._is_parallelized  # type: ignore[attr-defined]
    if hasattr(transformer, "_parallelism_config"):
        del transformer._parallelism_config  # type: ignore[attr-defined]
    return transformer


def maybe_pad_prompt(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompt: str,
    extra_prompt: Optional[str] = None,  # e.g., negative prompt
    num_parition: Optional[int] = None,  # e.g., dist.get_world_size()
    pad_token: Optional[str] = None,  # e.g., default tokenizer.pad_token
    num_extra_tokens: Optional[int] = 0,  # e.g., negative prompt tokens length
    verbose: bool = True,
) -> str:
    """Pad the prompt to make sure the number of tokens is divisible by num_partition."""
    assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), (
        f"tokenizer must be an instance of PreTrainedTokenizer or PreTrainedTokenizerFast, "
        f"but got {type(tokenizer)}"
    )
    inputs_ids = tokenizer(prompt, return_tensors="pt")

    if num_parition is None:
        if dist.is_initialized():
            num_parition = dist.get_world_size()
        else:
            num_parition = 1

    if num_parition <= 1:
        return prompt

    if pad_token is None:
        pad_token = tokenizer.pad_token
        if pad_token is None:
            pad_token = tokenizer.eos_token
            if pad_token is None:
                pad_token = " "
                logger.warning(
                    "pad_token and eos_token are not set in the tokenizer. "
                    "Using space ' ' as the pad_token."
                )

    seq_len = inputs_ids.input_ids.shape[1]  # [batch_size, seq_len]

    # Add extra tokens length, e.g., negative prompt tokens length
    partition_seq_len = seq_len
    partition_seq_len += num_extra_tokens
    if extra_prompt is not None:
        extra_inputs_ids = tokenizer(extra_prompt, return_tensors="pt")
        partition_seq_len += extra_inputs_ids.input_ids.shape[1]
        num_extra_tokens += extra_inputs_ids.input_ids.shape[1]

    if partition_seq_len % num_parition != 0:
        pad_len = num_parition - (partition_seq_len % num_parition)
        if verbose:
            logger.info(
                f"Padding the prompt from seq_len {seq_len} to "
                f"{seq_len + pad_len} to make {seq_len + pad_len} + "
                f"{num_extra_tokens} = {seq_len + pad_len + num_extra_tokens} "
                f"divisible by num_partition {num_parition}."
            )
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        assert isinstance(pad_token_id, int), f"pad_token {pad_token} has more than one token."

        pad_ids = torch.full(
            (1, pad_len),
            pad_token_id,
            dtype=inputs_ids.input_ids.dtype,
        )
        inputs_ids.input_ids = torch.cat([inputs_ids.input_ids, pad_ids], dim=1)

        prompt = tokenizer.decode(inputs_ids.input_ids[0])

        new_seq_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        new_partition_seq_len = new_seq_len + num_extra_tokens
        assert new_partition_seq_len % num_parition == 0, (
            f"Failed to pad the prompt to make it divisible by num_partition {num_parition}. "
            f"Got new_seq_len {new_seq_len}, new_partition_seq_len {new_partition_seq_len}."
        )
    return prompt
