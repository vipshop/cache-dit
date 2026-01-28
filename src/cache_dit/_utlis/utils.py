import torch
import argparse
import torch.distributed as dist
from diffusers import DiffusionPipeline
from typing import Optional, List, Tuple
from diffusers.quantizers import PipelineQuantizationConfig

from ..logger import init_logger
from ..platforms import current_platform
from ..quantize import normalize_quantize_type, quantize
from ..compile.utils import set_compile_configs
from ..parallelism import ParallelismBackend, ParallelismConfig
from ..caching import enable_cache, steps_mask, set_attn_backend
from ..caching import (
    BlockAdapter,
    DBCacheConfig,
    TaylorSeerCalibratorConfig,
    load_configs,
    load_parallelism_config,
)
from ..summary import strify as summary_strify

logger = init_logger(__name__)


class MemoryTracker:
    """Track peak GPU memory usage during execution."""

    def __init__(self, device=None):
        self.device = device if device is not None else current_platform.current_device()
        self.enabled = current_platform.is_accelerator_available()
        self.peak_memory = 0

    def __enter__(self):
        if self.enabled:
            current_platform.reset_peak_memory_stats(self.device)
            current_platform.synchronize(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            current_platform.synchronize(self.device)
            self.peak_memory = current_platform.max_memory_allocated(self.device)

    def get_peak_memory_gb(self):
        """Get peak memory in GB."""
        return self.peak_memory / (1024**3)

    def report(self):
        """Print memory usage report."""
        if self.enabled:
            peak_gb = self.get_peak_memory_gb()
            logger.info(f"Peak GPU memory usage: {peak_gb:.2f} GB")
            return peak_gb
        return 0


def GiB():
    try:
        if not current_platform.is_accelerator_available():
            return 0
        total_memory_bytes = current_platform.get_device_properties(
            current_platform.current_device(),
        ).total_memory
        total_memory_gib = total_memory_bytes / (1024**3)
        return int(total_memory_gib)
    except Exception:
        return 0


def get_args(
    parse: bool = True,
) -> argparse.ArgumentParser | argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Model and data paths
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override model path if provided",
    )
    parser.add_argument(
        "--controlnet-path",
        type=str,
        default=None,
        help="Override controlnet model path if provided",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Override lora model path if provided",
    )
    parser.add_argument(
        "--transformer-path",
        type=str,
        default=None,
        help="Override transformer model path if provided",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Override image path if provided",
    )
    parser.add_argument(
        "--mask-image-path",
        type=str,
        default=None,
        help="Override mask image path if provided",
    )
    # Acceleration Config path
    parser.add_argument(
        "--config-path",
        "--config",
        type=str,
        default=None,
        help="Path to CacheDiT configuration YAML file",
    )
    # Sampling settings
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override default prompt if provided",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Override default negative prompt if provided",
    )
    parser.add_argument(
        "--num_inference_steps",
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup steps before measuring performance",
    )
    parser.add_argument(
        "--warmup-num-inference-steps",
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of warmup inference steps per warmup before measuring performance",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the inference for performance measurement",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of the generated image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of the generated image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-frames",
        "--frames",
        type=int,
        default=None,
        help="Number of frames to generate for video",
    )
    # Output settings
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the generated output, e.g., output.png or output.mp4",
    )
    # Cache specific settings
    parser.add_argument(
        "--cache",
        action="store_true",
        default=False,
        help="Enable Cache Acceleration",
    )
    parser.add_argument(
        "--cache-summary",
        "--summary",
        action="store_true",
        default=False,
        help="Enable Cache Summary logging",
    )
    parser.add_argument(
        "--Fn-compute-blocks",
        "--Fn",
        type=int,
        default=1,
        help="CacheDiT Fn_compute_blocks parameter",
    )
    parser.add_argument(
        "--Bn-compute-blocks",
        "--Bn",
        type=int,
        default=0,
        help="CacheDiT Bn_compute_blocks parameter",
    )
    parser.add_argument(
        "--residual-diff-threshold",
        "--rdt",
        type=float,
        default=0.24,
        help="CacheDiT residual diff threshold",
    )
    parser.add_argument(
        "--max-warmup-steps",
        "--ws",
        type=int,
        default=8,
        help="Maximum warmup steps for CacheDiT",
    )
    parser.add_argument(
        "--warmup-interval",
        "--wi",
        type=int,
        default=1,
        help="Warmup interval for CacheDiT",
    )
    parser.add_argument(
        "--max-cached-steps",
        "--mc",
        type=int,
        default=-1,
        help="Maximum cached steps for CacheDiT",
    )
    parser.add_argument(
        "--max-continuous-cached-steps",
        "--mcc",
        type=int,
        default=3,
        help="Maximum continuous cached steps for CacheDiT",
    )
    parser.add_argument(
        "--taylorseer",
        action="store_true",
        default=False,
        help="Enable TaylorSeer for CacheDiT",
    )
    parser.add_argument(
        "--taylorseer-order",
        "-order",
        type=int,
        default=1,
        help="TaylorSeer order",
    )
    parser.add_argument(
        "--steps-mask",
        action="store_true",
        default=False,
        help="Enable steps mask for CacheDiT",
    )
    parser.add_argument(
        "--mask-policy",
        "--scm",
        type=str,
        default=None,
        choices=[
            None,
            "slow",
            "s",
            "medium",
            "m",
            "fast",
            "f",
            "ultra",
            "u",
        ],
        help="Pre-defined steps computation mask policy",
    )
    # Quantization settings
    parser.add_argument(
        "--quantize",
        "--q",
        action="store_true",
        default=False,
        help="Enable quantization for transformer",
    )
    # float8, float8_weight_only, int8, int8_weight_only, int4, int4_weight_only
    parser.add_argument(
        "--quantize-type",
        "--q-type",
        type=str,
        default=None,
        choices=[
            None,
            "float8",
            "float8_weight_only",
            "float8_wo",  # alias for float8_weight_only
            "int8",
            "int8_weight_only",
            "int8_wo",  # alias for int8_weight_only
            "int4",
            "int4_weight_only",
            "int4_wo",  # alias for int4_weight_only
            "bitsandbytes_4bit",
            "bnb_4bit",  # alias for bitsandbytes_4bit
        ],
    )
    parser.add_argument(
        "--quantize-text-encoder",
        "--q-text",
        action="store_true",
        default=False,
        help="Enable quantization for text encoder",
    )
    parser.add_argument(
        "--quantize-text-type",
        "--q-text-type",
        type=str,
        default=None,
        choices=[
            None,
            "float8",
            "float8_weight_only",
            "float8_wo",  # alias for float8_weight_only
            "int8",
            "int8_weight_only",
            "int8_wo",  # alias for int8_weight_only
            "int4",
            "int4_weight_only",
            "int4_wo",  # alias for int4_weight_only
            "bitsandbytes_4bit",
            "bnb_4bit",  # alias for bitsandbytes_4bit
        ],
    )
    parser.add_argument(
        "--quantize-controlnet",
        "--q-controlnet",
        action="store_true",
        default=False,
        help="Enable quantization for text encoder",
    )
    parser.add_argument(
        "--quantize-controlnet-type",
        "--q-controlnet-type",
        type=str,
        default=None,
        choices=[
            None,
            "float8",
            "float8_weight_only",
            "float8_wo",  # alias for float8_weight_only
            "int8",
            "int8_weight_only",
            "int8_wo",  # alias for int8_weight_only
            "int4",
            "int4_weight_only",
            "int4_wo",  # alias for int4_weight_only
            "bitsandbytes_4bit",
            "bnb_4bit",  # alias for bitsandbytes_4bit
        ],
    )
    # Parallelism settings
    parser.add_argument(
        "--parallel-type",
        "--parallel",
        type=str,
        default=None,
        choices=[
            None,
            "tp",
            "ulysses",
            "ring",
            "usp",
            # hybrid cp + tp
            "ulysses_tp",  # prefer ulysses first
            "ring_tp",
            "tp_ulysses",  # prefer tp first
            "tp_ring",
            "usp_tp",
        ],
    )
    parser.add_argument(
        "--parallel-vae",
        action="store_true",
        default=False,
        help="Enable VAE parallelism if applicable.",
    )
    parser.add_argument(
        "--parallel-text-encoder",
        "--parallel-text",
        action="store_true",
        default=False,
        help="Enable text encoder parallelism if applicable.",
    )
    parser.add_argument(
        "--parallel-controlnet",
        action="store_true",
        default=False,
        help="Enable ControlNet parallelism if applicable.",
    )
    parser.add_argument(
        "--attn",  # attention backend for context parallelism
        type=str,
        default=None,
        choices=[
            None,
            "flash",
            "_flash_3",  # FlashAttention-3
            # Based on this fix: https://github.com/huggingface/diffusers/pull/12563
            "native",  # native pytorch attention: sdpa
            "_native_cudnn",
            # '_sdpa_cudnn' is only in cache-dit to support context parallelism
            # with attn masks, e.g., ZImage. It is not in diffusers yet.
            "_sdpa_cudnn",
            "sage",  # Need install sageattention: https://github.com/thu-ml/SageAttention
            "_native_npu",  # native npu attention
        ],
    )
    # Ulysses context parallelism settings
    parser.add_argument(
        "--ulysses-anything",
        "--uaa",
        action="store_true",
        default=False,
        help="Enable Ulysses Anything Attention for context parallelism",
    )
    parser.add_argument(
        "--ulysses-float8",
        "--ufp8",
        action="store_true",
        default=False,
        help="Enable Ulysses Attention/UAA Float8 for context parallelism",
    )
    parser.add_argument(
        "--ulysses-async",
        "--uaqkv",
        action="store_true",
        default=False,
        help="Enabled experimental Async QKV Projection with Ulysses for context parallelism",
    )
    # Ring context parallelism settings
    parser.add_argument(
        "--ring-rotate-method",
        "--rotate",
        type=str,
        default="p2p",
        choices=[
            "allgather",
            "p2p",
        ],
        help="Ring Attention rotation method for context parallelism",
    )
    parser.add_argument(
        "--ring-no-convert-to-fp32",
        "--ring-no-fp32",
        "--no-fp32",
        action="store_true",
        default=False,
        help="Disable convert Ring Attention output and lse to fp32 for context parallelism",
    )
    # Offload settings
    parser.add_argument(
        "--cpu-offload",
        "--cpu-offload-model",
        action="store_true",
        default=False,
        help="Enable CPU offload for model if applicable.",
    )
    parser.add_argument(
        "--sequential-cpu-offload",
        action="store_true",
        default=False,
        help="Enable sequential GPU offload for model if applicable.",
    )
    parser.add_argument(
        "--device-map-balance",
        "--device-map",
        action="store_true",
        default=False,
        help="Enable automatic device map balancing model if multiple GPUs are available.",
    )
    # Vae tiling/slicing settings
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        default=False,
        help="Enable VAE tiling for low memory device.",
    )
    parser.add_argument(
        "--vae-slicing",
        action="store_true",
        default=False,
        help="Enable VAE slicing for low memory device.",
    )
    # Compiling settings
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable compile for transformer",
    )
    parser.add_argument(
        "--compile-repeated-blocks",
        action="store_true",
        default=False,
        help="Enable compile for repeated blocks in transformer",
    )
    parser.add_argument(
        "--compile-vae",
        action="store_true",
        default=False,
        help="Enable compile for VAE",
    )
    parser.add_argument(
        "--compile-text-encoder",
        "--compile-text",
        action="store_true",
        default=False,
        help="Enable compile for text encoder",
    )
    parser.add_argument(
        "--compile-controlnet",
        action="store_true",
        default=False,
        help="Enable compile for ControlNet",
    )
    parser.add_argument(
        "--max-autotune",
        action="store_true",
        default=False,
        help="Enable max-autotune mode for torch.compile",
    )
    # Profiling and memory tracking settings
    parser.add_argument(
        "--track-memory",
        action="store_true",
        default=False,
        help="Track and report peak GPU memory usage",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable profiling with torch.profiler",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default=None,
        help="Name for the profiling session",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Directory to save profiling results",
    )
    parser.add_argument(
        "--profile-activities",
        type=str,
        nargs="+",
        default=["CPU", "GPU"],
        choices=["CPU", "GPU", "MEM"],
        help="Activities to profile (CPU, GPU, MEM)",
    )
    parser.add_argument(
        "--profile-with-stack",
        action="store_true",
        default=True,
        help="profile with stack for better traceability",
    )
    parser.add_argument(
        "--profile-record-shapes",
        action="store_true",
        default=True,
        help="profile record shapes for better analysis",
    )
    # Lora settings
    parser.add_argument(
        "--disable-fuse-lora",
        type=str,
        default=None,
        help="Disable fuse_lora even if lora weights are provided.",
    )
    # Generator device
    parser.add_argument(
        "--generator-device",
        "--gen-device",
        type=str,
        default=None,
        help="Device for torch.Generator, e.g., 'cuda' or 'cpu'. "
        "If not set, use 'cpu' for better reproducibility across "
        "different hardware.",
    )

    args_or_parser = parser.parse_args() if parse else parser
    if parse:
        return maybe_postprocess_args(args_or_parser)
    return args_or_parser


def get_base_args(parse: bool = True) -> argparse.Namespace | argparse.ArgumentParser:
    return get_args(parse=parse)  # For future extension if needed


def maybe_postprocess_args(args: argparse.Namespace) -> argparse.Namespace:
    # Force enable quantization if quantize_type is specified
    if args.quantize_type is not None:
        args.quantize = True

    # Handle alias for quantize_type
    if args.quantize and args.quantize_type is None:
        args.quantize_type = "float8_weight_only"  # default type

    args.quantize_type = normalize_quantize_type(args.quantize_type)

    # Force enable quantization for text encoder if quantize_text_type is specified
    if args.quantize_text_type is not None:
        args.quantize_text_encoder = True
    # Handle alias for quantize_text_type
    if args.quantize_text_encoder and args.quantize_text_type is None:
        # default to same as quantize_type
        args.quantize_text_type = args.quantize_type

    args.quantize_text_type = normalize_quantize_type(args.quantize_text_type)

    # Force enable quantization for controlnet if quantize_controlnet_type is specified
    if args.quantize_controlnet_type is not None:
        args.quantize_controlnet = True
    # Handle alias for quantize_controlnet_type
    if args.quantize_controlnet and args.quantize_controlnet_type is None:
        # default to same as quantize_type
        args.quantize_controlnet_type = args.quantize_type

    args.quantize_controlnet_type = normalize_quantize_type(args.quantize_controlnet_type)

    if args.mask_policy is not None and not args.steps_mask:
        # Enable steps mask if mask_policy is specified
        args.steps_mask = True
    # Handle alias for mask_policy
    if args.mask_policy == "s":  # alias
        args.mask_policy = "slow"
    if args.mask_policy == "m":  # alias
        args.mask_policy = "medium"
    if args.mask_policy == "f":  # alias
        args.mask_policy = "fast"
    if args.mask_policy == "u":  # alias
        args.mask_policy = "ultra"
    return args


def get_text_encoder_from_pipe(
    pipe: DiffusionPipeline,
) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    pipe_cls_name = pipe.__class__.__name__
    if (
        hasattr(pipe, "text_encoder_2")
        and not pipe_cls_name.startswith("Hunyuan")
        and not pipe_cls_name.startswith("Kandinsky")
    ):
        # Specific for FluxPipeline, FLUX.1-dev
        return getattr(pipe, "text_encoder_2"), "text_encoder_2"
    elif hasattr(pipe, "text_encoder_3"):  # HiDream pipeline
        return getattr(pipe, "text_encoder_3"), "text_encoder_3"
    elif hasattr(pipe, "text_encoder"):  # General case
        return getattr(pipe, "text_encoder"), "text_encoder"
    else:
        return None, None


def prepare_extra_parallel_modules(
    args,
    pipe_or_adapter: DiffusionPipeline | BlockAdapter,
    custom_extra_modules: Optional[List[torch.nn.Module]] = None,
) -> list:
    if custom_extra_modules is not None:
        return custom_extra_modules

    if isinstance(pipe_or_adapter, BlockAdapter):
        pipe = pipe_or_adapter.pipe
        assert pipe is not None, "Please set extra_parallel_modules manually if pipe is None."
    else:
        pipe = pipe_or_adapter

    extra_parallel_modules = []

    if args.parallel_text_encoder:
        text_encoder, _ = get_text_encoder_from_pipe(pipe)
        if text_encoder is not None:
            extra_parallel_modules.append(text_encoder)
        else:
            logger.warning(
                "parallel-text-encoder is set but no text encoder found in the pipeline."
            )

    if args.parallel_vae:
        assert not args.vae_tiling, "VAE tiling is not compatible with VAE parallelism."
        assert not args.vae_slicing, "VAE slicing is not compatible with VAE parallelism."
        if hasattr(pipe, "vae"):
            extra_parallel_modules.append(getattr(pipe, "vae"))
        else:
            logger.warning("parallel-vae is set but no VAE found in the pipeline.")

    if args.parallel_controlnet:
        if hasattr(pipe, "controlnet"):
            extra_parallel_modules.append(getattr(pipe, "controlnet"))
        else:
            logger.warning("parallel-controlnet is set but no ControlNet found in the pipeline.")

    return extra_parallel_modules


def maybe_compile_text_encoder(
    args,
    pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
    if args.compile_text_encoder:
        torch.set_float32_matmul_precision("high")

        if isinstance(pipe_or_adapter, BlockAdapter):
            pipe = pipe_or_adapter.pipe
            assert pipe is not None, "Please compile text encoder manually if pipe is None."
        else:
            pipe = pipe_or_adapter

        text_encoder, name = get_text_encoder_from_pipe(pipe)
        if text_encoder is not None and not isinstance(
            text_encoder,
            torch._dynamo.OptimizedModule,  # already compiled
        ):
            # Find module to be compiled, [encoder, model, model.language_model, ...]
            _module_to_compile = text_encoder
            if hasattr(_module_to_compile, "model"):
                if hasattr(_module_to_compile.model, "language_model"):
                    _module_to_compile = _module_to_compile.model.language_model
                else:
                    _module_to_compile = _module_to_compile.model

            if hasattr(_module_to_compile, "encoder"):
                _module_to_compile = _module_to_compile.encoder

            _module_to_compile_cls_name = _module_to_compile.__class__.__name__
            _text_encoder_cls_name = text_encoder.__class__.__name__
            if isinstance(_module_to_compile, torch.nn.Module):
                logger.info(
                    f"Compiling text encoder module {name}:{_text_encoder_cls_name}:"
                    f"{_module_to_compile_cls_name} ..."
                )
                _module_to_compile = torch.compile(
                    _module_to_compile,
                    mode="max-autotune-no-cudagraphs" if args.max_autotune else "default",
                )
                # Set back the compiled text encoder
                if hasattr(text_encoder, "model"):
                    if hasattr(text_encoder.model, "language_model"):
                        text_encoder.model.language_model = _module_to_compile
                    else:
                        text_encoder.model = _module_to_compile
                if hasattr(text_encoder, "encoder"):
                    text_encoder.encoder = _module_to_compile

                setattr(pipe, name, text_encoder)
            else:
                logger.warning(
                    f"Cannot compile text encoder module {name}:{_text_encoder_cls_name}:"
                    f"{_module_to_compile_cls_name} Not a torch.nn.Module."
                )
        else:
            logger.warning("compile-text-encoder is set but no text encoder found in the pipeline.")
    return pipe_or_adapter


def maybe_compile_controlnet(
    args,
    pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
    if args.compile_controlnet:
        set_compile_configs()
        torch.set_float32_matmul_precision("high")

        if isinstance(pipe_or_adapter, BlockAdapter):
            pipe = pipe_or_adapter.pipe
            assert pipe is not None, "Please compile transformer manually if pipe is None."
        else:
            pipe = pipe_or_adapter

        if hasattr(pipe, "controlnet"):
            controlnet = getattr(pipe, "controlnet", None)
            if controlnet is not None and not isinstance(
                controlnet,
                torch._dynamo.OptimizedModule,  # already compiled
            ):
                controlnet_cls_name = controlnet.__class__.__name__
                if isinstance(controlnet, torch.nn.Module):
                    logger.info(f"Compiling controlnet module: {controlnet_cls_name} ...")
                    controlnet = torch.compile(
                        controlnet,
                        mode="max-autotune-no-cudagraphs" if args.max_autotune else "default",
                    )
                    setattr(pipe, "controlnet", controlnet)
                else:
                    logger.warning(
                        f"Cannot compile controlnet module: {controlnet_cls_name} Not a"
                        " torch.nn.Module."
                    )
            setattr(pipe, "controlnet", controlnet)
        else:
            logger.warning("compile is set but no controlnet found in the pipeline.")


def maybe_compile_vae(
    args,
    pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
    if args.compile_vae:
        torch.set_float32_matmul_precision("high")

        if isinstance(pipe_or_adapter, BlockAdapter):
            pipe = pipe_or_adapter.pipe
            assert pipe is not None, "Please compile VAE manually if pipe is None."
        else:
            pipe = pipe_or_adapter

        if hasattr(pipe, "vae"):
            vae = getattr(pipe, "vae", None)
            if vae is not None and not isinstance(
                vae,
                torch._dynamo.OptimizedModule,  # already compiled
            ):
                vae_cls_name = vae.__class__.__name__
                if hasattr(vae, "encoder"):
                    _encoder_to_compile = vae.encoder
                    if isinstance(_encoder_to_compile, torch.nn.Module):
                        logger.info(f"Compiling VAE encoder module: {vae_cls_name}.encoder ...")
                        vae.encoder = torch.compile(
                            _encoder_to_compile,
                            mode="max-autotune-no-cudagraphs" if args.max_autotune else "default",
                        )
                    else:
                        logger.warning(
                            f"Cannot compile VAE encoder module: {vae_cls_name}.encoder Not a"
                            " torch.nn.Module."
                        )
                if hasattr(vae, "decoder"):
                    _decoder_to_compile = vae.decoder
                    if isinstance(_decoder_to_compile, torch.nn.Module):
                        logger.info(f"Compiling VAE decoder module: {vae_cls_name}.decoder ...")
                        vae.decoder = torch.compile(
                            _decoder_to_compile,
                            mode="max-autotune-no-cudagraphs" if args.max_autotune else "default",
                        )
                    else:
                        logger.warning(
                            f"Cannot compile VAE decoder module: {vae_cls_name}.decoder Not a"
                            " torch.nn.Module."
                        )
                setattr(pipe, "vae", vae)
            else:
                logger.warning(f"Cannot compile VAE module: {vae_cls_name} Not a torch.nn.Module.")
        else:
            logger.warning("compile-vae is set but no VAE found in the pipeline.")
    return pipe_or_adapter


def maybe_compile_transformer(
    args,
    pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
    if args.compile:
        set_compile_configs()
        torch.set_float32_matmul_precision("high")

        if isinstance(pipe_or_adapter, BlockAdapter):
            pipe = pipe_or_adapter.pipe
            assert pipe is not None, "Please compile transformer manually if pipe is None."
        else:
            pipe = pipe_or_adapter

        if hasattr(pipe, "transformer"):
            transformer = getattr(pipe, "transformer", None)
            if transformer is not None and not isinstance(
                transformer,
                torch._dynamo.OptimizedModule,  # already compiled
            ):
                transformer_cls_name = transformer.__class__.__name__
                if isinstance(transformer, torch.nn.Module):
                    logger.info(f"Compiling transformer module: {transformer_cls_name} ...")
                    transformer = torch.compile(
                        transformer,
                        mode="max-autotune-no-cudagraphs" if args.max_autotune else "default",
                    )
                    setattr(pipe, "transformer", transformer)
                else:
                    logger.warning(
                        f"Cannot compile transformer module: {transformer_cls_name} Not a"
                        " torch.nn.Module."
                    )
            setattr(pipe, "transformer", transformer)
        else:
            logger.warning("compile is set but no transformer found in the pipeline.")

        if hasattr(pipe, "transformer_2"):
            transformer_2 = getattr(pipe, "transformer_2", None)
            if transformer_2 is not None and not isinstance(
                transformer_2,
                torch._dynamo.OptimizedModule,  # already compiled
            ):
                transformer_2_cls_name = transformer_2.__class__.__name__
                if isinstance(transformer_2, torch.nn.Module):
                    logger.info(f"Compiling transformer_2 module: {transformer_2_cls_name} ...")
                    transformer_2 = torch.compile(
                        transformer_2,
                        mode="max-autotune-no-cudagraphs" if args.max_autotune else "default",
                    )
                    setattr(pipe, "transformer_2", transformer_2)
                else:
                    logger.warning(
                        f"Cannot compile transformer_2 module: {transformer_2_cls_name} Not a"
                        " torch.nn.Module."
                    )
            setattr(pipe, "transformer_2", transformer_2)

    return pipe_or_adapter


def maybe_quantize_transformer(
    args,
    pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
    # Quantize transformer by default if quantization is enabled
    if args.quantize:
        if args.quantize_type in ("bitsandbytes_4bit", "bnb_4bit"):
            logger.debug(
                "bitsandbytes_4bit quantization should be handled by"
                " PipelineQuantizationConfig in from_pretrained."
            )
            return pipe_or_adapter

        if isinstance(pipe_or_adapter, BlockAdapter):
            pipe = pipe_or_adapter.pipe
            assert pipe is not None, "Please quantize transformer manually if pipe is None."
        else:
            pipe = pipe_or_adapter

        _class_not_supported_per_row = [
            "QwenImageTransformer2DModel",
        ]

        def is_per_row_supported(transformer):
            transformer_cls_name = transformer.__class__.__name__
            return transformer_cls_name not in _class_not_supported_per_row

        if hasattr(pipe, "transformer"):
            transformer = getattr(pipe, "transformer", None)
            if transformer is not None:
                transformer_cls_name = transformer.__class__.__name__
                if isinstance(transformer, torch.nn.Module):
                    logger.info(
                        f"Quantizing transformer module: {transformer_cls_name} to"
                        f" {args.quantize_type} ..."
                    )
                    transformer = quantize(
                        transformer,
                        quant_type=args.quantize_type,
                        per_row=is_per_row_supported(transformer),
                    )
                    setattr(pipe, "transformer", transformer)
                else:
                    logger.warning(
                        f"Cannot quantize transformer module: {transformer_cls_name} Not a"
                        " torch.nn.Module."
                    )
            setattr(pipe, "transformer", transformer)
        else:
            logger.warning("quantize is set but no transformer found in the pipeline.")

        if hasattr(pipe, "transformer_2"):
            transformer_2 = getattr(pipe, "transformer_2", None)
            if transformer_2 is not None:
                transformer_2_cls_name = transformer_2.__class__.__name__
                if isinstance(transformer_2, torch.nn.Module):
                    logger.info(
                        f"Quantizing transformer_2 module: {transformer_2_cls_name} to"
                        f" {args.quantize_type} ..."
                    )
                    transformer_2 = quantize(
                        transformer_2,
                        quant_type=args.quantize_type,
                        per_row=is_per_row_supported(transformer_2),
                    )
                    setattr(pipe, "transformer_2", transformer_2)
                else:
                    logger.warning(
                        f"Cannot quantize transformer_2 module: {transformer_2_cls_name} Not a"
                        " torch.nn.Module."
                    )
            setattr(pipe, "transformer_2", transformer_2)

    return pipe_or_adapter


def maybe_quantize_text_encoder(
    args,
    pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
    # Quantize text encoder by default if quantize_text_encoder is enabled
    if args.quantize_text_encoder:
        if args.quantize_text_type in ("bitsandbytes_4bit", "bnb_4bit"):
            logger.debug(
                "bitsandbytes_4bit quantization should be handled by"
                " PipelineQuantizationConfig in from_pretrained."
            )
            return pipe_or_adapter

        if isinstance(pipe_or_adapter, BlockAdapter):
            pipe = pipe_or_adapter.pipe
            assert pipe is not None, "Please quantize text encoder manually if pipe is None."
        else:
            pipe = pipe_or_adapter

        text_encoder, name = get_text_encoder_from_pipe(pipe)
        if text_encoder is not None:
            text_encoder_cls_name = text_encoder.__class__.__name__
            if isinstance(text_encoder, torch.nn.Module):
                logger.info(
                    f"Quantizing text encoder module: {name}:{text_encoder_cls_name} to"
                    f" {args.quantize_text_type} ..."
                )
                text_encoder = quantize(
                    text_encoder,
                    quant_type=args.quantize_text_type,
                )
                setattr(pipe, name, text_encoder)
            else:
                logger.warning(
                    f"Cannot quantize text encoder module: {name}:{text_encoder_cls_name} Not a"
                    " torch.nn.Module."
                )
        else:
            logger.warning("quantize is set but no text encoder found in the pipeline.")
    return pipe_or_adapter


def maybe_quantize_controlnet(
    args,
    pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
    # Quantize controlnet by default if quantize_controlnet is enabled
    if args.quantize_controlnet:
        if args.quantize_controlnet_type in ("bitsandbytes_4bit", "bnb_4bit"):
            logger.debug(
                "bitsandbytes_4bit quantization should be handled by"
                " PipelineQuantizationConfig in from_pretrained."
            )
            return pipe_or_adapter

        if isinstance(pipe_or_adapter, BlockAdapter):
            pipe = pipe_or_adapter.pipe
            assert pipe is not None, "Please quantize controlnet manually if pipe is None."
        else:
            pipe = pipe_or_adapter

        if hasattr(pipe, "controlnet"):
            controlnet = getattr(pipe, "controlnet", None)
            if controlnet is not None:
                controlnet_cls_name = controlnet.__class__.__name__
                if isinstance(controlnet, torch.nn.Module):
                    logger.info(
                        f"Quantizing controlnet module: {controlnet_cls_name} to"
                        f" {args.quantize_controlnet_type} ..."
                    )
                    controlnet = quantize(
                        controlnet,
                        quant_type=args.quantize_controlnet_type,
                    )
                    setattr(pipe, "controlnet", controlnet)
                else:
                    logger.warning(
                        f"Cannot quantize controlnet module: {controlnet_cls_name} Not a"
                        " torch.nn.Module."
                    )
            setattr(pipe, "controlnet", controlnet)
        else:
            logger.warning("quantize_controlnet is set but no controlnet found in the pipeline.")
    return pipe_or_adapter


def pipe_quant_bnb_4bit_config(
    args,
    components_to_quantize: Optional[List[str]] = ["text_encoder"],
) -> Optional[PipelineQuantizationConfig]:
    if not args.quantize_text_encoder and not args.quantize:
        return None

    if components_to_quantize:
        # Remove all components if quantize type is not bitsandbytes_4bit
        if args.quantize_type != "bitsandbytes_4bit":
            if "transformer" in components_to_quantize:
                components_to_quantize.remove("transformer")
            if "transformer_2" in components_to_quantize:
                components_to_quantize.remove("transformer_2")
        if args.quantize_text_type != "bitsandbytes_4bit":
            if "text_encoder" in components_to_quantize:
                components_to_quantize.remove("text_encoder")
            if "text_encoder_2" in components_to_quantize:
                components_to_quantize.remove("text_encoder_2")

        # Remove text encoder if parallel_text_encoder is enabled
        if args.parallel_text_encoder:
            if "text_encoder" in components_to_quantize:
                components_to_quantize.remove("text_encoder")
            if "text_encoder_2" in components_to_quantize:
                components_to_quantize.remove("text_encoder_2")

    if components_to_quantize:
        quantization_config = (
            (
                PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_compute_dtype": torch.bfloat16,
                    },
                    components_to_quantize=components_to_quantize,
                )
            )
            if args.quantize or args.quantize_text_encoder
            else None
        )
    else:
        quantization_config = None

    return quantization_config


def maybe_vae_tiling_or_slicing(
    args,
    pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
    if args.vae_tiling or args.vae_slicing:
        assert not args.parallel_vae, "VAE tiling/slicing is not compatible with VAE parallelism."

        if isinstance(pipe_or_adapter, BlockAdapter):
            pipe = pipe_or_adapter.pipe
            assert pipe is not None, "Please enable VAE tiling/slicing manually if pipe is None."
        else:
            pipe = pipe_or_adapter

        if hasattr(pipe, "vae"):
            vae = getattr(pipe, "vae", None)
            if vae is not None:
                vae_cls_name = vae.__class__.__name__
                if args.vae_tiling:
                    if hasattr(vae, "enable_tiling"):
                        logger.info(f"Enabling VAE tiling for module: {vae_cls_name} ...")
                        vae.enable_tiling()
                    else:
                        logger.warning(
                            f"Cannot enable VAE tiling for module: {vae_cls_name} No enable_tiling"
                            " method."
                        )
                if args.vae_slicing:
                    if hasattr(vae, "enable_slicing"):
                        logger.info(f"Enabling VAE slicing for module: {vae_cls_name} ...")
                        vae.enable_slicing()
                    else:
                        logger.warning(
                            f"Cannot enable VAE slicing for module: {vae_cls_name} No enable_slicing"
                            " method."
                        )
                setattr(pipe, "vae", vae)
            else:
                logger.warning("vae-tiling is set but no VAE found in the pipeline.")
    return pipe_or_adapter


def maybe_cpu_offload(
    args,
    pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> bool:
    _, device = get_rank_device()
    if args.cpu_offload or args.sequential_cpu_offload:
        if isinstance(pipe_or_adapter, BlockAdapter):
            pipe = pipe_or_adapter.pipe
            assert pipe is not None, "Please enable cpu offload manually if pipe is None."
        else:
            pipe = pipe_or_adapter

        pipe_cls_name = pipe.__class__.__name__
        if args.sequential_cpu_offload:
            logger.info(f"Enabling Sequential CPU offload for the model {pipe_cls_name} ...")
            pipe.enable_sequential_cpu_offload(device=device)
        else:
            logger.info(f"Enabling CPU offload for the model {pipe_cls_name} ...")
            pipe.enable_model_cpu_offload(device=device)

        return True

    return False


def maybe_apply_optimization(
    args,
    pipe_or_adapter,
    **kwargs,
):
    default_num_inference_steps = kwargs.pop("default_num_inference_steps", None)
    if args.cache or args.parallel_type is not None or args.config_path is not None:

        if args.config_path is None:
            # Construct acceleration configs from command line args if config path is not provided
            cache_config = kwargs.pop("cache_config", None)
            parallelism_config = kwargs.pop("parallelism_config", None)

            extra_parallel_modules = prepare_extra_parallel_modules(
                args,
                pipe_or_adapter,
                custom_extra_modules=kwargs.get("extra_parallel_modules", None),
            )

            parallel_kwargs = {
                "attention_backend": ("native" if not args.attn else args.attn),
                # e.g., text_encoder_2 in FluxPipeline, text_encoder in Flux2Pipeline
                "extra_parallel_modules": extra_parallel_modules,
                # experimental settings for context parallelism
                "experimental_ulysses_anything": args.ulysses_anything,
                "experimental_ulysses_float8": args.ulysses_float8,
                "experimental_ulysses_async": args.ulysses_async,
                "ring_rotate_method": args.ring_rotate_method,
                "ring_convert_to_fp32": not args.ring_no_convert_to_fp32,
            }

            # Caching and Parallelism
            if args.steps_mask and args.mask_policy is not None:
                logger.info(
                    f"Using steps computation mask with policy: {args.mask_policy} for caching."
                )
                if default_num_inference_steps is None:
                    assert (
                        args.num_inference_steps is not None
                    ), "num_inference_steps (--steps) must be provided for steps mask."
                    num_inference_steps = args.num_inference_steps
                else:
                    num_inference_steps = default_num_inference_steps
                steps_computation_mask = steps_mask(
                    total_steps=num_inference_steps,
                    mask_policy=args.mask_policy,
                )
            else:
                steps_computation_mask = None

            def _prepare_distributed_size():
                if args.parallel_type is not None:
                    world_size = dist.get_world_size() if dist.is_initialized() else 1
                    ulysses_size = world_size if args.parallel_type == "ulysses" else None
                    ring_size = world_size if args.parallel_type == "ring" else None
                    tp_size = world_size if args.parallel_type == "tp" else None
                    if args.parallel_type == "usp":
                        ulysses_size = max(1, world_size // 2)
                        ring_size = world_size // ulysses_size
                        tp_size = None
                    elif args.parallel_type == "ulysses_tp":
                        ulysses_size = max(1, world_size // 2)  # e.g, 4
                        ring_size = None
                        tp_size = world_size // ulysses_size  # e.g., 2
                    elif args.parallel_type == "tp_ulysses":
                        tp_size = max(1, world_size // 2)
                        ulysses_size = world_size // tp_size
                        ring_size = None
                    elif args.parallel_type == "ring_tp":
                        ulysses_size = None
                        ring_size = max(1, world_size // 2)
                        tp_size = world_size // ring_size
                    elif args.parallel_type == "tp_ring":
                        ulysses_size = None
                        tp_size = max(1, world_size // 2)
                        ring_size = world_size // tp_size
                    elif args.parallel_type == "usp_tp":
                        assert world_size == 8, "usp_tp currently only supports world size of 8."
                        ulysses_size = 2
                        ring_size = 2
                        tp_size = 2
                    return ulysses_size, ring_size, tp_size
                return None, None, None

            ulysses_size, ring_size, tp_size = _prepare_distributed_size()

            enable_cache(
                pipe_or_adapter,
                cache_config=(
                    DBCacheConfig(
                        Fn_compute_blocks=args.Fn_compute_blocks,
                        Bn_compute_blocks=args.Bn_compute_blocks,
                        max_warmup_steps=args.max_warmup_steps,
                        warmup_interval=args.warmup_interval,
                        max_cached_steps=args.max_cached_steps,
                        max_continuous_cached_steps=args.max_continuous_cached_steps,
                        residual_diff_threshold=args.residual_diff_threshold,
                        enable_separate_cfg=kwargs.get("enable_separate_cfg", None),
                        steps_computation_mask=steps_computation_mask,
                    )
                    if cache_config is None and args.cache
                    else cache_config
                ),
                calibrator_config=(
                    TaylorSeerCalibratorConfig(
                        taylorseer_order=args.taylorseer_order,
                    )
                    if args.taylorseer
                    else None
                ),
                params_modifiers=kwargs.get("params_modifiers", None),
                parallelism_config=(
                    ParallelismConfig(
                        ulysses_size=ulysses_size,
                        ring_size=ring_size,
                        tp_size=tp_size,
                        backend=ParallelismBackend.AUTO,
                        parallel_kwargs=parallel_kwargs,
                    )
                    if parallelism_config is None and args.parallel_type is not None
                    else parallelism_config
                ),
                # Allow attention backend for non-parallelism case
                attention_backend=(
                    ("native" if not args.attn else args.attn)
                    if args.parallel_type is None
                    else None
                ),
            )
        else:
            # Apply acceleration configs from config path
            enable_cache(
                pipe_or_adapter,
                **load_configs(args.config_path),
            )
            logger.info(f"Applied acceleration from {args.config_path}.")
    else:
        logger.info("No caching or parallelism is applied.")
        if args.attn is not None:
            logger.info(f"Applying custom attention backend: {args.attn} ...")
            set_attn_backend(pipe_or_adapter, attention_backend=args.attn)

    # Quantization
    # WARN: Must apply quantization after tensor parallelism is applied.
    # torchao is compatible with tensor parallelism but requires to be
    # applied after TP.
    maybe_quantize_transformer(args, pipe_or_adapter)
    maybe_quantize_text_encoder(args, pipe_or_adapter)
    maybe_quantize_controlnet(args, pipe_or_adapter)

    # VAE Tiling or Slicing
    maybe_vae_tiling_or_slicing(args, pipe_or_adapter)

    # Compilation
    maybe_compile_transformer(args, pipe_or_adapter)
    maybe_compile_text_encoder(args, pipe_or_adapter)
    maybe_compile_controlnet(args, pipe_or_adapter)
    maybe_compile_vae(args, pipe_or_adapter)

    # CPU Offload
    _, device = get_rank_device()
    if not maybe_cpu_offload(args, pipe_or_adapter):
        # Set device if no cpu offload
        if isinstance(pipe_or_adapter, BlockAdapter):
            pipe = pipe_or_adapter.pipe
        else:
            pipe = pipe_or_adapter
        if pipe is not None and not args.device_map_balance:
            pipe.to(device)

    return pipe_or_adapter


def strify(args, pipe_or_stats):
    base_str = ""
    if args.height is not None and args.width is not None:
        base_str += f"{args.height}x{args.width}_"
    quantize_type = args.quantize_type if args.quantize else ""
    if quantize_type != "":
        quantize_type = f"_{quantize_type}"
    base_str += (
        f"C{int(args.compile)}_Q{int(args.quantize)}{quantize_type}_"
        f"{summary_strify(pipe_or_stats)}"
    )
    if args.ulysses_anything:
        base_str += "_ulysses_anything"
        if args.ulysses_float8:
            base_str += "_float8"
    else:
        if args.ulysses_float8:
            base_str += "_ulysses_float8"
    if args.ulysses_async:
        base_str += "_ulysses_async"
    if args.parallel_type == "ring" or args.parallel_type == "usp":
        if args.ring_rotate_method is not None:
            base_str += f"_rotated_{args.ring_rotate_method}"
        if args.ring_no_convert_to_fp32:
            base_str += "_no_fp32"
    if args.parallel_text_encoder:
        if "_TEP" not in base_str:
            base_str += "_TEP"  # Text Encoder Parallelism
    if args.parallel_vae:
        if "_VAEP" not in base_str:
            base_str += "_VAEP"  # VAE Parallelism
    if args.parallel_controlnet:
        if "_CNP" not in base_str:
            base_str += "_CNP"  # ControlNet Parallelism
    if args.attn is not None:
        base_str += f"_{args.attn.strip('_')}"
    return base_str


def get_rank_device():
    available = current_platform.is_accelerator_available()
    device_type = current_platform.device_type
    if dist.is_initialized():
        rank = dist.get_rank()
        device = torch.device(device_type, rank % current_platform.device_count())
        return rank, device
    return 0, torch.device(device_type if available else "cpu")


def maybe_init_distributed(args=None):
    from ..platforms.platform import CpuPlatform

    platform_full_backend = current_platform.full_dist_backend
    cpu_full_backend = CpuPlatform.full_dist_backend
    # Always use hybrid comm backend since there no side effect.
    backend = f"{cpu_full_backend},{platform_full_backend}"
    if args is not None:
        if args.parallel_type is not None:
            dist.init_process_group(
                backend=backend,
            )
            rank, device = get_rank_device()
            current_platform.set_device(device)
            return rank, device
        elif args.config_path is not None:
            # check if distributed is needed from config file
            has_parallelism_config = load_parallelism_config(
                args.config_path,
                check_only=True,
            )
            if has_parallelism_config:
                if not dist.is_initialized():
                    dist.init_process_group(
                        backend=backend,
                    )
                rank, device = get_rank_device()
                current_platform.set_device(device)
                return rank, device
            else:
                # no distributed needed
                rank, device = get_rank_device()
                return rank, device
        else:
            # no distributed needed
            rank, device = get_rank_device()
            return rank, device
    else:
        # always init distributed for other examples
        if not dist.is_initialized():
            dist.init_process_group(
                backend=platform_full_backend,
            )
        rank, device = get_rank_device()
        current_platform.set_device(device)
        return rank, device


def maybe_destroy_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_profiler_from_args(args, profile_name=None):
    from ..profiler import ProfilerContext

    return ProfilerContext(
        enabled=args.profile,
        activities=getattr(args, "profile_activities", ["CPU", "GPU"]),
        output_dir=getattr(args, "profile_dir", None),
        profile_name=profile_name or getattr(args, "profile_name", None),
        with_stack=getattr(args, "profile_with_stack", True),
        record_shapes=getattr(args, "profile_record_shapes", True),
    )
