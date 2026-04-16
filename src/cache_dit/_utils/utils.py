import torch
import argparse
import torch.distributed as dist
from diffusers import DiffusionPipeline
from typing import Any, Dict, Optional, List, Tuple
from diffusers.quantizers import PipelineQuantizationConfig

from ..logger import init_logger
from ..platforms import current_platform
from ..compile.utils import set_compile_configs
from ..distributed import ParallelismBackend, ParallelismConfig
from ..caching import enable_cache, steps_mask
from ..attention import set_attn_backend
from ..caching import (
  BlockAdapter,
  DBCacheConfig,
  TaylorSeerCalibratorConfig,
  load_configs,
  load_parallelism_config,
)
from ..offload import _find_offload_related_hf_hook
from ..offload import layerwise_cpu_offload
from ..offload import remove_layerwise_offload
from ..quantization import quantize, QuantizeConfig

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
    """Get peak memory in GB.

    :returns: The resolved peak memory gb.
    """
    return self.peak_memory / (1024 ** 3)

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
      current_platform.current_device(), ).total_memory
    total_memory_gib = total_memory_bytes / (1024 ** 3)
    return int(total_memory_gib)
  except Exception:
    return 0


def get_args(parse: bool = True, ) -> argparse.ArgumentParser | argparse.Namespace:
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
  # Force skip negative prompt in some specific cases
  parser.add_argument(
    "--skip-negative_prompt",
    "--skip-neg",
    action="store_true",
    help="Force skip negative prompt even if negative prompt is provided.",
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
    "--warmup-seed",
    type=int,
    default=None,
    help=(
      "Optional random seed used only for warmup forwards. When set, warmup uses this seed while "
      "formal repeated inference continues to use --seed. This is especially useful for few-shot "
      "SVDQ runs where runtime quantization usually happens during warmup."),
  )
  parser.add_argument(
    "--warmup-prompt",
    type=str,
    default=None,
    help=("Optional prompt used only for warmup forwards. When set, warmup uses this prompt while "
          "formal repeated inference continues to use --prompt or the example default prompt."),
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
    "--input-height",
    type=int,
    default=None,
    help="Height of the input image",
  )
  parser.add_argument(
    "--input-width",
    type=int,
    default=None,
    help="Width of the input image",
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
    default=0.12,
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
  # per row quantization
  parser.add_argument(
    "--disable-per-row",
    "--no-per-row",
    action="store_true",
    default=False,
    help="Disable per row quantization for transformer",
  )
  # float8, float8_weight_only, int8, int8_weight_only, int4, int4_weight_only
  parser.add_argument(
    "--quantize-type",
    "--q-type",
    type=str,
    default=None,
    choices=[
      None,
      "float8_per_row",
      "float8_per_tensor",
      "float8_per_block",
      "float8_weight_only",
      "int8_per_row",
      "int8_per_tensor",
      "int8_weight_only",
      "int4_weight_only",
      "svdq_int4_r32_dq",
      "svdq_int4_r64_dq",
      "svdq_int4_r128_dq",
      "svdq_int4_r256_dq",
      "bitsandbytes_4bit",
    ],
  )
  parser.add_argument(
    "--disable-regional-quantize",
    "--disable-regional",
    "--no-regional",
    action="store_true",
    default=False,
    help="Disable quantization for repeated blocks in transformer",
  )
  parser.add_argument(
    "--disable-per-tensor-fallback",
    "--no-per-tensor-fallback",
    action="store_true",
    default=False,
    help="Disable (float8 only) per-tensor fallback quantization for transformer",
  )
  # some quick start flags for transformer quantization.
  parser.add_argument(
    "--float8-per-row",
    "--float8",
    action="store_true",
    default=False,
    help="Enable float8 per-row quantization for transformer",
  )
  parser.add_argument(
    "--float8-per-tensor",
    action="store_true",
    default=False,
    help="Enable float8 per-tensor quantization for transformer",
  )
  parser.add_argument(
    "--float8-per-block",
    action="store_true",
    default=False,
    help="Enable float8 per-block quantization for transformer",
  )
  parser.add_argument(
    "--float8-weight-only",
    "--float8-wo",
    action="store_true",
    default=False,
    help="Enable float8 weight-only quantization for transformer",
  )
  parser.add_argument(
    "--float8-blockwise",
    "--float8-bw",
    action="store_true",
    default=False,
    help="Enable float8 blockwise quantization for transformer",
  )
  parser.add_argument(
    "--int8-per-row",
    "--int8",
    action="store_true",
    default=False,
    help="Enable int8 per-row quantization for transformer",
  )
  parser.add_argument(
    "--int8-per-tensor",
    action="store_true",
    default=False,
    help="Enable int8 per-tensor quantization for transformer",
  )
  parser.add_argument(
    "--int8-weight-only",
    "--int8-wo",
    action="store_true",
    default=False,
    help="Enable int8 weight-only quantization for transformer",
  )
  parser.add_argument(
    "--int4-weight-only",
    "--int4-wo",
    action="store_true",
    default=False,
    help="Enable int4 weight-only quantization for transformer",
  )
  parser.add_argument(
    "--svdq-int4-r32-dq",
    "--svdq-r32",
    action="store_true",
    default=False,
    help="Enable SVDQ INT4 dynamic quantization with rank 32 for transformer",
  )
  parser.add_argument(
    "--svdq-int4-r64-dq",
    "--svdq-r64",
    action="store_true",
    default=False,
    help="Enable SVDQ INT4 dynamic quantization with rank 64 for transformer",
  )
  parser.add_argument(
    "--svdq-int4-r128-dq",
    "--svdq-r128",
    action="store_true",
    default=False,
    help="Enable SVDQ INT4 dynamic quantization with rank 128 for transformer",
  )
  parser.add_argument(
    "--svdq-int4-r256-dq",
    "--svdq-r256",
    action="store_true",
    default=False,
    help="Enable SVDQ INT4 dynamic quantization with rank 256 for transformer",
  )
  # quantization for extra modules: text encoder, vae, controlnet, etc.
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
      "float8_per_row",
      "float8_per_tensor",
      "float8_per_block",
      "float8_weight_only",
      "int8_per_row",
      "int8_per_tensor",
      "int8_weight_only",
      "int4_weight_only",
      "bitsandbytes_4bit",
    ],
  )
  parser.add_argument(
    "--quantize-controlnet",
    "--q-controlnet",
    action="store_true",
    default=False,
    help="Enable quantization for ControlNet",
  )
  parser.add_argument(
    "--quantize-controlnet-type",
    "--q-controlnet-type",
    type=str,
    default=None,
    choices=[
      None,
      "float8_per_row",
      "float8_per_tensor",
      "float8_per_block",
      "float8_weight_only",
      "int8_per_row",
      "int8_per_tensor",
      "int8_weight_only",
      "int4_weight_only",
      "bitsandbytes_4bit",
    ],
  )
  parser.add_argument(
    "--quantize-verbose",
    "--q-verbose",
    action="store_true",
    default=False,
    help="Print the verbose logs of the quantization process",
  )
  parser.add_argument(
    "--svdq-smooth-strategy",
    "--svdq-smooth",
    type=str,
    default="identity",
    choices=["identity", "weight", "weight_inv", "few_shot"],
    help="Smooth strategy for SVDQ DQ quantization. Default: identity.",
  )
  parser.add_argument(
    "--svdq-calibrate-precision",
    "--svdq-calib",
    type=str,
    default="low",
    choices=["low", "medium", "high"],
    help="Calibration / decomposition precision for SVDQ quantization. Default: low.",
  )
  parser.add_argument(
    "--svdq-runtime",
    type=str,
    default="v1",
    choices=["v1", "v2", "v3"],
    help=
    "Runtime SVDQ W4A4 GEMM kernel. Use v2 for the CUDA v2 plain path or v3 for the CuTe DSL rewrite path.",
  )
  parser.add_argument(
    "--svdq-few-shot-steps",
    type=int,
    default=1,
    help=("How many transformer/module forwards to observe before materializing SVDQ few-shot "
          "quantization. This counts cumulative root-module forwards on the armed transformer, not "
          "pipeline invocations."),
  )
  parser.add_argument(
    "--svdq-few-shot-relax-factor",
    type=float,
    default=1.5,
    help=
    ("Maximum few-shot relax factor applied to activation spans before smooth-scale recomputation. "
     "Must be >= 1.0 so the observed activation span is preserved or amplified, never reduced. "
     "Values above about 3.0 are often too aggressive and may oversmooth outputs; try 1.5-2.5 first."
     ),
  )
  parser.add_argument(
    "--svdq-few-shot-relax-top-ratio",
    type=float,
    default=0.25,
    help="Fraction of the largest few-shot activation spans used to define the relax threshold.",
  )
  parser.add_argument(
    "--svdq-few-shot-relax-strategy",
    type=str,
    default="auto",
    choices=["fixed", "top", "auto", "stable_auto", "power", "log", "rank", "top_q4"],
    help=(
      "How few-shot relax factors are assigned over activation spans. 'fixed' keeps the original "
      "activation span unchanged; 'top' keeps channels below the threshold unchanged; 'auto' uses "
      "a linear ramp; 'stable_auto' bucketizes that ramp for better run-to-run stability; "
      "'power', 'log', and 'rank' use stronger monotonic curves; 'top_q4' is accepted as an "
      "alias of 'top'."),
  )
  parser.add_argument(
    "--svdq-quantize-device",
    type=str,
    default="cuda",
    choices=["auto", "cpu", "cuda"],
    help=("Device used for SVDQ decomposition and packing math. The example CLI defaults to 'cuda' "
          "so CPU-resident models can still quantize layer-by-layer on GPU before the final move."),
  )
  parser.add_argument(
    "--svdq-offload-quantized-layers-to-cpu",
    dest="svdq_offload_quantized_layers_to_cpu",
    action="store_true",
    default=False,
    help=(
      "After each SVDQ linear layer is quantized, move it back to CPU immediately. This reduces "
      "peak memory during quantization and is forced on when --svdq-layerwise-offload is active."),
  )
  parser.add_argument(
    "--svdq-keep-quantized-layers-on-device",
    dest="svdq_offload_quantized_layers_to_cpu",
    action="store_false",
    help=(
      "Keep newly quantized SVDQ layers on the quantization device instead of offloading them to "
      "CPU. Ignored when --svdq-layerwise-offload is active."),
  )
  parser.add_argument(
    "--svdq-layerwise-offload",
    action="store_true",
    default=False,
    help=("Enable cache-dit layerwise CPU offload for SVDQ PTQ calibration and DQ few-shot "
          "activation collection when the root module is CPU-resident."),
  )
  parser.add_argument(
    "--svdq-defer-final-to-cuda",
    dest="svdq_defer_final_to_cuda",
    action="store_true",
    default=True,
    help=(
      "Defer the final pipeline move to CUDA until SVDQ few-shot runtime quantization finishes. "
      "This only takes effect when quantized layers are offloaded during the few-shot flow."),
  )
  parser.add_argument(
    "--svdq-no-defer-final-to-cuda",
    dest="svdq_defer_final_to_cuda",
    action="store_false",
    help="Do not defer the final pipeline move when using SVDQ few-shot quantization.",
  )
  parser.add_argument(
    "--svdq-few-shot-compile",
    action="store_true",
    default=False,
    help="Compile the transformer only after SVDQ few-shot runtime quantization completes.",
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
      "_npu_fia",  # npu fused infer attention
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
  offload_group = parser.add_mutually_exclusive_group()
  offload_group.add_argument(
    "--cpu-offload",
    "--cpu-offload-model",
    action="store_true",
    default=False,
    help="Enable diffusers model CPU offload for pipeline components when supported.",
  )
  offload_group.add_argument(
    "--sequential-cpu-offload",
    action="store_true",
    default=False,
    help="Enable diffusers sequential CPU offload for pipeline components when supported.",
  )
  offload_group.add_argument(
    "--module-layerwise-cpu-offload",
    "--layerwise-offload",
    dest="module_layerwise_cpu_offload",
    action="store_true",
    default=False,
    help=("Enable cache-dit generic sequential CPU offload for nn.Module components. This is "
          "intended for custom non-diffusers modules and is mutually exclusive with the diffusers "
          "pipeline offload switches."),
  )
  parser.add_argument(
    "--layerwise-async-transfer",
    "--svdq-layerwise-async-transfer",
    "--svdq-async-transfer",
    action="store_true",
    default=False,
    help=("Enable CUDA-stream-based async_transfer for cache-dit layerwise CPU offload. This "
          "shared setting applies to both generic module layerwise offload and SVDQ layerwise "
          "collection when --svdq-layerwise-offload is active."),
  )
  parser.add_argument(
    "--layerwise-transfer-buckets",
    "--svdq-layerwise-transfer-buckets",
    "--svdq-transfer-buckets",
    type=int,
    default=1,
    help=("Base async prefetch depth hint used to size layerwise copy-lane concurrency. "
          "Default is 1."),
  )
  parser.add_argument(
    "--layerwise-prefetch-limit",
    "--svdq-layerwise-prefetch-limit",
    action="store_true",
    default=False,
    help=("Enable the conservative future-prefetch target-count limit for layerwise async "
          "transfer. When set, runtime caps pending/ready future targets to "
          "min(4 * --layerwise-transfer-buckets, 8). By default this limit is disabled."),
  )
  parser.add_argument(
    "--layerwise-max-copy-streams",
    "--svdq-layerwise-max-copy-streams",
    type=int,
    default=None,
    help=("Maximum number of async CUDA copy streams used by cache-dit layerwise offload. This "
          "shared setting applies to both generic module layerwise offload and SVDQ layerwise "
          "collection when --svdq-layerwise-offload is active. When omitted, runtime derives "
          "it from --layerwise-transfer-buckets and applies its internal safety cap."),
  )
  parser.add_argument(
    "--layerwise-max-inflight-prefetch-bytes",
    "--svdq-layerwise-max-inflight-prefetch-bytes",
    type=int,
    default=None,
    help=("Maximum total CUDA residency budget, in bytes, for in-flight layerwise future-target "
          "prefetch. This shared setting applies to both generic module layerwise offload and "
          "SVDQ layerwise collection when --svdq-layerwise-offload is active. When omitted, "
          "runtime leaves the byte-budget limit disabled."),
  )
  parser.add_argument(
    "--layerwise-persistent-buckets",
    "--svdq-layerwise-persistent-buckets",
    "--svdq-persistent-buckets",
    type=int,
    default=0,
    help=("How many selected layerwise offload targets should stay resident on CUDA for the full "
          "handle lifetime. This shared setting applies to both generic module layerwise offload "
          "and SVDQ layerwise collection when --svdq-layerwise-offload is active. Default is 0."),
  )
  parser.add_argument(
    "--layerwise-persistent-bins",
    "--svdq-layerwise-persistent-bins",
    "--svdq-persistent-bins",
    type=int,
    default=1,
    help=("How many evenly distributed bins should be used when placing persistent layerwise "
          "offload targets across the selected target list. A value of 1 keeps the original "
          "prefix behavior. For example, with 32 targets, --layerwise-persistent-buckets 16 and "
          "--layerwise-persistent-bins 4, runtime keeps [0:4), [8:12), [16:20), and [24:28) "
          "resident. This shared setting applies to both generic module layerwise offload and "
          "SVDQ layerwise collection when --svdq-layerwise-offload is active. Default is 1."),
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
    help="Enable compile for transformer, only compile the repeated blocks by default.",
  )
  parser.add_argument(
    "--disable-compile-repeated-blocks",
    "--disable-compile-blocks",
    "--no-regional-compile",
    "--no-rc",
    action="store_true",
    default=False,
    help="Disable compile for repeated blocks in transformer",
  )
  # Force compile dynamic, this is useful for case PyTorch native TP + dynamic shape
  # + compile, where the shape of some inputs to transformer may change a lot during
  # inference, which can cause compile error if we let 'dynamic' to None (default),
  # we need to force it to be True to avoid the compile error. Note that setting
  # dynamic=True may cause some performance regression, so we only enable it when
  # necessary. e.g., Qwen-Image-Edit, FLUX.2-klein-9b-kv w/ TP, etc.
  parser.add_argument(
    "--force-compile-dynamic",
    action="store_true",
    default=False,
    help="Force set the compiled transformer to dynamic mode. ",
  )
  # cuda graph settings
  parser.add_argument(
    "--cuda-graph",
    action="store_true",
    default=False,
    help="Enable compile with CUDA Graph for transformer if applicable.",
  )
  parser.add_argument(
    "--compile-full-graph",
    "--full-graph",
    action="store_true",
    default=False,
    help="Enable compile with full graph requirement for transformer.",
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
    "--tune",
    action="store_true",
    default=False,
    help="Enable max-autotune mode for torch.compile",
  )
  # Profiling and memory tracking settings
  parser.add_argument(
    "--track-memory",
    "--mem",
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
  # Extra params
  parser.add_argument(
    "--saved-fps",
    "--fps",
    type=int,
    default=8,
    help="Export generated video with specified fps",
  )

  args_or_parser = parser.parse_args() if parse else parser
  if parse:
    return maybe_postprocess_args(args_or_parser)
  return args_or_parser


def get_base_args(parse: bool = True) -> argparse.Namespace | argparse.ArgumentParser:
  return get_args(parse=parse)  # For future extension if needed


def maybe_postprocess_args(args: argparse.Namespace) -> argparse.Namespace:
  # Force enable quantization if quantize_type is specified
  if args.float8_per_row:
    args.quantize_type = "float8_per_row"
  elif args.float8_per_tensor:
    args.quantize_type = "float8_per_tensor"
  elif args.float8_per_block:
    args.quantize_type = "float8_per_block"
  elif args.float8_weight_only:
    args.quantize_type = "float8_weight_only"
  elif args.int8_per_row:
    args.quantize_type = "int8_per_row"
  elif args.int8_per_tensor:
    args.quantize_type = "int8_per_tensor"
  elif args.int8_weight_only:
    args.quantize_type = "int8_weight_only"
  elif args.int4_weight_only:
    args.quantize_type = "int4_weight_only"
  elif args.svdq_int4_r32_dq:
    args.quantize_type = "svdq_int4_r32_dq"
  elif args.svdq_int4_r64_dq:
    args.quantize_type = "svdq_int4_r64_dq"
  elif args.svdq_int4_r128_dq:
    args.quantize_type = "svdq_int4_r128_dq"
  elif args.svdq_int4_r256_dq:
    args.quantize_type = "svdq_int4_r256_dq"

  if args.quantize_type is not None:
    args.quantize = True

  # Handle alias for quantize_type
  if args.quantize and args.quantize_type is None:
    args.quantize_type = "float8_per_row"  # default type

  # Force enable quantization for text encoder if quantize_text_type is specified
  if args.quantize_text_type is not None:
    args.quantize_text_encoder = True
  # Handle alias for quantize_text_type
  if args.quantize_text_encoder and args.quantize_text_type is None:
    # default to same as quantize_type
    args.quantize_text_type = args.quantize_type

  # Force enable quantization for controlnet if quantize_controlnet_type is specified
  if args.quantize_controlnet_type is not None:
    args.quantize_controlnet = True
  # Handle alias for quantize_controlnet_type
  if args.quantize_controlnet and args.quantize_controlnet_type is None:
    # default to same as quantize_type
    args.quantize_controlnet_type = args.quantize_type

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

  # Force enable compile if force_compile_dynamic is enabled
  if args.force_compile_dynamic:
    args.compile = True

  if getattr(args, "svdq_layerwise_offload", False):
    args.svdq_offload_quantized_layers_to_cpu = True

  return args


def get_text_encoder_from_pipe(
  pipe: DiffusionPipeline, ) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
  pipe_cls_name = pipe.__class__.__name__
  if (hasattr(pipe, "text_encoder_2") and not pipe_cls_name.startswith("Hunyuan")
      and not pipe_cls_name.startswith("Kandinsky")):
    # Specific for FluxPipeline, FLUX.1-dev
    return getattr(pipe, "text_encoder_2"), "text_encoder_2"
  elif hasattr(pipe, "text_encoder_3"):  # HiDream pipeline
    return getattr(pipe, "text_encoder_3"), "text_encoder_3"
  elif hasattr(
      pipe,
      "vision_language_encoder") and pipe_cls_name.startswith("GlmImage"):  # GLM Image pipeline
    return getattr(pipe, "vision_language_encoder"), "vision_language_encoder"
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
      logger.warning("parallel-text-encoder is set but no text encoder found in the pipeline.")

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


def _compile_mode(args):
  if args.max_autotune:
    if args.cuda_graph:
      return "max-autotune"
    return "max-autotune-no-cudagraphs"
  if args.force_compile_dynamic:
    return None  # let torch.compile to decide itself.
  return None


def _compile_options(args) -> Optional[Dict[str, Any]]:
  if args.cuda_graph and not args.max_autotune:
    return {"triton.cudagraphs": True}
  return None


def _force_compile_dynamic(args, pipe) -> bool:
  _class_maybe_force_compile_dynamic = [
    "QwenImage",
    "Flux2KleinKV",
  ]
  if args.force_compile_dynamic:
    return True
  if args.parallel_type is None or "tp" not in args.parallel_type.lower():
    return False
  # For some specific pipelines with PyTorch native TP.
  # Auto check if the pipeline class name starts with any
  # of the specified prefixes to decide whether to force
  # compile dynamic.
  return any(
    [pipe.__class__.__name__.startswith(prefix) for prefix in _class_maybe_force_compile_dynamic])


def _compile_transformer_module(args, pipe, transformer, name):
  if transformer is None or isinstance(
      transformer,
      torch._dynamo.OptimizedModule,  # already compiled
  ):
    logger.warning(f"{name} module is already compiled or None, skipping compilation.")
    return transformer

  from diffusers import ModelMixin

  transformer_cls_name = transformer.__class__.__name__
  if not isinstance(transformer, (torch.nn.Module, ModelMixin)):
    logger.warning(f"Cannot compile {name} module: {transformer_cls_name} Not a torch.nn.Module.")
    return transformer

  use_regional_compile = not args.disable_compile_repeated_blocks and hasattr(
    transformer, "compile_repeated_blocks")

  if args.cuda_graph and use_regional_compile:
    use_regional_compile = False

  if use_regional_compile:
    logger.info(f"Compiling repeated blocks in {name}: {transformer_cls_name} ...")
    transformer.compile_repeated_blocks(
      fullgraph=args.compile_full_graph,
      mode=_compile_mode(args),
      dynamic=_force_compile_dynamic(args, pipe),
      options=_compile_options(args),
    )
    setattr(pipe, name, transformer)
    return transformer

  if args.cuda_graph:
    logger.info(f"Compiling full {name}: {transformer_cls_name} with CUDA Graph enabled ...")
  else:
    logger.info(f"Compiling full {name}: {transformer_cls_name} ...")
  transformer = torch.compile(
    transformer,
    fullgraph=args.compile_full_graph,
    mode=_compile_mode(args),
    dynamic=_force_compile_dynamic(args, pipe),
    options=_compile_options(args),
  )
  setattr(pipe, name, transformer)
  return transformer


def _register_deferred_few_shot_compile_callback(args, pipe, transformer, name) -> None:
  controller = getattr(transformer, "_svdq_few_shot_controller", None)
  if controller is None:
    return

  def _register_named_callback(callback, callback_name: str) -> bool:
    callbacks = getattr(transformer, "_svdq_post_quantize_callbacks", None)
    if callbacks is None:
      transformer._svdq_post_quantize_callbacks = []
      callbacks = transformer._svdq_post_quantize_callbacks
    callback_names = getattr(transformer, "_svdq_post_quantize_callback_names", None)
    if callback_names is None:
      transformer._svdq_post_quantize_callback_names = set()
      callback_names = transformer._svdq_post_quantize_callback_names
    if callback_name in callback_names:
      return False
    callback_names.add(callback_name)
    callbacks.append(callback)
    return True

  def _callback(_quantized_module):
    current_transformer = getattr(pipe, name, None)
    if current_transformer is None:
      logger.warning("Deferred few-shot compile skipped because %s is no longer present.", name)
      return
    set_compile_configs(cuda_graphs=args.cuda_graph, ulysses_anything=args.ulysses_anything)
    torch.set_float32_matmul_precision("high")
    _compile_transformer_module(args, pipe, current_transformer, name)

  callback_name = f"compile:{name}"
  if not _register_named_callback(_callback, callback_name):
    return
  logger.info("Deferring compile for %s until SVDQ few-shot quantization finishes.", name)


def _should_defer_few_shot_pipeline_move(transformer) -> bool:
  if transformer is None or not getattr(transformer, "_svdq_pending_quantization", False):
    return False
  controller = getattr(transformer, "_svdq_few_shot_controller", None)
  if controller is None:
    return False
  if controller.context.svdq_kwargs.get("smooth_strategy") != "few_shot":
    return False
  if not bool(controller.context.svdq_kwargs.get("defer_move_to_execution_device", False)):
    return False
  return bool(
    controller.context.svdq_kwargs.get("layerwise_offload", False)
    or controller.context.svdq_kwargs.get("offload_quantized_layers_to_cpu", False))


def _should_skip_initial_few_shot_pipeline_move(transformer) -> bool:
  if transformer is None or not getattr(transformer, "_svdq_pending_quantization", False):
    return False
  controller = getattr(transformer, "_svdq_few_shot_controller", None)
  if controller is None:
    return False
  if controller.context.svdq_kwargs.get("smooth_strategy") != "few_shot":
    return False
  return bool(controller.context.svdq_kwargs.get("layerwise_offload", False))


def _register_deferred_few_shot_pipeline_move_callbacks(args, pipe) -> bool:
  if (pipe is None or args.device_map_balance or args.cpu_offload or args.sequential_cpu_offload
      or args.module_layerwise_cpu_offload):
    return False

  tracked_names: list[str] = []
  for name in ("transformer", "transformer_2"):
    transformer = getattr(pipe, name, None)
    if _should_defer_few_shot_pipeline_move(transformer):
      tracked_names.append(name)

  if not tracked_names:
    return False

  _, device = get_rank_device()
  target_device = torch.device(device)

  def _register_for_transformer(name: str) -> None:
    transformer = getattr(pipe, name, None)
    if transformer is None:
      return

    callbacks = getattr(transformer, "_svdq_post_quantize_callbacks", None)
    if callbacks is None:
      transformer._svdq_post_quantize_callbacks = []
      callbacks = transformer._svdq_post_quantize_callbacks
    callback_names = getattr(transformer, "_svdq_post_quantize_callback_names", None)
    if callback_names is None:
      transformer._svdq_post_quantize_callback_names = set()
      callback_names = transformer._svdq_post_quantize_callback_names
    callback_name = f"pipeline_move:{name}:{target_device}"
    if callback_name in callback_names:
      return

    def _callback(_quantized_module):
      remaining = [
        tracked_name for tracked_name in tracked_names
        if getattr(getattr(pipe, tracked_name, None), "_svdq_pending_quantization", False)
      ]
      if remaining:
        logger.info(
          "SVDQ few-shot quantization finished for %s; waiting for remaining modules before the "
          "final pipeline move: %s.",
          name,
          ", ".join(remaining),
        )
        return
      current_transformer = getattr(pipe, name, None)
      runtime_layerwise_handle = getattr(
        current_transformer,
        "_svdq_runtime_layerwise_offload_handle",
        None,
      )
      if runtime_layerwise_handle is not None:
        pipe._svdq_move_to_device_after_forward = target_device
        logger.info(
          "SVDQ few-shot quantization finished for %s; keeping temporary runtime layerwise "
          "offload active until the current pipeline call returns, then moving the pipeline to "
          "%s.",
          name,
          target_device,
        )
        return
      try:
        logger.info(
          "SVDQ few-shot quantization finished; moving pipeline to %s immediately so the "
          "current pipeline call can continue on the execution device.",
          target_device,
        )
        pipe.to(target_device)
      except Exception as exc:
        pipe._svdq_move_to_device_after_forward = target_device
        logger.warning(
          "Immediate pipeline move to %s after SVDQ few-shot quantization failed with %s: %s. "
          "Falling back to a deferred move after the current pipeline call returns.",
          target_device,
          type(exc).__name__,
          exc,
        )

    callback_names.add(callback_name)
    callbacks.append(_callback)

  for name in tracked_names:
    _register_for_transformer(name)

  logger.info(
    "Deferring final pipeline move to %s until SVDQ few-shot quantization finishes for %s.",
    target_device,
    ", ".join(tracked_names),
  )
  return True


def maybe_finalize_deferred_svdq_pipe_move(
  pipe_or_adapter: DiffusionPipeline | BlockAdapter, ) -> bool:
  if isinstance(pipe_or_adapter, BlockAdapter):
    pipe = pipe_or_adapter.pipe
  else:
    pipe = pipe_or_adapter
  if pipe is None:
    return False

  target_device = getattr(pipe, "_svdq_move_to_device_after_forward", None)
  if target_device is None:
    return False

  delattr(pipe, "_svdq_move_to_device_after_forward")
  for name in ("transformer", "transformer_2"):
    transformer = getattr(pipe, name, None)
    handle = getattr(transformer, "_svdq_runtime_layerwise_offload_handle", None)
    if handle is None:
      continue
    logger.info(
      "Removing temporary runtime layerwise offload from %s before moving pipeline to %s.",
      name,
      target_device,
    )
    handle.remove()
    delattr(transformer, "_svdq_runtime_layerwise_offload_handle")
  logger.info(
    "Moving pipeline to %s after SVDQ few-shot runtime quantization completed.",
    target_device,
  )
  pipe.to(target_device)
  return True


def _should_defer_few_shot_compile(args, transformer) -> bool:
  if transformer is None or not getattr(transformer, "_svdq_pending_quantization", False):
    return False
  controller = getattr(transformer, "_svdq_few_shot_controller", None)
  if controller is None:
    return False
  smooth_strategy = controller.context.svdq_kwargs.get("smooth_strategy")
  if smooth_strategy != "few_shot":
    return False
  return bool(controller.context.svdq_kwargs.get("few_shot_auto_compile", False))


def maybe_compile_transformer(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
  if args.compile or args.svdq_few_shot_compile:
    set_compile_configs(cuda_graphs=args.cuda_graph, ulysses_anything=args.ulysses_anything)
    torch.set_float32_matmul_precision("high")

    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
      assert pipe is not None, "Please compile transformer manually if pipe is None."
    else:
      pipe = pipe_or_adapter

    if hasattr(pipe, "transformer"):
      transformer = getattr(pipe, "transformer", None)
      if _should_defer_few_shot_compile(args, transformer):
        _register_deferred_few_shot_compile_callback(args, pipe, transformer, "transformer")
      else:
        _compile_transformer_module(args, pipe, transformer, "transformer")
    else:
      logger.warning("compile is set but no transformer found in the pipeline.")

    if hasattr(pipe, "transformer_2"):
      transformer_2 = getattr(pipe, "transformer_2", None)
      if _should_defer_few_shot_compile(args, transformer_2):
        _register_deferred_few_shot_compile_callback(args, pipe, transformer_2, "transformer_2")
      else:
        _compile_transformer_module(args, pipe, transformer_2, "transformer_2")

  return pipe_or_adapter


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
        logger.info(f"Compiling text encoder module {name}:{_text_encoder_cls_name}:"
                    f"{_module_to_compile_cls_name} ...")
        _module_to_compile = torch.compile(
          _module_to_compile,
          mode=_compile_mode(args),
          options=_compile_options(args),
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
        logger.warning(f"Cannot compile text encoder module {name}:{_text_encoder_cls_name}:"
                       f"{_module_to_compile_cls_name} Not a torch.nn.Module.")
    else:
      logger.warning("compile-text-encoder is set but no text encoder found in the pipeline.")
  return pipe_or_adapter


def maybe_compile_controlnet(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
  if args.compile_controlnet:
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
            mode=_compile_mode(args),
            options=_compile_options(args),
          )
          setattr(pipe, "controlnet", controlnet)
        else:
          logger.warning(f"Cannot compile controlnet module: {controlnet_cls_name} Not a"
                         " torch.nn.Module.")
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
              mode=_compile_mode(args),
              options=_compile_options(args),
            )
          else:
            logger.warning(f"Cannot compile VAE encoder module: {vae_cls_name}.encoder Not a"
                           " torch.nn.Module.")
        if hasattr(vae, "decoder"):
          _decoder_to_compile = vae.decoder
          if isinstance(_decoder_to_compile, torch.nn.Module):
            logger.info(f"Compiling VAE decoder module: {vae_cls_name}.decoder ...")
            vae.decoder = torch.compile(
              _decoder_to_compile,
              mode=_compile_mode(args),
              options=_compile_options(args),
            )
          else:
            logger.warning(f"Cannot compile VAE decoder module: {vae_cls_name}.decoder Not a"
                           " torch.nn.Module.")
        setattr(pipe, "vae", vae)
      else:
        logger.warning(f"Cannot compile VAE module: {vae_cls_name} Not a torch.nn.Module.")
    else:
      logger.warning("compile-vae is set but no VAE found in the pipeline.")
  return pipe_or_adapter


def maybe_quantize_transformer(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:

  def _resolve_cli_svdq_kwargs() -> Optional[Dict[str, Any]]:
    if args.quantize_type is None or not args.quantize_type.startswith("svdq"):
      return None
    few_shot_auto_compile = bool(args.svdq_few_shot_compile)
    if args.svdq_smooth_strategy == "few_shot" and args.compile:
      few_shot_auto_compile = True
    offload_quantized_layers_to_cpu = bool(args.svdq_offload_quantized_layers_to_cpu)
    defer_move_to_execution_device = bool(args.svdq_defer_final_to_cuda
                                          and args.svdq_smooth_strategy == "few_shot"
                                          and offload_quantized_layers_to_cpu)
    return {
      "smooth_strategy": args.svdq_smooth_strategy,
      "calibrate_precision": args.svdq_calibrate_precision,
      "runtime_kernel": args.svdq_runtime,
      "quantize_device": args.svdq_quantize_device,
      "offload_quantized_layers_to_cpu": offload_quantized_layers_to_cpu,
      "layerwise_offload": args.svdq_layerwise_offload,
      "async_transfer": args.layerwise_async_transfer,
      "transfer_buckets": args.layerwise_transfer_buckets,
      "prefetch_limit": args.layerwise_prefetch_limit,
      "max_copy_streams": args.layerwise_max_copy_streams,
      "max_inflight_prefetch_bytes": args.layerwise_max_inflight_prefetch_bytes,
      "persistent_buckets": args.layerwise_persistent_buckets,
      "persistent_bins": args.layerwise_persistent_bins,
      "defer_move_to_execution_device": defer_move_to_execution_device,
      "few_shot_steps": args.svdq_few_shot_steps,
      "few_shot_relax_factor": args.svdq_few_shot_relax_factor,
      "few_shot_relax_top_ratio": args.svdq_few_shot_relax_top_ratio,
      "few_shot_relax_strategy": args.svdq_few_shot_relax_strategy,
      "few_shot_auto_compile": few_shot_auto_compile,
    }

  # Quantize transformer by default if quantization is enabled
  if args.quantize:
    if args.compile and args.cuda_graph and args.quantize_type == "float8_per_row":
      from ..quantization.torchao._scaled_mm import (
        enable_opaque_torchao_float8_scaled_mm, )

      enable_opaque_torchao_float8_scaled_mm()

    if args.quantize_type in ("bitsandbytes_4bit", "bnb_4bit"):
      logger.debug("bitsandbytes_4bit quantization should be handled by"
                   " PipelineQuantizationConfig in from_pretrained.")
      return pipe_or_adapter

    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
      assert pipe is not None, "Please quantize transformer manually if pipe is None."
    else:
      pipe = pipe_or_adapter

    if hasattr(pipe, "transformer"):
      transformer = getattr(pipe, "transformer", None)
      if transformer is not None:
        transformer_cls_name = transformer.__class__.__name__
        if isinstance(transformer, torch.nn.Module):
          logger.info(f"Quantizing transformer module: {transformer_cls_name} to"
                      f" {args.quantize_type} ...")
          transformer = quantize(
            transformer,
            quantize_config=QuantizeConfig(
              quant_type=args.quantize_type,
              regional_quantize=not args.disable_regional_quantize,
              per_tensor_fallback=not args.disable_per_tensor_fallback,
              svdq_kwargs=_resolve_cli_svdq_kwargs(),
              verbose=args.quantize_verbose,
            ),
          )
          setattr(pipe, "transformer", transformer)
        else:
          logger.warning(f"Cannot quantize transformer module: {transformer_cls_name} Not a"
                         " torch.nn.Module.")
      setattr(pipe, "transformer", transformer)
    else:
      logger.warning("quantize is set but no transformer found in the pipeline.")

    if hasattr(pipe, "transformer_2"):
      transformer_2 = getattr(pipe, "transformer_2", None)
      if transformer_2 is not None:
        transformer_2_cls_name = transformer_2.__class__.__name__
        if isinstance(transformer_2, torch.nn.Module):
          logger.info(f"Quantizing transformer_2 module: {transformer_2_cls_name} to"
                      f" {args.quantize_type} ...")
          transformer_2 = quantize(
            transformer_2,
            quantize_config=QuantizeConfig(
              quant_type=args.quantize_type,
              regional_quantize=not args.disable_regional_quantize,
              per_tensor_fallback=not args.disable_per_tensor_fallback,
              svdq_kwargs=_resolve_cli_svdq_kwargs(),
              verbose=args.quantize_verbose,
            ),
          )
          setattr(pipe, "transformer_2", transformer_2)
        else:
          logger.warning(f"Cannot quantize transformer_2 module: {transformer_2_cls_name} Not a"
                         " torch.nn.Module.")
      setattr(pipe, "transformer_2", transformer_2)

  return pipe_or_adapter


def maybe_quantize_text_encoder(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
  # Quantize text encoder by default if quantize_text_encoder is enabled
  if args.quantize_text_encoder:
    if args.quantize_text_type in ("bitsandbytes_4bit", "bnb_4bit"):
      logger.debug("bitsandbytes_4bit quantization should be handled by"
                   " PipelineQuantizationConfig in from_pretrained.")
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
        logger.info(f"Quantizing text encoder module: {name}:{text_encoder_cls_name} to"
                    f" {args.quantize_text_type} ...")
        text_encoder = quantize(
          text_encoder,
          quantize_config=QuantizeConfig(
            quant_type=args.quantize_text_type,
            regional_quantize=not args.disable_regional_quantize,
            per_tensor_fallback=not args.disable_per_tensor_fallback,
            verbose=args.quantize_verbose,
          ),
        )
        setattr(pipe, name, text_encoder)
      else:
        logger.warning(f"Cannot quantize text encoder module: {name}:{text_encoder_cls_name} Not a"
                       " torch.nn.Module.")
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
      logger.debug("bitsandbytes_4bit quantization should be handled by"
                   " PipelineQuantizationConfig in from_pretrained.")
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
          logger.info(f"Quantizing controlnet module: {controlnet_cls_name} to"
                      f" {args.quantize_controlnet_type} ...")
          controlnet = quantize(
            controlnet,
            quantize_config=QuantizeConfig(
              quant_type=args.quantize_controlnet_type,
              regional_quantize=not args.disable_regional_quantize,
              per_tensor_fallback=not args.disable_per_tensor_fallback,
              verbose=args.quantize_verbose,
            ),
          )
          setattr(pipe, "controlnet", controlnet)
        else:
          logger.warning(f"Cannot quantize controlnet module: {controlnet_cls_name} Not a"
                         " torch.nn.Module.")
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
    quantization_config = ((PipelineQuantizationConfig(
      quant_backend="bitsandbytes_4bit",
      quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
      },
      components_to_quantize=components_to_quantize,
    )) if args.quantize or args.quantize_text_encoder else None)
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
            logger.warning(f"Cannot enable VAE tiling for module: {vae_cls_name} No enable_tiling"
                           " method.")
        if args.vae_slicing:
          if hasattr(vae, "enable_slicing"):
            logger.info(f"Enabling VAE slicing for module: {vae_cls_name} ...")
            vae.enable_slicing()
          else:
            logger.warning(f"Cannot enable VAE slicing for module: {vae_cls_name} No enable_slicing"
                           " method.")
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


def _append_generic_offload_target(
  targets: list[tuple[str, torch.nn.Module]],
  seen_module_ids: set[int],
  name: str,
  module: Any,
) -> None:
  if not isinstance(module, torch.nn.Module):
    return
  if id(module) in seen_module_ids:
    return
  seen_module_ids.add(id(module))
  targets.append((name, module))


def _resolve_generic_offload_targets(
  pipe_or_adapter: DiffusionPipeline | BlockAdapter | torch.nn.Module | Any,
) -> list[tuple[str, torch.nn.Module]]:
  targets: list[tuple[str, torch.nn.Module]] = []
  seen_module_ids: set[int] = set()

  if isinstance(pipe_or_adapter, torch.nn.Module):
    return [("module", pipe_or_adapter)]

  if isinstance(pipe_or_adapter, BlockAdapter):
    transformer = getattr(pipe_or_adapter, "transformer", None)
    if isinstance(transformer, list):
      for index, module in enumerate(transformer):
        _append_generic_offload_target(targets, seen_module_ids, f"transformer[{index}]", module)
    else:
      _append_generic_offload_target(targets, seen_module_ids, "transformer", transformer)
    if targets:
      return targets
    pipe_or_adapter = pipe_or_adapter.pipe

  components = getattr(pipe_or_adapter, "components", None)
  if isinstance(components, dict):
    for name, module in components.items():
      _append_generic_offload_target(targets, seen_module_ids, str(name), module)
    if targets:
      return targets

  if hasattr(pipe_or_adapter, "__dict__"):
    for name, module in vars(pipe_or_adapter).items():
      _append_generic_offload_target(targets, seen_module_ids, str(name), module)

  return targets


def maybe_generic_module_offload(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter | torch.nn.Module | Any,
) -> bool:
  if not getattr(args, "module_layerwise_cpu_offload", False):
    return False

  _, device = get_rank_device()
  targets = _resolve_generic_offload_targets(pipe_or_adapter)
  if not targets:
    raise ValueError(
      "--module-layerwise-cpu-offload was requested, but no nn.Module targets were found.")

  handles = []
  skipped_due_to_existing_offload = False
  for name, module in targets:
    remove_layerwise_offload(module)

    existing_offload_hook = _find_offload_related_hf_hook(module)
    if existing_offload_hook is not None:
      hook_module_name, hook = existing_offload_hook
      logger.warning(
        "Skipping cache-dit layerwise CPU offload for module %s because an existing accelerate "
        "offload hook %s is already registered on %s.",
        name,
        type(hook).__name__,
        hook_module_name or "<root>",
      )
      skipped_due_to_existing_offload = True
      continue

    logger.info(
      "Enabling cache-dit layerwise CPU offload for module %s: %s ...",
      name,
      module.__class__.__name__,
    )
    handles.append(
      layerwise_cpu_offload(
        module,
        onload_device=device,
        async_transfer=bool(getattr(args, "layerwise_async_transfer", False)),
        transfer_buckets=int(getattr(args, "layerwise_transfer_buckets", 1)),
        prefetch_limit=bool(getattr(args, "layerwise_prefetch_limit", False)),
        max_copy_streams=getattr(args, "layerwise_max_copy_streams", None),
        max_inflight_prefetch_bytes=getattr(args, "layerwise_max_inflight_prefetch_bytes", None),
        persistent_buckets=int(getattr(args, "layerwise_persistent_buckets", 0)),
        persistent_bins=int(getattr(args, "layerwise_persistent_bins", 1)),
      ))
  return bool(handles) or skipped_due_to_existing_offload


def maybe_apply_optimization(
  args,
  pipe_or_adapter,
  **kwargs,
):
  quantize_config = None
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

      # Caching and Parallelism
      if args.steps_mask and args.mask_policy is not None:
        logger.info(f"Using steps computation mask with policy: {args.mask_policy} for caching.")
        if default_num_inference_steps is None:
          assert (args.num_inference_steps
                  is not None), "num_inference_steps (--steps) must be provided for steps mask."
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
        cache_config=(DBCacheConfig(
          Fn_compute_blocks=args.Fn_compute_blocks,
          Bn_compute_blocks=args.Bn_compute_blocks,
          max_warmup_steps=args.max_warmup_steps,
          warmup_interval=args.warmup_interval,
          max_cached_steps=args.max_cached_steps,
          max_continuous_cached_steps=args.max_continuous_cached_steps,
          residual_diff_threshold=args.residual_diff_threshold,
          enable_separate_cfg=kwargs.get("enable_separate_cfg", None),
          steps_computation_mask=steps_computation_mask,
          force_refresh_step_hint=kwargs.get("force_refresh_step_hint", None),
          force_refresh_step_policy=kwargs.get("force_refresh_step_policy", "once"),
        ) if cache_config is None and args.cache else cache_config),
        calibrator_config=(TaylorSeerCalibratorConfig(taylorseer_order=args.taylorseer_order, )
                           if args.taylorseer else None),
        params_modifiers=kwargs.get("params_modifiers", None),
        parallelism_config=(ParallelismConfig(
          ulysses_size=ulysses_size,
          ring_size=ring_size,
          tp_size=tp_size,
          backend=ParallelismBackend.AUTO,
          attention_backend=("native" if not args.attn else args.attn),
          extra_parallel_modules=extra_parallel_modules,
          ulysses_anything=args.ulysses_anything,
          ulysses_float8=args.ulysses_float8,
          ulysses_async=args.ulysses_async,
          ring_rotate_method=args.ring_rotate_method,
          ring_convert_to_fp32=not args.ring_no_convert_to_fp32,
        ) if parallelism_config is None and args.parallel_type is not None else parallelism_config),
        # Allow attention backend for non-parallelism case
        attention_backend=(("native" if not args.attn else args.attn)
                           if args.parallel_type is None else None),
      )
    else:
      # Apply acceleration configs from config path
      configs = load_configs(args.config_path)
      quantize_config = configs.get("quantize_config", None)
      if quantize_config is not None:
        args.quantize = True
        try:
          args.quantize_type = quantize_config.component_quant_types()["transformer"]
        except Exception:
          args.quantize_type = list(quantize_config.component_quant_types().values())[0]
        logger.info(f"Quantization config from {args.config_path}: {quantize_config.strify()}")

      enable_cache(
        pipe_or_adapter,
        **configs,
      )
      logger.info(f"Applied acceleration config from file: {args.config_path}.")
  else:
    logger.info("No caching or parallelism is applied.")
    if args.attn is not None:
      logger.info(f"Applying custom attention backend: {args.attn} ...")
      set_attn_backend(pipe_or_adapter, attention_backend=args.attn)

  # Quantization
  # WARN: Must apply quantization after tensor parallelism is applied.
  # torchao is compatible with tensor parallelism but requires to be
  # applied after TP.
  # Avoid quantization if quant_config is already applied via config file.
  if quantize_config is None:
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

  if isinstance(pipe_or_adapter, BlockAdapter):
    pipe = pipe_or_adapter.pipe
  else:
    pipe = pipe_or_adapter
  should_skip_initial_pipeline_move = any(
    _should_skip_initial_few_shot_pipeline_move(getattr(pipe, name, None))
    for name in ("transformer", "transformer_2")) if pipe is not None else False
  _register_deferred_few_shot_pipeline_move_callbacks(args, pipe)

  # CPU Offload
  _, device = get_rank_device()
  used_diffusers_cpu_offload = maybe_cpu_offload(args, pipe_or_adapter)
  used_generic_module_offload = maybe_generic_module_offload(args, pipe_or_adapter)
  if not used_diffusers_cpu_offload and not used_generic_module_offload:
    # Set device if no cpu offload
    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
    else:
      pipe = pipe_or_adapter
    if pipe is not None and not args.device_map_balance:
      if should_skip_initial_pipeline_move:
        logger.info(
          "Skipping eager pipe.to(%s) because SVDQ few-shot quantization will finish "
          "\nduring warmup or the current run before the low-memory layerwise pipeline move.",
          device,
        )
      else:
        pipe.to(device)

  return pipe_or_adapter


def strify(args, pipe_or_stats):
  base_str = ""
  if args.height is not None and args.width is not None:
    base_str += f"{args.height}x{args.width}_"
  quantize_type = args.quantize_type if args.quantize else ""
  base_str = summary_strify(pipe_or_stats)
  if quantize_type not in base_str:
    base_str = f"C{int(args.compile)}_{quantize_type}_{base_str}"
  else:
    base_str = f"C{int(args.compile)}_{base_str}"
  if args.parallel_type == "ring" or args.parallel_type == "usp":
    if args.ring_rotate_method != "p2p":
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
  if args.cuda_graph:
    base_str += "_cuda_graph"
  # __ -> _, ___ -> _, etc.
  base_str = base_str.strip("_").replace("__", "_").replace("___", "_")
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
  from ..logger import suppress_stdout

  platform_full_backend = current_platform.full_dist_backend
  cpu_full_backend = CpuPlatform.full_dist_backend
  # Always use hybrid comm backend since there no side effect.
  backend = f"{cpu_full_backend},{platform_full_backend}"
  if args is not None:
    if args.parallel_type is not None:
      with suppress_stdout():
        dist.init_process_group(backend=backend, )
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
          with suppress_stdout():
            dist.init_process_group(backend=backend, )
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
      with suppress_stdout():
        dist.init_process_group(backend=backend, )
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
