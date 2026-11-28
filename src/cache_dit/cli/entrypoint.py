"""The generate CLI parser and its post-processing entry point."""

import argparse

from ..caching.cache_contexts.calibrators import normalize_calibrator_type
from ..offload import _parse_byte_size_arg
from .attention import _apply_attn_params
from .cache import (
  _resolve_cli_cache_params,
  parse_cli_calibrator_params,
  resolve_calibrator_params,
)
from .parallel import _resolve_cli_parallelism
from .params import pairs_parser, parse_attn_params, parse_cache_params
from .quantize import _resolve_cli_quantization, parse_quant_communicate_params
from .utils import parse_extra_input_kwargs


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
    "--prompt-enhancer",
    type=str,
    default=None,
    help=("Local path of an optional prompt-enhancer repository. When set, the examples that own "
          "such a step expand the prompt with it instead of rendering the prompt as given; the "
          "per-example docstring says where the weights come from and what else changes. Off by "
          "default: the expanded caption is a different input distribution, and the enhancer is "
          "18-19 GiB that no `--parallel-params` option covers."),
  )
  parser.add_argument(
    "--mask-image-path",
    type=str,
    default=None,
    help="Override mask image path if provided",
  )
  parser.add_argument(
    "--last-image-path",
    type=str,
    default=None,
    help=("Path of the *last* frame for first/last-frame conditioning (MiniMax-H3 `fl2va`, which "
          "takes `image` and `last_image` together)"),
  )
  parser.add_argument(
    "--driving-video-path",
    type=str,
    default=None,
    help=("Path of the driving video for a character-animation model (Wan-Animate-2). Falls back "
          "to `$WAN_ANIMATE_2_DEMO_DIR/template.mp4`, then to frames synthesized from the "
          "reference image"),
  )
  parser.add_argument(
    "--driving-video-fps",
    type=float,
    default=None,
    help="Frame rate of the driving video, used to resample it to the model fps",
  )
  parser.add_argument(
    "--audio-path",
    type=str,
    default=None,
    help=("Path of the driving waveform for a speech-to-video model (Wan2.2-S2V). Falls back to "
          "`$WAN_2_2_S2V_DEMO_DIR/*.wav`, then to the example wav next to the model"),
  )
  for _flag, _kind in (
    ("--reference-image-path", "image"),
    ("--reference-video-path", "video"),
    ("--reference-audio-path", "audio"),
  ):
    parser.add_argument(
      _flag,
      nargs="+",
      default=None,
      help=(f"Path(s) of {_kind} reference(s) for a reference-conditioned model (MiniMax-H3 "
            "`ref2va`). References are consumed in image -> video -> audio order"),
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
    "--extra-input-kwargs",
    type=parse_extra_input_kwargs,
    default=None,
    help=(
      "Extra input kwargs passed to the pipeline as a Python dict literal. "
      "Keys override both default and programmatic extra_input_kwargs. "
      "Example: --extra-input-kwargs \"{'cfg_normalization': False, 'max_sequence_length': 1536}\""
    ),
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
    "--cache-params",
    "--cache-config",
    type=parse_cache_params,
    default=None,
    help="Enable Cache Acceleration; options as a comma separated key=value list whose optional "
    "'type' key selects the cache: DBCache (default), SeaCache (DBCache plus the SEA probe) or "
    "SoftCache (SeaCache plus jump-step reuse). The type may also be the first bare entry, e.g. "
    "'DBCache,rdt=0.2'. DBCache knobs: 'Fn=1', 'Bn=0', 'rdt=0.12', "
    "'ws=8' (max warmup steps), 'wi=1' (warmup interval), 'mc=-1' (max cached steps), 'mcc=3' "
    "(max continuous cached steps), 'scm=fast' (steps mask policy: slow|medium|fast|ultra), "
    "'steps_mask=true'. SeaCache: 'seacache=true', 'seacache_mode=flow|vp', "
    "'seacache_feature_shape=64x64' (H W, or F H W for video), 'seacache_power_exp=2'. "
    "SoftCache: 'softcache=true', 'softcache_tau=0.5', 'softcache_jf_max=8'. The long names "
    "Fn_compute_blocks, Bn_compute_blocks, residual_diff_threshold, max_warmup_steps, "
    "warmup_interval, max_cached_steps, max_continuous_cached_steps and mask_policy are accepted too.",
  )
  parser.add_argument(
    "--cache-summary",
    "--summary",
    action="store_true",
    default=False,
    help="Enable Cache Summary logging",
  )
  parser.add_argument(
    "--calibrator-params",
    "--calibrator-config",
    type=parse_cli_calibrator_params,
    default=None,
    help="Enable a DBCache calibrator (requires --cache-params); "
    "options as a comma separated key=value "
    "list whose 'type' key selects the calibrator (it may also be the first bare entry): "
    "taylorseer, dmd, foca, svd (SVD-Cache), "
    "hicache (damped-Hermite), brace (barycentric rational), bezicast (Bernstein/Bezier), or "
    "residual_motion (sigma secant, SoftCache). Long forms such as 'svd-calibrator' are accepted. "
    "e.g. 'type=svd,tau=0.9,beta=0.9' or 'type=svd,predictor=taylor,order=1' for svd, "
    "'type=hicache,order=2,sigma=0.5' for hicache, 'type=brace,capacity=3,gamma=0.7' for brace, "
    "'type=bezicast,history=6,ridge=0.5,guardrail=false' for bezicast, "
    "'type=residual_motion,beta=0.5' for residual_motion, 'type=dmd,history=6,svd_precision=low' "
    "for dmd; 'calibrator_cache_type=hidden_states,encoder=false' may be added to any of them "
    "for the keys every calibrator accepts.",
  )
  # Quantization settings
  parser.add_argument(
    "--quantize-params",
    "--quantize-config",
    type=pairs_parser("--quantize-params",
                      "'quant_type=float8_per_row,backend=cache_dit'",
                      allow_empty=False,
                      head_key="quant_type"),
    default=None,
    help="Quantize the transformer; options as a comma separated key=value list, e.g. "
    "'quant_type=float8_per_row' (the default type when omitted; it may also be the first bare "
    "entry, e.g. 'float8_per_row,backend=cache_dit'), 'backend=cache_dit', "
    "'exclude_layers=adaln_proj+norm_out', "
    "'text_encoder=float8_weight_only' (or 'true' to reuse quant_type; quantizes only the text "
    "encoder when it is the only key), "
    "'regional_quantize=false', 'per_tensor_fallback=false', 'verbose=true', "
    "'fp8_fused_mlp=true', 'fp4_fused_mlp=true', 'mxfp4_quantize_device=cpu'.",
  )
  parser.add_argument(
    "--svdq-params",
    "--svdq-config",
    type=pairs_parser("--svdq-params", "'smooth_strategy=few_shot,runtime_kernel=cute'"),
    default=None,
    help="SVDQuant options as a comma separated key=value list, requires an svdq_* quant_type. "
    "Keys are the svdq_kwargs fields: smooth_strategy, calibrate_precision, runtime_kernel, "
    "lora_layout, quantize_device, offload_quantized_layers_to_cpu, layerwise_offload, "
    "async_transfer, transfer_buckets, prefetch_limit, max_copy_streams, "
    "max_inflight_prefetch_bytes, persistent_buckets, persistent_bins, "
    "defer_move_to_execution_device, few_shot_steps, few_shot_relax_factor, "
    "few_shot_relax_top_ratio, few_shot_relax_strategy, few_shot_auto_compile, fused_mlp, "
    "svd_refine_iters, packed_a4_stages, packed_a4_inner_group, streaming, "
    "activation_buffer_flush_sample_count, activation_buffer_flush_cpu_bytes. "
    "The layerwise keys inherit --layerwise-* when omitted.",
  )
  # per row quantization
  parser.add_argument(
    "--disable-per-row",
    "--no-per-row",
    action="store_true",
    default=False,
    help="Disable per row quantization for transformer",
  )
  # Parallelism settings
  parser.add_argument(
    "--parallel-params",
    "--parallel-config",
    type=pairs_parser("--parallel-params", "'ulysses=2,ring=1,tp=1'", allow_empty=False),
    default=None,
    help="Enable parallelism; options as a comma separated key=value list. The degrees are "
    "'ulysses', 'ring' and 'tp' (each defaults to 1 and their product must equal the number of "
    "processes), e.g. 'ulysses=2' or 'ulysses=2,tp=2'. Components: 'vae=true', 'text=true' "
    "(text encoder), 'controlnet=true'. 'local_weight_shards=true' initializes supported TP "
    "weights from CPU slices. Ulysses: 'ulysses_anything=true', "
    "'ulysses_async=true'. Ring: 'ring_rotate_method=allgather', 'ring_convert_to_fp32=false'.",
  )
  parser.add_argument(
    "--launcher",
    choices=["auto", "ray", "local"],
    default="auto",
    help="How to start the workers when --parallel-params needs several processes and the run is "
    "not already under torchrun: 'auto' uses Ray when installed and local subprocesses otherwise, "
    "'ray' requires Ray, 'local' always spawns local processes.",
  )
  parser.add_argument(
    "--attn-params",
    "--attn-config",
    type=parse_attn_params,
    default=None,
    help="Select the attention backend and its options as a comma separated key=value list whose "
    "'type' key is the backend (it may also be the first bare entry, e.g. 'native'): flash, "
    "flash_varlen, _flash_3 (FlashAttention-3), native (SDPA), "
    "_native_cudnn, _sdpa_cudnn, sage, sage3, sol_attn, ffpa, ffpa_fp8, ffpa_fp4, "
    "ffpa_fp8_per_block, _native_npu, _npu_fia, _mindiesd_laser. e.g. 'type=flash'. Backends with "
    "options: sol_attn: tau, thresh_type (diag|exact), strict, sink_tokens; ffpa: hybrid, "
    "hybrid_n_early; ffpa_fp8 / ffpa_fp8_per_block: hybrid, hybrid_n_early, fp8_hadamard, "
    "fp8_smooth_v, e.g. 'type=ffpa_fp8,fp8_hadamard=true'; ffpa_fp4: hybrid, hybrid_n_early, "
    "fp4_hadamard, fp4_pv_mm_type (fp4|fp8), fp4_smooth_v. The ffpa family also takes the 'bits' "
    "shorthand: 'ffpa,bits=8' is ffpa_fp8, 'ffpa,bits=4' is ffpa_fp4, 'ffpa,bits=16' is the plain "
    "fp16/bf16 ffpa, and the remaining keys follow the selected family, e.g. "
    "'ffpa,bits=8,fp8_hadamard=true'. The non-default options are reflected in the saved filename.",
  )
  # Ulysses context parallelism settings
  parser.add_argument(
    "--quant-communicate-params",
    "--quant-communicate-config",
    "--quant-comm-params",
    "--quant-comm-config",
    type=parse_quant_communicate_params,
    default=None,
    help="Ulysses low-precision communication options as k=v pairs, e.g. "
    "'bits=6,quant_type=per_group_32,quantize_k=false'",
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
  offload_group.add_argument(
    "--adaln-host-residency",
    dest="adaln_host_residency",
    action="store_true",
    default=False,
    help=(
      "Keep the transformer resident on the device but stream its AdaLN modulation tables "
      "(`adaln_proj`) from a pinned host mirror, and stream the text encoder as usual. "
      "Intended for joint video + audio models whose AdaLN tables are a large share of the "
      "transformer weights (e.g. MiniMax-H3: 24 GiB of 62 GiB), where it cuts both the "
      "resident footprint and the per-step host-to-device traffic. While the AdaLN tables are "
      "actually streamed (i.e. without transformer quantization or `--parallel-params tp=...`) each "
      "per-block copy is a graph break, so `--compile` still works but pays a smaller speedup; "
      "`--compile-full-graph` and `--cuda-graph` are rejected in that case."),
  )
  offload_group.add_argument(
    "--adaln-online",
    dest="adaln_online",
    action="store_true",
    default=False,
    help=("Precompute the AdaLN modulation tables of a MiniMax-H3 transformer for the checkpoint's "
          "fixed sampling grid, keep the small results on the device and park the 24 GiB of "
          "`adaln_proj` weights on the host. Exact, not an approximation: the tables are a pure "
          "function of the timestep embedding, and a timestep outside the announced grid falls "
          "back to recomputing with the weights. Removes the per-step host-to-device traffic "
          "`--adaln-host-residency` pays, so it also keeps `--compile` free of that graph break."),
  )
  parser.add_argument(
    "--layerwise-async-transfer",
    action="store_true",
    default=False,
    help=("Enable CUDA-stream-based async_transfer for cache-dit layerwise CPU offload. This "
          "shared setting applies to both generic module layerwise offload and SVDQ layerwise "
          "collection when --svdq-params layerwise_offload=true is active."),
  )
  parser.add_argument(
    "--layerwise-text-async-transfer",
    dest="layerwise_text_async_transfer",
    action="store_true",
    default=None,
    help=("Override text encoder layerwise async_transfer. When omitted, text encoder layerwise "
          "offload uses --layerwise-async-transfer."),
  )
  parser.add_argument(
    "--no-layerwise-text-async-transfer",
    dest="layerwise_text_async_transfer",
    action="store_false",
    help=("Disable text encoder layerwise async_transfer even if --layerwise-async-transfer is "
          "enabled."),
  )
  parser.add_argument(
    "--layerwise-transfer-buckets",
    type=int,
    default=1,
    help=("Base async prefetch depth hint used to size layerwise copy-lane concurrency. "
          "Default is 1."),
  )
  parser.add_argument(
    "--layerwise-text-transfer-buckets",
    type=int,
    default=None,
    help=("Override text encoder layerwise transfer_buckets. When omitted, text encoder "
          "layerwise offload uses --layerwise-transfer-buckets."),
  )
  parser.add_argument(
    "--layerwise-prefetch-limit",
    action="store_true",
    default=False,
    help=("Enable the conservative future-prefetch target-count limit for layerwise async "
          "transfer. When set, runtime caps pending/ready future targets to "
          "min(4 * --layerwise-transfer-buckets, 8). By default this limit is disabled."),
  )
  parser.add_argument(
    "--layerwise-text-prefetch-limit",
    dest="layerwise_text_prefetch_limit",
    action="store_true",
    default=None,
    help=("Override text encoder layerwise prefetch_limit. When omitted, text encoder layerwise "
          "offload uses --layerwise-prefetch-limit."),
  )
  parser.add_argument(
    "--no-layerwise-text-prefetch-limit",
    dest="layerwise_text_prefetch_limit",
    action="store_false",
    help=("Disable text encoder layerwise prefetch_limit even if --layerwise-prefetch-limit is "
          "enabled."),
  )
  parser.add_argument(
    "--layerwise-max-copy-streams",
    type=int,
    default=None,
    help=(
      "Maximum number of async CUDA copy streams used by cache-dit layerwise offload. This "
      "shared setting applies to both generic module layerwise offload and SVDQ layerwise "
      "collection when --svdq-params layerwise_offload=true is active. When omitted, runtime derives "
      "it from --layerwise-transfer-buckets and applies its internal safety cap."),
  )
  parser.add_argument(
    "--layerwise-text-max-copy-streams",
    type=int,
    default=None,
    help=("Override text encoder layerwise max_copy_streams. When omitted, text encoder "
          "layerwise offload uses --layerwise-max-copy-streams."),
  )
  parser.add_argument(
    "--layerwise-max-inflight-prefetch-bytes",
    type=_parse_byte_size_arg,
    default=None,
    help=(
      "Maximum total CUDA residency budget, in bytes, for in-flight layerwise future-target "
      "prefetch. This shared setting applies to both generic module layerwise offload and "
      "SVDQ layerwise collection when --svdq-params layerwise_offload=true is active. Accepts raw bytes "
      "or suffixes like 512MiB and 4GiB. When omitted, runtime leaves the byte-budget "
      "limit disabled."),
  )
  parser.add_argument(
    "--layerwise-text-max-inflight-prefetch-bytes",
    type=_parse_byte_size_arg,
    default=None,
    help=("Override text encoder layerwise max_inflight_prefetch_bytes. Accepts raw bytes or "
          "suffixes like 512MiB and 4GiB. When omitted, text encoder layerwise offload uses "
          "--layerwise-max-inflight-prefetch-bytes."),
  )
  parser.add_argument(
    "--layerwise-persistent-buckets",
    type=int,
    default=0,
    help=(
      "How many selected layerwise offload targets should stay resident on CUDA for the full "
      "handle lifetime. This shared setting applies to both generic module layerwise offload "
      "and SVDQ layerwise collection when --svdq-params layerwise_offload=true is active. Default is 0."
    ),
  )
  parser.add_argument(
    "--layerwise-text-persistent-buckets",
    type=int,
    default=None,
    help=("Override text encoder layerwise persistent_buckets. When omitted, text encoder "
          "layerwise offload uses --layerwise-persistent-buckets."),
  )
  parser.add_argument(
    "--layerwise-persistent-bins",
    type=int,
    default=1,
    help=(
      "How many evenly distributed bins should be used when placing persistent layerwise "
      "offload targets across the selected target list. A value of 1 keeps the original "
      "prefix behavior. For example, with 32 targets, --layerwise-persistent-buckets 16 and "
      "--layerwise-persistent-bins 4, runtime keeps [0:4), [8:12), [16:20), and [24:28) "
      "resident. This shared setting applies to both generic module layerwise offload and "
      "SVDQ layerwise collection when --svdq-params layerwise_offload=true is active. Default is 1."
    ),
  )
  parser.add_argument(
    "--layerwise-text-persistent-bins",
    type=int,
    default=None,
    help=("Override text encoder layerwise persistent_bins. When omitted, text encoder "
          "layerwise offload uses --layerwise-persistent-bins."),
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
    help="Device for torch.Generator, e.g., 'cuda' or 'cpu'. Defaults to the current "
    "rank's accelerator, falling back to CPU when there is none. CPU sampling is not "
    "reproducible across torch versions: torch 2.12 changed the bf16 `randn` kernel.",
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
  _resolve_cli_cache_params(args)
  _resolve_cli_parallelism(args)
  _resolve_cli_quantization(args)

  # Force enable compile if force_compile_dynamic is enabled
  if args.force_compile_dynamic:
    args.compile = True

  _apply_attn_params(args)

  # Resolve --calibrator-params; its `type` key selects the calibrator.
  raw_calibrator = args.calibrator_params
  if raw_calibrator is not None and args.config_path is not None:
    raise ValueError("--calibrator-params cannot be combined with --config-path; "
                     "set the calibrator inside the config file instead.")
  args.calibrator = None
  args.calibrator_fields = None
  if raw_calibrator is not None:
    if not args.cache:
      raise ValueError(f"--calibrator-params type={raw_calibrator['type']} requires "
                       f"--cache-params, e.g. type=DBCache.")
    args.calibrator = normalize_calibrator_type(raw_calibrator["type"])
    # Keep `calibrator_params` in its raw form so post-processing stays idempotent; the
    # resolved config-field mapping lives in `calibrator_fields`. `type` is the selector, not an
    # option, so it is dropped here rather than loosening `resolve_calibrator_params`.
    options = {key: text for key, text in raw_calibrator.items() if key != "type"}
    args.calibrator_fields = resolve_calibrator_params(args.calibrator, options)

  return args
