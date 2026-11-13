import torch
import torch.distributed as dist
from typing import Optional
from ..envs import ENV
from ..platforms import current_platform
from ..logger import init_logger

logger = init_logger(__name__)


def set_compile_configs(
    descent_tuning: bool = False,  # aggressive inductor tuning, ~17x longer compile.
    cuda_graphs: bool = False,
    force_disable_compile_caches: bool = False,
    fx_graph_cache: bool = True,
    fx_graph_remote_cache: bool = False,
    autotune_local_cache: bool = False,
    use_fast_math: bool = False,
    compute_comm_overlap: bool = True,
    ulysses_anything: bool = False,
    capture_scalar_outputs: Optional[bool] = None,
    capture_dynamic_output_shape_ops: Optional[bool] = None,
    cutedsl_enable_autotuning: Optional[bool] = None,  # >= PyTorch 2.11
    **kwargs,  # other kwargs
):
  """Apply cache-dit's recommended global `torch.compile` configuration tweaks.

  :param descent_tuning: Whether to enable the more aggressive inductor tuning set
    used by cache-dit for generative workloads (coordinate descent tuning,
    epilogue/prologue fusion ordering, dead code elimination, conv_1x1_as_mm).
    Disabled by default: measured on a diffusion transformer it inflated the compile
    time by roughly an order of magnitude (coordinate descent tuning benchmarks every
    generated kernel) without changing the steady-state step latency, so enable it
    only after measuring your own workload.
  :param cuda_graphs: Whether to enable CUDA graphs inside triton-generated kernels.
  :param force_disable_compile_caches: Whether to force-disable inductor compile
    caches for debugging or reproducibility.
  :param fx_graph_cache: Whether to enable the local FX graph cache.
  :param fx_graph_remote_cache: Whether to enable the remote FX graph cache.
  :param autotune_local_cache: Whether to enable the local autotune cache.
  :param use_fast_math: Whether to enable fast-math flags when supported by the
    current compiler backend.
  :param compute_comm_overlap: Whether to enable compile-time compute/communication
    overlap reordering when distributed execution is initialized.
  :param ulysses_anything: Whether compile-time guards should assume uneven Ulysses
    sequence partitioning is enabled.
  :param capture_scalar_outputs: Optional override for
    `torch._dynamo.config.capture_scalar_outputs`.
  :param capture_dynamic_output_shape_ops: Optional override for
    `torch._dynamo.config.capture_dynamic_output_shape_ops`.
  :param cutedsl_enable_autotuning: Optional override for CUTE DSL autotuning on
    newer PyTorch releases.
  :param kwargs: Extra compile-related toggles reserved for forward compatibility.
  """

  # Dynamo Configs

  try:
    import torch._dynamo.config as dynamo_config  # type: ignore[attr-defined]

    # Always increase recompile_limit for dynamic shape compilation
    dynamo_config.recompile_limit = 1024  # default is 8
    dynamo_config.accumulated_recompile_limit = 8192  # default is 256
    # https://docs.pytorch.org/docs/stable/nested.html#data-dependent-operation-within-torch-compile
    if hasattr(dynamo_config, "capture_scalar_outputs"):
      if capture_scalar_outputs is None:
        # Explicitly set capture_scalar_outputs to True to avoid graph break
        # while using Ulysses Anything Attention:
        # Graph break from `Tensor.item()`, consider setting:
        # torch._dynamo.config.capture_scalar_outputs = True
        if ulysses_anything:
          capture_scalar_outputs = True if torch.__version__ >= "2.10.0" else False
          if capture_scalar_outputs:
            logger.info("Ulysses Anything Attention is enabled. "
                        "Auto set capture_scalar_outputs as True "
                        "to avoid graph break from scalar outputs, "
                        "e.g., Tensor.item().")
            dynamo_config.capture_scalar_outputs = capture_scalar_outputs
      else:
        dynamo_config.capture_scalar_outputs = capture_scalar_outputs
    if hasattr(dynamo_config, "capture_dynamic_output_shape_ops"):
      if capture_dynamic_output_shape_ops is not None:
        dynamo_config.capture_dynamic_output_shape_ops = capture_dynamic_output_shape_ops
  except Exception:
    pass

  # Inductor Configs
  try:
    import torch._inductor.config as inductor_config  # type: ignore[attr-defined]

    # Handle compiler caches
    # https://github.com/vllm-project/vllm/blob/23baa2180b0ebba5ae94073ba9b8e93f88b75486/vllm/compilation/compiler_interface.py#L270
    inductor_config.fx_graph_cache = fx_graph_cache
    inductor_config.fx_graph_remote_cache = fx_graph_remote_cache
    # https://github.com/pytorch/pytorch/issues/153791
    inductor_config.autotune_local_cache = autotune_local_cache

    # Enable compute comm overlap (only meaningful once distributed is initialized).
    # The intra-node bandwidth is deliberately left unset: inductor auto-detects it
    # from the interconnect (`comm_analysis.get_intra_node_bw`) and feeds it to its
    # NCCL runtime estimation. Setting it here also used to override a user-provided
    # value, since the assignment had no `is None` guard.
    if dist.is_initialized():
      inductor_config.reorder_for_compute_comm_overlap = (
        compute_comm_overlap and ENV.CACHE_DIT_ENABLE_COMPILE_COMPUTE_COMM_OVERLAP)

    if cutedsl_enable_autotuning is not None and hasattr(
        inductor_config,
        "cutedsl_enable_autotuning",
    ):
      inductor_config.cutedsl_enable_autotuning = cutedsl_enable_autotuning

    # The toggles below are driven by explicit arguments or env vars, so they must stay
    # above the early return: they used to be unreachable whenever descent tuning was
    # disabled, which is now the default.
    if force_disable_compile_caches:
      inductor_config.force_disable_caches = True
      inductor_config.fx_graph_cache = False
      inductor_config.fx_graph_remote_cache = False
      inductor_config.autotune_local_cache = False  # default is True

    if hasattr(inductor_config, "use_fast_math"):
      inductor_config.use_fast_math = use_fast_math
    if hasattr(inductor_config, "cuda.use_fast_math"):
      inductor_config.cuda.use_fast_math = use_fast_math

    # Enable epilogue and prologue fusion by default. The environment variable
    # can disable it globally, while an explicit keyword takes precedence.
    fusion_enabled = kwargs.get(
      "epilogue_prologue_fusion",
      ENV.CACHE_DIT_EPILOGUE_PROLOGUE_FUSION,
    )
    if current_platform.device_type == "npu":
      # NPU: epilogue fusion is not supported on that backend.
      inductor_config.epilogue_fusion = False
    else:
      inductor_config.triton.cudagraphs = cuda_graphs
      inductor_config.epilogue_fusion = fusion_enabled
      inductor_config.prologue_fusion = fusion_enabled
      # Reordering the fusion schedule is part of the aggressive tuning set.
      if descent_tuning:
        inductor_config.epilogue_fusion_first = fusion_enabled

    if not descent_tuning:
      return

    if ENV.CACHE_DIT_FORCE_DISABLE_CUSTOM_COMPILE_CONFIG:
      logger.info(
        "CACHE_DIT_FORCE_DISABLE_CUSTOM_COMPILE_CONFIG is set to 1. \n"
        "Force disable custom compile config.", )
      return

    # Below are the aggressive inductor tuning defaults for generative workloads.
    inductor_config.max_fusion_size = 64
    inductor_config.max_pointwise_cat_inputs = 8

    if current_platform.device_type == "npu":
      # NPU: skip CUDA-specific inductor configs (triton, coordinate_descent, etc)
      inductor_config.dce = True  # default is False
    else:
      inductor_config.triton.use_block_ptr = False
      inductor_config.triton.codegen_upcast_to_fp32 = True

      # Copy from https://pytorch.org/blog/accelerating-generative-ai-3/
      inductor_config.conv_1x1_as_mm = True
      inductor_config.coordinate_descent_tuning = True
      inductor_config.coordinate_descent_check_all_directions = True

      # Dead code elimination
      inductor_config.dce = True  # default is False
  except Exception:
    pass


def mindiesd_compile_available():
  try:
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
      return False

    import mindiesd  # noqa F401
    from mindiesd.compilation import MindieSDBackend  # noqa F401

    return True
  except ImportError:
    return False


def mindiesd_compile(module, module_name, module_cls_name):
  # Auto-apply MindieSDBackend compile on NPU when mindiesd is available.
  # Returns the compiled module if compiled, None if MindIE-SD not applicable.
  try:
    import mindiesd  # noqa F401

    if not hasattr(torch, 'npu') or not torch.npu.is_available():
      return None
    from mindiesd.compilation import MindieSDBackend

    logger.info(f"Compiling {module_name}: {module_cls_name} with MindieSDBackend ...")
    return torch.compile(module, backend=MindieSDBackend(), dynamic=True)
  except Exception:
    return None
