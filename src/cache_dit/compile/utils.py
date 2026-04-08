import torch
import torch.distributed as dist
from typing import Optional
from ..envs import ENV
from ..platforms import current_platform
from ..parallelism.attention._templated_ulysses import is_ulysses_anything_enabled
from ..logger import init_logger

logger = init_logger(__name__)


def set_compile_configs(
    descent_tuning: bool = True,  # more compiling times but may get better performance.
    cuda_graphs: bool = False,
    force_disable_compile_caches: bool = False,
    fx_graph_cache: bool = True,
    fx_graph_remote_cache: bool = False,
    autotune_local_cache: bool = False,
    use_fast_math: bool = False,
    compute_comm_overlap: bool = True,
    capture_scalar_outputs: Optional[bool] = None,
    capture_dynamic_output_shape_ops: Optional[bool] = None,
    cutedsl_enable_autotuning: Optional[bool] = None,  # >= PyTorch 2.11
    **kwargs,  # other kwargs
):
    """Apply cache-dit's recommended global `torch.compile` configuration tweaks.

    Args:
        descent_tuning: Whether to enable the more aggressive inductor tuning set
            used by cache-dit for generative workloads.
        cuda_graphs: Whether to enable CUDA graphs inside triton-generated kernels.
        force_disable_compile_caches: Whether to force-disable inductor compile
            caches for debugging or reproducibility.
        fx_graph_cache: Whether to enable the local FX graph cache.
        fx_graph_remote_cache: Whether to enable the remote FX graph cache.
        autotune_local_cache: Whether to enable the local autotune cache.
        use_fast_math: Whether to enable fast-math flags when supported by the
            current compiler backend.
        compute_comm_overlap: Whether to enable compile-time compute/communication
            overlap reordering when distributed execution is initialized.
        capture_scalar_outputs: Optional override for
            `torch._dynamo.config.capture_scalar_outputs`.
        capture_dynamic_output_shape_ops: Optional override for
            `torch._dynamo.config.capture_dynamic_output_shape_ops`.
        cutedsl_enable_autotuning: Optional override for CUTE DSL autotuning on
            newer PyTorch releases.
        **kwargs: Extra compile-related toggles reserved for forward compatibility.

    Notes:
        This function mutates global dynamo/inductor configuration and is meant to
        be called before compiling cache-dit modules. It is intentionally tolerant of
        missing compiler internals so it can run across multiple PyTorch versions.
    """

    # Dynamo Configs

    try:
        import torch._dynamo.config as dynamo_config  # type: ignore[attr-defined]

        # Alway increase recompile_limit for dynamic shape compilation
        dynamo_config.recompile_limit = 1024  # default is 8
        dynamo_config.accumulated_recompile_limit = 8192  # default is 256
        # https://docs.pytorch.org/docs/stable/nested.html#data-dependent-operation-within-torch-compile
        if hasattr(dynamo_config, "capture_scalar_outputs"):
            if capture_scalar_outputs is None:
                # Exiplicitly set capture_scalar_outputs to True to avoid graph break
                # while using Ulysses Anything Attention:
                # Graph break from `Tensor.item()`, consider setting:
                # torch._dynamo.config.capture_scalar_outputs = True
                if is_ulysses_anything_enabled():
                    capture_scalar_outputs = True if torch.__version__ >= "2.10.0" else False
                    if capture_scalar_outputs:
                        logger.info(
                            "Ulysses Anything Attention is enabled. "
                            "Auto set capture_scalar_outputs as True "
                            "to avoid graph break from scalar outpus, "
                            "e.g., Tensor.item()."
                        )
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

        # Enable compute comm overlap
        if dist.is_initialized():
            inductor_config.reorder_for_compute_comm_overlap = (
                compute_comm_overlap and ENV.CACHE_DIT_ENABLE_COMPILE_COMPUTE_COMM_OVERLAP
            )
            # L20 64 GB/s, PCIe; A100/A800 NVLink 300 GB/s.
            if inductor_config.reorder_for_compute_comm_overlap:
                inductor_config.intra_node_bw = (
                    64 if "L20" in current_platform.get_device_name() else 300
                )

        if cutedsl_enable_autotuning is not None and hasattr(
            inductor_config,
            "cutedsl_enable_autotuning",
        ):
            inductor_config.cutedsl_enable_autotuning = cutedsl_enable_autotuning

        if not descent_tuning:
            return

        if ENV.CACHE_DIT_FORCE_DISABLE_CUSTOM_COMPILE_CONFIG:
            logger.info(
                "CACHE_DIT_FORCE_DISABLE_CUSTOM_COMPILE_CONFIG is set to 1. \n"
                "Force disable custom compile config.",
            )
            return

        # Below are default settings for torch.compile, you can change
        # them to your needs and test the performance
        inductor_config.max_fusion_size = 64
        inductor_config.max_pointwise_cat_inputs = 8
        inductor_config.triton.cudagraphs = cuda_graphs
        inductor_config.triton.use_block_ptr = False
        inductor_config.triton.codegen_upcast_to_fp32 = True

        # Copy from https://pytorch.org/blog/accelerating-generative-ai-3/
        inductor_config.conv_1x1_as_mm = True
        inductor_config.coordinate_descent_tuning = True
        inductor_config.coordinate_descent_check_all_directions = True
        inductor_config.epilogue_fusion = False

        # Enable epilogue and prologue fusion
        if ENV.CACHE_DIT_EPILOGUE_PROLOGUE_FUSION or kwargs.get(
            "epilogue_prologue_fusion",
            False,
        ):
            inductor_config.epilogue_fusion = True
            inductor_config.prologue_fusion = True
            inductor_config.epilogue_fusion_first = True

        # Dead code elimination
        inductor_config.dce = True  # default is False

        # May need to force disable all cache
        if force_disable_compile_caches:
            inductor_config.force_disable_caches = True
            inductor_config.fx_graph_cache = False
            inductor_config.fx_graph_remote_cache = False
            inductor_config.autotune_local_cache = False  # default is True

        # Use fast math
        if hasattr(inductor_config, "use_fast_math"):
            inductor_config.use_fast_math = use_fast_math
        if hasattr(inductor_config, "cuda.use_fast_math"):
            inductor_config.cuda.use_fast_math = use_fast_math
    except Exception:
        pass
