import torch
import torch.distributed as dist
from typing import Optional
from ..envs import ENV
from ..platforms import current_platform
from ..parallelism.attention._templated_ulysses import is_ulysses_anything_enabled
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def set_compile_configs(
    descent_tuning: bool = False,
    cuda_graphs: bool = False,
    force_disable_compile_caches: bool = False,
    fx_graph_cache: bool = True,
    fx_graph_remote_cache: bool = False,
    autotune_local_cache: bool = False,
    use_fast_math: bool = False,
    compute_comm_overlap: bool = True,
    capture_scalar_outputs: Optional[bool] = None,
    capture_dynamic_output_shape_ops: Optional[bool] = None,
    **kwargs,  # other kwargs
):
    # Alway increase recompile_limit for dynamic shape compilation
    torch._dynamo.config.recompile_limit = 1024  # default is 8
    torch._dynamo.config.accumulated_recompile_limit = 8192  # default is 256
    # Handle compiler caches
    # https://github.com/vllm-project/vllm/blob/23baa2180b0ebba5ae94073ba9b8e93f88b75486/vllm/compilation/compiler_interface.py#L270
    torch._inductor.config.fx_graph_cache = fx_graph_cache
    torch._inductor.config.fx_graph_remote_cache = fx_graph_remote_cache
    # https://github.com/pytorch/pytorch/issues/153791
    torch._inductor.config.autotune_local_cache = autotune_local_cache

    if dist.is_initialized():
        # Enable compute comm overlap
        torch._inductor.config.reorder_for_compute_comm_overlap = (
            compute_comm_overlap and ENV.CACHE_DIT_ENABLE_COMPILE_COMPUTE_COMM_OVERLAP
        )
        # L20 64 GB/s, PCIe; A100/A800 NVLink 300 GB/s.
        if torch._inductor.config.reorder_for_compute_comm_overlap:
            torch._inductor.config.intra_node_bw = (
                64 if "L20" in current_platform.get_device_name() else 300
            )

    # https://docs.pytorch.org/docs/stable/nested.html#data-dependent-operation-within-torch-compile
    if hasattr(torch._dynamo.config, "capture_scalar_outputs"):
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
                    torch._dynamo.config.capture_scalar_outputs = capture_scalar_outputs
        else:
            torch._dynamo.config.capture_scalar_outputs = capture_scalar_outputs
    if hasattr(torch._dynamo.config, "capture_dynamic_output_shape_ops"):
        if capture_dynamic_output_shape_ops is not None:
            torch._dynamo.config.capture_dynamic_output_shape_ops = capture_dynamic_output_shape_ops

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
    torch._inductor.config.max_fusion_size = 64
    torch._inductor.config.max_pointwise_cat_inputs = 8
    torch._inductor.config.triton.cudagraphs = cuda_graphs
    torch._inductor.config.triton.use_block_ptr = False
    torch._inductor.config.triton.codegen_upcast_to_fp32 = True

    # Copy from https://pytorch.org/blog/accelerating-generative-ai-3/
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.epilogue_fusion = False

    # Enable epilogue and prologue fusion
    if ENV.CACHE_DIT_EPILOGUE_PROLOGUE_FUSION or kwargs.get("epilogue_prologue_fusion", False):
        torch._inductor.config.epilogue_fusion = True
        torch._inductor.config.prologue_fusion = True
        torch._inductor.config.epilogue_fusion_first = True

    # Dead code elimination
    torch._inductor.config.dce = True  # default is False

    # May need to force disable all cache
    if force_disable_compile_caches:
        torch._inductor.config.force_disable_caches = True
        torch._inductor.config.fx_graph_cache = False
        torch._inductor.config.fx_graph_remote_cache = False
        torch._inductor.config.autotune_local_cache = False  # default is True

    # Use fast math
    if hasattr(torch._inductor.config, "use_fast_math"):
        torch._inductor.config.use_fast_math = use_fast_math
    if hasattr(torch._inductor.config, "cuda.use_fast_math"):
        torch._inductor.config.cuda.use_fast_math = use_fast_math
