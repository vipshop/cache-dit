import torch
import torch.distributed as dist
from cache_dit.envs import ENV
from cache_dit.logger import init_logger, logging_rank_0

logger = init_logger(__name__)


def epilogue_prologue_fusion_enabled(**kwargs) -> bool:
    mode = kwargs.get("epilogue_prologue_fusion", False)

    if ENV.CACHE_DIT_EPILOGUE_PROLOGUE_FUSION:
        logging_rank_0(
            logger,
            "CACHE_DIT_EPILOGUE_PROLOGUE_FUSION is set to 1. \n"
            "Force enable epilogue and prologue fusion.",
        )

    return ENV.CACHE_DIT_EPILOGUE_PROLOGUE_FUSION or mode


def enable_compile_compute_comm_overlap():
    ENV.CACHE_DIT_ENABLE_COMPILE_COMPUTE_COMM_OVERLAP = True
    logger.info("Enabled compile compute-communication overlap manually.")


def disable_compile_compute_comm_overlap():
    ENV.CACHE_DIT_ENABLE_COMPILE_COMPUTE_COMM_OVERLAP = False
    logger.info("Disabled compile compute-communication overlap manually.")


def is_compile_compute_comm_overlap_enabled() -> bool:
    return ENV.CACHE_DIT_ENABLE_COMPILE_COMPUTE_COMM_OVERLAP


def set_compile_configs(
    descent_tuning: bool = False,
    cuda_graphs: bool = False,
    force_disable_compile_caches: bool = False,
    use_fast_math: bool = False,
    compute_comm_overlap: bool = True,
    capture_scalar_outputs: bool = False,
    capture_dynamic_output_shape_ops: bool = False,
    **kwargs,  # other kwargs
):
    # Alway increase recompile_limit for dynamic shape compilation
    torch._dynamo.config.recompile_limit = 1024  # default is 8
    torch._dynamo.config.accumulated_recompile_limit = 8192  # default is 256
    # Handle compiler caches
    # https://github.com/vllm-project/vllm/blob/23baa2180b0ebba5ae94073ba9b8e93f88b75486/vllm/compilation/compiler_interface.py#L270
    torch._inductor.config.fx_graph_cache = True
    torch._inductor.config.fx_graph_remote_cache = False
    # https://github.com/pytorch/pytorch/issues/153791
    torch._inductor.config.autotune_local_cache = False

    if dist.is_initialized():
        # Enable compute comm overlap
        torch._inductor.config.reorder_for_compute_comm_overlap = (
            compute_comm_overlap and is_compile_compute_comm_overlap_enabled()
        )
        # L20 64 GB/s, PCIe; A100/A800 NVLink 300 GB/s.
        if torch._inductor.config.reorder_for_compute_comm_overlap:
            torch._inductor.config.intra_node_bw = (
                64 if "L20" in torch.cuda.get_device_name() else 300
            )

    # https://docs.pytorch.org/docs/stable/nested.html#data-dependent-operation-within-torch-compile
    if hasattr(torch._dynamo.config, "capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = capture_scalar_outputs
        torch._dynamo.config.capture_dynamic_output_shape_ops = capture_dynamic_output_shape_ops

    if not descent_tuning:
        return

    if ENV.CACHE_DIT_FORCE_DISABLE_CUSTOM_COMPILE_CONFIG:
        logging_rank_0(
            logger,
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
    if epilogue_prologue_fusion_enabled(**kwargs):
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
