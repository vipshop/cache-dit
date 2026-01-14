import importlib
from cache_dit.logger import init_logger
from .tp_plan_registers import TextEncoderTensorParallelismPlanner

logger = init_logger(__name__)


class ImportErrorTextEncoderTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
    def plan(
        self,
        text_encoder,
        **kwargs,
    ):
        raise ImportError(
            "This TextEncoderTensorParallelismPlanner requires latest diffusers to be installed. "
            "Please install diffusers from source."
        )


def _safe_import(module_name: str, class_name: str) -> type[TextEncoderTensorParallelismPlanner]:
    try:
        # e.g., module_name = ".tp_plan_t5_encoder", class_name = "T5EncoderTensorParallelismPlanner"
        package = __package__ if __package__ is not None else ""
        module = importlib.import_module(module_name, package=package)
        target_class = getattr(module, class_name)
        return target_class
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import {class_name} from {module_name}: {e}")
        return ImportErrorTextEncoderTensorParallelismPlanner


def _activate_text_encoder_tp_planners():
    """Function to register all built-in tensor parallelism planners."""
    T5EncoderTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_t5_encoder", "T5EncoderTensorParallelismPlanner"
    )
    UMT5EncoderTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_umt5_encoder", "UMT5EncoderTensorParallelismPlanner"
    )
    MistralTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_mistral", "MistralTensorParallelismPlanner"
    )
    Qwen2_5_VLTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_qwen2_5", "Qwen2_5_VLTensorParallelismPlanner"
    )
    Qwen3TensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_qwen3", "Qwen3TensorParallelismPlanner"
    )
    LlamaTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_llama", "LlamaTensorParallelismPlanner"
    )
    GemmaTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_gemma", "GemmaTensorParallelismPlanner"
    )
    GlmTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_glm", "GlmTensorParallelismPlanner"
    )


__all__ = ["_activate_text_encoder_tp_planners"]
