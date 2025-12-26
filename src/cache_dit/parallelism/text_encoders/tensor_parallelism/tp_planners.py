# NOTE: must import all planner classes to register them
def _activate_text_encoder_tp_planners():
    """Function to register all built-in tensor parallelism planners."""
    from .tp_plan_t5_encoder import T5EncoderTensorParallelismPlanner  # noqa: F401
    from .tp_plan_umt5_encoder import UMT5EncoderTensorParallelismPlanner  # noqa: F401
    from .tp_plan_mistral import MistralTensorParallelismPlanner  # noqa: F401
    from .tp_plan_qwen2_5 import Qwen2_5_VLTensorParallelismPlanner  # noqa: F401
    from .tp_plan_qwen3 import Qwen3TensorParallelismPlanner  # noqa: F401
    from .tp_plan_llama import LlamaTensorParallelismPlanner  # noqa: F401
    from .tp_plan_gemma import GemmaTensorParallelismPlanner  # noqa: F401
    from .tp_plan_glm import GlmTensorParallelismPlanner  # noqa: F401


__all__ = ["_activate_text_encoder_tp_planners"]
