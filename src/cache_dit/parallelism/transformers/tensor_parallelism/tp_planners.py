# NOTE: must import all planner classes to register them
def _activate_tp_planners():
    """Function to register all built-in tensor parallelism planners."""
    from .tp_plan_cogview import CogViewTensorParallelismPlanner  # noqa: F401
    from .tp_plan_flux import FluxTensorParallelismPlanner  # noqa: F401
    from .tp_plan_flux2 import Flux2TensorParallelismPlanner  # noqa: F401
    from .tp_plan_hunyuan_dit import HunyuanDiTTensorParallelismPlanner  # noqa: F401
    from .tp_plan_kandinsky5 import Kandinsky5TensorParallelismPlanner  # noqa: F401
    from .tp_plan_mochi import MochiTensorParallelismPlanner  # noqa: F401
    from .tp_plan_ltx_video import LTXVideoTensorParallelismPlanner  # noqa: F401
    from .tp_plan_pixart import PixArtTensorParallelismPlanner  # noqa: F401
    from .tp_plan_qwen_image import QwenImageTensorParallelismPlanner  # noqa: F401
    from .tp_plan_wan import WanTensorParallelismPlanner  # noqa: F401
    from .tp_plan_skyreels import SkyReelsV2TensorParallelismPlanner  # noqa: F401
    from .tp_plan_zimage import ZImageTensorParallelismPlanner  # noqa: F401
    from .tp_plan_ovis_image import OvisImageTensorParallelismPlanner  # noqa: F401
    from .tp_plan_longcat_image import LongCatImageTensorParallelismPlanner  # noqa: F401


__all__ = ["_activate_tp_planners"]
