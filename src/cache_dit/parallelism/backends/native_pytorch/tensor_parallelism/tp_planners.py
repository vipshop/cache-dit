# NOTE: must import all planner classes to register them
from .tp_plan_cogview3_plus import CogView3PlusTensorParallelismPlanner
from .tp_plan_cogview4 import CogView4TensorParallelismPlanner
from .tp_plan_flux import FluxTensorParallelismPlanner
from .tp_plan_hunyuan_dit import HunyuanDiTTensorParallelismPlanner
from .tp_plan_kandinsky5 import Kandinsky5TensorParallelismPlanner
from .tp_plan_mochi import MochiTensorParallelismPlanner
from .tp_plan_qwen_image import QwenImageTensorParallelismPlanner
from .tp_plan_registers import TensorParallelismPlannerRegister
from .tp_plan_wan import WanTensorParallelismPlanner

__all__ = [
    "CogView3PlusTensorParallelismPlanner",
    "CogView4TensorParallelismPlanner",
    "FluxTensorParallelismPlanner",
    "HunyuanDiTTensorParallelismPlanner",
    "Kandinsky5TensorParallelismPlanner",
    "MochiTensorParallelismPlanner",
    "QwenImageTensorParallelismPlanner",
    "TensorParallelismPlannerRegister",
    "WanTensorParallelismPlanner",
]
