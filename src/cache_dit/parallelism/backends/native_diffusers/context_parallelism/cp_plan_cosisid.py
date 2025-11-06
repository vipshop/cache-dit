import torch
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.consisid_transformer_3d import (
    ConsisIDTransformer3DModel,
)

try:
    from diffusers.models._modeling_parallel import (
        ContextParallelModelPlan,
    )
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )
from .cp_plan_registers import (
    ContextParallelismPlannerRegister,
)
from .cp_plan_cogvideox import CogVideoXContextParallelismPlanner

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("ConsisID")
class CosisIDContextParallelismPlanner(CogVideoXContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Diffusers native CP plan still not supported
        # for ConsisID now.
        self._cp_planner_preferred_native_diffusers = False

        if (
            transformer is not None
            and self._cp_planner_preferred_native_diffusers
        ):
            assert isinstance(
                transformer, ConsisIDTransformer3DModel
            ), "Transformer must be an instance of ConsisIDTransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        return super().apply(transformer=transformer, **kwargs)
