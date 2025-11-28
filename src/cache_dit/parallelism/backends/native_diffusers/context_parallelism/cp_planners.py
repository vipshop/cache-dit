# Docstring references: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/_modeling_parallel.py#L185
# A dictionary where keys denote the input to be split across context parallel region, and the
# value denotes the sharding configuration.
# If the key is a string, it denotes the name of the parameter in the forward function.
# If the key is an integer, split_output must be set to True, and it denotes the index of the output
# to be split across context parallel region.
# ContextParallelInputType = Dict[
#     Union[str, int], Union[ContextParallelInput, List[ContextParallelInput], Tuple[ContextParallelInput, ...]]
# ]

# A dictionary where keys denote the output to be gathered across context parallel region, and the
# value denotes the gathering configuration.
# ContextParallelOutputType = Union[
#     ContextParallelOutput, List[ContextParallelOutput], Tuple[ContextParallelOutput, ...]
# ]

# A dictionary where keys denote the module id, and the value denotes how the inputs/outputs of
# the module should be split/gathered across context parallel region.
# ContextParallelModelPlan = Dict[str, Union[ContextParallelInputType, ContextParallelOutputType]]

# Example of a ContextParallelModelPlan (QwenImageTransformer2DModel):
#
# Each model should define a _cp_plan attribute that contains information on how to shard/gather
# tensors at different stages of the forward:
#
# ```python
# _cp_plan = {
#     "": {
#         "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
#         "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
#         "encoder_hidden_states_mask": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
#     },
#     "pos_embed": {
#         0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
#         1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
#     },
#     "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
# }
# ```
#
# The dictionary is a set of module names mapped to their respective CP plan. The inputs/outputs of layers will be
# split/gathered according to this at the respective module level. Here, the following happens:
# - "":
#     we specify that we want to split the various inputs across the sequence dim in the pre-forward hook (i.e. before
#     the actual forward logic of the QwenImageTransformer2DModel is run, we will splitthe inputs)
# - "pos_embed":
#     we specify that we want to split the outputs of the RoPE layer. Since there are two outputs (imag & text freqs),
#     we can individually specify how they should be split
# - "proj_out":
#     before returning to the user, we gather the entire sequence on each rank in the post-forward hook (after the linear
#     layer forward has run).
#
# ContextParallelInput:
#     specifies how to split the input tensor in the pre-forward or post-forward hook of the layer it is attached to
#
# ContextParallelOutput:
#     specifies how to gather the input tensor in the post-forward hook in the layer it is attached to

from .cp_plan_registers import (
    ContextParallelismPlanner,
    ContextParallelismPlannerRegister,
)
from .cp_plan_flux import FluxContextParallelismPlanner
from .cp_plan_qwen_image import QwenImageContextParallelismPlanner
from .cp_plan_wan import WanContextParallelismPlanner
from .cp_plan_wan import WanVACEContextParallelismPlanner
from .cp_plan_ltxvideo import LTXVideoContextParallelismPlanner
from .cp_plan_hunyuan import HunyuanImageContextParallelismPlanner
from .cp_plan_hunyuan import HunyuanVideoContextParallelismPlanner
from .cp_plan_cogvideox import CogVideoXContextParallelismPlanner
from .cp_plan_cogview import CogView3PlusContextParallelismPlanner
from .cp_plan_cogview import CogView4ContextParallelismPlanner
from .cp_plan_cosisid import CosisIDContextParallelismPlanner
from .cp_plan_chroma import ChromaContextParallelismPlanner
from .cp_plan_pixart import PixArtContextParallelismPlanner
from .cp_plan_dit import DiTContextParallelismPlanner
from .cp_plan_kandinsky import Kandinsky5ContextParallelismPlanner
from .cp_plan_skyreels import SkyReelsV2ContextParallelismPlanner
from .cp_plan_flux2 import Flux2ContextParallelismPlanner
from .cp_plan_zimage import ZImageContextParallelismPlanner

try:
    import nunchaku  # noqa: F401

    _nunchaku_available = True
except ImportError:
    _nunchaku_available = False

if _nunchaku_available:
    from .cp_plan_nunchaku import (  # noqa: F401
        NunchakuFluxContextParallelismPlanner,
    )
    from .cp_plan_nunchaku import (  # noqa: F401
        NunchakuQwenImageContextParallelismPlanner,
    )


__all__ = [
    "ContextParallelismPlanner",
    "ContextParallelismPlannerRegister",
    "FluxContextParallelismPlanner",
    "QwenImageContextParallelismPlanner",
    "WanContextParallelismPlanner",
    "WanVACEContextParallelismPlanner",
    "LTXVideoContextParallelismPlanner",
    "HunyuanImageContextParallelismPlanner",
    "HunyuanVideoContextParallelismPlanner",
    "CogVideoXContextParallelismPlanner",
    "CogView3PlusContextParallelismPlanner",
    "CogView4ContextParallelismPlanner",
    "CosisIDContextParallelismPlanner",
    "ChromaContextParallelismPlanner",
    "PixArtContextParallelismPlanner",
    "DiTContextParallelismPlanner",
    "Kandinsky5ContextParallelismPlanner",
    "SkyReelsV2ContextParallelismPlanner",
    "Flux2ContextParallelismPlanner",
    "ZImageContextParallelismPlanner",
]

if _nunchaku_available:
    __all__.extend(
        [
            "NunchakuFluxContextParallelismPlanner",
            "NunchakuQwenImageContextParallelismPlanner",
        ]
    )
