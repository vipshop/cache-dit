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
import importlib
from cache_dit.logger import init_logger
from .cp_plan_registers import ContextParallelismPlanner

logger = init_logger(__name__)


class ImportErrorContextParallelismPlanner(ContextParallelismPlanner):
    def plan(
        self,
        transformer,
        **kwargs,
    ):
        raise ImportError(
            "This ContextParallelismPlanner requires latest diffusers to be installed. "
            "Please install diffusers from source."
        )


def _safe_import(module_name: str, class_name: str) -> type[ContextParallelismPlanner]:
    try:
        # e.g., module_name = ".cp_plan_dit", class_name = "DiTContextParallelismPlanner"
        package = __package__ if __package__ is not None else ""
        module = importlib.import_module(module_name, package=package)
        target_class = getattr(module, class_name)
        return target_class
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import {class_name} from {module_name}: {e}")
        return ImportErrorContextParallelismPlanner


def _activate_cp_planners():
    """Function to register all built-in context parallelism planners."""
    FluxContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_flux", "FluxContextParallelismPlanner"
    )
    QwenImageContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_qwen_image", "QwenImageContextParallelismPlanner"
    )
    WanContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_wan", "WanContextParallelismPlanner"
    )
    WanVACEContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_wan", "WanVACEContextParallelismPlanner"
    )
    LTXVideoContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_ltxvideo", "LTXVideoContextParallelismPlanner"
    )
    LTX2ContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_ltx2", "LTX2ContextParallelismPlanner"
    )
    HunyuanImageContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_hunyuan", "HunyuanImageContextParallelismPlanner"
    )
    HunyuanVideoContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_hunyuan", "HunyuanVideoContextParallelismPlanner"
    )
    CogVideoXContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_cogvideox", "CogVideoXContextParallelismPlanner"
    )
    CogView3PlusContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_cogview", "CogView3PlusContextParallelismPlanner"
    )
    CogView4ContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_cogview", "CogView4ContextParallelismPlanner"
    )
    CosisIDContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_cosisid", "CosisIDContextParallelismPlanner"
    )
    ChromaContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_chroma", "ChromaContextParallelismPlanner"
    )
    PixArtContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_pixart", "PixArtContextParallelismPlanner"
    )
    DiTContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_dit", "DiTContextParallelismPlanner"
    )
    Kandinsky5ContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_kandinsky", "Kandinsky5ContextParallelismPlanner"
    )
    SkyReelsV2ContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_skyreels", "SkyReelsV2ContextParallelismPlanner"
    )
    Flux2ContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_flux2", "Flux2ContextParallelismPlanner"
    )
    ZImageContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_zimage", "ZImageContextParallelismPlanner"
    )
    ChronoEditContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_chrono_edit", "ChronoEditContextParallelismPlanner"
    )
    OvisImageContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_ovis_image", "OvisImageContextParallelismPlanner"
    )
    LongCatImageContextParallelismPlanner = _safe_import(  # noqa: F841
        ".cp_plan_longcat_image", "LongCatImageContextParallelismPlanner"
    )

    try:
        import nunchaku  # noqa: F401

        _nunchaku_available = True
    except ImportError:
        _nunchaku_available = False

    if _nunchaku_available:
        NunchakuFluxContextParallelismPlanner = _safe_import(  # noqa: F841
            ".cp_plan_nunchaku", "NunchakuFluxContextParallelismPlanner"
        )
        NunchakuQwenImageContextParallelismPlanner = _safe_import(  # noqa: F841
            ".cp_plan_nunchaku", "NunchakuQwenImageContextParallelismPlanner"
        )
        NunchakuZImageContextParallelismPlanner = _safe_import(  # noqa: F841
            ".cp_plan_nunchaku", "NunchakuZImageContextParallelismPlanner"
        )


__all__ = ["_activate_cp_planners"]
