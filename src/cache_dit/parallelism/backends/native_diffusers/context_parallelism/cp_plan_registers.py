import torch
import logging
from abc import abstractmethod
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin

try:
    from diffusers.models._modeling_parallel import (
        ContextParallelInput,
        ContextParallelOutput,
        ContextParallelModelPlan,
    )
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>-0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )

from cache_dit.logger import init_logger

logger = init_logger(__name__)

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

__all__ = [
    "ContextParallelismPlaner",
    "ContextParallelismPlanerRegister",
    "FluxContextParallelismPlaner",
    "QwenImageContextParallelismPlaner",
    "WanContextParallelismPlaner",
    "LTXVideoContextParallelismPlaner",
]


class ContextParallelismPlaner:
    @abstractmethod
    def apply(
        self,
        # NOTE: Keep this kwarg for future extensions
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        # NOTE: This method should only return the CP plan dictionary.
        raise NotImplementedError(
            "apply method must be implemented by subclasses"
        )


class ContextParallelismPlanerRegister:
    _cp_planer_registry: dict[str, ContextParallelismPlaner] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(planer_cls: type[ContextParallelismPlaner]):
            assert (
                name not in cls._cp_planer_registry
            ), f"ContextParallelismPlaner with name {name} is already registered."
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Registering ContextParallelismPlaner: {name}")
            cls._cp_planer_registry[name] = planer_cls
            return planer_cls

        return decorator

    @classmethod
    def get_planer(
        cls, transformer: str | torch.nn.Module | ModelMixin
    ) -> type[ContextParallelismPlaner]:
        if isinstance(transformer, (torch.nn.Module, ModelMixin)):
            name = transformer.__class__.__name__
        else:
            name = transformer
        planer_cls = None
        for planer_name in cls._cp_planer_registry:
            if name.startswith(planer_name):
                planer_cls = cls._cp_planer_registry.get(planer_name)
                break
        if planer_cls is None:
            raise ValueError(f"No planer registered under name: {name}")
        return planer_cls


# Register context parallelism planer for models
@ContextParallelismPlanerRegister.register("Flux")
class FluxContextParallelismPlaner(ContextParallelismPlaner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        if transformer is not None:
            from diffusers import FluxTransformer2DModel

            assert isinstance(
                transformer, FluxTransformer2DModel
            ), "Transformer must be an instance of FluxTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                return transformer._cp_plan

        _cp_plan = {
            "": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "img_ids": ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=False
                ),
                "txt_ids": ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=False
                ),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


@ContextParallelismPlanerRegister.register("QwenImage")
class QwenImageContextParallelismPlaner(ContextParallelismPlaner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        if transformer is not None:
            from diffusers import QwenImageTransformer2DModel

            assert isinstance(
                transformer, QwenImageTransformer2DModel
            ), "Transformer must be an instance of QwenImageTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                return transformer._cp_plan

        _cp_plan = _cp_plan = {
            "": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states_mask": ContextParallelInput(
                    split_dim=1, expected_dims=2, split_output=False
                ),
            },
            "pos_embed": {
                0: ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=True
                ),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


# TODO: Add WanVACETransformer3DModel context parallelism planer.
# NOTE: We choice to use full name to avoid name conflict between
# WanTransformer3DModel and WanVACETransformer3DModel.
@ContextParallelismPlanerRegister.register("WanTransformer3D")
class WanContextParallelismPlaner(ContextParallelismPlaner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        if transformer is not None:
            from diffusers import WanTransformer3DModel

            assert isinstance(
                transformer, WanTransformer3DModel
            ), "Transformer must be an instance of WanTransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                return transformer._cp_plan

        _cp_plan = {
            "rope": {
                0: ContextParallelInput(
                    split_dim=1, expected_dims=4, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=1, expected_dims=4, split_output=True
                ),
            },
            "blocks.0": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "blocks.*": {
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


@ContextParallelismPlanerRegister.register("LTXVideo")
class LTXVideoContextParallelismPlaner(ContextParallelismPlaner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        if transformer is not None:
            from diffusers import LTXVideoTransformer3DModel

            assert isinstance(
                transformer, LTXVideoTransformer3DModel
            ), "Transformer must be an instance of LTXVideoTransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                return transformer._cp_plan

        _cp_plan = {
            "": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_attention_mask": ContextParallelInput(
                    split_dim=1, expected_dims=2, split_output=False
                ),
            },
            "rope": {
                0: ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=True
                ),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
