import torch
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin
from diffusers import ZImageTransformer2DModel

try:
    from diffusers.models._modeling_parallel import (
        ContextParallelInput,
        ContextParallelOutput,
        ContextParallelModelPlan,
    )
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )
from .cp_plan_registers import (
    ContextParallelismPlanner,
    ContextParallelismPlannerRegister,
)

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("ZImageTransformer2DModel")
class ZImageContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Diffusers native CP plan still not supported for ZImageTransformer2DModel
        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            assert isinstance(
                transformer, ZImageTransformer2DModel
            ), "Transformer must be an instance of ZImageTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
        n_noise_refiner_layers = len(transformer.noise_refiner)  # 2
        n_context_refiner_layers = len(transformer.context_refiner)  # 2
        num_layers = len(transformer.layers)  # 30
        _cp_plan = {
            # 0. Hooks for noise_refiner layers, 2
            "noise_refiner.0": {
                "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            "noise_refiner.*": {
                "freqs_cis": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            f"noise_refiner.{n_noise_refiner_layers - 1}": ContextParallelOutput(
                gather_dim=1, expected_dims=3
            ),
            # 1. Hooks for context_refiner layers, 2
            "context_refiner.0": {
                "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            "context_refiner.*": {
                "freqs_cis": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            f"context_refiner.{n_context_refiner_layers - 1}": ContextParallelOutput(
                gather_dim=1, expected_dims=3
            ),
            # 2. Hooks for main transformer layers, num_layers=30
            "layers.0": {
                "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            "layers.*": {
                "freqs_cis": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            # NOTE: The 'all_final_layer' is a ModuleDict of several final layers,
            # each for a specific patch size combination, so we do not add hooks for it here.
            # So, we have to gather the output of the last transformer layer.
            f"layers.{num_layers - 1}": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


# TODO: Add async Ulysses QKV proj for ZImage model
