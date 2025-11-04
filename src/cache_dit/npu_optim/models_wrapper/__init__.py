from .flux_transformer_block_wrapper import replace_flux_transformer_block_forward

NPU_MODELS_WRAPPER_MAP = {
    "flux_transformer_block_forward": replace_flux_transformer_block_forward,
}