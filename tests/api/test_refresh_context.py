import gc
import pytest
import torch
import cache_dit
from cache_dit import ForwardPattern, BlockAdapter, DBCacheConfig
from cache_dit.platforms import current_platform
from utils import RandPipeline


DEVICES = (
    ["cpu"]
    if not current_platform.is_accelerator_available()
    else ["cpu", current_platform.device_type]
)
PATTERNS = [
    ForwardPattern.Pattern_0,
    ForwardPattern.Pattern_1,
    ForwardPattern.Pattern_2,
    ForwardPattern.Pattern_3,
    ForwardPattern.Pattern_4,
    ForwardPattern.Pattern_5,
]

DTYPES = (
    [torch.float32]
    if not current_platform.is_accelerator_available()
    else [torch.float32, torch.bfloat16]
)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_refresh_context(device, pattern, dtype):
    gc.collect()
    pipe = RandPipeline(pattern=pattern)  # type: RandPipeline
    transformer = pipe.transformer

    transformer = pipe.transformer
    adapter = cache_dit.enable_cache(
        BlockAdapter(
            transformer=transformer,
            blocks=transformer.transformer_blocks,
            forward_pattern=pipe.pattern,
        ),
        cache_config=DBCacheConfig(
            Fn_compute_blocks=8,
            Bn_compute_blocks=0,
            residual_diff_threshold=0.05,
        ),
    )

    # Transformer only API
    bs, seq_len, headdim = 1, 1024, 64

    hidden_states = torch.normal(
        mean=100.0,
        std=20.0,
        size=(bs, seq_len, headdim),
        dtype=dtype,
    )

    encoder_hidden_states = None
    if pattern in [
        ForwardPattern.Pattern_0,
        ForwardPattern.Pattern_1,
        ForwardPattern.Pattern_2,
    ]:
        encoder_hidden_states = torch.normal(
            mean=100.0,
            std=20.0,
            size=(bs, seq_len, headdim),
            dtype=dtype,
        )

    if device == current_platform.device_type:
        pipe.to(device)
        hidden_states = hidden_states.to(device)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(device)

    STEPS = [16, 28, 50]
    if pattern in [
        ForwardPattern.Pattern_0,
        ForwardPattern.Pattern_1,
        ForwardPattern.Pattern_2,
    ]:
        for i, steps in enumerate(STEPS):
            # Refresh cache context
            if i == 0:
                # Test num_inference_steps only case
                cache_dit.refresh_context(
                    transformer,
                    num_inference_steps=steps,
                    verbose=True,
                )
            else:
                cache_dit.refresh_context(
                    transformer,
                    cache_config=DBCacheConfig(
                        Fn_compute_blocks=1,
                        Bn_compute_blocks=0,
                        residual_diff_threshold=0.08,
                        num_inference_steps=steps,
                        steps_computation_mask=cache_dit.steps_mask(
                            mask_policy="fast",
                            total_steps=steps,
                        ),
                        steps_computation_policy="dynamic",
                        enable_separate_cfg=False,
                    ),
                    verbose=True,
                )
            _ = pipe(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                num_inference_steps=steps,
            )
    else:
        for i, steps in enumerate(STEPS):
            if i == 0:
                # Test num_inference_steps only case
                cache_dit.refresh_context(
                    transformer,
                    num_inference_steps=steps,
                    verbose=True,
                )
            else:
                # Refresh cache context
                cache_dit.refresh_context(
                    transformer,
                    cache_config=DBCacheConfig(
                        Fn_compute_blocks=1,
                        Bn_compute_blocks=0,
                        residual_diff_threshold=0.08,
                        num_inference_steps=steps,
                        steps_computation_mask=cache_dit.steps_mask(
                            mask_policy="fast",
                            total_steps=steps,
                        ),
                        steps_computation_policy="dynamic",
                        enable_separate_cfg=False,
                    ),
                    verbose=True,
                )
            _ = pipe(
                hidden_states,
                num_inference_steps=steps,
            )

    cache_dit.summary(transformer)
    # We have to disable cache before deleting the pipe and adapter
    # using block adapter instance due to the fake pipe we used in
    # transformer only API.
    cache_dit.disable_cache(adapter)

    del pipe
    del adapter
    del hidden_states
    if encoder_hidden_states is not None:
        del encoder_hidden_states
    gc.collect()
