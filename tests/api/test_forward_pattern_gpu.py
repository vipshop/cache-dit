import gc
import pytest
import torch
import cache_dit
from cache_dit import ForwardPattern, BlockAdapter, DBCacheConfig
from rand_pipe import RandPipeline


DEVICES = ["cuda"]
PATTERNS = [
    ForwardPattern.Pattern_0,
    ForwardPattern.Pattern_1,
    ForwardPattern.Pattern_2,
    ForwardPattern.Pattern_3,
    ForwardPattern.Pattern_4,
    ForwardPattern.Pattern_5,
]

DTYPES = [torch.bfloat16]
STEPS = [50]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("steps", STEPS)
def test_forward_pattern_gpu(device, pattern, dtype, steps):
    assert device == "cuda"
    gc.collect()
    pipe = RandPipeline(pattern=pattern)

    cache_dit.enable_cache(
        BlockAdapter(
            pipe=pipe,
            transformer=pipe.transformer,
            blocks=pipe.transformer.transformer_blocks,
            blocks_name="transformer_blocks",
            forward_pattern=pipe.pattern,
        ),
        cache_config=DBCacheConfig(
            Fn_compute_blocks=1,
            Bn_compute_blocks=0,
            residual_diff_threshold=0.05,
        ),
    )
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

    pipe.to(device)
    hidden_states = hidden_states.to(device)
    if encoder_hidden_states is not None:
        encoder_hidden_states = encoder_hidden_states.to(device)

    if pattern in [
        ForwardPattern.Pattern_0,
        ForwardPattern.Pattern_1,
        ForwardPattern.Pattern_2,
    ]:
        _ = pipe(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            num_inference_steps=steps,
        )
    else:
        _ = pipe(
            hidden_states,
            num_inference_steps=steps,
        )

    cache_dit.summary(pipe, details=True)
    cache_dit.disable_cache(pipe)

    del pipe
    del hidden_states
    if encoder_hidden_states is not None:
        del encoder_hidden_states
    gc.collect()
