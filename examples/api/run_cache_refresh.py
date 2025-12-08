import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from utils import get_args, strify, MemoryTracker
import cache_dit


parser = get_args(parse=False)
parser.add_argument(
    "--no-adapt",
    action="store_true",
    default=False,
    help="Disable BlockAdapter or not",
)
parser.add_argument(
    "--summary",
    action="store_true",
    default=False,
    help="Print summary of the model after each inference",
)
args = parser.parse_args()
print(args)


pipe = FluxPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "FLUX_DIR",
            "black-forest-labs/FLUX.1-dev",
        )
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

if args.cache:
    from cache_dit import (
        BlockAdapter,
        ForwardPattern,
        ParamsModifier,
        DBCacheConfig,
    )

    assert isinstance(pipe.transformer, FluxTransformer2DModel)

    cache_dit.enable_cache(
        (
            BlockAdapter(
                pipe=None,
                transformer=pipe.transformer,
                blocks=[
                    pipe.transformer.transformer_blocks,
                    pipe.transformer.single_transformer_blocks,
                ],
                forward_pattern=[
                    ForwardPattern.Pattern_1,
                    ForwardPattern.Pattern_1,
                ],
            )
            if not args.no_adapt
            else pipe.transformer
        ),
        cache_config=(
            DBCacheConfig(
                Fn_compute_blocks=args.Fn,
                Bn_compute_blocks=args.Bn,
                max_warmup_steps=args.max_warmup_steps,
                max_cached_steps=args.max_cached_steps,
                max_continuous_cached_steps=args.max_continuous_cached_steps,
                residual_diff_threshold=args.rdt,
                # NOTE: num_inference_steps can be None here, we will
                # set it properly during cache refreshing.
                num_inference_steps=None,
            )
            if args.cache
            else None
        ),
        params_modifiers=[
            ParamsModifier(
                cache_config=DBCacheConfig().reset(
                    residual_diff_threshold=args.rdt,
                ),
            ),
            ParamsModifier(
                # NOTE: single_transformer_blocks should have higher
                # residual_diff_threshold because of the precision error
                # accumulation from previous transformer_blocks
                cache_config=DBCacheConfig().reset(
                    residual_diff_threshold=args.rdt * 3,
                ),
            ),
        ],
    )

# Set default prompt
prompt = "A cat holding a sign that says hello world"
if args.prompt is not None:
    prompt = args.prompt


def run_pipe(steps: int = 28):
    cache_dit.refresh_context(
        pipe.transformer,
        num_inference_steps=steps,
        verbose=True,
    )
    image = pipe(
        prompt,
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)


memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

steps = [8, 16, 28, 40, 50]
for i in range(len(steps)):
    print("-" * 150)
    start = time.time()
    image = run_pipe(steps=steps[i])
    end = time.time()
    time_cost = end - start

    save_path = f"flux.steps{steps[i]}.{strify(args, pipe.transformer)}.png"
    image.save(save_path)

    if args.summary:
        cache_dit.summary(pipe.transformer)
        print(f"Time cost: {time_cost:.2f}s")
        print(f"Saving image to {save_path}")


if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()
