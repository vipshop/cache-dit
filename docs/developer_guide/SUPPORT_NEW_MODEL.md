# Support New Model  

Please make sure you have install and initialize pre-commit before adding any new commit. Refer [PRE_COMMIT](PRE_COMMIT.md) for more details.

## Cache Acceleration

In order to support <span style="color:#c77dff;">cache acceleration</span> for new model, we have to register it's BlockAdapter at [caching/block_adapters/adapter.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/block_adapters/adapters.py) and use `_safe_import` func to import it at [caching/block_adapters/\_\_init\_\_.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/block_adapters/__init__.py). For example:

- step 1: Implement the `qwenimage_adapter` at [caching/block_adapters/adapters.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/block_adapters/adapters.py)

```python
@BlockAdapterRegister.register("QwenImage")
def qwenimage_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import QwenImageTransformer2DModel
  except ImportError:
    QwenImageTransformer2DModel = None  # requires diffusers>=0.35.2

  _relaxed_assert(pipe.transformer, QwenImageTransformer2DModel)
  
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_1,
    check_forward_pattern=True,
    has_separate_cfg=True,
    **kwargs,
  )
```

- step 2: use `_safe_import` to import it at [caching/block_adapters/\_\_init\_\_.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/block_adapters/__init__.py).

```python
qwenimage_adapter = _safe_import(".adapters", "qwenimage_adapter")
```


## Context Parallelism

In order to support <span style="color:#c77dff;">context parallelism</span> for new model, we have to register it's ContextParallelismPlanner at [context_parallelism](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/transformers) and use `_safe_import` func to import it at [cp_planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/transformers/planners.py). For example:

- step 1: Implement the `FluxContextParallelismPlanner`
 at FLUX.1 CP planner at [flux.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/transformers/flux.py)
- step 2: use `_safe_import` func to import it at [planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/transformers/planners.py).

## Tensor Parallelism

In order to support <span style="color:#c77dff;">tensor parallelism</span> for new model, we have to register it's TensorParallelismPlanner at [tensor_parallelism](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/transformers) and use `_safe_import` func to import it at [planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/transformers/planners.py). For example:

- step 1: Implement the `FluxTensorParallelismPlanner`
 at FLUX.1 TP planner at [flux.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/transformers/flux.py)
- step 2: use `_safe_import` func to import it at [planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/transformers/planners.py).

## Text Encoder Parallelism

In order to support <span style="color:#c77dff;">text encoder tensor parallelism</span> for new model, we have to register it's TextEncoderTensorParallelismPlanner at [text_encoders](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/text_encoders) and use `_safe_import` func to import it at [planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/text_encoders/planners.py). For example:

- step 1: Implement the `T5EncoderTensorParallelismPlanner`
 at T5 TP planner at [t5_encoder.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/text_encoders/t5_encoder.py)
- step 2: use `_safe_import` func to import it at [text_encoders/planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/text_encoders/planners.py).


## Auto Encoder (VAE) Parallelism

In order to support <span style="color:#c77dff;">auto encoder (VAE) data parallelism</span> for new model, we have to register it's AutoEncoderDateParallelismPlanner at [autoencoders](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/autoencoders) and use `_safe_import` func to import it at [planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/autoencoders/planners.py). For example:

- step 1: Implement the `AutoencoderKLDataParallelismPlanner`
 at AutoencoderKL DP planner at [autoencoder_kl.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/autoencoders/autoencoder_kl.py)
- step 2: use `_safe_import` func to import it at [autoencoders/planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/distributed/autoencoders/planners.py).


## Examples and Tests

Once the acceleration support for the new model is completed, we should add the new models to the [Examples](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/_utils/examples.py) and perform the necessary tests.
