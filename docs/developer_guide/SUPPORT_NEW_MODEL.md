# Support New Model  

Please make sure you have install and initialize pre-commit before adding any new commit. Refer [PRE_COMMIT](PRE_COMMIT.md) for more details.

## Cache Acceleration

In order to support cache acceleration for new model, we have to register it's BlockAdapter at [caching/block_adapters/adapter.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/block_adapters/adapters.py) and use `_safe_import` func to import it at [caching/block_adapters/\_\_init\_\_.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/block_adapters/__init__.py). For example:

- step 1: Implement the `qwenimage_adapter` at [caching/block_adapters/adapter.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/block_adapters/adapters.py)

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

In order to support context parallelism for new model, we have to register it's ContextParallelismPlanner at [context_parallelism](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/transformers/context_parallelism) and use `_safe_import` func to import it at [cp_planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/transformers/context_parallelism/cp_planners.py). For example:

- step 1: Implement the `FluxContextParallelismPlanner`
 at FLUX.1 CP planner at [cp_plan_flux.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/transformers/context_parallelism/cp_plan_flux.py)
- step 2: use `_safe_import` func to import it at [cp_planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/transformers/context_parallelism/cp_planners.py).

## Tensor Parallelism

In order to support tensor parallelism for new model, we have to register it's TensorParallelismPlanner at [tensor_parallelism](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/transformers/tensor_parallelism) and use `_safe_import` func to import it at [tp_planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/transformers/tensor_parallelism/tp_planners.py). For example:

- step 1: Implement the `FluxTensorParallelismPlanner`
 at FLUX.1 TP planner at [tp_plan_flux.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/transformers/tensor_parallelism/tp_plan_flux.py)
- step 2: use `_safe_import` func to import it at [tp_planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/transformers/tensor_parallelism/tp_planners.py).

## Text Encoder Parallelism

In order to support text encoder tensor parallelism for new model, we have to register it's TextEncoderTensorParallelismPlanner at [text_encoders/tensor_parallelism](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/text_encoders/tensor_parallelism) and use `_safe_import` func to import it at [text_encoders/tensor_parallelism/tp_planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/text_encoders/tensor_parallelism/tp_planners.py). For example:

- step 1: Implement the `T5EncoderTensorParallelismPlanner`
 at T5 TP planner at [tp_plan_t5_encoder.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/transformers/tensor_parallelism/tp_plan_t5_encoder.py)
- step 2: use `_safe_import` func to import it at [text_encoders/tensor_parallelism/tp_planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/text_encoders/tensor_parallelism/tp_planners.py).


## Auto Encoder (VAE) Parallelism

In order to support auto encoder (VAE) data parallelism for new model, we have to register it's AutoEncoderDateParallelismPlanner at [autoencoders/data_parallelism](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/autoencoders/data_parallelism) and use `_safe_import` func to import it at [autoencoders/data_parallelism/dp_planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/autoencoders/dp_parallelism/dp_planners.py). For example:

- step 1: Implement the `AutoencoderKLDataParallelismPlanner`
 at AutoencoderKL DP planner at [dp_plan_autoencoder_kl.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/autoencoders/data_parallelism/dp_plan_autoencoder_kl.py)
- step 2: use `_safe_import` func to import it at [autoencoders/data_parallelism/dp_planners.py](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/autoencoders/data_parallelism/dp_planners.py).


## Examples and Tests

Once the acceleration support for the new model is completed, we should add the new models to the [Examples](https://github.com/vipshop/cache-dit/blob/main/examples) and perform the necessary tests.
