## ⚡️DBCache: Dual Block Cache  

<div id="dbcache"></div>

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-v1.png)


**DBCache**: **Dual Block Caching** for Diffusion Transformers. We have enhanced `FBCache` into a more general and customizable cache algorithm, namely `DBCache`, enabling it to achieve fully `UNet-style` cache acceleration for DiT models. Different configurations of compute blocks (**F8B12**, etc.) can be customized in DBCache. Moreover, it can be entirely **training**-**free**. DBCache can strike a perfect **balance** between performance and precision!

<div align="center">
  <p align="center">
    DBCache, <b> L20x1 </b>, Steps: 28, "A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|F1B0 (0.08)|F1B0 (0.20)|F8B8 (0.15)|F12B12 (0.20)|F16B16 (0.20)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|15.59s|8.58s|15.41s|15.11s|17.74s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.08_S11.png width=105px> | <img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.2_S19.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F8B8S1_R0.15_S15.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F12B12S4_R0.2_S16.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F16B16S4_R0.2_S13.png width=105px>|
|**Baseline(L20x1)**|**F1B0 (0.08)**|**F8B8 (0.12)**|**F8B12 (0.12)**|**F8B16 (0.20)**|**F8B20 (0.20)**|
|27.85s|6.04s|5.88s|5.77s|6.01s|6.20s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_NONE_R0.08.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F1B0_R0.08.png width=105px> |<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B8_R0.12.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B12_R0.12.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B16_R0.2.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B20_R0.2.png width=105px>|

<div align="center">
  <p align="center">
    DBCache, <b> L20x4 </b>, Steps: 20, case to show the texture recovery ability of DBCache
  </p>
</div>

These case studies demonstrate that even with relatively high thresholds (such as 0.12, 0.15, 0.2, etc.) under the DBCache **F12B12** or **F8B16** configuration, the detailed texture of the kitten's fur, colored cloth, and the clarity of text can still be preserved. This suggests that users can leverage DBCache to effectively balance performance and precision in their workflows! 


**DBCache** provides configurable parameters for custom optimization, enabling a balanced trade-off between performance and precision:

- **Fn**: Specifies that DBCache uses the **first n** Transformer blocks to fit the information at time step t, enabling the calculation of a more stable L1 diff and delivering more accurate information to subsequent blocks.
- **Bn**: Further fuses approximate information in the **last n** Transformer blocks to enhance prediction accuracy. These blocks act as an auto-scaler for approximate hidden states that use residual cache.

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-fnbn-v1.png)

- **warmup_steps**: (default: 0) DBCache does not apply the caching strategy when the number of running steps is less than or equal to this value, ensuring the model sufficiently learns basic features during warmup.
- **max_cached_steps**:  (default: -1) DBCache disables the caching strategy when the previous cached steps exceed this value to prevent precision degradation.
- **residual_diff_threshold**: The value of residual diff threshold, a higher value leads to faster performance at the cost of lower precision.

For a good balance between performance and precision, DBCache is configured by default with **F8B0**, 8 warmup steps, and unlimited cached steps.

```python
import cache_dit
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Default options, F8B0, good balance between performance and precision
cache_options = cache_dit.default_options()

# Custom options, F8B8, higher precision
cache_options = {
    "cache_type": cache_dit.DBCache,
    "warmup_steps": 8,
    "max_cached_steps": -1, # -1 means no limit
    "Fn_compute_blocks": 8, # Fn, F8, etc.
    "Bn_compute_blocks": 8, # Bn, B8, etc.
    "residual_diff_threshold": 0.12,
}

cache_dit.enable_cache(pipe, **cache_options)
```
Moreover, users configuring higher **Bn** values (e.g., **F8B16**) while aiming to maintain good performance can specify **Bn_compute_blocks_ids** to work with Bn. DBCache will only compute the specified blocks, with the remaining estimated using the previous step's residual cache.

```python
# Custom options, F8B16, higher precision with good performance.
cache_options = {
    # 0, 2, 4, ..., 14, 15, etc. [0,16)
    "Bn_compute_blocks_ids": cache_dit.block_range(0, 16, 2),
    # If the L1 difference is below this threshold, skip Bn blocks 
    # not in `Bn_compute_blocks_ids`(1, 3,..., etc), Otherwise, 
    # compute these blocks.
    "non_compute_blocks_diff_threshold": 0.08,
}
```
