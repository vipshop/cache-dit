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

- **max_warmup_steps**: (default: 0) DBCache does not apply the caching strategy when the number of running steps is less than or equal to this value, ensuring the model sufficiently learns basic features during warmup.
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

# Default options, F8B0, 8 warmup steps, and unlimited cached 
# steps for good balance between performance and precision
cache_dit.enable_cache(pipe)

# Custom options, F8B8, higher precision
cache_dit.enable_cache(
    pipe,
    max_warmup_steps=8,  # steps do not cache
    max_cached_steps=-1, # -1 means no limit
    Fn_compute_blocks=8, # Fn, F8, etc.
    Bn_compute_blocks=8, # Bn, B8, etc.
    residual_diff_threshold=0.12,
)
```
<div align="center">
  <p align="center">
    DBCache, <b> L20x1 </b>, Steps: 28, "A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|F1B0 (0.08)|F1B0 (0.20)|F8B8 (0.15)|F12B12 (0.20)|F16B16 (0.20)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|15.59s|8.58s|15.41s|15.11s|17.74s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.08_S11.png width=105px> | <img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.2_S19.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F8B8S1_R0.15_S15.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F12B12S4_R0.2_S16.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F16B16S4_R0.2_S13.png width=105px>|

## 🔥Hybrid TaylorSeer

<div id="taylorseer"></div>

We have supported the [TaylorSeers: From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/pdf/2503.06923) algorithm to further improve the precision of DBCache in cases where the cached steps are large, namely, **Hybrid TaylorSeer + DBCache**. At timesteps with significant intervals, the feature similarity in diffusion models decreases substantially, significantly harming the generation quality. 

$$
\mathcal{F}\_{\text {pred }, m}\left(x_{t-k}^l\right)=\mathcal{F}\left(x_t^l\right)+\sum_{i=1}^m \frac{\Delta^i \mathcal{F}\left(x_t^l\right)}{i!\cdot N^i}(-k)^i
$$

**TaylorSeer** employs a differential method to approximate the higher-order derivatives of features and predict features in future timesteps with Taylor series expansion. The TaylorSeer implemented in cache-dit supports both hidden states and residual cache types. That is $\mathcal{F}\_{\text {pred }, m}\left(x_{t-k}^l\right)$ can be a residual cache or a hidden-state cache.

```python
cache_dit.enable_cache(
    pipe,
    enable_taylorseer=True,
    enable_encoder_taylorseer=True,
    # Taylorseer cache type cache be hidden_states or residual.
    taylorseer_cache_type="residual",
    # Higher values of order will lead to longer computation time
    taylorseer_order=1, # default is 1.
    max_warmup_steps=3, # prefer: >= order + 1
    residual_diff_threshold=0.12
)
``` 

> [!Important]
> Please note that if you have used TaylorSeer as the calibrator for approximate hidden states, the **Bn** param of DBCache can be set to **0**. In essence, DBCache's Bn is also act as a calibrator, so you can choose either Bn > 0 or TaylorSeer. We recommend using the configuration scheme of **TaylorSeer** + **DBCache FnB0**.

<div align="center">
  <p align="center">
    <b>DBCache F1B0 + TaylorSeer</b>, L20x1, Steps: 28, <br>"A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|F1B0 (0.12)|+TaylorSeer|F1B0 (0.15)|+TaylorSeer|+compile| 
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|12.85s|12.86s|10.27s|10.28s|8.48s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T0ET0_R0.12_S14_T12.85s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T1ET1_R0.12_S14_T12.86s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T0ET0_R0.15_S17_T10.27s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T1ET1_R0.15_S17_T10.28s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C1_DBCACHE_F1B0S1W0T1ET1_R0.15_S17_T8.48s.png width=105px>|

## ⚡️Hybrid Cache CFG

<div id="cfg"></div>

cache-dit supports caching for **CFG (classifier-free guidance)**. For models that fuse CFG and non-CFG into a single forward step, or models that do not include CFG (classifier-free guidance) in the forward step, please set `enable_separate_cfg` param to **False (default)**. Otherwise, set it to True. For examples:

```python
cache_dit.enable_cache(
    pipe, 
    ...,
    # CFG: classifier free guidance or not
    # For model that fused CFG and non-CFG into single forward step,
    # should set enable_separate_cfg as False. For example, set it as True 
    # for Wan 2.1/Qwen-Image and set it as False for FLUX.1, HunyuanVideo, 
    # CogVideoX, Mochi, etc.
    enable_separate_cfg=True, # Wan 2.1, Qwen-Image
    # Compute cfg forward first or not, default False, namely, 
    # 0, 2, 4, ..., -> non-CFG step; 1, 3, 5, ... -> CFG step.
    cfg_compute_first=False,
    # Compute separate diff values for CFG and non-CFG step, 
    # default True. If False, we will use the computed diff from 
    # current non-CFG transformer step for current CFG step.
    cfg_diff_compute_separate=True,
)
```
