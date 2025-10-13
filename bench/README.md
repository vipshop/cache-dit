# 🤖Benchmarks

- [🔥Hightlight](#hightlight)
- [📚DrawBench](#benchmark-flux)
- [📚Distillation DrawBench](#benchmark-lightning)
- [📚How to Reproduce?](#how-to-reproduce)
  - [⚙️Installation](#installation)
  - [📖Download](#download)
  - [📖Evaluation](#evaluation)
- [📚Detailed Bnechmark](#detailed-benchmark)
  - [📖Clip Score](#clipscore)
  - [📖Image Reward](#imagereward)
  - [📖PSNR](#psnr)
  - [📖SSIM](#ssim)
  - [📖LPIPS](#lpips)

## 🔥Hightlight  

Comparisons between different FnBn compute block configurations show that **more compute blocks result in higher precision**. For example, the F8B0_W8MC0 configuration achieves the best Clip Score (33.007) and ImageReward (1.0333). The meaning of parameter configuration is as follows (such as F8B0_W8M0MC0_T0O1_R0.08): (**Device**: NVIDIA L20.) 
  - **F**: Fn_compute_blocks, `int`, Specifies that `DBCache` uses the**first n**Transformer blocks to fit the information at time step t, enabling the calculation of a more stable L1 difference and delivering more accurate information to subsequent blocks.
  - **B**: Bn_compute_blocks, `int`, Further fuses approximate information in the**last n**Transformer blocks to enhance prediction accuracy. These blocks act as an auto-scaler for approximate hidden states that use residual cache.
  - **W**: max_warmup_steps, `int`, DBCache does not apply the caching strategy when the number of running steps is less than or equal to this value, ensuring the model sufficiently learns basic features during warmup.
  - **I**: warmup_interval, `int`, defaults to 1, Skip interval in warmup steps, e.g., when warmup_interval is 2, only 0, 2, 4, ... steps in warmup steps will be computed, others will use dynamic cache.
  - **M**: max_cached_steps, `int`, DBCache disables the caching strategy when the previous cached steps exceed this value to prevent precision degradation.
  - **MC**: max_continuous_cached_steps, `int`, DBCache disables the caching strategy when the previous continuous cached steps exceed this value to prevent precision degradation. (namely, hybird dynamic cache and static cache)
  - **T**: enable talyorseer or not (namely, hybrid taylorseer w/ dynamic cache - DBCache). DBCache acts as the Indicator to decide when to cache, while the Calibrator decides how to cache. 
  - **O**: The taylorseer order, `int`, e.g., O1 means order 1.
  - **R**: The residual diff threshold of DBCache, range [0, 1.0)
  - **Latency(s)**: Recorded compute time (eager mode) that **w/o** other optimizations
  - **TFLOPs**: Recorded compute FLOPs using [calflops](https://github.com/chengzegang/calculate-flops.pytorch.git)'s [calculate_flops](./utils.py) API.

> [!Note]   
> Among all the accuracy indicators, the overall accuracy has slightly improved after using TaylorSeer.

![](https://github.com/vipshop/cache-dit/raw/main/assets/image-reward-bench.png)
![](https://github.com/vipshop/cache-dit/raw/main/assets/clip-score-bench.png)

## 📚Text2Image DrawBench: FLUX.1-dev

<div id="benchmark-flux"></div>

| Config | Clip Score(↑) | ImageReward(↑) | PSNR(↑) | TFLOPs(↓) | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 32.9217 | 1.0412 | INF | 3726.87 | 1.00x |
| F8B0_W4MC0_T0O1_R0.08 | 32.9871 | 1.0370 | 33.8317 | 2064.81 | 1.80x |
| F8B0_W4MC2_T0O1_R0.12 | 32.9535 | 1.0185 | 32.7346 | 1935.73 | 1.93x |
| F8B0_W4MC3_T0O1_R0.12 | 32.9234 | 1.0085 | 32.5385 | 1816.58 | 2.05x |
| F4B0_W4MC3_T0O1_R0.12 | 32.8981 | 1.0130 | 31.8031 | 1507.83 | 2.47x |
| F4B0_W4MC4_T0O1_R0.12 | 32.8384 | 1.0065 | 31.5292 | 1400.08 | 2.66x |

The comparison between **cache-dit: DBCache** and algorithms such as Δ-DiT, Chipmunk, FORA, DuCa, TaylorSeer and FoCa is as follows. Now, in the comparison with a speedup ratio less than **3x**, cache-dit achieved the best accuracy. Please check [📚How to Reproduce?](#how-to-reproduce) for more details.

| Method | TFLOPs(↓) | SpeedUp(↑) | ImageReward(↑) | Clip Score(↑) |
| --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 3726.87 | 1.00× | 0.9898 | 32.404 |
| [**FLUX.1**-dev]: 60% steps | 2231.70 | 1.67× | 0.9663 | 32.312 |
| Δ-DiT(N=2) | 2480.01 | 1.50× | 0.9444 | 32.273 |
| Δ-DiT(N=3) | 1686.76 | 2.21× | 0.8721 | 32.102 |
| [**FLUX.1**-dev]: 34% steps | 1264.63 | 3.13× | 0.9453 | 32.114 |
| Chipmunk | 1505.87 | 2.47× | 0.9936 | 32.776 |
| FORA(N=3) | 1320.07 | 2.82× | 0.9776 | 32.266 |
| **[DBCache(Slow)](https://github.com/vipshop/cache-dit)** | 1400.08 | **2.66×** | **1.0065** | 32.838 |
| DuCa(N=5) | 978.76 | 3.80× | 0.9955 | 32.241 |
| TaylorSeer(N=4,O=2) | 1042.27 | 3.57× | 0.9857 | 32.413 |
| **[DBCache(Slow)+TaylorSeer(O=1)](https://github.com/vipshop/cache-dit)** | 1153.05 | **3.23×** | **1.0221** | 32.819 |
| **[DBCache(Medium)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | 0.9997 | 32.849 |
| **[DBCache(Medium)+TaylorSeer(O=1)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | **1.0107** | 32.865 |
| **[FoCa(N=5): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 893.54 | **4.16×** | 1.0029 | **32.948** |
| [**FLUX.1**-dev]: 22% steps | 818.29 | 4.55× | 0.8183 | 31.772 |
| FORA(N=7) | 670.14 | 5.55× | 0.7418 | 31.519 |
| ToCa(N=12) | 644.70 | 5.77× | 0.7155 | 31.808 |
| DuCa(N=10) | 606.91 | 6.13× | 0.8382 | 31.759 |
| TeaCache(l=1.2) | 669.27 | 5.56× | 0.7394 | 31.704 |
| TaylorSeer(N=7,O=2) | 670.44 | 5.54× | 0.9128 | 32.128 |
| **[DBCache(Fast)](https://github.com/vipshop/cache-dit)** | 651.90 | **5.72x** | 0.9271 | 32.552 |
| **[FoCa(N=8): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 596.07 | 6.24× | 0.9502 | 32.706 |
| **[DBCache(Fast)+TaylorSeer(O=1)](https://github.com/vipshop/cache-dit)** | 651.90 | **5.72x** | **0.9526** | 32.568 |
| **[DBCache(Ultra)+TaylorSeer(O=1)](https://github.com/vipshop/cache-dit)** | 505.47 | **7.37x** | 0.8645 | **32.719** |

NOTE: Except for DBCache, other performance data are referenced from the paper [FoCa, arxiv.2508.16211](https://arxiv.org/pdf/2508.16211). The configurations of DBCache listed as belows: 
- **DBCache(Slow)**: F=4,B=0,W=4,I=1,MC=4; DBCache(Slow)+TaylorSeer(O=1): F=1,B=0,W=4,I=1,MC=4,O=1; 
- **DBCache(Medium)**: F=1,B=0,W=4,I=1,MC=6; DBCache(Medium)+TaylorSeer(O=1): F=1,B=0,W=4,I=1,MC=6,O=1;
- **DBCache(Fast)**: F=1,B=0,W=8,I=2,MC=8; DBCache(Fast)+TaylorSeer(O=1): F=1,B=0,W=8,I=2,MC=8,O=1; 
- **DBCache(Ultra)**: F=1,B=0,W=8,I=4,MC=10,O=1.

## 📚Text2Image Distillation DrawBench: Qwen-Image-Lightning

<div id="benchmark-lightning"></div>

Surprisingly, cache-dit: DBCache still works in the extremely few-step distill model. For example,  **Qwen-Image-Lightning w/ 4 steps**, with the F16B16 configuration, the PSNR is 34.8163, the Clip Score is 35.6109, and the ImageReward is 1.2614. It maintained a relatively high precision.

| Config                     |  PSNR(↑)      | Clip Score(↑) | ImageReward(↑) | TFLOPs(↓)   | SpeedUp(↑) |
|----------------------------|-----------|------------|--------------|----------|------------|
| [**Lightning**]: 4 steps   | INF       | 35.5797    | 1.2630       | 274.33   | 1.00x       |
| F24B24_W2MC1_T0O1_R0.8          | 36.3242   | 35.6224    | 1.2630       | 264.74   | 1.04x       |
| F16B16_W2MC1_T0O1_R0.8          | 34.8163   | 35.6109    | 1.2614       | 244.25   | 1.12x       |
| F12B12_W2MC1_T0O1_R0.8          | 33.8953   | 35.6535    | 1.2549       | 234.63   | 1.17x       |
| F8B8_W2MC1_T0O1_R0.8            | 33.1374   | 35.7284    | 1.2517       | 224.29   | 1.22x       |
| F1B0_W2MC1_T0O1_R0.8            | 31.8317   | 35.6651    | 1.2397       | 206.90   | 1.33x       |


## 📚How to Reproduce?

### ⚙️Installation

<div id="installation"></div>

```bash
# install requirements
pip3 install git+https://github.com/openai/CLIP.git
pip3 install git+https://github.com/chengzegang/calculate-flops.pytorch.git
pip3 install image-reward
pip3 install git+https://github.com/vipshop/cache-dit.git
```

### 📖Download

<div id="donwload"></div>

```bash
git clone https://github.com/vipshop/cache-dit.git
cd cache-dit/bench && mkdir tmp && mkdir log && mkdir hf_models && cd hf_models

# FLUX.1-dev
modelscope download black-forest-labs/FLUX.1-dev --local_dir ./FLUX.1-dev
hf download black-forest-labs/FLUX.1-dev --local-dir ./FLUX.1-dev
export FLUX_DIR=$PWD/FLUX.1-dev

# Qwen-Image-Lightning
modelscope download Qwen/Qwen-Image --local_dir ./Qwen-Image
modelscope download lightx2v/Qwen-Image-Lightning --local_dir ./Qwen-Image-Lightning
hf download Qwen/Qwen-Image --local-dir ./Qwen-Image
hf download lightx2v/Qwen-Image-Lightning --local-dir ./Qwen-Image-Lightning
export QWEN_IMAGE_DIR=$PWD/Qwen-Image
export QWEN_IMAGE_LIGHT_DIR=$PWD/Qwen-Image-Lightning

# Clip Score & Image Reward
modelscope download laion/CLIP-ViT-g-14-laion2B-s12B-b42K --local_dir ./CLIP-ViT-g-14-laion2B-s12B-b42K
modelscope download ZhipuAI/ImageReward --local_dir ./ImageReward
hf download laion/CLIP-ViT-g-14-laion2B-s12B-b42K --local-dir ./CLIP-ViT-g-14-laion2B-s12B-b42K
hf download ZhipuAI/ImageReward --local-dir ./ImageReward
export CLIP_MODEL_DIR=$PWD/CLIP-ViT-g-14-laion2B-s12B-b42K
export IMAGEREWARD_MODEL_DIR=$PWD/ImageReward

cd ..
```


### 📖Evaluation

<div id="evaluation"></div>

```bash
# NOTE: The reported benchmark was run on NVIDIA L20 device.

# FLUX.1-dev DrawBench w/ low speedup ratio
export CUDA_VISIBLE_DEVICES=0
nohup bash bench.sh default > log/cache_dit_bench_default.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1
nohup bash bench.sh taylorseer > log/cache_dit_bench_taylorseer.log 2>&1 &
bash ./metrics.sh

# FLUX.1-dev DrawBench w/ high speedup ratio
export CUDA_VISIBLE_DEVICES=0
nohup bash bench_fast.sh default > log/cache_dit_bench_fast.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1
nohup bash bench_fast.sh taylorseer > log/cache_dit_bench_taylorseer_fast.log 2>&1 &
bash ./metrics_fast.sh

# Qwen-Image-Lightning DrawBench
export CUDA_VISIBLE_DEVICES=0,1
nohup bash bench_distill.sh 8_steps > log/cache_dit_bench_distill_8_steps.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2,3
nohup bash bench_distill.sh 4_steps > log/cache_dit_bench_distill_4_steps.log 2>&1 &
bash ./metrics_distill.sh
```

## 📚Detailed Benchmark 

### 📖ClipScore: DBCache(Slow) FnBn

<div id="clipscore"></div>

| Config | CLIP_SCORE | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Baseline: [FLUX.1 50 steps] | 32.9217 | 42.74 | 1.00 | 3726.87 | 1.00 |
| F1B0_W4M0MC0_R0.08_T0O0 | 33.0745 | 23.21 | 1.84 | 1729.60 | 2.15 |
| F8B0_W4M0MC3_R0.32_T0O0 | 33.0055 | 22.07 | 1.94 | 1548.80 | 2.41 |
| F8B0_W4M0MC4_R0.24_T0O0 | 32.9921 | 20.79 | 2.06 | 1421.96 | 2.62 |
| F8B0_W4M0MC6_R0.32_T0O0 | 32.9804 | 19.38 | 2.21 | 1262.77 | 2.95 |
| F4B0_W4M0MC4_R0.24_T0O0 | 32.9720 | 18.77 | 2.28 | 1231.01 | 3.03 |
| F4B0_W4M0MC4_R0.32_T0O0 | 32.9680 | 18.10 | 2.36 | 1163.10 | 3.20 |
| F4B0_W4M0MC6_R0.32_T0O0 | 32.8926 | 16.62 | 2.57 | 1024.52 | 3.64 |
| F1B0_W4M0MC4_R0.32_T0O0 | 32.8640 | 16.30 | 2.62 | 1017.97 | 3.66 |
| F1B0_W4M0MC6_R0.24_T0O0 | 32.8496 | 15.54 | 2.75 | 944.75 | 3.94 |
| F1B0_W4M0MC6_R0.32_T0O0 | 32.8098 | 15.65 | 2.73 | 942.92 | 3.95 |

### 📖ClipScore: DBCache(Slow) FnBn + TaylorSeer O(1)

| Config | CLIP_SCORE | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Baseline: [FLUX.1 50 steps] | 32.9217 | 42.92 | 1.00 | 3726.87 | 1.00 |
| F1B0_W4M0MC0_R0.08_T1O1 | 33.0398 | 23.29 | 1.84 | 1730.70 | 2.15 |
| F4B0_W4M0MC3_R0.12_T1O1 | 32.9795 | 21.36 | 2.01 | 1499.51 | 2.49 |
| F1B0_W4M0MC6_R0.32_T1O1 | 32.9589 | 15.70 | 2.73 | 944.02 | 3.95 |

### 📖ClipScore: DBCache(Fast) FnBn

| Config | CLIP_SCORE | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| C0_Q0_NONE | 32.9616 | 44.61 | 1.00 | 3726.87 | 1.00 |
| F1B0_W16I4M0MC10_R0.8_T0O0 | 32.6306 | 10.48 | 4.26 | 578.69 | 6.44 |
| F1B0_W16I4M0MC8_R0.8_T0O0 | 32.5522 | 11.26 | 3.96 | 651.90 | 5.72 |
| F1B0_W8I4M0MC8_R0.8_T0O0 | 32.5420 | 10.46 | 4.26 | 578.69 | 6.44 |
| F1B0_W8I4M0MC10_R0.8_T0O0 | 32.4479 | 9.71 | 4.59 | 505.47 | 7.37 |
| F1B0_W8I2M0MC8_R0.8_T0O0 | 32.4395 | 11.40 | 3.91 | 651.90 | 5.72 |
| F1B0_W8I2M0MC10_R0.8_T0O0 | 32.4207 | 10.62 | 4.20 | 578.69 | 6.44 |

### 📖ClipScore: DBCache(Fast) FnBn + TaylorSeer O(1)

| Config | CLIP_SCORE | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| C0_Q0_NONE | 32.9616 | 44.94 | 1.00 | 3726.87 | 1.00 |
| F1B0_W16I4M0MC8_R0.8_T1O1 | 33.0988 | 11.42 | 3.94 | 651.90 | 5.72 |
| F1B0_W16I4M0MC10_R0.8_T1O1 | 32.9764 | 10.65 | 4.22 | 578.69 | 6.44 |
| F1B0_W8I4M0MC10_R0.8_T1O1 | 32.7195 | 9.83 | 4.57 | 505.47 | 7.37 |
| F1B0_W8I4M0MC8_R0.8_T1O1 | 32.6448 | 10.56 | 4.26 | 578.69 | 6.44 |
| F1B0_W8I2M0MC8_R0.8_T1O1 | 32.5682 | 11.44 | 3.93 | 651.90 | 5.72 |
| F1B0_W8I2M0MC10_R0.8_T1O1 | 32.3999 | 10.66 | 4.22 | 578.69 | 6.44 |

### 📖ImageReward: DBCache(Slow) FnBn

<div id="imagereward"></div>

| Config | IMAGE_REWARD | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Baseline: [FLUX.1 50 steps] | 1.0412 | 42.74 | 1.00 | 3726.87 | 1.00 |
| F1B0_W4M0MC2_R0.16_T0O0 | 1.0461 | 21.29 | 2.01 | 1530.46 | 2.44 |
| F1B0_W4M0MC2_R0.24_T0O0 | 1.0418 | 20.51 | 2.08 | 1457.25 | 2.56 |
| F1B0_W4M0MC2_R0.2_T0O0 | 1.0418 | 20.53 | 2.08 | 1457.25 | 2.56 |
| F1B0_W4M0MC2_R0.32_T0O0 | 1.0418 | 20.63 | 2.07 | 1457.25 | 2.56 |
| F1B0_W4M0MC3_R0.16_T0O0 | 1.0360 | 19.09 | 2.24 | 1310.82 | 2.84 |
| F1B0_W4M0MC3_R0.2_T0O0 | 1.0323 | 18.57 | 2.30 | 1237.61 | 3.01 ||
| F1B0_W4M0MC4_R0.16_T0O0 | 1.0232 | 17.75 | 2.41 | 1164.76 | 3.20 |
| F1B0_W4M0MC4_R0.2_T0O0 | 1.0229 | 17.71 | 2.41 | 1153.05 | 3.23 ||
| F1B0_W4M0MC4_R0.24_T0O0 | 1.0096 | 16.98 | 2.52 | 1091.18 | 3.42 |
| F1B0_W4M0MC4_R0.32_T0O0 | 1.0083 | 16.30 | 2.62 | 1017.97 | 3.66 |
| F1B0_W4M0MC6_R0.24_T0O0 | 0.9997 | 15.54 | 2.75 | 944.75 | 3.94 |
| F1B0_W4M0MC6_R0.32_T0O0 | 0.9921 | 15.65 | 2.73 | 942.92 | 3.95 |

### 📖ImageReward: DBCache(Slow) FnBn + TaylorSeer O(1)

| Config | IMAGE_REWARD | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Baseline: [FLUX.1 50 steps] | 1.0412 | 42.92 | 1.00 | 3726.87 | 1.00 |
| F1B0_W4M0MC0_R0.08_T1O1 | 1.0591 | 23.29 | 1.84 | 1730.70 | 2.15 |
| F1B0_W4M0MC2_R0.16_T1O1 | 1.0389 | 21.35 | 2.01 | 1530.46 | 2.44 |
| F1B0_W4M0MC2_R0.24_T1O1 | 1.0379 | 20.61 | 2.08 | 1457.25 | 2.56 |
| F1B0_W4M0MC2_R0.2_T1O1 | 1.0379 | 20.59 | 2.08 | 1457.25 | 2.56 |
| F1B0_W4M0MC2_R0.32_T1O1 | 1.0379 | 20.70 | 2.07 | 1457.25 | 2.56 |
| F1B0_W4M0MC3_R0.12_T1O1 | 1.0292 | 20.70 | 2.07 | 1457.62 | 2.56 |
| F1B0_W4M0MC3_R0.2_T1O1 | 1.0257 | 18.63 | 2.30 | 1237.61 | 3.01 |
| F1B0_W4M0MC4_R0.2_T1O1 | 1.0221 | 17.71 | 2.42 | 1153.05 | 3.23 |
| F1B0_W4M0MC4_R0.32_T1O1 | 1.0149 | 16.33 | 2.63 | 1017.97 | 3.66 |
| F1B0_W4M0MC6_R0.24_T1O1 | 1.0107 | 15.61 | 2.75 | 944.75 | 3.94 |
| F1B0_W4M0MC6_R0.32_T1O1 | 1.0025 | 15.70 | 2.73 | 944.02 | 3.95 |

### 📖ImageReward: DBCache(Fast) FnBn

| Config | IMAGE_REWARD | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| C0_Q0_NONE | 1.0445 | 44.61 | 1.00 | 3726.87 | 1.00 |
| F1B0_W8I2M0MC8_R0.8_T0O0 | 0.9271 | 11.40 | 3.91 | 651.90 | 5.72 |
| F1B0_W8I2M0MC10_R0.8_T0O0 | 0.9234 | 10.62 | 4.20 | 578.69 | 6.44 |
| F1B0_W16I4M0MC8_R0.8_T0O0 | 0.9115 | 11.26 | 3.96 | 651.90 | 5.72 |
| F1B0_W16I4M0MC10_R0.8_T0O0 | 0.8644 | 10.48 | 4.26 | 578.69 | 6.44 |
| F1B0_W8I4M0MC8_R0.8_T0O0 | 0.8301 | 10.46 | 4.26 | 578.69 | 6.44 |
| F1B0_W8I4M0MC10_R0.8_T0O0 | 0.8092 | 9.71 | 4.59 | 505.47 | 7.37 |

### 📖ImageReward: DBCache(Fast) FnBn + TaylorSeer O(1)

| Config | IMAGE_REWARD | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| C0_Q0_NONE | 1.0445 | 44.94 | 1.00 | 3726.87 | 1.00 |
| F1B0_W8I2M0MC8_R0.8_T1O1 | 0.9526 | 11.44 | 3.93 | 651.90 | 5.72 |
| F1B0_W8I2M0MC10_R0.8_T1O1 | 0.9360 | 10.66 | 4.22 | 578.69 | 6.44 |
| F1B0_W16I4M0MC8_R0.8_T1O1 | 0.9285 | 11.42 | 3.94 | 651.90 | 5.72 |
| F1B0_W8I4M0MC8_R0.8_T1O1 | 0.9088 | 10.56 | 4.26 | 578.69 | 6.44 |
| F1B0_W16I4M0MC10_R0.8_T1O1 | 0.8941 | 10.65 | 4.22 | 578.69 | 6.44 |
| F1B0_W8I4M0MC10_R0.8_T1O1 | 0.8645 | 9.83 | 4.57 | 505.47 | 7.37 |

### 📖PSNR: DBCache(Slow) FnBn

<div id="psnr"></div>

| Config | PSNR | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Baseline: [FLUX.1.dev, 50 steps] | - | 42.74 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_R0.08_T0O0 | 35.2008 | 27.87 | 1.53 | 2162.19 | 1.72 |
| F8B0_W8M0MC2_R0.12_T0O0 | 34.7449 | 27.22 | 1.57 | 2072.18 | 1.80 |
| F8B0_W8M0MC2_R0.16_T0O0 | 34.6659 | 26.34 | 1.62 | 2002.67 | 1.86 |
| F8B0_W8M0MC2_R0.2_T0O0 | 34.6579 | 26.33 | 1.62 | 1994.67 | 1.87 |
| F8B0_W8M0MC2_R0.24_T0O0 | 34.5399 | 25.62 | 1.67 | 1933.17 | 1.93 |
| F8B0_W8M0MC2_R0.32_T0O0 | 34.4860 | 25.66 | 1.67 | 1933.17 | 1.93 |
| F1B0_W4M0MC0_R0.08_T0O0 | 33.9639 | 23.21 | 1.84 | 1729.60 | 2.15 |
| F1B0_W4M0MC2_R0.12_T0O0 | 33.1898 | 21.92 | 1.95 | 1604.78 | 2.32 |
| F1B0_W4M0MC3_R0.12_T0O0 | 33.0037 | 20.61 | 2.07 | 1457.62 | 2.56 |
| F1B0_W4M0MC4_R0.12_T0O0 | 32.9462 | 20.01 | 2.14 | 1401.61 | 2.66 |
| F1B0_W4M0MC3_R0.16_T0O0 | 31.9664 | 19.09 | 2.24 | 1310.82 | 2.84 |
| F1B0_W4M0MC3_R0.2_T0O0 | 31.9315 | 18.57 | 2.30 | 1237.61 | 3.01 |
| F1B0_W4M0MC4_R0.16_T0O0 | 31.7174 | 17.75 | 2.41 | 1164.76 | 3.20 |
| F1B0_W4M0MC4_R0.2_T0O0 | 31.7109 | 17.71 | 2.41 | 1153.05 | 3.23 |
| F1B0_W4M0MC4_R0.24_T0O0 | 31.3081 | 16.98 | 2.52 | 1091.18 | 3.42 |
| F1B0_W4M0MC6_R0.24_T0O0 | 31.0524 | 15.54 | 2.75 | 944.75 | 3.94 |
| F1B0_W4M0MC6_R0.32_T0O0 | 30.6637 | 15.65 | 2.73 | 942.92 | 3.95 |

### 📖PSNR: DBCache(Slow) FnBn + TaylorSeer O(1)

| Config | PSNR | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Baseline: [FLUX.1 50 steps] | - | 42.74 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_R0.08_T1O1 | 36.7296 | 28.06 | 1.53 | 2172.76 | 1.72 |
| F8B0_W8M0MC2_R0.12_T1O1 | 36.1337 | 27.31 | 1.57 | 2074.10 | 1.80 |
| F8B0_W8M0MC2_R0.16_T1O1 | 36.1062 | 26.46 | 1.62 | 2005.56 | 1.86 |
| F8B0_W8M0MC2_R0.2_T1O1 | 36.0838 | 26.22 | 1.64 | 1985.38 | 1.88 |
| F8B0_W8M0MC2_R0.24_T1O1 | 35.9808 | 25.64 | 1.67 | 1933.17 | 1.93 |
| F8B0_W8M0MC2_R0.32_T1O1 | 35.9008 | 25.71 | 1.67 | 1933.17 | 1.93 |
| F8B0_W8M0MC4_R0.12_T1O1 | 35.4352 | 25.59 | 1.68 | 1912.99 | 1.95 |
| F8B0_W8M0MC3_R0.16_T1O1 | 34.7351 | 24.67 | 1.74 | 1815.62 | 2.05 |
| F8B0_W8M0MC3_R0.2_T1O1 | 34.6739 | 24.46 | 1.75 | 1794.16 | 2.08 |
| F8B0_W8M0MC3_R0.24_T1O1 | 34.6091 | 23.84 | 1.80 | 1740.99 | 2.14 |
| F8B0_W8M0MC3_R0.32_T1O1 | 34.5803 | 23.98 | 1.79 | 1740.99 | 2.14 |
| F1B0_W4M0MC0_R0.08_T1O1 | 34.5305 | 23.29 | 1.84 | 1730.70 | 2.15 |
| F1B0_W4M0MC2_R0.12_T1O1 | 34.1546 | 22.07 | 1.94 | 1604.04 | 2.32 |
| F1B0_W4M0MC4_R0.12_T1O1 | 33.8600 | 20.08 | 2.14 | 1404.17 | 2.65 |
| F1B0_W4M0MC3_R0.16_T1O1 | 31.8628 | 19.20 | 2.24 | 1310.82 | 2.84 |
| F1B0_W4M0MC3_R0.2_T1O1 | 31.8328 | 18.63 | 2.30 | 1237.61 | 3.01 |
| F1B0_W4M0MC4_R0.16_T1O1 | 31.6688 | 17.82 | 2.41 | 1164.76 | 3.20 |
| F1B0_W4M0MC4_R0.2_T1O1 | 31.6624 | 17.71 | 2.42 | 1153.05 | 3.23 |
| F1B0_W4M0MC4_R0.24_T1O1 | 30.8172 | 17.04 | 2.52 | 1091.18 | 3.42 |
| F1B0_W4M0MC6_R0.24_T1O1 | 30.5131 | 15.61 | 2.75 | 944.75 | 3.94 |
| F1B0_W4M0MC6_R0.32_T1O1 | 29.8860 | 15.70 | 2.73 | 944.02 | 3.95 |

### 📖SSIM: DBCache(Slow) FnBn

<div id="ssim"></div>

| Config | SSIM | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Baseline: [FLUX.1 50 steps] | - | 42.74 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_R0.08_T0O0 | 0.9131 | 27.87 | 1.53 | 2162.19 | 1.72 |
| F8B0_W8M0MC2_R0.12_T0O0 | 0.9017 | 27.22 | 1.57 | 2072.18 | 1.80 |
| F8B0_W8M0MC2_R0.16_T0O0 | 0.8999 | 26.34 | 1.62 | 2002.67 | 1.86 |
| F8B0_W8M0MC2_R0.2_T0O0 | 0.8996 | 26.33 | 1.62 | 1994.67 | 1.87 |
| F8B0_W8M0MC2_R0.24_T0O0 | 0.8953 | 25.62 | 1.67 | 1933.17 | 1.93 |
| F8B0_W8M0MC2_R0.32_T0O0 | 0.8936 | 25.66 | 1.67 | 1933.17 | 1.93 |
| F8B0_W8M0MC4_R0.12_T0O0 | 0.8858 | 25.41 | 1.68 | 1897.61 | 1.96 |
| F8B0_W8M0MC3_R0.16_T0O0 | 0.8774 | 24.55 | 1.74 | 1811.77 | 2.06 |
| F8B0_W8M0MC3_R0.2_T0O0 | 0.8761 | 24.56 | 1.74 | 1803.45 | 2.07 |
| F1B0_W4M0MC0_R0.08_T0O0 | 0.8727 | 23.21 | 1.84 | 1729.60 | 2.15 |
| F8B0_W8M0MC4_R0.2_T0O0 | 0.8571 | 23.40 | 1.83 | 1678.21 | 2.22 |
| F1B0_W4M0MC2_R0.12_T0O0 | 0.8536 | 21.92 | 1.95 | 1604.78 | 2.32 |
| F1B0_W4M0MC3_R0.12_T0O0 | 0.8521 | 20.61 | 2.07 | 1457.62 | 2.56 |
| F1B0_W4M0MC4_R0.12_T0O0 | 0.8461 | 20.01 | 2.14 | 1401.61 | 2.66 |
| F1B0_W4M0MC3_R0.16_T0O0 | 0.8092 | 19.09 | 2.24 | 1310.82 | 2.84 |
| F1B0_W4M0MC3_R0.2_T0O0 | 0.8053 | 18.57 | 2.30 | 1237.61 | 3.01 |
| F1B0_W4M0MC4_R0.16_T0O0 | 0.7964 | 17.75 | 2.41 | 1164.76 | 3.20 |
| F1B0_W4M0MC4_R0.2_T0O0 | 0.7952 | 17.71 | 2.41 | 1153.05 | 3.23 |
| F1B0_W4M0MC6_R0.24_T0O0 | 0.7620 | 15.54 | 2.75 | 944.75 | 3.94 |
| F1B0_W4M0MC6_R0.32_T0O0 | 0.7438 | 15.65 | 2.73 | 942.92 | 3.95 |

### 📖SSIM: DBCache(Slow) FnBn + TaylorSeer O(1)

| Config | SSIM | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Baseline: [FLUX.1 50 steps] | - | 42.74 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_R0.08_T1O1 | 0.9340 | 28.06 | 1.53 | 2172.76 | 1.72 |
| F8B0_W8M0MC2_R0.12_T1O1 | 0.9201 | 27.31 | 1.57 | 2074.10 | 1.80 |
| F8B0_W8M0MC2_R0.16_T1O1 | 0.9194 | 26.46 | 1.62 | 2005.56 | 1.86 |
| F8B0_W8M0MC2_R0.2_T1O1 | 0.9189 | 26.22 | 1.64 | 1985.38 | 1.88 |
| F8B0_W8M0MC2_R0.24_T1O1 | 0.9169 | 25.64 | 1.67 | 1933.17 | 1.93 |
| F8B0_W8M0MC2_R0.32_T1O1 | 0.9157 | 25.71 | 1.67 | 1933.17 | 1.93 |
| F8B0_W8M0MC3_R0.12_T1O1 | 0.9116 | 26.09 | 1.65 | 1974.49 | 1.89 |
| F8B0_W8M0MC4_R0.12_T1O1 | 0.9098 | 25.59 | 1.68 | 1912.99 | 1.95 |
| F8B0_W8M0MC3_R0.16_T1O1 | 0.8953 | 24.67 | 1.74 | 1815.62 | 2.05 |
| F8B0_W8M0MC3_R0.2_T1O1 | 0.8926 | 24.46 | 1.75 | 1794.16 | 2.08 |
| F8B0_W8M0MC3_R0.24_T1O1 | 0.8907 | 23.84 | 1.80 | 1740.99 | 2.14 |
| F8B0_W8M0MC3_R0.32_T1O1 | 0.8905 | 23.98 | 1.79 | 1740.99 | 2.14 |
| F1B0_W4M0MC0_R0.08_T1O1 | 0.8845 | 23.29 | 1.84 | 1730.70 | 2.15 |
| F1B0_W4M0MC2_R0.12_T1O1 | 0.8802 | 22.07 | 1.94 | 1604.04 | 2.32 |
| F1B0_W4M0MC3_R0.12_T1O1 | 0.8752 | 20.70 | 2.07 | 1457.62 | 2.56 |
| F1B0_W4M0MC4_R0.12_T1O1 | 0.8745 | 20.08 | 2.14 | 1404.17 | 2.65 |
| F1B0_W4M0MC3_R0.16_T1O1 | 0.8222 | 19.20 | 2.24 | 1310.82 | 2.84 |
| F1B0_W4M0MC3_R0.2_T1O1 | 0.8198 | 18.63 | 2.30 | 1237.61 | 3.01 |
| F1B0_W4M0MC4_R0.16_T1O1 | 0.8125 | 17.82 | 2.41 | 1164.76 | 3.20 |
| F1B0_W4M0MC4_R0.2_T1O1 | 0.8119 | 17.71 | 2.42 | 1153.05 | 3.23 |
| F1B0_W4M0MC4_R0.24_T1O1 | 0.7837 | 17.04 | 2.52 | 1091.18 | 3.42 |
| F1B0_W4M0MC6_R0.24_T1O1 | 0.7718 | 15.61 | 2.75 | 944.75 | 3.94 |
| F1B0_W4M0MC6_R0.32_T1O1 | 0.7423 | 15.70 | 2.73 | 944.02 | 3.95 |

### 📖LPIPS: DBCache(Slow) FnBn

<div id="lpips"></div>

| Config | LPIPS | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Baseline: [FLUX.1 50 steps] | - | 42.74 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_R0.08_T0O0 | 0.0786 | 27.87 | 1.53 | 2162.19 | 1.72 |
| F8B0_W8M0MC2_R0.12_T0O0 | 0.0895 | 27.22 | 1.57 | 2072.18 | 1.80 |
| F8B0_W8M0MC2_R0.16_T0O0 | 0.0919 | 26.34 | 1.62 | 2002.67 | 1.86 |
| F8B0_W8M0MC2_R0.2_T0O0 | 0.0923 | 26.33 | 1.62 | 1994.67 | 1.87 |
| F8B0_W8M0MC2_R0.24_T0O0 | 0.0963 | 25.62 | 1.67 | 1933.17 | 1.93 |
| F8B0_W8M0MC2_R0.32_T0O0 | 0.0990 | 25.66 | 1.67 | 1933.17 | 1.93 |
| F8B0_W8M0MC4_R0.12_T0O0 | 0.1085 | 25.41 | 1.68 | 1897.61 | 1.96 |
| F8B0_W8M0MC3_R0.16_T0O0 | 0.1182 | 24.55 | 1.74 | 1811.77 | 2.06 |
| F1B0_W4M0MC0_R0.08_T0O0 | 0.1196 | 23.21 | 1.84 | 1729.60 | 2.15 |
| F1B0_W4M0MC2_R0.12_T0O0 | 0.1390 | 21.92 | 1.95 | 1604.78 | 2.32 |
| F1B0_W4M0MC3_R0.12_T0O0 | 0.1426 | 20.61 | 2.07 | 1457.62 | 2.56 |
| F1B0_W4M0MC4_R0.12_T0O0 | 0.1500 | 20.01 | 2.14 | 1401.61 | 2.66 |
| F1B0_W4M0MC3_R0.16_T0O0 | 0.1909 | 19.09 | 2.24 | 1310.82 | 2.84 |
| F1B0_W4M0MC3_R0.2_T0O0 | 0.1942 | 18.57 | 2.30 | 1237.61 | 3.01 |
| F1B0_W4M0MC4_R0.16_T0O0 | 0.2082 | 17.75 | 2.41 | 1164.76 | 3.20 |
| F1B0_W4M0MC4_R0.2_T0O0 | 0.2094 | 17.71 | 2.41 | 1153.05 | 3.23 |
| F1B0_W4M0MC4_R0.24_T0O0 | 0.2289 | 16.98 | 2.52 | 1091.18 | 3.42 |
| F1B0_W4M0MC6_R0.24_T0O0 | 0.2553 | 15.54 | 2.75 | 944.75 | 3.94 |
| F1B0_W4M0MC6_R0.32_T0O0 | 0.2826 | 15.65 | 2.73 | 942.92 | 3.95 |

### 📖LPIPS: DBCache(Slow) FnBn + TaylorSeer O(1)

| Config | LPIPS | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Baseline: [FLUX.1 50 steps] | - | 42.74 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_R0.08_T1O1 | 0.0558 | 28.06 | 1.53 | 2172.76 | 1.72 |
| F8B0_W8M0MC2_R0.12_T1O1 | 0.0687 | 27.31 | 1.57 | 2074.10 | 1.80 |
| F8B0_W8M0MC2_R0.16_T1O1 | 0.0690 | 26.46 | 1.62 | 2005.56 | 1.86 |
| F8B0_W8M0MC2_R0.2_T1O1 | 0.0692 | 26.22 | 1.64 | 1985.38 | 1.88 |
| F8B0_W8M0MC2_R0.24_T1O1 | 0.0699 | 25.64 | 1.67 | 1933.17 | 1.93 |
| F8B0_W8M0MC2_R0.32_T1O1 | 0.0714 | 25.71 | 1.67 | 1933.17 | 1.93 |
| F8B0_W8M0MC3_R0.12_T1O1 | 0.0772 | 26.09 | 1.65 | 1974.49 | 1.89 |
| F8B0_W8M0MC4_R0.12_T1O1 | 0.0792 | 25.59 | 1.68 | 1912.99 | 1.95 |
| F8B0_W8M0MC3_R0.16_T1O1 | 0.0933 | 24.67 | 1.74 | 1815.62 | 2.05 |
| F8B0_W8M0MC3_R0.2_T1O1 | 0.0958 | 24.46 | 1.75 | 1794.16 | 2.08 |
| F8B0_W8M0MC3_R0.24_T1O1 | 0.0964 | 23.84 | 1.80 | 1740.99 | 2.14 |
| F8B0_W8M0MC3_R0.32_T1O1 | 0.0974 | 23.98 | 1.79 | 1740.99 | 2.14 |
| F1B0_W4M0MC0_R0.08_T1O1 | 0.1032 | 23.29 | 1.84 | 1730.70 | 2.15 |
| F1B0_W4M0MC2_R0.12_T1O1 | 0.1091 | 22.07 | 1.94 | 1604.04 | 2.32 |
| F1B0_W4M0MC3_R0.12_T1O1 | 0.1147 | 20.70 | 2.07 | 1457.62 | 2.56 |
| F1B0_W4M0MC4_R0.12_T1O1 | 0.1154 | 20.08 | 2.14 | 1404.17 | 2.65 |
| F1B0_W4M0MC3_R0.2_T1O1 | 0.1733 | 18.63 | 2.30 | 1237.61 | 3.01 |
| F1B0_W4M0MC4_R0.16_T1O1 | 0.1837 | 17.82 | 2.41 | 1164.76 | 3.20 |
| F1B0_W4M0MC4_R0.2_T1O1 | 0.1839 | 17.71 | 2.42 | 1153.05 | 3.23 |
| F1B0_W4M0MC4_R0.24_T1O1 | 0.2155 | 17.04 | 2.52 | 1091.18 | 3.42 |
| F1B0_W4M0MC6_R0.24_T1O1 | 0.2310 | 15.61 | 2.75 | 944.75 | 3.94 |
| F1B0_W4M0MC6_R0.32_T1O1 | 0.2677 | 15.70 | 2.73 | 944.02 | 3.95 |
