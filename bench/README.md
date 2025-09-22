# 🤖Benchmarks

## 📖Contents

- [📚DrawBench](#benchmark-flux)
- [📚Distillation DrawBench](#benchmark-lightning)
- [📚How to Reproduce?](#how-to-reproduce)
  - [⚙️Installation](#installation)
  - [📖Download](#download)
  - [📖Evaluation](#evaluation)

## 📚Text2Image DrawBench: FLUX.1-dev

<div id="benchmark-flux"></div>

Comparisons between different FnBn compute block configurations show that **more compute blocks result in higher precision**. For example, the F8B0_W8MC0 configuration achieves the best Clip Score (33.007) and ImageReward (1.0333). The meaning of parameter configuration is as follows (such as F8B0_W8M0MC0_T0O1_R0.08): (**Device**: NVIDIA L20.) 
  - **F**: Fn_compute_blocks 
  - **B**: Bn_compute_blocks
  - **W**: max_warmup_steps
  - **M**: max_cached_steps
  - **MC**: max_continuous_cached_steps (namely, hybird dynamic cache and static cache)
  - **T**: enable talyorseer or not (namely, hybrid taylorseer w/ dynamic cache - DBCache) 
  - **O**: taylorseer order, O1 means order 1.
  - **R**: residual diff threshold, range [0, 1.0)
  - **Latency(s)**: Recorded compute time (eager mode) that **w/o** other optimizations
  - **TFLOPs**: Recorded compute FLOPs using [calflops](https://github.com/chengzegang/calculate-flops.pytorch.git)'s [calculate_flops](./utils.py) API.


> [!Note]   
> Among all the accuracy indicators, the overall accuracy has slightly improved after using TaylorSeer.


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
| **[DBCache(F=4,B=0,W=4,MC=4)](https://github.com/vipshop/cache-dit)** | 1400.08 | **2.66×** | **1.0065** | 32.838 |
| DuCa(N=5) | 978.76 | 3.80× | 0.9955 | 32.241 |
| TaylorSeer(N=4,O=2) | 1042.27 | 3.57× | 0.9857 | 32.413 |
| **[DBCache+TaylorSeer(F=1,B=0,O=1)](https://github.com/vipshop/cache-dit)** | 1153.05 | **3.23×** | **1.0221** | 32.819 |
| **[DBCache(F=1,B=0,W=4,MC=6)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | 0.9997 | 32.849 |
| **[DBCache+TaylorSeer(F=1,B=0,O=1)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | **1.0107** | 32.865 |
| **[FoCa(N=5): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 893.54 | **4.16×** | **1.0029** | **32.948** |
| [**FLUX.1**-dev]: 22% steps | 818.29 | 4.55× | 0.8183 | 31.772 |
| FORA(N=4) | 967.91 | 3.84× | 0.9730 | 32.142 |
| ToCa(N=8) | 784.54 | 4.74× | 0.9451 | 31.993 |
| DuCa(N=7) | 760.14 | 4.89× | 0.9757 | 32.066 |
| TeaCache(l=0.8) | 892.35 | 4.17× | 0.8683 | 31.704 |
| **[DBCache(F=4,B=0,W=4,MC=10)](https://github.com/vipshop/cache-dit)** | 816.65 | 4.56x | 0.8245 | 32.191 |
| TaylorSeer(N=5,O=2) | 893.54 | 4.16× | 0.9768 | 32.467 |
| **[FoCa(N=7): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 670.44 | **5.54×** | **0.9891** | **32.920** |
| FORA(N=7) | 670.14 | 5.55× | 0.7418 | 31.519 |
| ToCa(N=12) | 644.70 | 5.77× | 0.7155 | 31.808 |
| DuCa(N=10) | 606.91 | 6.13× | 0.8382 | 31.759 |
| TeaCache(l=1.2) | 669.27 | 5.56× | 0.7394 | 31.704 |
| **[DBCache(F=1,B=0,W=4,MC=10)](https://github.com/vipshop/cache-dit)** | 651.90 | **5.72x** | 0.8796 | **32.318** |
| TaylorSeer(N=7,O=2) | 670.44 | 5.54× | 0.9128 | 32.128 |
| **[FoCa(N=8): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 596.07 | **6.24×** | **0.9502** | **32.706** |

NOTE: Except for DBCache, other performance data are referenced from the paper [FoCa, arxiv.2508.16211](https://arxiv.org/pdf/2508.16211).

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
export CUDA_VISIBLE_DEVICES=0 && nohup bash bench.sh default > log/cache_dit_bench_default.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1 && nohup bash bench.sh taylorseer > log/cache_dit_bench_taylorseer.log 2>&1 &
bash ./metrics.sh

# FLUX.1-dev DrawBench w/ high speedup ratio
export CUDA_VISIBLE_DEVICES=0 && nohup bash bench_fast.sh default > log/cache_dit_bench_fast.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1 && nohup bash bench_fast.sh taylorseer > log/cache_dit_bench_taylorseer_fast.log 2>&1 &
bash ./metrics_fast.sh

# Qwen-Image-Lightning DrawBench
export CUDA_VISIBLE_DEVICES=0,1 && nohup bash bench_distill.sh 8_steps > log/cache_dit_bench_distill_8_steps.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2,3 && nohup bash bench_distill.sh 4_steps > log/cache_dit_bench_distill_4_steps.log 2>&1 &
bash ./metrics_distill.sh
```
