# Hybrid Cache Acceleration Benchmark

|Baseline|SCM S S*|SCM F D*|SCM U D*|+TS|+compile|+FP8*|   
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|15.4s|11.4s|8.2s|8.2s|**🎉7.1s**|**🎉4.5s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.NONE.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/static.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.2_SCM1111110100010000100000100000_dynamic_T0O0_S15.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.3_SCM111101000010000010000001000000_dynamic_T0O0_S19.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main//assets/steps_mask/flux.C1_Q1_float8_DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=90px>|


<p align="center">
  Scheme: <b>DBCache + SCM(steps_computation_mask) + TS(TaylorSeer) + FP8*</b>, L20x1, S*: static cache, <br><b>D*: dynamic cache</b>, <b>S</b>: Slow, <b>F</b>: Fast, <b>U</b>: Ultra Fast, <b>TS</b>: TaylorSeer, <b>FP8*</b>: FP8 DQ + Sage, <b>FLUX.1</b>-Dev
</p>


![clip-score-bench](https://github.com/vipshop/cache-dit/raw/main/assets/clip-score-bench.png)


<div id="benchmarks"></div>

cache-dit will support more mainstream Cache acceleration algorithms in the future. More benchmarks will be released, please stay tuned for update. Here, only the results of some precision and performance benchmarks are presented. The test dataset is **DrawBench**. For a complete benchmark, please refer to [📚Benchmarks](https://github.com/vipshop/cache-dit/tree/main/bench/).

## Text2Image DrawBench

Comparisons between different FnBn compute block configurations show that **more compute blocks result in higher precision**. For example, the F8B0_W8MC0 configuration achieves the best Clip Score (33.007) and ImageReward (1.0333). **Device**: NVIDIA L20. **F**: Fn_compute_blocks, **B**: Bn_compute_blocks, 50 steps.

| Config | Clip Score(↑) | ImageReward(↑) | PSNR(↑) | TFLOPs(↓) | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 32.9217 | 1.0412 | INF | 3726.87 | 1.00x |
| F8B0_W4MC0_R0.08 | 32.9871 | 1.0370 | 33.8317 | 2064.81 | 1.80x |
| F8B0_W4MC2_R0.12 | 32.9535 | 1.0185 | 32.7346 | 1935.73 | 1.93x |
| F8B0_W4MC3_R0.12 | 32.9234 | 1.0085 | 32.5385 | 1816.58 | 2.05x |
| F4B0_W4MC3_R0.12 | 32.8981 | 1.0130 | 31.8031 | 1507.83 | 2.47x |
| F4B0_W4MC4_R0.12 | 32.8384 | 1.0065 | 31.5292 | 1400.08 | 2.66x |

## SOTA Performance

The comparison between **cache-dit: DBCache** and algorithms such as Δ-DiT, Chipmunk, FORA, DuCa, TaylorSeer and FoCa is as follows. Now, in the comparison with a speedup ratio less than **4x**, cache-dit achieved the best accuracy. Surprisingly, cache-dit: DBCache still works in the extremely few-step distill model. For a complete benchmark, please refer to [📚Benchmarks](https://github.com/vipshop/cache-dit/raw/main/bench/). NOTE: Except for DBCache, other performance data are referenced from the paper [FoCa, arxiv.2508.16211](https://arxiv.org/pdf/2508.16211).

| Method | TFLOPs(↓) | SpeedUp(↑) | ImageReward(↑) | Clip Score(↑) |
| --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 3726.87 | 1.00× | 0.9898 | 32.404 |
| [**FLUX.1**-dev]: 60% steps | 2231.70 | 1.67× | 0.9663 | 32.312 |
| Δ-DiT(N=2) | 2480.01 | 1.50× | 0.9444 | 32.273 |
| Δ-DiT(N=3) | 1686.76 | 2.21× | 0.8721 | 32.102 |
| [**FLUX.1**-dev]: 34% steps | 1264.63 | 3.13× | 0.9453 | 32.114 |
| Chipmunk | 1505.87 | 2.47× | 0.9936 | 32.776 |
| FORA(N=3) | 1320.07 | 2.82× | 0.9776 | 32.266 |
| **[DBCache(S)](https://github.com/vipshop/cache-dit)** | 1400.08 | **2.66×** | **1.0065** | 32.838 |
| DuCa(N=5) | 978.76 | 3.80× | 0.9955 | 32.241 |
| TaylorSeer(N=4,O=2) | 1042.27 | 3.57× | 0.9857 | 32.413 |
| **[DBCache(S)+TS](https://github.com/vipshop/cache-dit)** | 1153.05 | **3.23×** | **1.0221** | 32.819 |
| **[DBCache(M)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | 0.9997 | 32.849 |
| **[DBCache(M)+TS](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | **1.0107** | 32.865 |
| **[FoCa(N=5): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 893.54 | **4.16×** | 1.0029 | **32.948** |
| [**FLUX.1**-dev]: 22% steps | 818.29 | 4.55× | 0.8183 | 31.772 |
| FORA(N=7) | 670.14 | 5.55× | 0.7418 | 31.519 |
| ToCa(N=12) | 644.70 | 5.77× | 0.7155 | 31.808 |
| DuCa(N=10) | 606.91 | 6.13× | 0.8382 | 31.759 |
| TeaCache(l=1.2) | 669.27 | 5.56× | 0.7394 | 31.704 |
| TaylorSeer(N=7,O=2) | 670.44 | 5.54× | 0.9128 | 32.128 |
| **[DBCache(F)](https://github.com/vipshop/cache-dit)** | 651.90 | **5.72x** | 0.9271 | 32.552 |
| **[FoCa(N=8): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 596.07 | 6.24× | 0.9502 | 32.706 |
| **[DBCache(F)+TS](https://github.com/vipshop/cache-dit)** | 651.90 | **5.72x** | **0.9526** | 32.568 |
| **[DBCache(U)+TS](https://github.com/vipshop/cache-dit)** | 505.47 | **7.37x** | 0.8645 | **32.719** |

## Text2Image Distillation DrawBench

Surprisingly, cache-dit: DBCache still works in the extremely few-step distill model. For example,  **Qwen-Image-Lightning w/ 4 steps**, with the F16B16 configuration, the PSNR is 34.8163, the Clip Score is 35.6109, and the ImageReward is 1.2614. It maintained a relatively high precision.

| Config                     |  PSNR(↑)      | Clip Score(↑) | ImageReward(↑) | TFLOPs(↓)   | SpeedUp(↑) |
|----------------------------|-----------|------------|--------------|----------|------------|
| [**Lightning**]: 4 steps   | INF       | 35.5797    | 1.2630       | 274.33   | 1.00x       |
| F24B24_W2MC1_R0.8          | 36.3242   | 35.6224    | 1.2630       | 264.74   | 1.04x       |
| F16B16_W2MC1_R0.8          | 34.8163   | 35.6109    | 1.2614       | 244.25   | 1.12x       |
| F12B12_W2MC1_R0.8          | 33.8953   | 35.6535    | 1.2549       | 234.63   | 1.17x       |
| F8B8_W2MC1_R0.8            | 33.1374   | 35.7284    | 1.2517       | 224.29   | 1.22x       |
| F1B0_W2MC1_R0.8            | 31.8317   | 35.6651    | 1.2397       | 206.90   | 1.33x       |
