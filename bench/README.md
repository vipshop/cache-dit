# 🤖Benchmarks

## 📖Contents

- [📚DrawBench](#drawbench)
  - [🎉CLIP Score](#clip-score-)
  - [🎉Image Reward](#image-reward-)
  - [🎉PSNR](#psnr-)
  - [🎉SSIM](#ssim-)
  - [🎉LPIPS](#lpips-)
- [📚How to Reproduce?](#reproduce)
  - [⚙️Installation](#installation)
  - [📖Download](#download)
  - [📖Evaluation](#evaluation)

## 📚DrawBench

The meaning of parameter configuration is as follows (such as F8B0_W8M0MC0_T0O1_R0.08):  
  - **Device**: NVIDIA L20. 
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

> [!Tips]   
> Among all the accuracy indicators, the overall accuracy has slightly improved after using TaylorSeer.

### 🎉CLIP Score (↑)

| Config | CLIP_SCORE | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Base: FLUX.1-dev, 50 steps | 32.9217 | 42.63 | 1.00 | 3726.87 | 1.00 |
| F1B0_W4M0MC0_T0O1_R0.08 | 33.0745 | 23.19 | 1.84 | 1729.60 | 2.15 |
| F8B0_W8M0MC0_T0O1_R0.08 | 33.0070 | 27.80 | 1.53 | 2162.19 | 1.72 |
| F8B0_W4M0MC0_T0O1_R0.08 | 32.9871 | 26.91 | 1.58 | 2064.81 | 1.80 |
| F4B0_W4M0MC2_T0O1_R0.12 | 32.9718 | 22.87 | 1.86 | 1678.98 | 2.22 |
| F8B0_W8M0MC3_T0O1_R0.12 | 32.9613 | 26.04 | 1.64 | 1977.69 | 1.88 |
| F8B0_W4M0MC2_T0O1_R0.12 | 32.9535 | 25.72 | 1.66 | 1935.73 | 1.93 |
| F8B0_W8M0MC2_T0O1_R0.12 | 32.9302 | 26.91 | 1.58 | 2072.18 | 1.80 |
| F8B0_W4M0MC3_T0O1_R0.12 | 32.9234 | 24.53 | 1.74 | 1816.58 | 2.05 |
| F8B0_W8M0MC4_T0O1_R0.12 | 32.9041 | 25.26 | 1.69 | 1897.61 | 1.96 |
| F4B0_W4M0MC3_T0O1_R0.12 | 32.8981 | 21.27 | 2.00 | 1507.83 | 2.47 |
| F4B0_W4M0MC0_T0O1_R0.08 | 32.8544 | 22.69 | 1.88 | 1654.72 | 2.25 |
| F8B0_W4M0MC4_T0O1_R0.12 | 32.8443 | 23.84 | 1.79 | 1753.48 | 2.13 |
| F4B0_W4M0MC4_T0O1_R0.12 | 32.8384 | 20.21 | 2.11 | 1400.08 | 2.66 |
| F1B0_W4M0MC4_T0O1_R0.12 | 32.8291 | 19.99 | 2.13 | 1401.61 | 2.66 |
| F1B0_W4M0MC3_T0O1_R0.12 | 32.8236 | 20.58 | 2.07 | 1457.62 | 2.56 |  


| Config | CLIP_SCORE | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Base: FLUX.1-dev, 50 steps | 32.9217 | 42.63 | 1.00 | 3726.87 | 1.00 |
| F1B0_W4M0MC0_T1O1_R0.08 | 33.0398 | 23.18 | 1.84 | 1730.70 | 2.15 |
| F8B0_W8M0MC3_T1O1_R0.12 | 33.0191 | 26.01 | 1.64 | 1974.49 | 1.89 |
| F8B0_W4M0MC0_T1O1_R0.08 | 33.0037 | 27.09 | 1.58 | 2079.23 | 1.79 |
| F4B0_W4M0MC3_T1O1_R0.12 | 32.9795 | 21.18 | 2.02 | 1499.51 | 2.49 |
| F8B0_W8M0MC0_T1O1_R0.08 | 32.9541 | 27.92 | 1.53 | 2172.76 | 1.72 |
| F8B0_W8M0MC2_T1O1_R0.12 | 32.9475 | 27.00 | 1.58 | 2074.10 | 1.80 |
| F1B0_W4M0MC3_T1O1_R0.12 | 32.9302 | 20.59 | 2.07 | 1457.62 | 2.56 |
| F4B0_W4M0MC4_T1O1_R0.12 | 32.9144 | 20.11 | 2.12 | 1388.30 | 2.68 |
| F8B0_W8M0MC4_T1O1_R0.12 | 32.8962 | 25.43 | 1.68 | 1912.99 | 1.95 |
| F4B0_W4M0MC2_T1O1_R0.12 | 32.8839 | 22.81 | 1.87 | 1668.58 | 2.23 |
| F1B0_W4M0MC4_T1O1_R0.12 | 32.8787 | 19.97 | 2.14 | 1404.17 | 2.65 |
| F4B0_W4M0MC0_T1O1_R0.08 | 32.8591 | 22.73 | 1.88 | 1659.92 | 2.25 |
| F8B0_W4M0MC3_T1O1_R0.12 | 32.8072 | 24.51 | 1.74 | 1817.86 | 2.05 |
| F8B0_W4M0MC2_T1O1_R0.12 | 32.7674 | 25.68 | 1.66 | 1936.69 | 1.92 |
| F8B0_W4M0MC4_T1O1_R0.12 | 32.6280 | 23.96 | 1.78 | 1762.45 | 2.11 |

### 🎉Image Reward (↑)

| Config | IMAGE_REWARD | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Base: FLUX.1-dev, 50 steps | 1.0412 | 42.63 | 1.00 | 3726.87 | 1.00 |
| F1B0_W4M0MC0_T0O1_R0.08 | 1.0418 | 23.19 | 1.84 | 1729.60 | 2.15 |
| F8B0_W4M0MC0_T0O1_R0.08 | 1.0370 | 26.91 | 1.58 | 2064.81 | 1.80 |
| F8B0_W8M0MC0_T0O1_R0.08 | 1.0333 | 27.80 | 1.53 | 2162.19 | 1.72 |
| F4B0_W4M0MC2_T0O1_R0.12 | 1.0301 | 22.87 | 1.86 | 1678.98 | 2.22 |
| F8B0_W8M0MC3_T0O1_R0.12 | 1.0270 | 26.04 | 1.64 | 1977.69 | 1.88 |
| F8B0_W8M0MC2_T0O1_R0.12 | 1.0227 | 26.91 | 1.58 | 2072.18 | 1.80 |
| F8B0_W4M0MC2_T0O1_R0.12 | 1.0185 | 25.72 | 1.66 | 1935.73 | 1.93 |
| F1B0_W4M0MC4_T0O1_R0.12 | 1.0181 | 19.99 | 2.13 | 1401.61 | 2.66 |
| F1B0_W4M0MC3_T0O1_R0.12 | 1.0166 | 20.58 | 2.07 | 1457.62 | 2.56 |
| F8B0_W8M0MC4_T0O1_R0.12 | 1.0140 | 25.26 | 1.69 | 1897.61 | 1.96 |
| F4B0_W4M0MC3_T0O1_R0.12 | 1.0130 | 21.27 | 2.00 | 1507.83 | 2.47 |
| F8B0_W4M0MC4_T0O1_R0.12 | 1.0102 | 23.84 | 1.79 | 1753.48 | 2.13 |
| F8B0_W4M0MC3_T0O1_R0.12 | 1.0085 | 24.53 | 1.74 | 1816.58 | 2.05 |
| F4B0_W4M0MC0_T0O1_R0.08 | 1.0065 | 22.69 | 1.88 | 1654.72 | 2.25 |
| F4B0_W4M0MC4_T0O1_R0.12 | 1.0065 | 20.21 | 2.11 | 1400.08 | 2.66 |


| Config | IMAGE_REWARD | Latency(s) | Latency(s)(↑) | TFLOPs | TFLOPs(↑) |
| --- | --- | --- | --- | --- | --- |
| Base: FLUX.1-dev, 50 steps | 1.0412 | 42.69 | 1.00 | 3726.87 | 1.00 |
| F1B0_W4M0MC0_T1O1_R0.08 | 1.0591 | 23.18 | 1.84 | 1730.70 | 2.15 |
| F8B0_W8M0MC0_T1O1_R0.08 | 1.0558 | 27.92 | 1.53 | 2172.76 | 1.72 |
| F8B0_W4M0MC0_T1O1_R0.08 | 1.0497 | 27.09 | 1.58 | 2079.23 | 1.79 |
| F4B0_W4M0MC2_T1O1_R0.12 | 1.0356 | 22.81 | 1.87 | 1668.58 | 2.23 |
| F4B0_W4M0MC3_T1O1_R0.12 | 1.0347 | 21.18 | 2.02 | 1499.51 | 2.49 |
| F8B0_W8M0MC2_T1O1_R0.12 | 1.0306 | 27.00 | 1.58 | 2074.10 | 1.80 |
| F1B0_W4M0MC3_T1O1_R0.12 | 1.0292 | 20.59 | 2.07 | 1457.62 | 2.56 |
| F4B0_W4M0MC4_T1O1_R0.12 | 1.0287 | 20.11 | 2.12 | 1388.30 | 2.68 |
| F8B0_W8M0MC3_T1O1_R0.12 | 1.0268 | 26.01 | 1.64 | 1974.49 | 1.89 |
| F8B0_W4M0MC3_T1O1_R0.12 | 1.0263 | 24.51 | 1.74 | 1817.86 | 2.05 |
| F8B0_W4M0MC2_T1O1_R0.12 | 1.0226 | 25.68 | 1.66 | 1936.69 | 1.92 |
| F8B0_W8M0MC4_T1O1_R0.12 | 1.0221 | 25.43 | 1.68 | 1912.99 | 1.95 |
| F4B0_W4M0MC0_T1O1_R0.08 | 1.0199 | 22.73 | 1.88 | 1659.92 | 2.25 |
| F1B0_W4M0MC4_T1O1_R0.12 | 1.0182 | 19.97 | 2.14 | 1404.17 | 2.65 |
| F8B0_W4M0MC4_T1O1_R0.12 | 1.0138 | 23.96 | 1.78 | 1762.45 | 2.11 |

### 🎉PSNR (↑)

| Config | PSNR | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Base: FLUX.1-dev, 50 steps | INF | 42.63 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_T0O1_R0.08 | 35.2008 | 27.80 | 1.53 | 2162.19 | 1.72 |
| F8B0_W8M0MC2_T0O1_R0.12 | 34.7449 | 26.91 | 1.58 | 2072.18 | 1.80 |
| F8B0_W8M0MC3_T0O1_R0.12 | 34.2834 | 26.04 | 1.64 | 1977.69 | 1.88 |
| F1B0_W4M0MC0_T0O1_R0.08 | 33.9639 | 23.19 | 1.84 | 1729.60 | 2.15 |
| F8B0_W8M0MC4_T0O1_R0.12 | 33.9466 | 25.26 | 1.69 | 1897.61 | 1.96 |
| F8B0_W4M0MC0_T0O1_R0.08 | 33.8317 | 26.91 | 1.58 | 2064.81 | 1.80 |
| F1B0_W4M0MC3_T0O1_R0.12 | 33.0037 | 20.58 | 2.07 | 1457.62 | 2.56 |
| F1B0_W4M0MC4_T0O1_R0.12 | 32.9462 | 19.99 | 2.13 | 1401.61 | 2.66 |
| F8B0_W4M0MC2_T0O1_R0.12 | 32.7346 | 25.72 | 1.66 | 1935.73 | 1.93 |
| F8B0_W4M0MC3_T0O1_R0.12 | 32.5385 | 24.53 | 1.74 | 1816.58 | 2.05 |
| F8B0_W4M0MC4_T0O1_R0.12 | 32.4231 | 23.84 | 1.79 | 1753.48 | 2.13 |
| F4B0_W4M0MC0_T0O1_R0.08 | 32.3555 | 22.69 | 1.88 | 1654.72 | 2.25 |
| F4B0_W4M0MC2_T0O1_R0.12 | 31.9394 | 22.87 | 1.86 | 1678.98 | 2.22 |
| F4B0_W4M0MC3_T0O1_R0.12 | 31.8031 | 21.27 | 2.00 | 1507.83 | 2.47 |
| F4B0_W4M0MC4_T0O1_R0.12 | 31.5292 | 20.21 | 2.11 | 1400.08 | 2.66 |


| Config | PSNR | Latency(s) | Latency(s)(↑) | TFLOPs | TFLOPs(↑) |
| --- | --- | --- | --- | --- | --- |
| Base: FLUX.1-dev, 50 steps | INF | 42.63 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_T1O1_R0.08 | 36.7296 | 27.92 | 1.53 | 2172.76 | 1.72 |
| F8B0_W8M0MC2_T1O1_R0.12 | 36.1337 | 27.00 | 1.58 | 2074.10 | 1.80 |
| F8B0_W8M0MC3_T1O1_R0.12 | 35.4623 | 26.01 | 1.64 | 1974.49 | 1.89 |
| F8B0_W8M0MC4_T1O1_R0.12 | 35.4352 | 25.43 | 1.68 | 1912.99 | 1.95 |
| F8B0_W4M0MC0_T1O1_R0.08 | 34.8411 | 27.09 | 1.58 | 2079.23 | 1.79 |
| F1B0_W4M0MC0_T1O1_R0.08 | 34.5305 | 23.18 | 1.84 | 1730.70 | 2.15 |
| F1B0_W4M0MC4_T1O1_R0.12 | 33.8600 | 19.97 | 2.14 | 1404.17 | 2.65 |
| F1B0_W4M0MC2_T1O1_R0.12 | 33.8273 | 21.87 | 1.95 | 1604.04 | 2.32 |
| F1B0_W4M0MC3_T1O1_R0.12 | 33.7795 | 20.59 | 2.07 | 1457.62 | 2.56 |
| F8B0_W4M0MC2_T1O1_R0.12 | 33.5948 | 25.68 | 1.66 | 1936.69 | 1.92 |
| F8B0_W4M0MC3_T1O1_R0.12 | 33.4969 | 24.51 | 1.74 | 1817.86 | 2.05 |
| F8B0_W4M0MC4_T1O1_R0.12 | 33.1565 | 23.96 | 1.78 | 1762.45 | 2.11 |
| F4B0_W4M0MC0_T1O1_R0.08 | 33.0380 | 22.73 | 1.88 | 1659.92 | 2.25 |
| F4B0_W4M0MC2_T1O1_R0.12 | 31.8955 | 22.81 | 1.87 | 1668.58 | 2.23 |
| F4B0_W4M0MC3_T1O1_R0.12 | 31.6260 | 21.18 | 2.02 | 1499.51 | 2.49 |
| F4B0_W4M0MC4_T1O1_R0.12 | 31.4629 | 20.11 | 2.12 | 1388.30 | 2.68 |

### 🎉SSIM (↑)

| Config | SSIM | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Base: FLUX.1-dev, 50 steps | INF | 42.63 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_T0O1_R0.08 | 0.9131 | 27.80 | 1.53 | 2162.19 | 1.72 |
| F8B0_W8M0MC2_T0O1_R0.12 | 0.9017 | 26.91 | 1.58 | 2072.18 | 1.80 |
| F8B0_W8M0MC3_T0O1_R0.12 | 0.8951 | 26.04 | 1.64 | 1977.69 | 1.88 |
| F8B0_W8M0MC4_T0O1_R0.12 | 0.8858 | 25.26 | 1.69 | 1897.61 | 1.96 |
| F1B0_W4M0MC0_T0O1_R0.08 | 0.8727 | 23.19 | 1.84 | 1729.60 | 2.15 |
| F8B0_W4M0MC0_T0O1_R0.08 | 0.8721 | 26.91 | 1.58 | 2064.81 | 1.80 |
| F1B0_W4M0MC3_T0O1_R0.12 | 0.8521 | 20.58 | 2.07 | 1457.62 | 2.56 |
| F1B0_W4M0MC4_T0O1_R0.12 | 0.8461 | 19.99 | 2.13 | 1401.61 | 2.66 |
| F8B0_W4M0MC2_T0O1_R0.12 | 0.8444 | 25.72 | 1.66 | 1935.73 | 1.93 |
| F8B0_W4M0MC3_T0O1_R0.12 | 0.8392 | 24.53 | 1.74 | 1816.58 | 2.05 |
| F8B0_W4M0MC4_T0O1_R0.12 | 0.8354 | 23.84 | 1.79 | 1753.48 | 2.13 |
| F4B0_W4M0MC0_T0O1_R0.08 | 0.8324 | 22.69 | 1.88 | 1654.72 | 2.25 |
| F4B0_W4M0MC2_T0O1_R0.12 | 0.8116 | 22.87 | 1.86 | 1678.98 | 2.22 |
| F4B0_W4M0MC3_T0O1_R0.12 | 0.8094 | 21.27 | 2.00 | 1507.83 | 2.47 |
| F4B0_W4M0MC4_T0O1_R0.12 | 0.7973 | 20.21 | 2.11 | 1400.08 | 2.66 |

| Config | SSIM | Latency(s) | Latency(s)(↑) | TFLOPs | TFLOPs(↑) |
| --- | --- | --- | --- | --- | --- |
| Base: FLUX.1-dev, 50 steps | INF | 42.63 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_T1O1_R0.08 | 0.9340 | 27.92 | 1.53 | 2172.76 | 1.72 |
| F8B0_W8M0MC2_T1O1_R0.12 | 0.9201 | 27.00 | 1.58 | 2074.10 | 1.80 |
| F8B0_W8M0MC3_T1O1_R0.12 | 0.9116 | 26.01 | 1.64 | 1974.49 | 1.89 |
| F8B0_W8M0MC4_T1O1_R0.12 | 0.9098 | 25.43 | 1.68 | 1912.99 | 1.95 |
| F8B0_W4M0MC0_T1O1_R0.08 | 0.8954 | 27.09 | 1.58 | 2079.23 | 1.79 |
| F1B0_W4M0MC0_T1O1_R0.08 | 0.8845 | 23.18 | 1.84 | 1730.70 | 2.15 |
| F1B0_W4M0MC2_T1O1_R0.12 | 0.8802 | 21.87 | 1.95 | 1604.04 | 2.32 |
| F1B0_W4M0MC3_T1O1_R0.12 | 0.8752 | 20.59 | 2.07 | 1457.62 | 2.56 |
| F1B0_W4M0MC4_T1O1_R0.12 | 0.8745 | 19.97 | 2.14 | 1404.17 | 2.65 |
| F8B0_W4M0MC2_T1O1_R0.12 | 0.8672 | 25.68 | 1.66 | 1936.69 | 1.92 |
| F8B0_W4M0MC3_T1O1_R0.12 | 0.8643 | 24.51 | 1.74 | 1817.86 | 2.05 |
| F8B0_W4M0MC4_T1O1_R0.12 | 0.8597 | 23.96 | 1.78 | 1762.45 | 2.11 |
| F4B0_W4M0MC0_T1O1_R0.08 | 0.8527 | 22.73 | 1.88 | 1659.92 | 2.25 |
| F4B0_W4M0MC2_T1O1_R0.12 | 0.8231 | 22.81 | 1.87 | 1668.58 | 2.23 |
| F4B0_W4M0MC3_T1O1_R0.12 | 0.8154 | 21.18 | 2.02 | 1499.51 | 2.49 |
| F4B0_W4M0MC4_T1O1_R0.12 | 0.8087 | 20.11 | 2.12 | 1388.30 | 2.68 |

### 🎉LPIPS (↓)

| Config | LPIPS | Latency(s) | SpeedUp(↑) | TFLOPs | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| Base: FLUX.1-dev, 50 steps | INF | 42.63 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_T0O1_R0.08 | 0.0786 | 27.80 | 1.53 | 2162.19 | 1.72 |
| F8B0_W8M0MC2_T0O1_R0.12 | 0.0895 | 26.91 | 1.58 | 2072.18 | 1.80 |
| F8B0_W8M0MC3_T0O1_R0.12 | 0.0978 | 26.04 | 1.64 | 1977.69 | 1.88 |
| F8B0_W8M0MC4_T0O1_R0.12 | 0.1085 | 25.26 | 1.69 | 1897.61 | 1.96 |
| F1B0_W4M0MC0_T0O1_R0.08 | 0.1196 | 23.19 | 1.84 | 1729.60 | 2.15 |
| F8B0_W4M0MC0_T0O1_R0.08 | 0.1201 | 26.91 | 1.58 | 2064.81 | 1.80 |
| F1B0_W4M0MC3_T0O1_R0.12 | 0.1426 | 20.58 | 2.07 | 1457.62 | 2.56 |
| F1B0_W4M0MC4_T0O1_R0.12 | 0.1500 | 19.99 | 2.13 | 1401.61 | 2.66 |
| F8B0_W4M0MC2_T0O1_R0.12 | 0.1511 | 25.72 | 1.66 | 1935.73 | 1.93 |
| F8B0_W4M0MC3_T0O1_R0.12 | 0.1587 | 24.53 | 1.74 | 1816.58 | 2.05 |
| F8B0_W4M0MC4_T0O1_R0.12 | 0.1637 | 23.84 | 1.79 | 1753.48 | 2.13 |
| F4B0_W4M0MC0_T0O1_R0.08 | 0.1659 | 22.69 | 1.88 | 1654.72 | 2.25 |
| F4B0_W4M0MC2_T0O1_R0.12 | 0.1884 | 22.87 | 1.86 | 1678.98 | 2.22 |
| F4B0_W4M0MC3_T0O1_R0.12 | 0.1938 | 21.27 | 2.00 | 1507.83 | 2.47 |
| F4B0_W4M0MC4_T0O1_R0.12 | 0.2108 | 20.21 | 2.11 | 1400.08 | 2.66 |


| Config | LPIPS | Latency(s) | Latency(s)(↑) | TFLOPs | TFLOPs(↑) |
| --- | --- | --- | --- | --- | --- |
| Base: FLUX.1-dev, 50 steps | INF | 42.63 | 1.00 | 3726.87 | 1.00 |
| F8B0_W8M0MC0_T1O1_R0.08 | 0.0558 | 27.92 | 1.53 | 2172.76 | 1.72 |
| F8B0_W8M0MC2_T1O1_R0.12 | 0.0687 | 27.00 | 1.58 | 2074.10 | 1.80 |
| F8B0_W8M0MC3_T1O1_R0.12 | 0.0772 | 26.01 | 1.64 | 1974.49 | 1.89 |
| F8B0_W8M0MC4_T1O1_R0.12 | 0.0792 | 25.43 | 1.68 | 1912.99 | 1.95 |
| F8B0_W4M0MC0_T1O1_R0.08 | 0.0925 | 27.09 | 1.58 | 2079.23 | 1.79 |
| F1B0_W4M0MC0_T1O1_R0.08 | 0.1032 | 23.18 | 1.84 | 1730.70 | 2.15 |
| F1B0_W4M0MC2_T1O1_R0.12 | 0.1091 | 21.87 | 1.95 | 1604.04 | 2.32 |
| F1B0_W4M0MC3_T1O1_R0.12 | 0.1147 | 20.59 | 2.07 | 1457.62 | 2.56 |
| F1B0_W4M0MC4_T1O1_R0.12 | 0.1154 | 19.97 | 2.14 | 1404.17 | 2.65 |
| F8B0_W4M0MC2_T1O1_R0.12 | 0.1229 | 25.68 | 1.66 | 1936.69 | 1.92 |
| F8B0_W4M0MC3_T1O1_R0.12 | 0.1261 | 24.51 | 1.74 | 1817.86 | 2.05 |
| F8B0_W4M0MC4_T1O1_R0.12 | 0.1322 | 23.96 | 1.78 | 1762.45 | 2.11 |
| F4B0_W4M0MC0_T1O1_R0.08 | 0.1387 | 22.73 | 1.88 | 1659.92 | 2.25 |
| F4B0_W4M0MC2_T1O1_R0.12 | 0.1729 | 22.81 | 1.87 | 1668.58 | 2.23 |
| F4B0_W4M0MC3_T1O1_R0.12 | 0.1820 | 21.18 | 2.02 | 1499.51 | 2.49 |
| F4B0_W4M0MC4_T1O1_R0.12 | 0.1901 | 20.11 | 2.12 | 1388.30 | 2.68 |


## 📚How to Reproduce?

### ⚙️Installation

```bash
# install requirements
pip3 install git+https://github.com/openai/CLIP.git
pip3 install git+https://github.com/chengzegang/calculate-flops.pytorch.git
pip3 install image-reward
pip3 install git+https://github.com/vipshop/cache-dit.git
```

### 📖Download

```bash
git clone https://github.com/vipshop/cache-dit.git
cd cache-dit/bench && mkdir tmp && mkdir log && mkdir hf_models && cd hf_models

# download from modelscope
modelscope download black-forest-labs/FLUX.1-dev --local_dir ./FLUX.1-dev
modelscope download laion/CLIP-ViT-g-14-laion2B-s12B-b42K --local_dir ./CLIP-ViT-g-14-laion2B-s12B-b42K
modelscope download ZhipuAI/ImageReward --local_dir ./ImageReward
# download from huggingface
hf download black-forest-labs/FLUX.1-dev --local-dir ./FLUX.1-dev
hf download laion/CLIP-ViT-g-14-laion2B-s12B-b42K --local-dir ./CLIP-ViT-g-14-laion2B-s12B-b42K
hf download ZhipuAI/ImageReward --local-dir ./ImageReward

export FLUX_DIR=$PWD/FLUX.1-dev
export CLIP_MODEL_DIR=$PWD/CLIP-ViT-g-14-laion2B-s12B-b42K
export IMAGEREWARD_MODEL_DIR=$PWD/ImageReward
cd ..
```


### 📖Evaluation

```bash
# NOTE: The reported benchmark was run on NVIDIA L20 device.
export CUDA_VISIBLE_DEVICES=0 && nohup bash bench.sh default > log/cache_dit_bench_default.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1 && nohup bash bench.sh taylorseer > log/cache_dit_bench_taylorseer.log 2>&1 &
bash ./metrics.sh
```
