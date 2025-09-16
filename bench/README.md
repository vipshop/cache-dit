# 🤖Benchmarks

## 📚DrawBench

Device: NVIDIA L20, F: Fn_compute_blocks, B: Bn_compute_blocks, W: max_warmup_steps, M: max_cached_steps, MC: max_continuous_cached_steps, T: enable talyorseer or not, O: taylorseer order.

- 🎉CLIP Score (↑)

| Config | CLIP_SCORE | Latency(s) | Latency(s)(↑) | TFLOPs | TFLOPs(↑) |
| --- | --- | --- | --- | --- | --- |
| C0_Q0_NONE | 32.9217 | 42.63 | 1.00 | 3726.87 | 1.00 |
| F1B0_W4M0MC0_T0O1_R0.08 | 33.0745 | 23.19 | 1.84 | 1729.60 | 2.15 |
| F8B0_W8M0MC0_T0O1_R0.08 | 33.0070 | 27.80 | 1.53 | 2162.19 | 1.72 |
| F8B0_W4M0MC0_T0O1_R0.08 | 32.9871 | 26.91 | 1.58 | 2064.81 | 1.80 |
| F8B0_W8M0MC3_T0O1_R0.12 | 32.9613 | 26.04 | 1.64 | 1977.69 | 1.88 |
| F8B0_W8M0MC2_T0O1_R0.12 | 32.9302 | 26.91 | 1.58 | 2072.18 | 1.80 |
| F8B0_W4M0MC3_T0O1_R0.12 | 32.9234 | 24.53 | 1.74 | 1816.58 | 2.05 |
| F8B0_W8M0MC4_T0O1_R0.12 | 32.9041 | 25.26 | 1.69 | 1897.61 | 1.96 |
| F4B0_W4M0MC3_T0O1_R0.12 | 32.8981 | 21.27 | 2.00 | 1507.83 | 2.47 |
| F4B0_W4M0MC0_T0O1_R0.08 | 32.8544 | 22.69 | 1.88 | 1654.72 | 2.25 |
| F8B0_W4M0MC4_T0O1_R0.12 | 32.8443 | 23.84 | 1.79 | 1753.48 | 2.13 |
| F4B0_W4M0MC4_T0O1_R0.12 | 32.8384 | 20.21 | 2.11 | 1400.08 | 2.66 |
| F1B0_W4M0MC4_T0O1_R0.12 | 32.8291 | 19.99 | 2.13 | 1401.61 | 2.66 |
| F1B0_W4M0MC3_T0O1_R0.12 | 32.8236 | 20.58 | 2.07 | 1457.62 | 2.56 |

- 🎉Image Reward (↑)

| Config | IMAGE_REWARD | Latency(s) | Latency(s)(↑) | TFLOPs | TFLOPs(↑) |
| --- | --- | --- | --- | --- | --- |
| C0_Q0_NONE | 1.0412 | 42.63 | 1.00 | 3726.87 | 1.00 |
| F1B0_W4M0MC0_T0O1_R0.08 | 1.0418 | 23.19 | 1.84 | 1729.60 | 2.15 |
| F8B0_W4M0MC0_T0O1_R0.08 | 1.0370 | 26.91 | 1.58 | 2064.81 | 1.80 |
| F8B0_W8M0MC0_T0O1_R0.08 | 1.0333 | 27.80 | 1.53 | 2162.19 | 1.72 |
| F8B0_W8M0MC3_T0O1_R0.12 | 1.0270 | 26.04 | 1.64 | 1977.69 | 1.88 |
| F8B0_W8M0MC2_T0O1_R0.12 | 1.0227 | 26.91 | 1.58 | 2072.18 | 1.80 |
| F1B0_W4M0MC4_T0O1_R0.12 | 1.0181 | 19.99 | 2.13 | 1401.61 | 2.66 |
| F1B0_W4M0MC3_T0O1_R0.12 | 1.0166 | 20.58 | 2.07 | 1457.62 | 2.56 |
| F8B0_W8M0MC4_T0O1_R0.12 | 1.0140 | 25.26 | 1.69 | 1897.61 | 1.96 |
| F4B0_W4M0MC3_T0O1_R0.12 | 1.0130 | 21.27 | 2.00 | 1507.83 | 2.47 |
| F8B0_W4M0MC4_T0O1_R0.12 | 1.0102 | 23.84 | 1.79 | 1753.48 | 2.13 |
| F8B0_W4M0MC3_T0O1_R0.12 | 1.0085 | 24.53 | 1.74 | 1816.58 | 2.05 |
| F4B0_W4M0MC0_T0O1_R0.08 | 1.0065 | 22.69 | 1.88 | 1654.72 | 2.25 |
| F4B0_W4M0MC4_T0O1_R0.12 | 1.0065 | 20.21 | 2.11 | 1400.08 | 2.66 |

## 📚Reproduce

```bash
# download models
cd cache-dit/bench && mkdir tmp && mkdir log && mkdir hf_models
cd hf_models
modelscope download --model black-forest-labs/FLUX.1-dev --local_dir ./FLUX.1-dev
modelscope download --model laion/CLIP-ViT-g-14-laion2B-s12B-b42K --local_dir ./CLIP-ViT-g-14-laion2B-s12B-b42K
modelscope download --model ZhipuAI/ImageReward --local_dir ./ImageReward
export FLUX_DIR=$PWD/FLUX.1-dev
export CLIP_MODEL_DIR=$PWD/CLIP-ViT-g-14-laion2B-s12B-b42K
export IMAGEREWARD_MODEL_DIR=$PWD/ImageReward
cd ..

# run benchmark
export CUDA_VISIBLE_DEVICES=0 && nohup bash bench.sh default > log/cache_dit_bench_default.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1 && nohup bash bench.sh taylorseer > log/cache_dit_bench_taylorseer.log 2>&1 &

# run metrics
bash ./metrics.sh
```
