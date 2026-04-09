#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# DBCache
# 8 Steps
cache-dit-metrics \
  clip_score image_reward --summary \
  --cal-speedup --gen-markdown-table \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache_Distill_8_Steps \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_distill_8_steps.log

cache-dit-metrics \
  psnr ssim lpips --summary \
  --cal-speedup --gen-markdown-table \
  --ref-img-dir ./tmp/DrawBench200_DBCache_Distill_8_Steps/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache_Distill_8_Steps \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_distill_8_steps.log


# 4 Steps
cache-dit-metrics \
  clip_score image_reward --summary \
  --cal-speedup --gen-markdown-table \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache_Distill_4_Steps \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_distill_4_steps.log

# psnr ssim lpips
cache-dit-metrics \
  psnr ssim lpips --summary \
  --cal-speedup --gen-markdown-table \
  --ref-img-dir ./tmp/DrawBench200_DBCache_Distill_4_Steps/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache_Distill_4_Steps \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_distill_4_steps.log
