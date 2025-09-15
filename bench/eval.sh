# DBCache
# TFLOPs
cache-dit-metrics-cli \
  clip_score image_reward --summary \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache \
  --perf-tag "Mean pipeline TFLOPs" \
  --perf-log ./log/cache_dit_bench_default.log

cache-dit-metrics-cli \
  psnr ssim lpips --summary \
  --ref-img-dir ./tmp/DrawBench200_DBCache/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache \
  --perf-tag "Mean pipeline TFLOPs" \
  --perf-log ./log/cache_dit_bench_default.log

# Latency
cache-dit-metrics-cli \
  clip_score image_reward --summary \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache \
  --perf-tag "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_default.log

cache-dit-metrics-cli \
  psnr ssim lpips --summary \
  --ref-img-dir ./tmp/DrawBench200_DBCache/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache \
  --perf-tag "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_default.log

# DBCache_TaylorSeer_O(1)
# TFLOPs
cache-dit-metrics-cli \
  clip_score image_reward --summary \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache_TaylorSeer \
  --perf-tag "Mean pipeline TFLOPs" \
  --perf-log ./log/cache_dit_bench_default.log

cache-dit-metrics-cli \
  psnr ssim lpips --summary \
  --ref-img-dir ./tmp/DrawBench200_DBCache_TaylorSeer/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache_TaylorSeer \
  --perf-tag "Mean pipeline TFLOPs" \
  --perf-log ./log/cache_dit_bench_default.log

# Latency
cache-dit-metrics-cli \
  clip_score image_reward --summary \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache_TaylorSeer \
  --perf-tag "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_default.log

cache-dit-metrics-cli \
  psnr ssim lpips --summary \
  --ref-img-dir ./tmp/DrawBench200_DBCache_TaylorSeer/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache_TaylorSeer \
  --perf-tag "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_default.log
