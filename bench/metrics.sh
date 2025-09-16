# DBCache
# TFLOPs + Latency
cache-dit-metrics-cli \
  clip_score image_reward --summary --cal-speedup \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_default.log

cache-dit-metrics-cli \
  psnr ssim lpips --summary --cal-speedup \
  --ref-img-dir ./tmp/DrawBench200_DBCache/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_default.log

# DBCache_TaylorSeer_O(1)
# TFLOPs + Latency
cache-dit-metrics-cli \
  clip_score image_reward --summary --cal-speedup \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache_TaylorSeer \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_taylorseer.log

cache-dit-metrics-cli \
  psnr ssim lpips --summary --cal-speedup \
  --ref-img-dir ./tmp/DrawBench200_DBCache_TaylorSeer/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache_TaylorSeer \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_taylorseer.log
