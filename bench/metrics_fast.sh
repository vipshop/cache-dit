# DBCache
# TFLOPs + Latency
cache-dit-metrics-cli \
  clip_score image_reward --summary \
  --cal-speedup --gen-markdown-table \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache_Fast \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_fast.log

cache-dit-metrics-cli \
  psnr ssim lpips --summary \
  --cal-speedup --gen-markdown-table \
  --ref-img-dir ./tmp/DrawBench200_DBCache_Fast/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache_Fast \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_fast.log

# DBCache_TaylorSeer_O(1)
# TFLOPs + Latency
cache-dit-metrics-cli \
  clip_score image_reward --summary \
  --cal-speedup --gen-markdown-table \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache_TaylorSeer_Fast \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_taylorseer_fast.log

cache-dit-metrics-cli \
  psnr ssim lpips --summary \
  --cal-speedup --gen-markdown-table \
  --ref-img-dir ./tmp/DrawBench200_DBCache_TaylorSeer_Fast/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache_TaylorSeer_Fast \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_taylorseer_fast.log
