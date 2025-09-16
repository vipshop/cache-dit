# DBCache
# TFLOPs + Latency
cache-dit-metrics-cli \
  clip_score image_reward --summary \
  --cal-speedup --gen-markdown-table \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache_Distill \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_distill.log

cache-dit-metrics-cli \
  psnr ssim lpips --summary \
  --cal-speedup --gen-markdown-table \
  --ref-img-dir ./tmp/DrawBench200_DBCache_Distill/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache_Distill \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_distill.log
