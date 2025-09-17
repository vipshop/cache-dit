# DBCache
# 8 Steps
cache-dit-metrics-cli \
  clip_score image_reward --summary \
  --cal-speedup --gen-markdown-table \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache_Distill_8_Steps \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_distill_8_steps.log

cache-dit-metrics-cli \
  psnr ssim lpips --summary \
  --cal-speedup --gen-markdown-table \
  --ref-img-dir ./tmp/DrawBench200_DBCache_Distill_8_Steps/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache_Distill_8_Steps \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_distill_8_steps.log


# 4 Steps
cache-dit-metrics-cli \
  clip_score image_reward --summary \
  --cal-speedup --gen-markdown-table \
  --ref-prompt-true ./prompts/DrawBench200.txt \
  --img-source-dir ./tmp/DrawBench200_DBCache_Distill_4_Steps \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_distill_4_steps.log

cache-dit-metrics-cli \
  psnr ssim lpips --summary \
  --cal-speedup --gen-markdown-table \
  --ref-img-dir ./tmp/DrawBench200_DBCache_Distill_4_Steps/C0_Q0_NONE \
  --img-source-dir ./tmp/DrawBench200_DBCache_Distill_4_Steps \
  --perf-tags "Mean pipeline TFLOPs" "Mean pipeline time" \
  --perf-log ./log/cache_dit_bench_distill_4_steps.log
