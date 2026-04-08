# FLUX.2-klein-4B SVDQ PTQ example report

This report compares one deterministic validation prompt across the baseline, memory-quantized, reloaded, and compiled quantized transformer stages.

## Run metadata

| field | value |
| :---: | :---: |
| model_source | /workspace/dev/vipdev/hf_models/FLUX.2-klein-4B |
| output_dir | /workspace/dev/vipshop/cache-dit/examples/ptq/FLUX.2-klein-4B-svdq |
| quant_type | svdq_int4_r32 |
| torch_dtype | bfloat16 |
| prompts_path | /workspace/dev/vipshop/cache-dit/examples/data/prompts/DrawBench200.txt |
| calibration_prompt_count | 2 |
| validation_prompt | A white rabbit on the beach, minimalist illustration. |
| height | 256 |
| width | 256 |
| num_inference_steps | 2 |
| calibration_height | 256 |
| calibration_width | 256 |
| calibration_steps | 2 |
| benchmark_runs | 1 |
| seed | 0 |
| quantization_time_s | 95.6097 |
| checkpoint_size_gb | 1.9128 |

## Stage metrics

| stage | run_count | avg_latency_s | total_latency_s | peak_memory_gb | transformer_weight_cuda_gb | status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| baseline | 1 | 0.9942 | 0.9942 | 15.1851 | 7.2188 | ok |
| memory_quantized | 1 | 0.1954 | 0.1954 | 10.2514 | 2.2761 | ok |
| loaded_quantized | 1 | 0.1542 | 0.1542 | 10.2499 | 2.2761 | ok |
| compiled_quantized | 1 | 0.2449 | 0.2449 | 10.2499 | 2.2761 | ok (warmup_latency_s=127.1302) |

## PSNR comparisons

| comparison | psnr | max_abs_diff | mean_abs_diff | status |
| :---: | :---: | :---: | :---: | :---: |
| baseline vs memory_quantized | 33.0433 | 255 | 8.6627 | ok |
| baseline vs loaded_quantized | 30.6256 | 255 | 10.3064 | ok |
| baseline vs compiled_quantized | 32.9000 | 255 | 7.8516 | ok (warmup_latency_s=127.1302) |

## Artifacts

| artifact | path |
| :---: | :---: |
| checkpoint | checkpoint/svdq_int4_r32.safetensors |
| quant_config_json | checkpoint/quant_config.json |
| baseline_image | images/baseline.png |
| memory_quantized_image | images/memory_quantized.png |
| loaded_quantized_image | images/loaded_quantized.png |
| compiled_quantized_image | images/compiled_quantized.png |
| comparison_grid | images/comparison_grid.png |
| report | reports/report.md |
