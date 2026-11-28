# `cache_dit.generate` option reference

Source of truth: `src/cache_dit/cli/entrypoint.py` (`python3 -m cache_dit.generate --help`) and the
per-family specs in `src/cache_dit/cli/{params,cache,quantize,parallel,attention}.py`. Regenerate the
tables after changing any of them.

## Entry and task selection

```bash
python3 -m cache_dit.generate [task] [example] [options]
cache-dit-generate [task] [example] [options]
```

| Argument | Default | Meaning |
|---|---|---|
| `task` (positional) | `generate` | `list`, a registered example name (e.g. `flux`), or a model directory / repo id |
| `example` (positional) | None | Second positional for the `generate <example>` form |
| `--task` | None | Force the task when a model path matches several examples: `T2V, T2VA, I2V, T2I, IE2I, FLF2V, VACE, FL2VA, REF2VA, T2A, ANIMATE, S2V` |
| `--example-summary`, `--esummary` | False | Log the example configuration before running |
| `--model-path` | None | Override the example/model path |
| `--controlnet-path` | None | Override ControlNet weights |
| `--lora-path` | None | Override LoRA weights |
| `--transformer-path` | None | Override the transformer checkpoint |
| `--config-path`, `--config` | None | CacheDiT YAML configuration |

Model/example resolution: an exact registered name wins; otherwise the selector is matched as a model
path against the registered default models. `list` prints every registered example.

## Inputs

| Flag | Default | Meaning |
|---|---|---|
| `--prompt` | None | Override the example prompt |
| `--negative-prompt` | None | Override the negative prompt |
| `--skip-negative_prompt`, `--skip-neg` | False | Drop the negative prompt even if provided |
| `--image-path` | None | Input image (I2V / edit / VACE) |
| `--mask-image-path` | None | Inpainting mask |
| `--last-image-path` | None | Last frame for first/last-frame conditioning |
| `--driving-video-path` | None | Driving video for character animation |
| `--driving-video-fps` | None | Resample rate for the driving video |
| `--audio-path` | None | Driving audio for speech-to-video |
| `--reference-image-path ...` | None | One or more image references (order: image → video → audio) |
| `--reference-video-path ...` | None | One or more video references |
| `--reference-audio-path ...` | None | One or more audio references |
| `--prompt-enhancer` | None | Local prompt-enhancer repository; examples that own the step expand the prompt |
| `--extra-input-kwargs` | None | Python dict literal passed to the pipeline; keys override defaults |
| `--input-height`, `--input-width` | None | Input image geometry |

`--extra-input-kwargs` needs a Python literal inside shell quotes, e.g.
`--extra-input-kwargs "{'cfg_normalization': False, 'max_sequence_length': 1536}"`.

## Generation and output

| Flag | Default | Meaning |
|---|---|---|
| `--num_inference_steps`, `--steps` | example default | Denoising steps |
| `--warmup` | 1 | Warmup runs before timing |
| `--warmup-num-inference-steps`, `--warmup-steps` | None | Steps used only during warmup |
| `--warmup-seed` | None | Seed used only during warmup (useful for few-shot SVDQ) |
| `--warmup-prompt` | None | Prompt used only during warmup |
| `--repeat` | 1 | Timed repeats |
| `--height`, `--width` | example default | Output geometry |
| `--num-frames`, `--frames` | example default | Video frame count |
| `--seed` | example default | Sampling seed |
| `--generator-device`, `--gen-device` | rank accelerator, else CPU | Device for `torch.Generator`; CPU sampling is not reproducible across torch versions |
| `--save-path` | None | Output image/video path |
| `--saved-fps`, `--fps` | 8 | FPS used when exporting video |

## Cache and calibrator

`--cache-params` / `--cache-config` — optional `type` (also accepted as the first bare entry):
`DBCache` (default), `SeaCache`, `SoftCache`.

| Key | Default | Meaning |
|---|---|---|
| `Fn` / `Fn_compute_blocks` | 1 | First N blocks always computed |
| `Bn` / `Bn_compute_blocks` | 0 | Last N blocks always computed |
| `rdt` / `residual_diff_threshold` | 0.12 | Cache reuse threshold |
| `ws` / `max_warmup_steps` | 8 | Steps always computed at the start |
| `wi` / `warmup_interval` | 1 | Warmup interval |
| `mc` / `max_cached_steps` | -1 | Max cached steps (-1 = unlimited) |
| `mcc` / `max_continuous_cached_steps` | 3 | Max consecutive cached steps |
| `scm` / `mask_policy` | None | Steps mask policy: `slow`, `medium`, `fast`, `ultra` |
| `steps_mask` | False | Enable the steps computation mask |
| `seacache` | False | Enable SeaCache |
| `seacache_mode` | `flow` | `flow` or `vp` |
| `seacache_feature_shape` | None | `HxW` for images, `FxHxW` for video |
| `seacache_power_exp` | 2 | SEA probe exponent |
| `softcache` | False | Enable SoftCache |
| `softcache_tau` | 0.5 | SoftCache threshold |
| `softcache_jf_max` | 8 | SoftCache JF cap |

`--calibrator-params` / `--calibrator-config` — requires `--cache-params`; `type` can be the first
bare entry.

| Type | Keys |
|---|---|
| `taylorseer` | `order` |
| `dmd` | `history`, `rank`, `ridge`, `svd_precision` |
| `foca` | — |
| `svd` | `tau`, `beta`, `max_rank`, `predictor`, `order` |
| `hicache` | `order`, `sigma` |
| `brace` | `capacity`, `gamma` |
| `bezicast` | `history`, `control_points`, `ridge`, `guardrail`, `u_max`, `beta_max`, `rho`, `gamma` |
| `residual_motion` | `beta` |

Every calibrator also accepts `calibrator_cache_type=residual|hidden_states` and
`encoder=true|false`. Long names such as `svd-calibrator` are normalized.

## Quantization

`--quantize-params` / `--quantize-config` — any key except `text_encoder` enables transformer
quantization; omitting `quant_type` means `float8_per_row`.

| Key | Meaning |
|---|---|
| `quant_type` | `float8_per_row` (default), `float8_per_tensor`, `float8_per_block`, `int8_per_row`, `int8_per_tensor`, `nvfp4`, `mxfp4_weight_only`, `svdq_int4_r{32,64,128,256}_dq`, `svdq_nvfp4_r{32,64,128}_dq`, `svdq_fp8_r{32,64,128}_dq`, `svdq_w4a8_*`, `svdq_w6a6_*`, `svdq_fp6_*`, `svdq_w1a4_*`, `svdq_w2a4_*`, `svdq_w3a4_*`, ... |
| `backend` | `auto` (default), `torchao`, `cache_dit` |
| `regional_quantize` | Per-region quantization switch |
| `per_tensor_fallback` | Fallback unsupported layers to FP8 per-tensor |
| `exclude_layers` | `+`-separated substrings to skip |
| `text_encoder` | `true` (reuse `quant_type`) or a separate text-encoder quant type |
| `verbose` | Per-layer logging |
| `fp8_fused_mlp`, `fp4_fused_mlp` | Fused MLP epilogues |
| `mxfp4_quantize_device` | `auto` (default), `cpu`, `cuda` |

`--svdq-params` / `--svdq-config` — only with an `svdq_*` `quant_type`.

| Key | Meaning |
|---|---|
| `runtime_kernel` | `native` (default), `cute`, `flashinfer`, `flashinfer+cute-dsl`, `flashinfer+cutlass-sm120` |
| `lora_layout` | `native` or `row_major`; `row_major` requires `runtime_kernel=cute` |
| `smooth_strategy` | `identity` (default), `activation`, `weight`, `weight_inv`, `few_shot` |
| `calibrate_precision` | `low` (default), `medium`, `high` |
| `few_shot_steps`, `few_shot_relax_factor`, `few_shot_relax_top_ratio`, `few_shot_relax_strategy`, `few_shot_auto_compile` | Few-shot DQ controls |
| `fused_mlp` | Fuse GELU/SwiGLU epilogues |
| `layerwise_offload`, `async_transfer`, `transfer_buckets`, `prefetch_limit`, `max_copy_streams`, `max_inflight_prefetch_bytes`, `persistent_buckets`, `persistent_bins` | SVDQ layerwise collection; omitted keys inherit `--layerwise-*` |
| `offload_quantized_layers_to_cpu`, `defer_move_to_execution_device`, `quantize_device` | Placement |
| `streaming`, `activation_buffer_flush_sample_count`, `activation_buffer_flush_cpu_bytes` | Streaming calibration |
| `svd_refine_iters`, `packed_a4_stages`, `packed_a4_inner_group` | Packed-A4 controls |

`--disable-per-row` / `--no-per-row` turns off per-row quantization for the transformer.

## Attention and communication

`--attn-params` / `--attn-config` — `type` can be the first bare entry. Backends:
`flash`, `flash_varlen`, `_flash_3`, `native`, `_native_cudnn`, `_sdpa_cudnn`, `sage`, `sage3`,
`sol_attn`, `ffpa`, `ffpa_fp8`, `ffpa_fp4`, `ffpa_fp8_per_block`, `_native_npu`, `_npu_fia`,
`_mindiesd_laser`.

| Backend | Extra keys |
|---|---|
| `sol_attn` | `tau`, `thresh_type=diag|exact`, `strict`, `sink_tokens` |
| `ffpa` | `hybrid`, `hybrid_n_early` |
| `ffpa_fp8`, `ffpa_fp8_per_block` | `hybrid`, `hybrid_n_early`, `fp8_hadamard`, `fp8_smooth_v` |
| `ffpa_fp4` | `hybrid`, `hybrid_n_early`, `fp4_hadamard`, `fp4_pv_mm_type=fp4|fp8`, `fp4_smooth_v` |

`--quant-communicate-params` (aliases `--quant-communicate-config`, `--quant-comm-params`,
`--quant-comm-config`) — requires Ulysses `> 1`.

| Key | Default | Meaning |
|---|---|---|
| `bits` | 8 | `8`, `6`, or `4` |
| `quant_type` | `per_group_16` | `per_token`, `per_group_16`, `per_group_32`; FP4 only supports G16 |
| `quantize_k` | True | Quantize K in addition to Q/V/O |

## Parallelism

`--parallel-params` / `--parallel-config` — degrees multiply to the number of processes.

| Key | Default | Meaning |
|---|---|---|
| `ulysses` | 1 | Ulysses context parallel degree |
| `ring` | 1 | Ring context parallel degree |
| `tp` | 1 | Tensor parallel degree |
| `vae` | False | Add the VAE to the parallel modules |
| `text` | False | Add the text encoder (TE-P) |
| `controlnet` | False | Add ControlNet |
| `local_weight_shards` | False | Initialize supported MiniMax-H3 video TP and Qwen3-VL TE-P weights from CPU slices |
| `ulysses_anything` | False | Uneven Ulysses split (UAA) |
| `ulysses_async` | False | Async Ulysses |
| `ring_rotate_method` | `p2p` | `p2p` or `allgather` |
| `ring_convert_to_fp32` | True | Keep Ring rotation in fp32 |

Component keys are modifiers: a value with only `vae=true`, `text=true`, `controlnet=true` or
`ulysses_anything=true` builds no parallel config and is a silent no-op. Degree-1 axes are omitted.

`--launcher` accepts `auto`, `ray`, and `local`. The default `auto` uses Ray if installed and local
processes otherwise. CLI auto-launch reads degrees from `--parallel-params` or `--parallel-config`.
Use external workers for YAML-only multi-GPU configuration and set `local_weight_shards` under
`parallelism_config` when passing `--config`.
[Multi-GPU examples](../../../../docs/EXAMPLES.md#multi-gpu)

## CPU offload

The five primary switches are mutually exclusive.

| Flag | Meaning |
|---|---|
| `--cpu-offload`, `--cpu-offload-model` | diffusers model CPU offload |
| `--sequential-cpu-offload` | diffusers sequential CPU offload |
| `--module-layerwise-cpu-offload`, `--layerwise-offload` | cache-dit generic layerwise offload |
| `--adaln-host-residency` | Keep the transformer resident, stream AdaLN tables from pinned host memory |
| `--adaln-online` | Precompute AdaLN tables for a fixed grid and park the projections on the host |

| Knob | Default | Meaning |
|---|---|---|
| `--layerwise-async-transfer` | False | Stream-based async transfer |
| `--layerwise-text-async-transfer` / `--no-layerwise-text-async-transfer` | inherit / force off | Text-encoder override |
| `--layerwise-transfer-buckets` | 1 | Prefetch depth hint |
| `--layerwise-text-transfer-buckets` | inherit | Text-encoder override |
| `--layerwise-prefetch-limit` / `--no-layerwise-text-prefetch-limit` | off / force off | Future-prefetch target cap |
| `--layerwise-max-copy-streams` | auto | Async copy-stream cap |
| `--layerwise-max-inflight-prefetch-bytes` | unlimited | Residency budget (`512MiB`, `4GiB`, ...) |
| `--layerwise-persistent-buckets` | 0 | Targets kept resident for the handle lifetime |
| `--layerwise-persistent-bins` | 1 | Distribution bins for persistent targets |

## Compile and graphs

| Flag | Default | Meaning |
|---|---|---|
| `--compile` | False | Compile the transformer (repeated blocks by default) |
| `--disable-compile-repeated-blocks`, `--disable-compile-blocks`, `--no-regional-compile`, `--no-rc` | False | Disable repeated-block compile |
| `--force-compile-dynamic` | False | Force dynamic shapes |
| `--cuda-graph` | False | Capture CUDA Graph; use with `--compile` |
| `--compile-full-graph`, `--full-graph` | False | Require a full graph; incompatible with graph breaks such as offload/streaming |
| `--compile-vae` | False | Compile the VAE |
| `--compile-text-encoder`, `--compile-text` | False | Compile the text encoder |
| `--compile-controlnet` | False | Compile ControlNet |
| `--max-autotune`, `--tune` | False | `max-autotune` mode |

`--adaln-host-residency` rejects `--cuda-graph` and `--compile-full-graph` while the AdaLN tables are
actually streamed (no transformer quantization and no `tp`).

## Diagnostics and memory

| Flag | Default | Meaning |
|---|---|---|
| `--cache-summary`, `--summary` | False | Log cache summary |
| `--track-memory`, `--mem` | False | Report peak GPU memory |
| `--profile` | False | Enable `torch.profiler` |
| `--profile-name` | None | Profile session name |
| `--profile-dir` | None | Output directory |
| `--profile-activities` | `CPU GPU` | Any of `CPU`, `GPU`, `MEM` |
| `--profile-with-stack` | True | Record stacks |
| `--profile-record-shapes` | True | Record tensor shapes |
| `--disable-fuse-lora` | None | Skip `fuse_lora` even when LoRA weights are given |
| `--device-map-balance`, `--device-map` | False | Balanced device map when multiple GPUs exist |
| `--vae-tiling`, `--vae-slicing` | False | Low-memory VAE decoding |

## Removed forms

Do not use: `--cache`, `--Fn`, `--Bn`, `--rdt`, `--parallel`, `--parallel-type`, `--parallel-vae`,
`--parallel-text`, `--parallel-controlnet`, `--ulysses-anything`, `--uaa`, `--ulysses-async`,
`--uaqkv`, `--ring-rotate-method`, `--rotate`, `--ring-no-convert-to-fp32`, `--no-fp32`,
`--quantize`, `--q`, `--quantize-type`, `--quantize-backend`, `--attn`, `--calibrator`,
`--taylorseer`, `--dmd`, `--foca`, `--svd`, `--hicache`, `--steps-mask`, `--mask-policy`.

They are covered by the `--xxx-params` lists and `--xxx-config` aliases above. The convert CLI keeps
its own `--quant-type`; that flag is unrelated and still valid.
