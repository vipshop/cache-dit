# Audio Models — TTS / text-to-audio DiT integration

When to read this: the target model produces **audio instead of images or video frames** — a TTS
(text-to-speech) or text-to-audio diffusion transformer, e.g. LongCat-AudioDiT. Read it **before**
the Cache chapter: the caching mechanics are nearly unchanged, but the request shape, the parallel
payoff, the CLI output path and the acceptance criteria are all different.

Worked case, integrated and measured: **LongCat-AudioDiT** (Meituan LongCat, 1B / 3.5B,
waveform-latent flow-matching TTS), 2026-10-29. Treat the *shapes of the problems* as representative
of audio-only DiTs, not the specific numbers as model-independent constants.

---

## 1. What is actually different from an image / video DiT

| Axis | Image / video DiT | Audio-only DiT (TTS) | Consequence |
|---|---|---|---|
| Latent shape | `(B, C, H/p, W/p)` or `(B, C, T, H/p, W/p)`, patchified | **`(B, T, latent_dim)`** — one token per waveform chunk, no patchify, no grid | No spatial axes, no `img_ids`/`txt_ids`, no patch-embed geometry |
| Sequence length | 4k–64k tokens | **58–351** (5–30 s at 11.72 latent frames per second) | Compute per step is tiny → launches and Python overhead dominate |
| Duration | fixed by `height`/`width`/`num_frames` flags | a **request parameter** (`audio_duration_s`), seq len changes run to run | Masks, CP divisibility and cache shapes must be derived per call, not baked in |
| Text conditioning | cross-attn over a long caption, often re-projected per block | cross-attn over ~20 tokens, projected **once outside the block loop** (`text_embed` + `text_conv_layer` + `adaln_global_mlp`) | The block loop carries only loop-invariant extras → very cache-friendly |
| Guidance | usually one forward (fused CFG) or two | **two full forwards per step** (cond, then null when `guidance_scale > 1`) | `has_separate_cfg` must be inferred from the pipeline's default guidance |
| Decoder | 2-D / 3-D VAE, typically a large share of runtime (not measured in this task) | **1-D convolutional decoder, ~2 %** of runtime (measured) | VAE-P is almost always not worth implementing (§4.7) |
| Output | `PIL`/frames → png/mp4 | **waveform `(channels, samples)`** → wav | The generate CLI has no audio writer until you add one (§4.3) |
| Acceptance | PSNR / SSIM on pixels, ≥30 dB / ≥0.90 | log-mel PSNR/SSIM + SI-SDR, **with a model-specific noise floor** | See §6 — the image thresholds do not transfer |

Two of these deserve emphasis because they invalidate common assumptions:

1. **The workload is latency-bound, not FLOP-bound.** LongCat-AudioDiT 1B runs 16 steps × 2 CFG =
   32 forwards over 117 tokens in 0.71 s on a PRO 5000; the 3.5B model (3.5× the DiT parameters:
   0.98B → 3.40B) needs only 0.90 s. Parallelism buys almost nothing and often loses (§5). Do not
   promise a speedup before measuring, and do not call TP/CP "supported" only because they produce
   audio.
2. **The model is chaotic across denoising steps.** Any kernel-level rounding difference changes the
   sampling trajectory, so the waveform is a *different, equally valid* sample. Measured on
   LongCat-AudioDiT: swapping only the attention backend (`_sdpa_cudnn` vs `native`) drops mel PSNR
   to 20.9 dB / SSIM 0.750, and `torch.compile` is not reproducible across processes. Every
   acceptance number therefore needs a **same-configuration rerun** next to it as the noise floor.

## 2. Gate-check additions (before the plan)

Besides the skill's standard gate (local path + pipeline/transformer source), for audio confirm:

- The **latent frame rate**: `sample_rate / vae_downsampling_ratio` (LongCat: `24000 / 2048` =
  11.72 frames/s). It decides seq len, CP divisibility and the cache workload.
- The **duration cap** in the pipeline (LongCat hardcodes `max_wav_duration = 30.0`; the official
  repo config says 60, but that value does not survive `from_pretrained` because the attribute is not
  a registered pipeline config, so the effective cap is always 30 s). Pick a test duration well under
  it.
- Whether **one or two transformer forwards** run per step (guidance), and whether that count is
  constant. `guidance_scale <= 1` must flip `has_separate_cfg` off, otherwise cache-dit halves the
  counted steps for the cond/uncond parity.
- Whether the transformer **returns a tensor or a dataclass** — it decides where CP can split (§4.4a).
- Whether a duration mask is passed to the transformer, and whether it is ever padded (§4.4c).

## 3. The reference architecture (LongCat-AudioDiT)

Upstream code (diffusers ≥ 0.38; the 3.5B conversion records `_diffusers_version 0.39.0.dev0`):

```
diffusers/pipelines/longcat_audio_dit/pipeline_longcat_audio_dit.py
diffusers/models/transformers/transformer_longcat_audio_dit.py   # LongCatAudioDiTTransformer
diffusers/models/autoencoders/autoencoder_longcat_audio_dit.py   # LongCatAudioDiTVae
```

Pipeline: `UMT5EncoderModel` (+ `layer_norm` + first hidden state, `text_norm_feat`/`text_add_embed`)
→ `duration = int(audio_duration_s * 24000 // 2048)` → latents `(B, duration, 64)` → `FlowMatchEuler`
with `sigmas = linspace(1.0, 1/n, n)` → per step `transformer(...)` for cond and (if guidance > 1)
null → `vae.decode(latents.permute(0, 2, 1))` → `(B, 1, samples)`.

Transformer (inside one forward, all loop-invariant work happens **once**):

```
hidden_states (B,T,64) --input_embed--> (B,T,dim)
latent_cond   (B,T,64) --latent_embed--> (B,T,dim) --cat--> latent_cond_embedder
text (B,L,768) --text_embed--> --text_conv_layer (4x ConvNeXtV2)-->
rope = rotary_embed(hidden_states, T)          # full table, positions = frame index
cond_rope = rotary_embed(text, L)
adaln_global_out = adaln_global_mlp(timestep_embed + text_mean)
for block in self.blocks:  AudioDiTBlock(hidden_states=..., timestep_embed=..., cond=...,
                                        mask=..., cond_mask=..., rope=..., cond_rope=...,
                                        adaln_global_out=...)          # keywords only
hidden_states = hidden_states + residual        # long skip
norm_out(adaln) -> proj_out(dim -> 64) -> * attention_mask
```

`AudioDiTBlock` = self-attn (RoPE, QK-RMSNorm) + cross-attn over the projected text + a plain GELU
MLP FFN (`ff.0` Linear → GELU(tanh) → Dropout → `ff.3` Linear), all modulated by a per-step
`adaln_global_out` that the transformer computes once. The only ConvNeXtV2 blocks sit in the text
branch (`text_conv_layer`).

## 4. Integration recipe

### 4.1 Weights: check the checkpoint before writing a converter

The official `meituan-longcat/LongCat-AudioDiT-1B|3.5B` ships **one `model.safetensors` whose keys are
already diffusers-named** (`transformer.*`, `vae.*`, `text_encoder.*`). Getting a diffusers layout is
therefore not a weight-renaming exercise: split by prefix and add the only alias
`text_encoder.encoder.embed_tokens.weight -> shared.weight`. Cross-checked against the community
conversions (`ruixiangma/...-1B-Diffusers`, `saddam213/...-3.5B-diffusers`) by file sizes: 1B splits
into `3.928 / 0.624 / 1.127 GB` (transformer / vae / text_encoder, summing to the 5.68 GB official
single file), and the 3.5B transformer is `13.58 GB`. Those conversion repos are also where the small
`config.json`s / tokenizer can be taken from. Seed `config.json` values to sanity-check: 1B =
`dit_dim 1536 / depth 24 / heads 24`, 3.5B = `2560 / 32 / 32 / ff_mult 3.6`, both `latent_dim 64`,
`dit_text_dim 768`.

Download note: the HF mirror is heavily throttled for these repos (≈0.2–1 MB/s, 0.95 MB/s measured on
a single stream) while ModelScope serves the official checkpoints at ≈5–8 MB/s. Take the big
`model.safetensors` from ModelScope and only the few-KB configs from the mirror
(`./model_download.md`).

### 4.2 Cache: a plain `Pattern_3` adapter, no PatchFunctor

`AudioDiTBlock.forward(hidden_states, timestep_embed, cond, mask, cond_mask, rope, cond_rope,
adaln_global_out) -> Tensor` is called with **keywords only**, and every extra argument is computed
outside the loop. cache-dit's `CachedBlocks_Pattern_3_4_5.forward(hidden_states, *args, **kwargs)`
binds `hidden_states` by name and replays the rest through `**kwargs`, so the wrapper is transparent.
The real adapter (snippet elides the `try/except ImportError` guard that keeps the module importable
on older diffusers):

```python
@BlockAdapterRegister.register("LongCatAudioDiT")
def longcat_audio_dit_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import LongCatAudioDiTTransformer
  _relaxed_assert(pipe.transformer, LongCatAudioDiTTransformer)
  # The pipeline always runs the conditional forward and runs the null forward only when
  # `guidance_scale > 1` (default 4.0), so the bookkeeping has to follow the reachable default.
  guidance_scale = _default_call_float(pipe, "guidance_scale")
  has_separate_cfg = True if guidance_scale is None else guidance_scale > 1.0
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.blocks,          # NOT `transformer_blocks`
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    has_separate_cfg=has_separate_cfg,       # `guidance <= 1` runs a single forward per step
    **kwargs,
  )
```

Checklist: the registration name must not prefix-collide (never register a bare `LongCat*`), and the
`guidance_scale` used for `has_separate_cfg` must be the *pipeline default* — the attribute only
exists after `__call__` starts, which is why the default is read from `inspect.signature`.

### 4.3 CLI: audio-only output needs a new task type

`ExampleOutputData` already carried `audio` / `audio_sampling_rate`, but `save()` only wrote it when
`video is not None`, `_default_save_path()` never produced a `.wav`, and `ExampleType` had no audio
task, so a pure-audio pipeline silently dropped its result. What had to be added:

- `ExampleType.T2A` (+ an `_AUDIO_TASKS` set, disjoint from `_AUDIO_VIDEO_TASKS`);
- `save()` → `_save_audio()`: lazy `soundfile` import, `(channels, samples)` transposed to
  `(frames, channels)`, rate taken from `audio_sampling_rate` — filled by `parse_audio_output` from
  `pipe.sample_rate`;
- `_default_save_path()` → `f"{model_tag}.{strify_tag}.wav"`;
- `parse_audio_output(output, task, pipe)` → `{"audio": output.audios[0], "audio_sampling_rate": ...}`
  (works for `output_type="np"` and `"pt"`);
- `Example.run()` branch + the example itself, with the duration passed through
  `--extra-input-kwargs '{"audio_duration_s": 10.0}'` (there is no dedicated CLI field).

### 4.4 CP: three audio-specific obstacles

LongCat-AudioDiT is the first audio-only CP integration; the obstacles generalize.

**(a) `forward` returns a dataclass.** The split/gather hook of a root plan (`""`) validates the
module output and raises on anything that is not a tensor / tuple of tensors. Split the **embedder
outputs** instead, and gather at the projection that restores the latent layout:

```python
{
  "input_embed":  {0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)},
  "latent_embed": {0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)},
  "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
}
```

Both embedders must use the same sharder: their outputs are concatenated along the feature axis, so
different boundaries would mis-pair frames. Everything downstream (long-skip residual, rope, block
outputs) is then already local.

**(b) RoPE is rebuilt per forward from the local length.** After (a), rank `k` sees `seq_len_local`
frames, so `rotary_embed(hidden_states, T_local)` would restart positions at 0 and every rank would
share rank 0's positions. Keep the module but make it return the **full table**, and slice
`[offset : offset + local_len]` inside the processors (`offset = rank * local_len` for an
equipartition split, the prefix sum of all gathered sizes for `--ulysses-anything`; cache that
prefix sum **keyed by the whole size vector** — a local length alone is ambiguous, 17 frames and 16
frames both give rank 1 a length of 8).

**(c) Cross attention must stay local, and the duration mask must go.** The text stream is replicated,
so the cross-attn processor must null both `_parallel_config` and its cache-dit alias `_cp_config`
(same object in cache-dit's dispatcher); letting Ulysses run there all-to-alls the text on every rank
and the attention dies on a K/mask length mismatch. The self-attn keeps the CP config, because it owns
the sharded latent sequence. The duration mask (`(B, T)` bool) is used at four points — `input_embed`,
`latent_embed`, the cross-attn post-mask and the trailing multiply, all of which see the local shard,
plus the self-attn, which sees the gathered global sequence — so a single mask tensor cannot serve
them. Assert it is all-True (the pipeline derives it from the duration and never pads) and pass `None`.

Also reject a mixed ring×ulysses mesh explicitly: the latent shard is a flat chunk of the flattened
mesh, so a per-mesh offset would repeat positions across the ring dimension.

### 4.5 TP: two attributes and one norm

```python
shard_div_attr(block.self_attn, "heads", tp_size)
shard_div_attr(block.self_attn, "inner_dim", tp_size)   # processors do head_dim = inner_dim // heads
# self_attn / cross_attn (both): to_q / to_k / to_v -> ColwiseParallel(); to_out.0 -> RowwiseParallel()
# ffn.ff.0 -> ColwiseParallel(); ffn.ff.3 -> RowwiseParallel()
# norm_out.linear -> ColwiseParallel(output_layouts=Replicate())   # output is chunk(2)
# adaln_global_mlp / adaln_scale_shift / text_conv_layer (Conv1d+GRN): keep replicated
```

Dividing only `heads` doubles the effective head dimension and garbles the audio. Shard the cross
attention too — its projections are the same four Linears, and skipping them gives a numerically
correct plan that halves the savings. The QK `RMSNorm` normalizes the full `inner_dim`, so after a
column-wise shard it needs a **distributed** norm: reduce the mean square with `ReduceOp.AVG` over
the TP group and slice the weight with the same chunking. Every `_shard_attention` call replaces both
`q_norm` and `k_norm`, for self-attn and cross-attn alike.

Tiny-model parity is the cheapest proof (a single forward, no denoising): divisible CP is **bit-exact**
(`max_abs_diff = 0`), `--ulysses-anything` on 17 frames is `O(1e-5)` relative, TP is `O(1e-6)`
relative. The parity script and the assertions live in
`tests/parallelism/test_longcat_audio_dit_cp_tp_tiny.py`; record its printed numbers in the report.

### 4.6 Quantization / offload / compile

- **`float8_per_row` requires bf16 weights** — load the checkpoint in bf16 or the backend refuses.
- Quantize/compile discover the stack from `_repeated_blocks` (LongCat declares `["AudioDiTBlock"]`);
  confirm the `Compiling repeated blocks` log line, otherwise you are timing a whole-module compile.
- A FFN wrapped in `nn.Sequential` is still found by `named_modules()`, but a class not named
  `FeedForward` misses svdq's fused-GELU path — a lost opportunity, not a bug.
- **Two configurations are bit-exact and are the correctness anchors**: layerwise offload
  (`SI-SDR = inf`, output file byte-identical to the baseline) and cache with `rdt=0`. Every *speed*
  configuration (compile, cache at the shipped threshold, TP/CP, quantization) is not, so a metric
  drop there has to be read against the noise floor (§6). Offload is a memory tool: on a 2–3 GB model
  with a 72 GB card it is a pure slowdown (2.4–2.8 s vs 0.71 s), async transfer included.

### 4.7 VAE-P: measure before implementing

Compute the decode share first (a `component_timing.py`-style probe: time `vae.decode` on a fixed
latent and compare against the full pipeline). LongCat-AudioDiT: **0.014 s of 0.688 s = 2.0 %**
(measured separately from the table in §5, whose total is 0.71 s) — a perfect 2× VAE-P would save
1 %, while an audio VAE is a 1-D convolutional decoder that would need overlapping time-shard
stitching. Record "N/A + measured share" in the docs instead of writing a planner. Also do not list
`vae` in `extra_parallel_modules`: the module path matches the "parallelizable VAE" probe and an
unregistered planner raises at startup.

## 5. Measured reference (LongCat-AudioDiT 1B, 10 s / 16 steps / guidance 4.0, PRO 5000 72 GB)

Metrics from `cache_dit.metrics` (§6). Latency = `--warmup 1` then the **mean of 3** runs, taken from
the CLI's `Inference Time` (host clock; the CUDA-event median in §4.7 is a separate probe).

| Config | Latency | Speedup | mel PSNR | mel SSIM | SI-SDR |
|---|---|---|---|---|---|
| single-GPU bf16 (reference) | 0.71 s | 1.00× | – | 1.000 | – |
| + compile | 0.32 s | 2.22× | 33.42 | 0.963 | 4.94 |
| + cache (rdt 0.12, 4/16 steps hit) | 0.57 s | 1.25× | 39.83 | 0.985 | 14.02 |
| + cache (rdt 0.30) | 0.49 s | 1.45× | 31.93 | 0.931 | 8.07 |
| + cache + compile | 0.26 s | 2.73× | 19.58 | 0.702 | −8.26 |
| TP2 / TP2+compile / TP2+TE-P | 2.81 / 0.73 / 2.86 s | 0.25–0.97× | 19.5 / 19.6 / 27.0 | 0.69–0.90 | −8.1 / −7.9 / 4.7 |
| CP2 (UAA) / CP2+compile / CP2+cache | 1.69 / 1.30 / 1.30 s | 0.42–0.55× | 21.8 / 21.6 / 21.8 | 0.76–0.77 | −5.0 |
| **CP2 on 5 s (58 frames, no UAA)** | 1.14 s | 0.62×¹ | **34.32** | **0.967** | 13.69 |
| float8_per_row + compile | 0.44 s | 1.61× | 21.48 | 0.731 | −5.57 |
| svdq nvfp4 r32 + compile | 0.49 s | 1.45× | 18.97 | 0.610 | −10.32 |
| layerwise offload (sync / +async) | 2.37 / 2.79 s | 0.30 / 0.25× | inf | 1.000 | inf |
| offload + async + compile | 2.75 s | 0.26× | 35.51 | 0.971 | 8.71 |
| *noise floor: swap attention backend only* | 0.79 s | – | *20.91* | *0.750* | *−4.89* |

¹ denominator is the 10 s baseline (0.71 s); the 5 s baseline measured 0.71 s as well.

3.5B reads the same way: 0.90 s base → 0.74 s with cache (36.08 / 0.962 / 18.79) → 0.54 s with
**compile alone** (36.16 / 0.973 / 11.30). The cache+compile combination was not run on 3.5B.

What the table means, and what to write in the report:

- **Cache is the only lever that pays** on this workload (1.25× at 40 dB, 1.45× at 32 dB).
- **Compile pays on wall-clock** (2.2×) and stays above the noise floor.
- **TP/CP are correctness-verified but fail the §6 accuracy gate at 117 frames** (19.5–21.8 dB /
  0.69–0.76 vs the 20.9 / 0.750 floor) and lose on latency. Report them as
  verified-but-not-recommended, with the divisible-frame CP row (34 dB) as the witness that the
  planner is correct rather than lucky.
- **Quantization sits at or below the floor**; nvfp4 at SSIM 0.610 is measurably below it — flag as
  degraded.
- **cache + compile at SSIM 0.702** is below the floor too: do not recommend the combination.

## 6. Metrics for audio (`cache_dit.metrics`)

An image-style PSNR on the waveform is useless — a phase flip or a one-sample shift destroys it while
nobody hears a difference. cache-dit therefore ships four audio metrics, split by comparison domain:

```python
from cache_dit.metrics import compute_si_sdr, compute_snr, compute_mel_psnr, compute_mel_ssim

si_sdr,    n = compute_si_sdr("base.wav", "cache.wav")     # inf = bit-identical
snr,       n = compute_snr("base.wav", "cache.wav")
mel_psnr,  n = compute_mel_psnr("base_dir", "test_dir")
mel_ssim,  n = compute_mel_ssim("base.wav", "cache.wav")
```

```bash
cache-dit-metrics si_sdr snr mel_psnr mel_ssim -a1 base.wav -a2 cache.wav
cache-dit-metrics all -a1 base_dir -a2 test_dir          # all four audio metrics
```

| Metric | Domain | Definition | Insensitive to | Sensitive to |
|---|---|---|---|---|
| `si_sdr` | waveform | `10log10(‖αs‖²/‖ŝ−αs‖²)`, `α = <ŝ,s>/‖s‖²`, both zero-mean | a global gain | sample-level error / phase |
| `snr` | waveform | `10log10(‖s‖²/‖ŝ−s‖²)` | nothing | gain change and sample-level error |
| `mel_psnr` | 80-dim log-mel | PSNR with the peak = the reference spectrogram's dynamic range | phase, sub-hop shift (one hop or more is **not** compensated) | spectral envelope distance |
| `mel_ssim` | 80-dim log-mel | SSIM on the `(n_mels, frames)` image | global gain, small shifts | time-frequency structure (harmonics, formants, onsets) |

Details that matter when reading the numbers:

- `si_sdr` is **not symmetric** (`audio_true` is the target) and is gain-compensated, so it answers
  "is this the same samples", not "is this good audio". For a *reconstruction* task (offload, tiled
  decode, vocoder) it is the sharpest gate: identical output gives `inf`.
- `mel_psnr` is only comparable under the same STFT/mel settings — the implementation fixes
  `n_fft=1024`, `hop_length=256`, `n_mels=80`, `log(clamp(1e-5))` and normalizes by the reference
  dynamic range. A scratch script that used `log10` and `1e-10` reads ~2 dB higher for the same pair.
- `mel_ssim` can go **below 0** on a very peaky spectrogram (negative local correlation); do not
  clamp it and do not assume `[0, 1]`.
- Array inputs carry no sample rate and fall back to 24 kHz; file inputs always use their own rate and
  raise on a mismatch rather than silently resampling.
- Directory pairs follow the image convention (same basename) and, **like the image path,
  bit-identical pairs are excluded from the average** (`inf` is not a distance). The one difference is
  the all-identical case: audio returns `inf` together with the pair count instead of `(None, None)`,
  because for audio `inf` is the signal, not a degenerate input. `tests/api/test_metrics_audio.py`
  pins this behaviour.

How to gate an audio integration (the thresholds that survived this task):

| Gate | Criterion | Why |
|---|---|---|
| cache / offload correctness | `si_sdr = inf` (rdt=0, offload) or ≥ 10 dB at the shipped threshold | these configurations should be bit-exact or nearly so |
| parallel / quantized acceptance | mel SSIM ≥ 0.70 **and** no worse than the same-config rerun, mel PSNR within ~2 dB of that rerun | the rerun *is* the noise floor (measured 20.9 dB / 0.750) |
| never | a fixed "PSNR > 30 / SSIM > 0.90" copied from the image path | parallel/quantized rows sit at 19–22 dB / 0.61–0.76 by construction |
| always | listen to one wav per configuration + spectrogram panels | numbers cannot tell "different sample" from "broken decoder" |

A row below the floor is a **failed** row, not an accepted one — do not report "verified" for a
config that fails this table.

Reporting discipline: put speed and metrics in the same rows, print the noise-floor row explicitly,
and state which metric decided the recommendation.

## 7. Pitfalls that only show up on audio models

1. **Judging by waveform SI-SDR alone.** It collapses on any kernel change (below 0 dB here) while
   the audio is fine. Pair it with a spectrogram metric.
2. **A fixed PSNR/SSIM gate copied from images.** See §6; define the gate against a same-config rerun.
3. **Assuming `guidance_scale` is constant.** `has_separate_cfg` must follow the pipeline's actual
   guidance, or the cond/uncond step parity breaks and cache statistics halve.
4. **Splitting the transformer at its entry when `forward` returns a dataclass.** The hook raises;
   split the embedder outputs instead (§4.4a).
5. **Rebuilding RoPE from the local length under CP.** Positions restart at 0 on every rank, so the
   frames are no longer positioned by their global index — the run stays plausible and wrong. Keep
   the full table and slice.
6. **Letting cross attention run through the CP collective.** The replicated text gets concatenated
   per rank; K grows to `2 × text_len` while the mask stays at `text_len` and the attention dies on a
   shape mismatch. Null `_parallel_config` / `_cp_config` for the cross-attn call.
7. **Reusing an image `attention_mask` path under CP.** The all-True duration mask must be dropped
   (assert it), not sharded.
8. **Choosing a test duration that is not divisible by `cp_size`.** 10 s = 117 frames is odd; either
   use `--ulysses-anything` (slightly larger drift) or a divisible duration (5 s = 58, 6 s = 70) when
   the point is to prove CP correctness.
9. **Promising a parallel speedup.** 58–351 tokens are launch-bound; report the negative result.
10. **Writing a VAE-P planner without measuring.** The decoder is 2 % of the run; document N/A.
11. **Forgetting the CLI can only write images and videos.** A pure-audio run "succeeds" and saves
    nothing — add the T2A path first, or the whole matrix is unverifiable.
12. **Downloading the audio checkpoints from the HF mirror.** ≈0.2–1 MB/s vs ≈5–8 MB/s on ModelScope
    for the same bytes.

## 8. Where things live in cache-dit

The paths below are the LongCat-AudioDiT integration (added with the model support itself). For a new
audio model, copy the *shapes* — adapter, planner split points, T2A writer, metric wiring — not the
file names.

| Concern | Path |
|---|---|
| BlockAdapter | `src/cache_dit/caching/block_adapters/adapters.py` (`longcat_audio_dit_adapter`) |
| CP + TP planners | `src/cache_dit/distributed/transformers/longcat_audio_dit.py` (+ `planners.py`) |
| T2A task + wav writer | `src/cache_dit/_utils/registers.py` (`ExampleType.T2A`, `_save_audio`, `parse_audio_output`) |
| CLI example | `src/cache_dit/_utils/examples.py` (`longcat_audio_dit_example`) |
| Audio metrics | `src/cache_dit/metrics/metrics.py` (`calculate_si_sdr` / `calculate_snr` / `calculate_mel_psnr` / `calculate_mel_ssim`, `compute_audio_metric`) |
| CPU tests | `tests/api/test_longcat_audio_dit_adapter.py`, `tests/api/test_metrics_audio.py` |
| Multi-rank parity | `tests/parallelism/test_longcat_audio_dit_cp_tp_tiny.py` |

## 9. Verification checklist for an audio integration

- [ ] baseline wav written and non-silent (`peak`/`rms` sane, duration matches the request)
- [ ] `--rtd 0` gives `si_sdr = inf` (the cache wrapper is exact; §4.6)
- [ ] cache at the shipped threshold: speedup recorded with mel PSNR/SSIM **and** hit count
- [ ] compile: warmup time and `Compiling repeated blocks` confirmed; a rerun in a **new process**
      recorded as the noise floor
- [ ] TP/CP: tiny parity (bit-exact or ≤1e-5 relative) **before** the end-to-end run; divisible-frame
      CP row included as the correctness witness
- [ ] quantization: `float8_per_row` and one 4-bit tier with bf16 weights; note which rows fall below
      the floor
- [ ] offload: `si_sdr = inf`; async transfer timed; state that it is a memory tool, not a speedup
- [ ] VAE-P: decode share measured and the N/A decision documented
- [ ] one wav + one spectrogram per configuration handed to the user, plus the speed/metrics table
      with the noise-floor row
