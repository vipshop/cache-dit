# RFC: Pending Model Support — diffusers v0.38+ Models Not Yet Supported by cache-dit

When to read this: when picking the next model to integrate into cache-dit, or when starting/finishing work on a model listed here (update its Status entry and add a line to R.5). Return to `../SKILL.md` for the integration workflow; see [`./model_download.md`](./model_download.md) for fetching the weights.

## R.1 Background, Scope and Method

- Analysis dates: **2026-10-22** (three passes: diffusers v0.40+ scan, the v0.38/v0.39 window scan, then unification into one difficulty-ordered list).
- Scope: models **added to diffusers v0.38.0 and later** — tag chain `v0.37.1` (2026-03-25) → `v0.38.0` (2026-05-01) → `v0.39.0` (2026-07-03) → `v0.40.0` (2026-08-20) → `v0.41.0` (2026-10-06) → `0.42.0.dev0` (HEAD `5378870`, 2026-10-20) — that have **no BlockAdapter coverage** in cache-dit.
- Method: diffusers git log between upstream release tags (`--diff-filter=A` on `pipelines/`, `modular_pipelines/`, `models/transformers/`) cross-checked against `@BlockAdapterRegister.register(...)` in `cache-dit/src/cache_dit/caching/block_adapters/adapters.py`. A model counts as "supported" only when a registered adapter matches its transformer class — or a class it reuses.
- **All models — verification-only quick wins and full-adapter work — are listed in ONE unified difficulty-ordered table (R.2), easiest first.** The Type column distinguishes Verification-only (backbone class already registered; work = A/B test + CLI example + matrix entry) from Full adapter (new `BlockAdapter` needed). Within the same difficulty tier, verification-only items come first because they are cheaper.
- Difficulty assessment factors: pipeline type (regular diffusers pipeline < modular-only), block structure (uniform `ModuleList` < PatchFunctor needed < non-uniform), family reference (same-family adapter in cache-dit < analogous only < none), verification method (image PSNR/SSIM < video frames < audio < domain-specific), and interaction risks (KV-cache interplay, multi-stage pipelines, MoE routing). Business value is a separate axis — video models remain cache-dit's main workload; jump the queue when value demands it.
- Models from the v0.38+ window that are **already covered and therefore excluded**: MiniMax-H3 (v0.40, #14355), Qwen-Image 2.1 (v0.41, #14804 + modular #14976), FLUX.3 (0.42-dev, #15046), ERNIE-Image (v0.38, #13432 + modular #13498), Helios (v0.38 modular, #13216), HunyuanVideo 1.5 (v0.38 modular, #13389), LTX Video modular (v0.38, #13378), FLUX.2 Klein Inpaint / Klein-9b-kv (v0.38), SD3 modular (v0.39, #13324), AnyFlow / AnyFlow-FAR (v0.39, #13745), Cosmos3 (v0.39, #13818, incl. Cosmos3 Omni), JoyImage-Edit (v0.39, #13444), Krea 2 (v0.39, #14045), LTX-2 IC LoRA / HDR variants (v0.39, #13572), plus v0.40-era family-variant work (Cosmos3 Distilled/FP8, Wan VACE modular blockset, Krea2 modular, JoyImage Edit Plus).

Status legend: ⬜ pending · 🔄 in progress · ✅ done (move the row to R.5 with a one-line summary).

## R.2 Unified Model List — Ordered by Integration Difficulty

Verification-only rows: the work is a cache A/B test (PSNR > 30, SSIM > 0.90 vs baseline) + CLI example + support-matrix entry; model-specific caveats in R.3. Full-adapter rows follow the full `../SKILL.md` workflow (Cache → CP → TP → TE-P → VAE-P → CLI → Testing).

| # | Model | Difficulty | Type | diffusers | Task | Transformer class | HF | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | LTX-2.5 (incl. DFR pipeline) | ★☆☆☆☆ | Verification-only | v0.40.0 (#14447, #14567, 2026-08) | t2v / i2v | `LTX2VideoTransformer3DModel` — `LTX` adapter | [Lightricks/LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) | ✅ |
| 2 | PRXPixel | ★★☆☆☆ | Verification-only | v0.39.0 (#13928, 2026-06-17) | t2i (pixel space, no VAE) | `PRXTransformer2DModel` — `PRX` adapter | [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i) | ✅ |
| 3 | Anima | ★★☆☆☆ | Verification-only (modular) | v0.39.0 (#13732, 2026-05-29) | t2i | `CosmosTransformer3DModel` — `Cosmos` adapter | [circlestone-labs/Anima-Base-v1.0-Diffusers](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers) | ✅ |
| 4 | Echo (JoyAI-Echo) | ★★☆☆☆ | Verification-only (modular) | v0.41.0 (#14941, 2026-10-05) | long video + audio (DMD few-step) | `LTX2VideoTransformer3DModel` — `LTX` adapter | [jdopensource/JoyAI-Echo](https://huggingface.co/jdopensource/JoyAI-Echo) | ⬜ |
| 5 | Bria Fibo 2.0 | ★★☆☆☆ | Full adapter | 0.42-dev (#14991, 2026-10-20) | t2i / edit (8.1B) | `BriaFibo2Transformer2DModel` | [briaai/Fibo-2.0-dev](https://huggingface.co/briaai/Fibo-2.0-dev), [briaai/Fibo-2.0-edit-turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) | ⬜ |
| 6 | Motif-Video | ★★☆☆☆ | Full adapter | v0.39.0 (#13551, 2026-05-15) | t2v + i2v (2B) | `MotifVideoTransformer3DModel` | [Motif-Technologies/Motif-Video-2B](https://huggingface.co/Motif-Technologies/Motif-Video-2B) | ⬜ |
| 7 | Stable Audio 3 | ★★★☆☆ | Full adapter | v0.40.0 (#14119, 2026-08-18) | audio t2a / audio2audio / inpaint | `StableAudio3DiTModel` | [stabilityai/stable-audio-3-medium-base](https://huggingface.co/stabilityai/stable-audio-3-medium-base), [stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium) | ⬜ |
| 8 | LongCat-AudioDiT | ★★★☆☆ | Full adapter | v0.38.0 (#13390, 2026-04-15) | text-to-audio (1B) | `LongCatAudioDiTTransformer` | [ruixiangma/LongCat-AudioDiT-1B-Diffusers](https://huggingface.co/ruixiangma/LongCat-AudioDiT-1B-Diffusers) | ⬜ |
| 9 | Ideogram 4 | ★★★☆☆ | Full adapter (regular + modular) | v0.39.0 (#13859 + #13860, 2026-06-03) | t2i | `Ideogram4Transformer2DModel` | [ideogram-ai/ideogram-v4](https://huggingface.co/ideogram-ai/ideogram-v4) (+ `ideogram-4-nf4`) | ⬜ |
| 10 | NucleusMoE-Image | ★★★☆☆ | Full adapter (Cache-only scope) | v0.38.0 (#13317, 2026-04-03) | t2i (MoE) | `NucleusMoEImageTransformer2DModel` | [NucleusAI/NucleusMoE-Image](https://huggingface.co/NucleusAI/NucleusMoE-Image) | ⬜ |
| 11 | Kandinsky 6.0 | ★★★☆☆ | Full adapter | 0.42-dev (#14949, 2026-10-06) | text/image→video+audio (TI2VA), VSR | `Kandinsky6Transformer3DModel` (+ separate SR transformer) | [kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers), [Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers) | ⬜ |
| 12 | MiniMax Music 3 | ★★★☆☆ | Full adapter (modular) | v0.40.0 (#14456, 2026-08-14) | music generation | `MiniMaxMusic3Transformer1DModel` | [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) | ⬜ |
| 13 | SANA-WM | ★★★★☆ | Full adapter | 0.42-dev (#13881, 2026-10-13) | camera-controlled image-to-video world model | `SanaWMTransformer3DModel` | [Efficient-Large-Model/SANA-WM_bidirectional-diffusers](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional-diffusers) (+ [refiner](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional-diffusers-refiner)) | ⬜ |
| 14 | Wan-Animate-2 | ★★★★☆ | Full adapter (modular) | v0.40.0 (#14413, 2026-08-12) | character animation (ref image + driving video) | `WanAnimate2Transformer3DModel` | [Wan-AI/Wan2.2-Animate-2-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Diffusers) | ⬜ |
| 15 | ACE-Step 1.5 | ★★★★☆ | Full adapter | v0.38.0 (#13095, 2026-05-01) | text-to-music | `AceStepTransformer1DModel` | [ACE-Step/acestep-v15-xl-turbo-diffusers](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers) | ⬜ |
| 16 | RFD4-Proteina | ★★★★★ | Full adapter | 0.42-dev (#15088, 2026-10-20) | protein sequence + all-atom structure generation | `RFD4ProteinaTransformer` | [nvidia/RFD4-Proteina-Diffusers](https://huggingface.co/nvidia/RFD4-Proteina-Diffusers) (+ [prep](https://huggingface.co/nvidia/RFD4-Proteina-prep)) | ⬜ |

## R.3 Per-Model Integration Notes

All class names below were verified against diffusers source at HEAD `5378870`. Numbers match the R.2 rows.

### 3.1 LTX-2.5 (incl. DFR pipeline)
- Difficulty ★☆☆☆☆ (verification-only): `LTX2VideoTransformer3DModel` is covered by the `LTX` adapter (Pattern_0 + `LTX2PatchFunctor` on the `is_ltx2` path).
- Work: cache A/B test (PSNR > 30, SSIM > 0.90) on `Lightricks/LTX-2.5-Diffusers`, including the DFR pipeline variants (#14567); add CLI example; add support-matrix entry.

### 3.2 PRXPixel
- Difficulty ★★☆☆☆ (verification-only): `prx_adapter` already asserts `PRXTransformer2DModel` — the same class `PRXPixelPipeline` uses.
- ⚠️ Pixel-space pipeline (no VAE; the denoised output *is* the image): confirm the block loop and CFG handling match latent-space PRX before trusting the A/B result. Requires `transformers >= 4.57` (Qwen3-VL text encoder).
- Done (2026-10-23). Cache needed no change; CP/TP/TE-P were added because the shared class had none. Two model quirks are worth remembering: the text sequence and the RoPE must be split *outside* the block loop (`txt_in` output / `pe_embedder` ids) because PRX keeps `txt` and `pe` as loop-invariant locals, and the joint text+image key mask has to be rebuilt in the Ulysses all-to-all order. diffusers never declares `_repeated_blocks` for PRX, so regional quantization and `compile_repeated_blocks()` need the declaration cache-dit now adds.

### 3.3 Anima
- Difficulty ★★☆☆☆ (verification-only, modular): Cosmos adapter covers `CosmosTransformer3DModel`. Read [`./modular_pipeline.md`](./modular_pipeline.md) first.
- t2i → standard image metrics, fast iteration.
- Done (2026-10-23). Cache went through a thin `"Anima"` prefix adapter on the shared Cosmos `Pattern_2` branch (the registry matches pipeline class-name prefixes, so `AnimaModularPipeline` never matched `"Cosmos"`; `has_separate_cfg` derives from the modular guider's condition count). CP/TP for `CosmosTransformer3DModel` were new (`distributed/transformers/cosmos.py`). Anima quirks worth remembering: the root input is 5D latents and `padding_mask` is concatenated as an extra channel inside `patch_embed` (NOT an attention mask — no CP reorder needed), so CP is a single-point split at `transformer_blocks.0` plus a `rope` output-split (`(cos, sin)`, each `[S, D]`, split on dim 0) and a `proj_out` gather. `attn1`/`attn2` are independent `Attention` instances sharing `CosmosAttnProcessor2_0` (no img-context → no `CosmosAttention`), so one class-level dispatch patch — inert without `_cp_config` — covers both; TP needs `shard_div_attr` on BOTH attention layers and the GELU `FeedForward` keys `ff.net.0.proj` (Colwise) / `ff.net.2` (Rowwise) — there is no gate/up pair. Planner must be registered under the exact class name: the prefix `"Cosmos"` would also match `Cosmos3OmniTransformer`, which has no planner. `_repeated_blocks` is undeclared upstream, so `declare_repeated_blocks` now maps `CosmosTransformer3DModel` (fp8 regional scope: 448 layers, 0 skipped). The repo ships `modular_model_index.json`, so the CLI example needs no `modular_blocks_factory`; guidance lives in the guider component (`ModularPipeline.__call__` does not route `guidance_scale`), fixed at 4.0.

### 3.4 Echo (JoyAI-Echo)
- Difficulty ★★☆☆☆ (verification-only, modular): LTX2 backbone; modular path as in 3.3.
- ⚠️ DMD few-step sampling: few denoise steps → caching gains shrink and per-step accuracy matters more; verify per-frame PSNR/SSIM and report speedup honestly. Long video + audio output → frame-level metrics (audio checks once 3.7/3.8 land the audio methodology).

### 3.5 Bria Fibo 2.0
- Difficulty ★★☆☆☆: regular pipeline, uniform 2D blocks, direct same-family adapter reference, standard image verification.
- Blocks: `BriaFibo2TransformerBlock` + `BriaFibo2PerceiverResampler`; check whether the Perceiver resampler runs inside the block loop (PatchFunctor territory if so).
- Reference adapter: `briafibo_adapter` (Fibo 1.0). Start with `Fibo-2.0-edit-turbo` (fewer steps → faster verification).

### 3.6 Motif-Video
- Difficulty ★★☆☆☆: regular pipelines (t2v + i2v, no modular), standard double→single stream blocks (Flux paradigm), 2B small model → fast iteration; no same-family adapter but the in-repo double/single pattern (`flux_adapter`) is mature; SVDQuant mapping pre-exists.
- Blocks: `MotifVideoTransformerBlock` + `MotifVideoSingleTransformerBlock` (double→single stream; cf. `flux_adapter`'s dual-block handling).
- SVDQuant `src/cache_dit/quantization/svdquant/passes.py` already maps `MotifVideoTransformerBlock` and `MotifVideoSingleTransformerBlock` — pre-work done on the quantization side.
- Verification: video — sample frames, per-frame PSNR/SSIM (precedent: Wan / LTX test flows).

### 3.7 Stable Audio 3
- Difficulty ★★★☆☆: family reference exists (`stabledudio_adapter`), but audio verification methodology is new territory.
- `StableAudio3DiTModel` works on 1D audio tokens; three pipeline variants (t2a / audio2audio / inpaint).
- Reference adapter: `stabledudio_adapter` covers only the old `StableAudioDiTModel` (Stable Audio Open) — same family, different class; audit the block `forward()` signature before assuming the same pattern.
- ⚠️ PSNR/SSIM are image metrics. Audio correctness needs a different check (waveform/spectrogram comparison; check what the `cache_dit.metrics` module offers, extend if needed).

### 3.8 LongCat-AudioDiT
- Difficulty ★★★☆☆: regular pipeline, 1B model; no family reference; block IO is standard (self-attn + cross-attn + ConvNeXtV2 FFN) so the adapter itself is easy — the cost is the audio verification methodology, which is SHARED with Stable Audio 3. **Do 3.7 and 3.8 as one effort**: build the audio metrics once, integrate both.
- Blocks: `AudioDiTBlock` inside `LongCatAudioDiTTransformer` (⚠️ block class is not model-prefixed — grep `AudioDiTBlock`, not `LongCatAudio...`). ConvNeXtV2 FFN with GRN is non-standard, but block-IO caching does not care about the FFN implementation.
- 1D audio tokens, single pipeline (`LongCatAudioDiTPipeline`).

### 3.9 Ideogram 4
- Difficulty ★★★☆☆: dual form (regular pipeline + modular); asymmetric CFG with an **independent `unconditional_transformer` instance** — precedent exists (`wan_adapter`'s Wan 2.2 MoE `transformer_2` list, `has_separate_cfg` flag), but audit how the pipeline invokes the uncond transformer before wiring.
- Blocks: `Ideogram4TransformerBlock` with MRoPE.
- Structured prompt-upsampling stage (#13860, `prompt_enhancer` component) — same pitfall class as FLUX.3 `pup` (SKILL.md pitfall 9): pop the stage for benchmarking, inject for production, and record which one the example uses.
- Variants: `ideogram-ai/ideogram-v4` (bf16) and `ideogram-4-nf4` (quantized).

### 3.10 NucleusMoE-Image
- Difficulty ★★★☆☆ for **Cache**, ★★★★☆+ if CP/TP is attempted: MoE routing lives inside blocks. Block-IO caching stays valid (a skipped block skips its routing too), but CP/TP requires expert sharding — a research-level task. Recommend **Cache-only scope initially**, revisit parallelism later.
- Class: `NucleusMoEImageTransformer2DModel`.
- No family reference. HF id `NucleusAI/NucleusMoE-Image` comes from the diffusers pipeline docstring; ⚠️ not mirrored on ModelScope as of 2026-10-22 — fetch via the verified HF mirror flow (D.2 of [`./model_download.md`](./model_download.md)): `HF_ENDPOINT=https://hf-mirror.com hf download NucleusAI/NucleusMoE-Image --local_dir ./NucleusMoE-Image`.

### 3.11 Kandinsky 6.0
- Difficulty ★★★☆☆: heterogeneous transformer (trunk + audio decoder + SR), but the fix is a clear scoping decision (cache trunk blocks only); regular pipeline.
- Beyond the video trunk, `Kandinsky6Transformer3DModel` carries audio decoder blocks (`Kandinsky6TransformerDecoderBlock`, `Kandinsky6FusedTransformerDecoderBlock`); VSR is a separate transformer (`transformer_kandinsky6_sr.py`).
- Decide scope first: cache the main TI2VA transformer; treat SR and audio decoder as separate follow-ups.
- Reference adapter: `kandinsky5_adapter`. Start with `Kandinsky-6.0-Pro-distill-5s-Diffusers` (distilled = fewer steps → faster verification).

### 3.12 MiniMax Music 3
- Difficulty ★★★☆☆: modular-only, but the MiniMax-H3 integration is the closest in-repo reference (same org, same plumbing); uniform blocks; scope is clear; audio metrics as in 3.7.
- Modular-only pipeline → read [`./modular_pipeline.md`](./modular_pipeline.md) first.
- Architecture: AR LM + 2.4B flow DiT + RVQ depth decoder; **only the flow DiT is a cache target**. Block class: `MiniMaxMusic3TransformerBlock`.
- The `MiniMaxH3` adapter covers the 3D joint video+audio class; Music3 is 1D — needs its own adapter. The MiniMax-H3 modular integration (multi-transformer) is the closest in-repo reference for the modular plumbing.

### 3.13 SANA-WM
- Difficulty ★★★★☆: camera-control cross-attention inside blocks means likely PatchFunctor surgery; two-stage main+refiner orchestration.
- Blocks: `SanaVideoMSCamCtrlBlock` — camera-control **cross-attention inside blocks** → likely PatchFunctor territory (extra inputs in the block loop).
- Two-stage pipeline: main model + LTX-2 refiner (`SanaWMLTX2Refiner`); the refiner backbone `LTX2VideoTransformer3DModel` is already covered by the `LTX` adapter.
- Reference adapters: `sana_adapter` (2D Sana) for block style, `ltxvideo_adapter` for the refiner.

### 3.14 Wan-Animate-2
- Difficulty ★★★★☆: modular-only, and the segmented KV cache × block cache interaction is a genuine correctness risk (skipped blocks never update KV entries for cross-segment attention); CLI needs new input types.
- Blocks: `WanAnimate2TransformerBlock` (Wan-family derivative); reference adapter: `wan_adapter` (Wan family uses `ForwardPattern.Pattern_2`).
- ⚠️ The pipeline ships a **segmented KV cache** (`WanAnimate2KVCache` / `WanAnimate2KVLayerCache`) — cache-dit's block-level cache interacts with attention-level KV caching. Audit where the KV-cache ops run relative to the block loop before wiring; a PatchFunctor may be needed if they sit inside the loop body.
- SVDQuant `src/cache_dit/quantization/svdquant/passes.py` already maps the older `WanAnimateTransformerBlock` (v1) — extend the mapping for v2 blocks if quantization support is in scope.
- CLI: needs ref-image + driving-video inputs; check available `ExampleType`s in `src/cache_dit/_utils/registers.py`.

### 3.15 ACE-Step 1.5
- Difficulty ★★★★☆: mixed block types (linear transformer blocks + umbrella attention layers) → non-uniform stack, likely multiple block lists or PatchFunctor surgery; multi-component pipeline → explicit scoping needed.
- Architecture: beat/lyrics encoders + flow DiT + RVQ decoder; **scope: cache the flow DiT only**.
- `AceStepTransformer1DModel` carries diffusers' own `CacheMixin` — audit the interplay with cache-dit's mechanism before wiring (two caching layers must not fight over the same forward).
- Music verification: waveform/spectrogram comparison; share the audio metrics built for 3.7/3.8.

### 3.16 RFD4-Proteina
- Difficulty ★★★★★: non-uniform structure that `BlockAdapter`'s flat `ModuleList` assumption may not hold; domain-specific verification.
- `RFD4ProteinaTransformer` = trunk layers (`RFD4ProteinaTrunkLayer`) + pair modules (`RFD4ProteinaPairBuilder`, triangle multiplication, pair transition) — **NOT a uniform repeated-block DiT**. `BlockAdapter` assumes a flat homogeneous `ModuleList`; this model likely needs a custom PatchFunctor or a different integration approach entirely.
- Schedule last; budget extra design time. Outputs are protein structures — correctness verification needs domain-appropriate checks, not PSNR/SSIM.

## R.4 Out of Scope — In diffusers v0.38/0.39 but Excluded from cache-dit

Documented so future refreshes do not re-add them:

- **LLaDA2** (v0.38.0, #13226, 2026-03-25) and **DiffusionGemma** (v0.39.0, #13986, 2026-06-25): discrete-diffusion text LMs (masked-token remasking, not continuous flow matching). Block caching exploits the temporal smoothness of continuous DiT trajectories; discrete remasking breaks that assumption. Excluded unless a discrete-diffusion caching strategy is ever designed.
- **DreamLite** (v0.39.0, #13815, 2026-06-12): UNet backbone (`unets/unet_dreamlite.py`) with only an auxiliary 2D transformer — cache-dit is a DiT engine, and the auxiliary transformer is not the compute bottleneck. Out of scope.

## R.5 Completed Log

| Date | Model | Summary (adapter / coverage / PR or commit) |
|---|---|---|
| 2026-10-23 | PRXPixel (`Photoroom/prxpixel-t2i`) | Reused the `PRX` adapter (no change). Added CP + TP planners (`distributed/transformers/prx.py`), `Qwen3VLTextModel` TE-P, `prx_pixel` CLI example, and an `_repeated_blocks` declaration so regional quant/compile can find `PRXBlock`. Verified on RTX PRO 5000: CP2 PSNR 43.4 / SSIM 0.989, TP2 43.1 / 0.989, fp8_per_row 35.3 / 0.962 (regional scope; whole-module scope loses ~3 dB and whole-module compile is slower than eager), fp8 pays off only together with `--compile` (eager 18.4s vs compiled 10.9s), svdq nvfp4 r64 reaches 9.6s / 2.03x with 29.2 dB / SSIM 0.835 and needs `--svdq-smooth-strategy few_shot` (identity: SSIM 0.21). Both quantization numbers require the `_repeated_blocks` declaration: without it they fall back to whole-module scope plus full-module compile and look ~40% slower. |
| 2026-10-23 | Anima (`circlestone-labs/Anima-Base-v1.0-Diffusers`) | Added the `"Anima"` prefix adapter (thin wrapper on the shared Cosmos `Pattern_2` branch, guider-derived `has_separate_cfg`), `anima` CLI example, CP + TP planners (`distributed/transformers/cosmos.py`, exact class-name registration), `_repeated_blocks` declaration, CPU adapter tests + 2-GPU tiny parity tests. Verified on RTX PRO 5000 (1024², 30 steps, CFG 4.0): baseline 8.21s, cache rdt0.12 4.41s (15/30 cached, 33.4 dB / 0.927), CP2 6.85s (40.9 / 0.983), TP2 7.44s (40.8 / 0.983), fp8+compile 4.20s (30.8 dB / 0.815 — SSIM is the visible cost, keep for speed-critical runs only), svdq nvfp4+compile 4.13s (30.0 dB / 0.651 — coherent image, fur/texture softening only, no garbling; visual-checked), layerwise offload bit-exact (inf / 1.000); TE-P (Qwen3) TP2+text 7.53s (36.5 / 0.962) and VAE-P (QwenImage VAE) TP2+vae 7.42s (40.7 / 0.981) run as-is. Weights ~5.7 GB, so 2-GPU runs verify function/precision, not speed. |

## R.6 How to Refresh This List

Run on each new diffusers release (paths relative to `/workspace/dev/vipshop/diffusers` and `/workspace/dev/vipshop/cache-dit`):

```bash
# 1. Discover new release tags from upstream (the local fork lags behind on tags)
git ls-remote --tags https://github.com/huggingface/diffusers.git | grep -E "refs/tags/v[0-9.]+$" | sort -V

# 2. Fetch the boundary tags you want to compare (the v0.37.1..0.41 chain was fetched
#    in the 2026-10-22 passes as upstream-v0.37.1/38/39/40/41)
git fetch --no-tags https://github.com/huggingface/diffusers.git \
  refs/tags/v0.42.0:refs/tags/upstream-v0.42.0 \
  refs/tags/v0.43.0:refs/tags/upstream-v0.43.0

# 3. List newly added pipelines / transformers between the tags.
#    ⚠️ modular-only models live under modular_pipelines/ — do not scan pipelines/ only.
git --no-pager log upstream-v0.42.0..upstream-v0.43.0 --diff-filter=A \
  --pretty=format:"COMMIT %h %ad %s" --date=short \
  -- src/diffusers/pipelines src/diffusers/modular_pipelines src/diffusers/models/transformers

# 4. Supported side: the authoritative list of registered adapters
grep -n "BlockAdapterRegister.register" src/cache_dit/caching/block_adapters/adapters.py
```

Also check for models that **reuse** an already-registered backbone class (like LTX-2.5 / PRXPixel / Anima / Echo above) — they only need verification, not an adapter. New rows must be inserted into R.2 at the position their difficulty dictates, and R.3 renumbered accordingly.
