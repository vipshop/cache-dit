# Performance & Accuracy Matrix (acceptance runs)

When to read this: the user wants **speed + accuracy numbers** for a model that is **already
integrated** — "跑一遍全量测试矩阵", "4 卡验收一下", "量化掉多少精度", "和 bf16 比加速比". Routed in
from `../SKILL.md` ROUTE FIRST.

Stop and go back to `../SKILL.md` GATE CHECK if the model still has no `BlockAdapter`, no
`cdit.generate` example, or no CP/TP planner — that is an **integration** task, and its per-phase
numbers live in `./testing.md`.

Run order (do not reorder):

1. `./perf_analysis.md` — FLOPs budget (attention vs GEMM share). It decides **in advance** which
   optimizations can pay off, so a flat fp8 number later is not misdiagnosed as an integration bug.
   An attention-dominated model additionally needs the `--attn ffpa_fp8` / `ffpa_fp4` rows (§1 note),
   because GEMM-only quantization is capped by its GEMM share.
2. Tier A 1-GPU rows, starting with the baseline **and** the determinism baseline (§2).
3. Tier A 2-GPU rows.
4. Conclusions + a recommended entry (not just a row list) — §7.
5. Tier B (4 GPUs) / Tier C (8 GPUs) — **only when the user explicitly asks for them**.

## 1. Scope tiers — 1/2 GPUs by default

| Tier | GPUs | When | Distribution |
|---|---|---|---|
| **A** | 1 and 2 | **Default, always.** Every model-support task ends with a Tier A table. | single card, `--parallel tp`, `--parallel ulysses` |
| **B** | 4 | Only when the user asks to extend to 4 cards. | `tp4`, `ulysses4`, hybrid TP2+CP2 (`--parallel ulysses_tp`) |
| **C** | 8 | Only when the user asks to extend to 8 cards. | `tp8`, `ulysses8`, hybrid TP2+CP4 (`ulysses_tp`) / TP4+CP2 (`tp_ulysses`), optional `usp_tp` |

Rules:

- **Never open Tier B/C on your own.** An 8-card video row costs hours; the user decides.
- A machine with a single card still produces a valid Tier A table: drop the 2-GPU rows and say so
  in the caption.
- 4 cards **cannot** run TP2+CP4 (that needs 8 processes). On 4 cards `CP4` is pure `--parallel
  ulysses`, and the only hybrid layout is TP2+CP2.
- On 4 cards `--parallel ulysses_tp` and `--parallel tp_ulysses` resolve to the **same** layout
  (U2×T2) — run one of them, not both. The 8-card tier is the only place where "Ulysses first" and
  "TP first" differ, which is exactly why Tier C keeps both.
- Card assignment: 1–2 GPU rows use the cache-dit cards assigned by the user (this workspace: 4 and
  5); Tier B/C need more cards, so use the block the user assigns. **Within one table, every row
  must run on the same physical cards** — cross-tier comparison inherits clock/temperature noise.

### Tier A checklist (1/2 GPUs)

| | Configuration | tag | flags | expected outcome |
|---|---|---|---|---|
| [ ] | single-GPU bf16 reference | `U1_C0_Q0_NONE_a` / `_b` | – | the reference itself; run it **twice** for the determinism baseline (§2) |
| [ ] | single-GPU + compile | `U1_C1_Q0_NONE` | `--compile` | ≈30 dB (bf16 is not bit-exact under compile) |
| [ ] | cache, default threshold | `U1_C0_Q0_DB` | `--cache` | hits > 0 on a 30+ step model; record the real `_R` value |
| [ ] | cache, raised threshold | `U1_C0_Q0_DB_HI` | `--cache --rdt <hi>` | required when the default caches 0 steps (distilled models) |
| [ ] | cache raised + compile | `U1_C1_Q0_DB_HI` | `--cache --rdt <hi> --compile` | fastest cache entry |
| [ ] | single-GPU fp8 + compile | `U1_C1_FP8_NONE` | `--compile --quantize-type float8_per_row` | ~26 dB |
| [ ] | single-GPU W4A4 + compile | `U1_C1_W4A4_NONE` | `--compile --quantize-type svdq_nvfp4_r64_dq --svdq-smooth-strategy few_shot` (sm_89 → `svdq_int4_r64_dq`) | ~23 dB; add `--quantize-exclude-layers` if a gate layer has `min(in, out) < rank` |
| [ ] | layerwise offload (sync) | `U1_C0_Q0_OFF` | `--layerwise-offload` | **PSNR inf** (bit-exact) |
| [ ] | layerwise offload + async | `U1_C0_Q0_OFFA` | `--layerwise-offload --layerwise-async-transfer` | inf, and faster than the sync row |
| [ ] | offload + compile | `U1_C1_Q0_OFF` | `--layerwise-offload --compile` | compile-tier drift (≈32 dB), **not** inf |
| [ ] | offload + async + compile | `U1_C1_Q0_OFFA` | `--layerwise-offload --layerwise-async-transfer --compile` | fastest offload entry |
| [ ] | TP2 | `TP2_C0_Q0_NONE` | `--parallel tp` | ≈30 dB; usually a *memory* win, not a latency win |
| [ ] | TP2 + compile | `TP2_C1_Q0_NONE` | `--parallel tp --compile` | |
| [ ] | CP2 | `U2_C0_Q0_NONE` | `--parallel ulysses` | ≈30 dB; the real latency lever |
| [ ] | CP2 + compile | `U2_C1_Q0_NONE` | `--parallel ulysses --compile` | |
| [ ] | CP2 + fp8 + compile | `U2_C1_FP8_NONE` | `--parallel ulysses --compile --quantize-type float8_per_row` | |
| [ ] | CP2 + W4A4 + compile | `U2_C1_W4A4_NONE` | `--parallel ulysses --compile --quantize-type svdq_nvfp4_r64_dq --svdq-smooth-strategy few_shot` | |
| [ ] | TE-P (run-only) | `TP2_C0_Q0_TEP` | `--parallel tp --parallel-text` | runs, ≈30 dB |
| [ ] | VAE-P (run-only) | `TP2_C0_Q0_VAEP` | `--parallel tp --parallel-vae` | runs, ≈30 dB |

- TE-P / VAE-P are **run-only** rows (user requirement: 跑通就行): record pass/fail + PSNR/SSIM,
  never quote them as a speedup.
- `--parallel-text` / `--parallel-vae` are **no-ops without `--parallel`** — `parallelism_config` is
  only built when `--parallel` is set (`_utils/utils.py::maybe_apply_optimization`), so always hang
  them under `--parallel tp`.
- **Attention-dominated models** (long video sequences, see `./perf_analysis.md`): the GEMM-side rows
  above have a hard ceiling of the model's GEMM share. Add two single-GPU rows
  `A1_C1_Q0_ATTN8` (`--attn ffpa_fp8`) and, if acceptable, `A1_C1_Q0_ATTN4` (`--attn ffpa_fp4`), each
  ± `--cache`. Without them the table under-reports the ceiling by 2× on such models.

### Tier B checklist (4 GPUs, on request)

| | Configuration | tag | flags |
|---|---|---|---|
| [ ] | TP4 | `TP4_C0_Q0_NONE` | `--parallel tp` |
| [ ] | TP4 + compile | `TP4_C1_Q0_NONE` | `--parallel tp --compile` |
| [ ] | CP4 | `U4_C0_Q0_NONE` | `--parallel ulysses` |
| [ ] | CP4 + compile | `U4_C1_Q0_NONE` | `--parallel ulysses --compile` |
| [ ] | CP4 + cache | `U4_C0_Q0_DB` | `--parallel ulysses --cache` |
| [ ] | **TP2 + CP2 (Ulysses)** | `U2T2_C0_Q0_NONE` | `--parallel ulysses_tp` |
| [ ] | TP2 + CP2 + compile | `U2T2_C1_Q0_NONE` | `--parallel ulysses_tp --compile` |
| [ ] | TP2 + CP2 + fp8 + compile | `U2T2_C1_FP8_NONE` | `--parallel ulysses_tp --compile --quantize-type float8_per_row` |

### Tier C checklist (8 GPUs, on request)

| | Configuration | tag | flags |
|---|---|---|---|
| [ ] | TP8 | `TP8_C0_Q0_NONE` | `--parallel tp` |
| [ ] | TP8 + compile | `TP8_C1_Q0_NONE` | `--parallel tp --compile` |
| [ ] | CP8 | `U8_C0_Q0_NONE` | `--parallel ulysses` |
| [ ] | CP8 + compile | `U8_C1_Q0_NONE` | `--parallel ulysses --compile` |
| [ ] | CP8 + cache | `U8_C0_Q0_DB` | `--parallel ulysses --cache` |
| [ ] | **TP2 + CP4 (Ulysses first)** | `U4T2_C0_Q0_NONE` | `--parallel ulysses_tp` |
| [ ] | TP2 + CP4 + compile | `U4T2_C1_Q0_NONE` | `--parallel ulysses_tp --compile` |
| [ ] | **TP4 + CP2 (TP first)** | `T4U2_C0_Q0_NONE` | `--parallel tp_ulysses` |
| [ ] | TP4 + CP2 + compile | `T4U2_C1_Q0_NONE` | `--parallel tp_ulysses --compile` |
| [ ] | TP2 + CP4 + fp8 + compile | `U4T2_C1_FP8_NONE` | `--parallel ulysses_tp --compile --quantize-type float8_per_row` |
| [ ] | optional: USP+TP | `U2R2T2_C0_Q0_NONE` | `--parallel usp_tp` (8 ranks only) |

## 2. Fixed conditions — every row must share them

- **Same workload**: identical resolution / frames / steps / seed across all rows. Video models:
  open with a small size, then run the acceptance table at the official size.
- **Timing**: `--warmup 2 --repeat 2`; quote only `- Inference Time: X.XXs`, which is the mean over
  `--repeat` (`_utils/registers.py::Example.summary`). Compile cost is paid once during warmup, so
  a compiled row measured without warmup is garbage.
- Flag disambiguation — these three are different things:
  - `--warmup N` / `--repeat N`: **timing** warmup / repeats.
  - `--ws` / `--max-warmup-steps`: **cache** warmup steps.
  - `--warmup-steps`: inference steps used inside each warmup run.
- **One reference for everything**: the single-GPU bf16 output of the same seed (`U1_C0_Q0_NONE_a`).
  Every row's PSNR/SSIM is computed against it, never against another accelerated row.
- **Determinism baseline first**: run the single-GPU bf16 row twice under two different tags
  (`U1_C0_Q0_NONE_a`, `U1_C0_Q0_NONE_b` — same tag twice would overwrite the artifact) and compare.
  - `PSNR = inf` → the model is deterministic. Then **any non-quantized, non-cache row** (bf16
    parallel / compile / offload+compile) below ~25 dB mean is an integration bug, not drift;
    quantization and cache rows are judged against their own tiers (`./testing.md` §7.6) and are
    expected to land far lower.
  - otherwise (some models are run-to-run unstable, e.g. Triton autotune over reductions) →
    record `baseline run-to-run = X dB` in the caption and judge every other row relative to it.
- `--track-memory` is on by default in the template below: it gives `- Memory Usage: X.XXGiB`, worth
  a caption note for TP/offload rows (their whole point is memory). Drop it if you do not need it.

## 3. Distributed size mapping and preflight

`--parallel` is resolved into (ulysses, ring, tp) sizes by
`_utils/utils.py::_prepare_distributed_size()`; the mapping is fixed, not free-form:

| `--parallel` | ws=2 | ws=4 | ws=8 | notes |
|---|---|---|---|---|
| `tp` | T2 | T4 | T8 | shards weights; all-gather in every block |
| `ulysses` | U2 | U4 | U8 | CP; the latency lever |
| `ring` | R2 | R4 | R8 | CP; rotates KV, fp32 by default |
| `usp` | U1×R2 (= pure Ring2) | U2×R2 | U4×R2 | Ulysses + Ring |
| `ulysses_tp` | U1×T2 (= **pure TP2**) | U2×T2 | U4×T2 | Ulysses first |
| `tp_ulysses` | T1×U2 (= **pure CP2**) | T2×U2 | T4×U2 | TP first; same as `ulysses_tp` at ws=4 |
| `ring_tp` | R1×T2 (= pure TP2) | R2×T2 | R4×T2 | Ring first |
| `tp_ring` | T1×R2 (= pure CP2) | T2×R2 | T4×R2 | TP first |
| `usp_tp` | assert | assert | U2×R2×T2 | asserts `world_size == 8` (plain `assert`, skipped under `python -O`) |

Preflight before blaming a crash on the integration:

1. `ulysses_size * tp_size == world_size` — an odd card count cannot form a hybrid layout
   (ws=5 → 2×2=4 ≠ 5).
2. `heads % tp_size == 0` — TP itself must shard the heads evenly (a model may raise here before any
   hybrid check).
3. **Post-TP head divisibility**: CP runs before TP and Ulysses shards heads, so the hybrid
   constraint is `(heads // tp_size) % ulysses_size == 0`. Model planners enforce it and raise with
   a suggestion (MiniMax-H3, Kandinsky 6.0). 56 heads → `tp2+cp2` (28), `tp2+cp4` (14), `tp4+cp2`
   (14) all work; 12 heads → **neither** 8-card hybrid works, so record that combination as
   "unsupported by model shape", not as a bug.
4. If (3) fails, `--ulysses-anything` (`--uaa`) is the escape hatch — unless the model rejects UAA
   outright (Kandinsky 6.0 raises `NotImplementedError`, and it also refuses Ring CP entirely), in
   which case the combination is simply not supported.
5. The model must have **both** a CP planner and a TP planner registered; with only one, hybrid is
   unavailable (it will not silently degrade).

Reference scale-out direction (8×L20, FLUX.2-dev, `docs/user_guide/HYBRID_PARALLEL.md`): TP8
21.37 s → Ulysses-4-TP-2 **15.21 s** (1.40×), Ulysses-2-TP-4 17.98 s, USP-2-2-TP-2 16.06 s. On 4
cards (LTX-2): TP4 143.49 s → Ulysses-2-TP-2 **110.95 s** (1.29×). So the expectation for Tier B/C is
"hybrid beats the pure-TP layout it replaces" — a hybrid row that loses to pure TP is worth a
second look at preflight (1)/(3) and at whether all-gathers dominate at that sequence length.

## 4. Command template

Tag scheme: `<parallel>_<compile>_<quant>_<variant>`, e.g. `U2_C1_FP8_NONE`, `U1_C0_Q0_DB_HI`,
`U4T2_C1_Q0_NONE`. Slot 4 carries the non-quantization variant: `NONE`, `DB` / `DB_HI` (cache),
`OFF` / `OFFA` (offload, ± async), `TEP` / `VAEP` (run-only), `ATTN8` / `ATTN4` (attention backend).
The **file/dir name must be the tag** — §6 builds the table from those names.

```bash
export TASK=perf_MYMODEL_1024x1024x1     # task dir; avoid <angle brackets>, they are shell operators
export MODEL=MYMODEL                     # the cdit.generate example name, e.g. flux2
export SEED=42
export EXT=png                           # video models: mp4
export LAYOUT=dir                        # image 1-vs-N mode: dir; video/flat: file
export CUDA_VISIBLE_DEVICES=4            # 1 GPU; 2-GPU block below re-exports 4,5
mkdir -p .tmp/$TASK

NPROC=1

run() {                                  # run <tag> [extra args...]
  local tag=$1; shift
  local out
  if [ "$LAYOUT" = "dir" ]; then
    mkdir -p ".tmp/$TASK/$tag"
    out=".tmp/$TASK/$tag/out.$EXT"
  else
    out=".tmp/$TASK/$tag.$EXT"
  fi
  local launcher=(python3 -m)
  if [ "$NPROC" -gt 1 ]; then
    launcher=(torchrun --nproc_per_node=$NPROC -m)
  fi
  "${launcher[@]}" cache_dit.generate "$MODEL" --seed "$SEED" \
    --warmup 2 --repeat 2 --track-memory --summary \
    --save-path "$out" "$@" > ".tmp/$TASK/$tag.log" 2>&1
  printf '%s %s %s\n' "$tag" \
    "$(grep -o 'Inference Time: [0-9.]*s' ".tmp/$TASK/$tag.log" | tail -1)" \
    "$(grep -o 'Memory Usage: [0-9.]*GiB' ".tmp/$TASK/$tag.log" | tail -1)" \
    | tee -a ".tmp/$TASK/perf.log"
}
```

Artifact naming — it is what §6 consumes:

- image, automatic 1-vs-N table (`LAYOUT=dir`): `.tmp/$TASK/<tag>/out.png`
- video (`LAYOUT=file`, `EXT=mp4`): `.tmp/$TASK/<tag>.mp4`
- single-pair image comparison: `LAYOUT=file`, `EXT=png` → `.tmp/$TASK/<tag>.png`, then use `-i1/-i2`

Tier A:

```bash
run U1_C0_Q0_NONE_a                                           # reference, run 1
run U1_C0_Q0_NONE_b                                           # reference, run 2 (determinism, §2)
run U1_C1_Q0_NONE            --compile
run U1_C0_Q0_DB              --cache                          # read the effective _R from the tag
run U1_C0_Q0_DB_HI           --cache --rdt 0.20               # distilled models need the raised one
run U1_C1_Q0_DB_HI           --cache --rdt 0.20 --compile
run U1_C1_FP8_NONE           --compile --quantize-type float8_per_row
run U1_C1_W4A4_NONE          --compile --quantize-type svdq_nvfp4_r64_dq --svdq-smooth-strategy few_shot
run U1_C0_Q0_OFF             --layerwise-offload
run U1_C0_Q0_OFFA            --layerwise-offload --layerwise-async-transfer
run U1_C1_Q0_OFF             --layerwise-offload --compile
run U1_C1_Q0_OFFA            --layerwise-offload --layerwise-async-transfer --compile
# attention-dominated models only (§1 note):
# run A1_C1_Q0_ATTN8         --compile --attn ffpa_fp8

export CUDA_VISIBLE_DEVICES=4,5
NPROC=2
run TP2_C0_Q0_NONE           --parallel tp
run TP2_C1_Q0_NONE           --parallel tp --compile
run U2_C0_Q0_NONE            --parallel ulysses
run U2_C1_Q0_NONE            --parallel ulysses --compile
run U2_C1_FP8_NONE           --parallel ulysses --compile --quantize-type float8_per_row
run U2_C1_W4A4_NONE          --parallel ulysses --compile --quantize-type svdq_nvfp4_r64_dq --svdq-smooth-strategy few_shot
run TP2_C0_Q0_TEP            --parallel tp --parallel-text     # run-only
run TP2_C0_Q0_VAEP           --parallel tp --parallel-vae      # run-only
```

Notes on the quantization flags:

- `--svdq-smooth-strategy few_shot` matters on pixel-space / few-shot-calibrated models — the
  default `identity` can garble the image while PSNR still reads ~28 (SKILL.md pitfall #11).
- **Check the gate layers first**: `svdq_*_r64_dq` aborts with `rank 64 exceeds the maximum
  supported rank 32` on a `Linear` whose `min(in, out) < rank`. DFR needs
  `--quantize-exclude-layers to_gate_logits`; any model with a similar `heads`-sized gate needs its
  own exclusion list. This is per-model, so it cannot be baked into the template.
- `fp8_per_row` only pays off together with `--compile`; measuring it eager is a wasted row.
- sm_89 cards (L20/L40S) → `svdq_int4_r64_dq`; sm_120 (RTX PRO 5000) → `svdq_nvfp4_r64_dq`.

Tier B (4 GPUs):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3     # use the block the user assigned
NPROC=4
run TP4_C0_Q0_NONE   --parallel tp
run TP4_C1_Q0_NONE   --parallel tp --compile
run U4_C0_Q0_NONE    --parallel ulysses
run U4_C1_Q0_NONE    --parallel ulysses --compile
run U4_C0_Q0_DB      --parallel ulysses --cache
run U2T2_C0_Q0_NONE  --parallel ulysses_tp
run U2T2_C1_Q0_NONE  --parallel ulysses_tp --compile
run U2T2_C1_FP8_NONE --parallel ulysses_tp --compile --quantize-type float8_per_row
```

Tier C (8 GPUs):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=8
run TP8_C0_Q0_NONE    --parallel tp
run TP8_C1_Q0_NONE    --parallel tp --compile
run U8_C0_Q0_NONE     --parallel ulysses
run U8_C1_Q0_NONE     --parallel ulysses --compile
run U8_C0_Q0_DB       --parallel ulysses --cache
run U4T2_C0_Q0_NONE   --parallel ulysses_tp      # TP2 + CP4, Ulysses first
run U4T2_C1_Q0_NONE   --parallel ulysses_tp --compile
run T4U2_C0_Q0_NONE   --parallel tp_ulysses      # TP4 + CP2, TP first
run T4U2_C1_Q0_NONE   --parallel tp_ulysses --compile
run U4T2_C1_FP8_NONE  --parallel ulysses_tp --compile --quantize-type float8_per_row
run U2R2T2_C0_Q0_NONE --parallel usp_tp          # optional
```

## 5. Table template

Authoritative acceptance table (PSNR/SSIM in **mean / min**). The integration-phase short table in
`./testing.md` §7.5 is a different, coarser shape — do not mix the two. Rows below are written
human-readably; the `<tag>` from §1/§4 is only the artifact name that ties a row to its file and to
its line in `perf.log`.

| Configuration | Inference | Speedup | PSNR mean/min | SSIM mean/min |
|---|---|---|---|---|
| single-GPU bf16 (reference) | – | 1.00× | – | – |
| single-GPU + compile | | | | |
| single-GPU + cache (thr=R, hits=n/N) | | | | |
| single-GPU + cache (thr=R_hi, hits=n/N) | | | | |
| single-GPU + cache (thr=R_hi) + compile | | | | |
| TP2 | | | | |
| TP2 + compile | | | | |
| CP2 | | | | |
| CP2 + compile | | | | |
| single-GPU + fp8_per_row + compile | | | | |
| single-GPU + W4A4 + compile | | | | |
| CP2 + fp8_per_row + compile | | | | |
| CP2 + W4A4 + compile | | | | |
| single-GPU + layerwise offload | | | inf | 1.000 |
| single-GPU + offload + async transfer | | | inf | 1.000 |
| single-GPU + offload + compile | | | | |
| single-GPU + offload + async + compile | | | | |
| TP2 + TE-P (run-only) | | – | | |
| TP2 + VAE-P (run-only) | | – | | |

Tier B/C tables are the same columns with the rows from §1 (B: TP4 / CP4 / TP2+CP2; C: TP8 / CP8 /
TP2+CP4 / TP4+CP2, plus the optional USP+TP row). Keep them in a separate table — a 4-card row must
not be mixed into a 1/2-card baseline table.

No filled example table here on purpose: a set of numbers belongs to one model, one schedule and one
card count, and reusing them as "the expected level" is how wrong conclusions start. The caption of
a finished table must still carry:

- the workload (resolution / frames / steps), the card count and the layout (`tp2`, `u2`, `u2t2`, ...);
- the flags that changed behaviour: excluded quant layers, `--svdq-smooth-strategy`, `--ulysses-anything`,
  effective cache threshold;
- the determinism baseline (`inf` or `X dB`) — it sets the floor for every other row;
- whether offload was a win or a loss **on this hardware**: on a card that already holds the weights
  it is a negative result that exists to prove correctness, and that has to be said out loud.

## 6. Measuring: mean, min, and the auto table

**Mean** (both metrics, always both):

```bash
# video: one file per tag (single pair → -v1/-v2)
cache-dit-metrics psnr ssim --summary \
  -vd .tmp/$TASK -rv .tmp/$TASK/U1_C0_Q0_NONE_a.mp4

# images: one directory per tag, same basename inside (single pair → -i1/-i2)
cache-dit-metrics psnr ssim --summary \
  -d .tmp/$TASK -r .tmp/$TASK/U1_C0_Q0_NONE_a
```

**Auto table with speedup** (image directory layout; needs `perf.log` lines that contain *both* the
tag and `Inference Time:` — the `run()` helper above writes exactly that):

```bash
cache-dit-metrics psnr ssim --summary --cal-speedup -table \
  -plog .tmp/$TASK/perf.log -ptags "Inference Time" \
  -d .tmp/$TASK -r .tmp/$TASK/U1_C0_Q0_NONE_a
```

- `-table` sorts rows by metric value, not by the matrix's logical order — re-order before pasting.
- In video mode the logged label carries the `.mp4` basename while `perf.log` carries the tag, so
  the perf column does not bind: fill the video table by hand (or join `perf.log` with awk).

**Min** (worst frame / worst prompt — the column that exposes localized corruption):

```python
import cv2
from cache_dit.metrics.metrics import compute_psnr_file, compute_ssim_file

def frames(path):
  cap, out = cv2.VideoCapture(path), []
  while True:
    ok, frame = cap.read()
    if not ok:
      break
    out.append(frame)
  cap.release()
  return out

ref = frames(".tmp/$TASK/U1_C0_Q0_NONE_a.mp4")
for tag in ["U2_C0_Q0_NONE", "U2_C1_FP8_NONE"]:
  pairs = list(zip(ref, frames(f".tmp/$TASK/{tag}.mp4")))
  psnr = [compute_psnr_file(a, b) for a, b in pairs]
  ssim = [compute_ssim_file(a, b) for a, b in pairs]
  print(f"{tag}: PSNR {sum(psnr) / len(psnr):.2f} / {min(psnr):.2f} | "
        f"SSIM {sum(ssim) / len(ssim):.3f} / {min(ssim):.3f}")
```

- Image models: the same loop over prompts instead of frames (one image per prompt).
- Bit-exact rows print `inf`; that is the intended reading, not a bug in the script.
- mp4 is re-encoded (H.264), which adds its own noise to every frame. For quantization tiers where
  you need to attribute a 1–2 dB loss, dump PNG frames and compare frame directories instead.

**Where the numbers in the caption come from**

- Effective cache threshold: the `- Optimization:` line's tag carries `_R<threshold>`
  (`DBCacheConfig.strify()`); `--rdt` left at its default is replaced by the model's own preset when
  that model ships one, so never write "0.12" from memory.
- Cache hits: `--summary` logs `Cache Steps and Residual Diffs Statistics: <cls>, Executed Steps: N`
  plus a `| Cache Steps | ... |` row; the tag also ends with `_S<hits>`.
- Counts differ by mode: image mode prints `Num:` = **total pairs including** the ones skipped as
  `inf` (the mean itself is over valid pairs only), video mode prints `Frames:` = non-`inf` frames
  only. The two `n` values are not interchangeable.

## 7. How to write the table (conventions)

- **Speed and accuracy always pair up.** A row with time but no PSNR/SSIM is not a result.
- PSNR gets `mean / min`; the `min` is what exposes a locally broken frame or prompt.
- bf16 parallel/compile is **not** bit-exact: ≈30 dB (lower bound 28) is normal drift once the
  determinism baseline passes. A `min` collapsing while the mean looks fine means localized
  corruption, not drift.
- Quantization degrades by tier — quote the tier's expectation, not a single global threshold
  (gates: `./testing.md` §7.6).
- **Cache needs two rows**: the default threshold and a raised one, each with its hit count. A
  distilled model can hit 0 steps at the default, and "cache gives 1.00×" is then a wrong reading.
- **Offload needs two rows** (sync + `--layerwise-async-transfer`); async must be on for the shipped
  recommendation. Offload without compile is bit-exact (`inf`) — that is its correctness proof;
  offload + compile is *not* bit-exact, judge it against the compile tier.
- Write the per-model pitfalls into the caption (excluded quant layers, required `--svdq-smooth-strategy`,
  UAA requirement, hybrid head divisibility). A number that needs a hidden flag is a trap for the
  next person.
- End with a **recommended entry** and the reason: "at equal speed pick the higher quality"
  (CP2 + fp8 + compile over svdq), "TP only if the weights do not fit", "offload only under VRAM
  pressure". A bare table without a verdict is not finished.
- Tier B/C tables get their own caption stating GPUs, layout, and the layout preflight result (§3),
  plus one line on whether hybrid beat the pure-TP layout it replaced.
- Keep the artifacts and `perf.log` under `.tmp/$TASK` and name the directory in the caption —
  the numbers are only auditable if the run is findable.
