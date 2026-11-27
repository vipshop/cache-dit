# Agentic Workflows

Cache-DiT ships a set of SKILLs under [`.agents/skills`](https://github.com/vipshop/cache-dit/tree/main/.agents/skills) so a coding agent can carry out the repetitive parts of model support, attention backends, distributed parallelism, operator work and quantization by itself — including the parts that are easy to get wrong by hand (the BlockAdapter registration, the CP/TP planners, the backend registration and dispatch contract, the CP plan DSL, the test matrix).

Each skill is a `SKILL.md` router plus a `references/` folder, and works with any agent that can read files and run commands, e.g. [GitHub Copilot](https://docs.github.com/en/copilot), [Claude Code](https://claude.ai), [Open Code](https://opencode.ai/) or CodeBuddy.

## Available Skills

| Skill | What it covers |
|:--|:--|
| [`cache-dit-model-integration`](https://github.com/vipshop/cache-dit/tree/main/.agents/skills/cache-dit-model-integration) | Supporting a new DiT: BlockAdapter / ForwardPattern, Context Parallelism, Tensor Parallelism, TE-P, VAE-P, the `generate` CLI entry and the tests. Also runs the speed + PSNR/SSIM performance matrix for a model that is already supported. |
| [`cache-dit-quantization`](https://github.com/vipshop/cache-dit/tree/main/.agents/skills/cache-dit-quantization) | Choosing or extending quantization: TorchAO (fp8/int8/int4/nvfp4), cache-dit's own FP8 path, SVDQuant W4A4 PTQ and DQ, `QuantizeConfig`, regional quantize, precision plans and per-layer observers. |
| [`cache-dit-custom-kernel`](https://github.com/vipshop/cache-dit/tree/main/.agents/skills/cache-dit-custom-kernel) | Operators under `csrc/kernels/`: the CUDA → pybind → Python → `torch.library` integration chain, ports from other projects, reproducible builds, profiling and optimization. |
| [`cache-dit-cache-integration`](https://github.com/vipshop/cache-dit/tree/main/.agents/skills/cache-dit-cache-integration) | Deciding whether a new caching paper is worth integrating, and where it lands: the three-layer split (skip timing / skip granularity / replacement operator), the DBCache / `CalibratorBase` seams, the 12-point paper triage, the already-judged cases (HiCache / MagCache / SVD-Cache / DiCache), and the acceptance gate (raw-float PSNR, `cached_steps > 0`, bit-exact at `max_cached_steps=0`). |
| [`cache-dit-attention`](https://github.com/vipshop/cache-dit/tree/main/.agents/skills/cache-dit-attention) | Adding or changing an attention backend under `src/cache_dit/attention/`: the `_AttnBackend` enum and `_AttnBackendRegistry` (signature filtering, constraints, CP capability), the NHD dispatch contract, the templated-CP forward/backward ops, the diffusers bridge and `set_attn_backend` wiring, the `--attn-params type=` CLI choices, `torch.library` wrapping for `torch.compile`, and the acceptance matrix. Uses ffpa-attn, SageAttention-2/3, Sol-Attn and flash-varlen as worked precedents. |
| [`cache-dit-distributed`](https://github.com/vipshop/cache-dit/tree/main/.agents/skills/cache-dit-distributed) | Understanding or extending `src/cache_dit/distributed/`: `ParallelismConfig` / `ParallelismBackend` derivation, the CP plan DSL and split/gather hooks, Ring / Ulysses / USP attention, the all-to-all and P2P primitives (FP4/FP6/FP8 Ulysses communication, async Ulysses), tensor-parallel planners and the CP+TP hybrid 3D mesh, plus the text-encoder / VAE / ControlNet parallelism paths. Includes the per-model planner inventory and the add-a-planner runbook. |

## Install

```bash
gh skill install vipshop/cache-dit cache-dit-model-integration --allow-hidden-dirs
```

`--allow-hidden-dirs` is required because the skills live under the hidden `.agents/` directory. Install one skill per command; add the skill name you need.

## Typical Prompts

```bash
# Support a new model, then verify it.
/cache-dit-model-integration "Download and support MiniMax-H3 in Cache-DiT."
/cache-dit-model-integration "Run the perf matrix for MiniMax-H3 on 8 GPUs."

# Quantize an existing model and report the accuracy drop.
/cache-dit-quantization "Quantize FLUX.2-Klein with svdq nvfp4 r128 vs bf16."

# Work on an operator.
/cache-dit-custom-kernel "Speed up the SVDQ NVFP4 producer on sm120 (NCU)."

# Add or debug an attention backend.
/cache-dit-attention "Add the vLLM FlashAttention-4 kernel as a --attn-params type=<backend> backend."
/cache-dit-attention "--attn-params type=sage runs but the output matches SDPA bit-for-bit, find out why."
/cache-dit-attention "Make the new backend work with Ulysses context parallelism."

# Understand or extend the distributed module.
/cache-dit-distributed "Add a CP planner for the new transformer and a matching tiny CP+TP test."
/cache-dit-distributed "--parallel-params tp=4 on 4 GPUs gives wrong output for Wan2.2-S2V, narrow it down."
/cache-dit-distributed "Explain how VAE parallelism tiles the decoder and why rank 0 blends."

# Evaluate a caching paper before writing any code.
/cache-dit-cache-integration "Is this paper worth adding as a DBCache calibrator?"
/cache-dit-cache-integration "Triage the 2026 cache papers and rank them by integration value."
```

The prompt does not need to spell out the procedure: each skill starts with a route check (is this a new model, a second-pass optimization of a model diffusers already ships, a benchmark, or a fix?), then loads only the references that route needs.

## Structure

- `SKILL.md` — the router. It states the scope, the routes, the hard rules for the task and the validation gates, and points at the references.
- `references/*.md` — the details per topic: `cp.md`, `tp.md`, `tep.md`, `vaep.md`, `testing.md`, `custom_models.md`, `perf_matrix.md`, `kernel-integration-chain.md`, `dispatch-chain.md`, `backend-contract.md`, `architecture.md`, `context-parallel.md`, `planner-inventory.md`, ...
- Paths inside the skills are repository-relative, and the commands are the ones already used by `examples/` and `docs/`, so a skill run produces the same commands a maintainer would type.

## Working on the Skills

The skills live with the code. If your change alters a workflow a skill describes — a renamed flag, a new planner, a new extension family, a changed acceptance gate — update the matching `SKILL.md` or reference in the same pull request, so the next agent run does not follow a stale procedure. Keep user-facing behaviour in the user docs ([`docs/user_guide`](../user_guide/OVERVIEWS.md)) and let the skill link to it instead of duplicating it.

## See Also

- [Support New Model](./SUPPORT_NEW_MODEL.md) — the manual version of the model-integration skill.
- [Attention Backend](../user_guide/ATTENTION.md) — the user-facing backend list, hardware scope and per-backend limitations the attention skill keeps in sync.
- [Context Parallelism](../user_guide/CONTEXT_PARALLEL.md) — the user-facing `--parallel-params` combinations, UAA / async / low-precision communication and Ring / USP notes the distributed skill keeps in sync.
- [Prepare before commit](./PRE_COMMIT.md) — pre-commit setup, which every skill assumes.
- [Quick Examples](../EXAMPLES.md) — the commands and checkpoints the skills drive.
