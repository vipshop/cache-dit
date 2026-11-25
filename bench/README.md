# Benchmarks

`bench/` now uses a split layout:

- `cache/`: pipeline-level cache, distillation, plotting, metrics, prompts, and legacy benchmark assets.
- `kernels/`: reusable kernel microbenchmarks. Scripts under this directory follow the `bench_xxx.py` naming style.
- [`profile/`](profile/README.md): request/step profiling and explicit model instrumentation through a shared runner.
- Full-clip H3 video VAE decode benchmark. [h3_media/](h3_media/bench_vae.py)

Typical entry points:

```bash
cd cache-dit/bench/cache
bash bench.sh default
bash metrics.sh
```

Temporary benchmark outputs should continue to go under `cache-dit/.tmp/`.

## H3 video VAE

Run from the repository root with the MiniMax-H3 video VAE checkpoint and a Diffusers build that provides `AutoencoderKLMiniMaxH3`, the video VAE shared by MiniMax-H3 and Pika-H3-Flex. All modes use the same FP32 weights and seeded latent for a 1344 × 768, 124-frame video.

```bash
for mode in replicated parallel batched; do
  PYTHONPATH=src torchrun --standalone --nproc-per-node=2 bench/h3_media/bench_vae.py \
    --vae-dir "$MINIMAX_H3/vae" --mode "$mode"
done
```

Each GPU decodes all tiles in `replicated`, its own tiles one at a time in `parallel` (B1), or all 14 local tiles per temporal window in `batched` (B14). Each run warms up once and times three full `vae.decode()` calls, including tile exchange and stitching in the parallel modes and temporal blending in all modes. Rank 0 reports its timings and median, along with the largest peak allocation across both GPUs.
