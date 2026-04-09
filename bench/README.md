# Benchmarks

`bench/` now uses a split layout:

- `cache/`: pipeline-level cache, distillation, plotting, metrics, prompts, and legacy benchmark assets.
- `kernels/`: reusable kernel microbenchmarks. Scripts under this directory follow the `bench_xxx.py` naming style.

Typical entry points:

```bash
cd cache-dit/bench/cache
bash bench.sh default
bash metrics.sh
```

```bash
cd cache-dit
python bench/kernels/bench_svdq_runtime.py
```

Temporary benchmark outputs should continue to go under `cache-dit/.tmp/`.
