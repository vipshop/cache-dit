# Benchmarks

`bench/` now uses a split layout:

- `cache/`: pipeline-level cache, distillation, plotting, metrics, prompts, and legacy benchmark assets.
- `kernels/`: reusable kernel microbenchmarks. Scripts under this directory follow the `bench_xxx.py` naming style.
- [`profile/`](profile/README.md): request/step profiling and explicit model instrumentation through a shared runner.

Typical entry points:

```bash
cd cache-dit/bench/cache
bash bench.sh default
bash metrics.sh
```

Temporary benchmark outputs should continue to go under `cache-dit/.tmp/`.
