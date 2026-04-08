# cache-dit Copilot Instructions

- Use the `operator-migration` skill for requests about operator or kernel migration in cache-dit.
- This includes CUDA or Triton operator ports, nunchaku or deepcompressor kernel imports, `torch.library` registration, public wrapper design, optional native extension packaging, and layered kernel or module validation.
- Treat that skill as the default workflow for these tasks instead of starting with blind copy-paste migration.
- Keep migration references portable: use repo-relative paths for cache-dit files, and repository-relative or GitHub-searchable paths for external repos.

- Use the `cuda-cpp-kernel` skill for low-level CUDA C++ or PTX kernel work in cache-dit.
- This includes handwritten CUDA kernel implementation, debugging, Nsight Systems or Nsight Compute analysis, bank-conflict and occupancy investigation, shared-memory and register-pressure tuning, and architecture-specific analysis on sm89, sm90, sm100, sm103, and sm120.
- Treat that skill as the default workflow for CUDA or PTX kernel implementation and profiling tasks instead of giving generic CUDA advice from memory.

- Use the `cutlass-cpp-kernel` skill for CUTLASS or CuTe C++ kernel work in cache-dit.
- This includes CUTLASS example navigation, collective-builder decisions, GEMM schedule or epilogue tuning, CuTe C++ layout or atom analysis, and rewrite reviews for operators implemented with CUTLASS-style C++ kernels.
- Treat that skill as the default workflow when the task centers on CUTLASS or CuTe C++ structure rather than generic CUDA runtime behavior.

- Use the `cute-dsl-kernel` skill for CuTe DSL kernel work in cache-dit.
- This includes CuTe DSL Python kernel authoring, reading bundled CuTe DSL references, integrating generated kernels into cache-dit, and rewriting an existing CUDA or C++ operator into CuTe DSL.
- Treat that skill as the default workflow for CuTe DSL tasks instead of treating CuTe DSL as interchangeable with CUTLASS C++ or handwritten CUDA.

- When a task combines repository integration with kernel implementation, pair `operator-migration` with the most relevant kernel skill instead of treating them as alternatives.
- For operator or kernel implementation and rewrite tasks, follow the skill-level validation rules: add unit tests, compare accuracy and performance against a PyTorch baseline when applicable, and for rewrites compare the new operator against the pre-rewrite implementation on both accuracy and performance.

- Use the `ptq-workflow-integration` skill for requests about adding or reviewing a PTQ workflow in cache-dit.
- This includes quantize/load API design, `QuantizeConfig` validation, backend-specific PTQ orchestration, serialization manifests such as `quant_config.json`, public docs, and fast/slow/compile test coverage planning.
- Treat that skill as the default workflow for PTQ integration tasks instead of copying the SVDQ PTQ implementation mechanically.
- Use the SVDQ PTQ integration only as a reference for API design style, file placement, test layering, and coverage scope. Do not treat it as a copy-paste template.
