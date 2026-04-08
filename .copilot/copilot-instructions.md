# cache-dit Copilot Instructions

- Use the `operator-migration` skill for requests about operator or kernel migration in cache-dit.
- This includes CUDA or Triton operator ports, nunchaku or deepcompressor kernel imports, `torch.library` registration, public wrapper design, optional native extension packaging, and layered kernel or module validation.
- Treat that skill as the default workflow for these tasks instead of starting with blind copy-paste migration.
- Keep migration references portable: use repo-relative paths for cache-dit files, and repository-relative or GitHub-searchable paths for external repos.

- Use the `ptq-workflow-integration` skill for requests about adding or reviewing a PTQ workflow in cache-dit.
- This includes quantize/load API design, `QuantizeConfig` validation, backend-specific PTQ orchestration, serialization manifests such as `quant_config.json`, public docs, and fast/slow/compile test coverage planning.
- Treat that skill as the default workflow for PTQ integration tasks instead of copying the SVDQ PTQ implementation mechanically.
- Use the SVDQ PTQ integration only as a reference for API design style, file placement, test layering, and coverage scope. Do not treat it as a copy-paste template.
