# cache-dit Copilot Instructions

- Use the `operator-migration` skill for requests about operator or kernel migration in cache-dit.
- This includes CUDA or Triton operator ports, nunchaku or deepcompressor kernel imports, `torch.library` registration, public wrapper design, optional native extension packaging, and layered kernel or module validation.
- Treat that skill as the default workflow for these tasks instead of starting with blind copy-paste migration.
- Keep migration references portable: use repo-relative paths for cache-dit files, and repository-relative or GitHub-searchable paths for external repos.
