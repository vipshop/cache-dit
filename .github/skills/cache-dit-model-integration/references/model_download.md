# Model Download Reference (ModelScope first, HuggingFace fallback)

When to read this: before Phase 0 ("Model files downloaded to local path") of the integration TODO list, or whenever model weights need to be fetched to the local model root. Return to `../SKILL.md` for the integration workflow.

All local models live under `/workspace/dev/vipdev/hf_models/` (see [`./testing.md`](./testing.md) for the env-var wiring). Both flows below produce the same directory layout. **Never let the generate CLI or diffusers download from the network at runtime** — always pass the local path via environment variables or `--model-path`.

Default choice: **ModelScope**. HuggingFace is often unreachable from this network; ModelScope mirrors most HF repos (including `-Diffusers` weights) and is fast.

## D.1 Default Flow — ModelScope

### Step 1 — check the model exists on ModelScope

ModelScope repo ids usually mirror HF ids (`org/name`), but not always. Check before downloading.

**Human / browser** — search page: `https://www.modelscope.cn/search?search=<Model-Name>`, e.g. [https://www.modelscope.cn/search?search=Qwen-Image-2.1](https://www.modelscope.cn/search?search=Qwen-Image-2.1). Open the model page and confirm the exact repo id (e.g. `Qwen/Qwen-Image-2.1`).

**Agent** — three verified ways (as of 2026-10-22):

1. **Playwright MCP (renders the SPA)** — navigate to `https://www.modelscope.cn/search?search=<Model-Name>` with `playwright_navigate` (headless), then `playwright_get_visible_text` to read the result cards (repo ids, sizes, download counts). ⚠️ The MCP server name registered in `~/.codebuddy/mcp.json` is **`Playwright MCP Server`** (with spaces) — passing `"Playwright"` returns "Server not found".
2. **Exact existence check (most reliable)**: `200` = exists (full metadata: downloads, license, file list), `404` = missing:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://www.modelscope.cn/api/v1/models/Qwen/Qwen-Image-2.1
```

3. **Keyword search via the API behind the search box.** ⚠️ Only the `"Name"` field filters; `"Search"` / `"Keyword"` / `"SearchWord"` are silently ignored (they return the unfiltered hot list):

```bash
curl -s -X PUT https://www.modelscope.cn/api/v1/dolphin/models \
  -H "Content-Type: application/json" \
  -d '{"PageSize":10,"PageNumber":1,"Name":"Qwen-Image-2.1"}'
```

Plain HTTP fetch (web_fetch / mcp fetch) of the search page does NOT work — it is a JS-rendered SPA and only returns an empty JS shell ("Page failed to be simplified").

```bash
# Exact existence check: 200 = exists (full metadata: downloads, license, file list), 404 = missing
curl -s -o /dev/null -w "%{http_code}\n" https://www.modelscope.cn/api/v1/models/Qwen/Qwen-Image-2.1

# Keyword search — the API behind the search box. ⚠️ only the "Name" field filters;
# "Search"/"Keyword"/"SearchWord" are silently ignored (they return the unfiltered hot list)
curl -s -X PUT https://www.modelscope.cn/api/v1/dolphin/models \
  -H "Content-Type: application/json" \
  -d '{"PageSize":10,"PageNumber":1,"Name":"Qwen-Image-2.1"}'
```

If the id differs from HF, use the ModelScope id in the download command. If the model is not on ModelScope at all (404 on the exact check, no relevant rows in the search), fall back to D.2.

### Step 2 — download into the local model root

```bash
cd /workspace/dev/vipdev/hf_models
modelscope-hub download --model Qwen/Qwen-Image-2.1 --local_dir ./Qwen-Image-2.1
```

⚠️ Use the **`modelscope-hub` CLI** (package `modelscope-hub`, verified v0.4.6). The old `modelscope` CLI is deprecated — do not use it.

Verified variants of the same command (all accepted by the CLI):

```bash
modelscope-hub download Qwen/Qwen-Image-2.1 --local-dir ./Qwen-Image-2.1   # repo id as positional arg
modelscope-hub download --model Qwen/Qwen-Image-2.1 --local_dir ./Qwen-Image-2.1   # --model + underscore --local_dir
```

Useful options: `--include` / `--exclude` glob patterns for partial snapshots (e.g. skip original-format weight folders when a repo hosts both formats), `--revision` for a branch/tag, `--force` to re-download. Private repos: `ms login` or `--token`.

### Step 3 — verify and wire the env var

```bash
ls /workspace/dev/vipdev/hf_models/Qwen-Image-2.1   # expect model_index.json / *.json / *.safetensors
export QWEN_IMAGE_21_DIR=/workspace/dev/vipdev/hf_models/Qwen-Image-2.1      # see ./testing.md, ./cli.md
```

## D.2 Fallback Flow — HuggingFace via hf-mirror.com (field-verified)

Direct HF is blocked from this network (`curl https://huggingface.co` times out), but the mirror endpoint is **field-verified (2026-10-22, `Qwen/Qwen-Image-2.1`, 17.6 GB)**. Use this flow for HF-only weights (e.g. `NucleusAI/NucleusMoE-Image`, which has no ModelScope mirror); prefer D.1 when the model is on ModelScope.

```bash
cd /workspace/dev/vipdev/hf_models
HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen-Image-2.1 --local_dir ./Qwen-Image-2.1
```

- `HF_ENDPOINT=https://hf-mirror.com` routes all huggingface_hub API/CDN requests through the mirror — **required, not optional**, on this network.
- A benign `Exception ignored in: module 'threading'` warning may appear mid-download (hub worker threads); the download continues and completes — ignore it unless files are missing at the end.
- `hf download` is the current huggingface_hub CLI; the legacy `huggingface-cli download` alias still works but is deprecated.
- Same options as ModelScope's CLI conceptually: `--include` / `--exclude` globs, `--revision`, resume on re-run.

## D.3 Conventions and Pitfalls

- **Keep the local dir name equal to the repo name** (matches the `testing.md` convention `export <MODEL>_DIR=/workspace/dev/vipdev/hf_models/<your-model-dir>`).
- **Prefer `-Diffusers`-suffix weight repos** when a model ships both original and diffusers layouts (e.g. `Wan-AI/Wan2.2-Animate-2-14B-Diffusers`) — cache-dit and diffusers load the diffusers layout.
- Some large repos host multiple formats side by side; use `--exclude` to skip the format you do not need instead of downloading everything.
- After download, re-check `model_index.json` exists for diffusers-format repos — its absence usually means you grabbed the original-format repo.
