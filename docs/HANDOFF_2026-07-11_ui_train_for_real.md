# HANDOFF 2026-07-11 — browser-only "train for real" golden path (COMPLETE)

Executes `serenitymojo/docs/HANDOFF_2026-07-11_ui_completion.md` (the spec).
Everything below is MEASURED this session (curls, unit tests, and a full
Playwright browser voyage on the live :8188 supervisor). New trainer binary is
deployed on :8188.

## Result: the maiden voyage PASSED, browser-only

raw image folder → caption → prepare cache → launch → **LoRA file**, driven
entirely through the `:8188` UI (Playwright clicks → real jobs → real artifacts):

- **Caption** (Captioner tab, engine=**Mojo**): 4 real `.txt` sidecars written by
  the pure-Mojo `qwen3vl_caption` (e.g. "A serene, intimate portrait of a young
  woman…"). ~15-20 s/image (per-image model load; one-shot binary).
- **Prepare** (Dataset tab, "Prepare cache"): Stage A `krea2_stage_images.py`
  (raw jpg+txt → `_staged/images.safetensors`, CPU) → Stage B `krea2_prepare_cache`
  (VAE + Qwen3-VL encode → `browser_cache.safetensors`, GPU). `#cache` auto-fills.
- **Train** (Start training, max_steps=3): `[fp8cache] 28/28 blocks` · steps
  loss 0.065/0.439/0.054, grad_norm nonzero · peak VRAM 16.5 GB · exited clean.
  **`[save] FINAL LoRA: 224 pairs (448 tensors) → voyage_krea2_3.safetensors`**
  (429 MB, valid krea2 PEFT: keys `diffusion_model.blocks.N.attn.wq.lora_A/B.weight`,
  BF16, rank 64). 0 page/HTTP errors.

Artifacts live under `/home/alex/mojodiffusion/output/voyage*` (raw folder,
`_staged/`, `browser_cache.safetensors`, `voyage_krea2/voyage_krea2_3.safetensors`).

## What shipped (all built + tested)

### T1 — preset↔launch truth matrix (dry-run curls to live :8188)
- Launch-ready (dry_run 200): krea2, klein9b, chroma, ernie, anima, hidream,
  ideogram4, sdxl, sd35.
- 422 (no base-config template / no cache): **zimage, l2p**.
- 501 (unwired, honest): **ltx2, wan22**. (wan22 T2V-14B still needs the ~28 GB
  low-noise ckpt not on disk.)

### T2 — cache-prep in the UI (the missing link)  → `webui/src/prepare.rs` (NEW)
- `POST /api/prepare/run|status|abort`. A preset's `prepare` block is a **step
  list** (`presets.json`): each step is `{interp|bin, cwd, env, argv, produces}`,
  run SEQUENTIALLY — each must exit 0 AND write its `produces` file before the
  next. krea2 = 2 steps (Stage A python + Stage B mojo). Placeholders
  `{stage_dir}/{staged}/{out}/{n}/{size}`; `{staged}` = `<out-dir>/_staged`.
- Single-job mutex + `gpu_busy()>1GB` refusal (shared with captioner/training).
- Honest 4xx: unknown preset 404, out-of-scope 403, n=0 422, **"prepare binary
  not built: <bin>" 422** (klein9b/zimage carry data-only prepare blocks — their
  Stage-B bins aren't built yet).
- **`krea2_prepare_cache` had to be built** (was unbuilt): `mojo build -O2 -I .
  -I /home/alex/MOJO-libs -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib
  -Xlinker -lserenity_cudnn_sdpa serenitymojo/models/krea2/krea2_prepare_cache.mojo
  -o output/bin/krea2_prepare_cache`. Contract:
  `krea2_prepare_cache <stage_dir> <out.safetensors> <n> <SIZE(512|1024)>`.
- Frontend: Dataset tab gains stage-dir/out/n/size + "Prepare cache" → polls
  `/api/prepare/status` → on success sets `#cache` (closes the loop).

### T3 — Mojo captioner engine  → `webui/src/captioner.rs`
- `RunReq.engine` = `"python"` (default, ai-toolkit venv — unchanged) | `"mojo"`
  (loops `output/bin/qwen3vl_caption <image> [prompt] [max_new]`, extracts text
  between `=== CAPTION ===`/`=== END ===`, writes `.txt` sidecars, honors
  `skip_existing`, updates the SAME CapStatus, abortable). Frontend engine select.

### Verification
- `cargo build --release` clean. **17/17 unit tests pass** (incl.
  `prepare::krea2_steps_substitute_the_two_step_chain` against the real
  presets.json, and caption marker extraction).
- Endpoint validation curls all honest (see T2). Full browser voyage above.

## Notes / remaining
- **Test-harness gotcha (not a product bug):** Playwright `waitForFunction` with a
  numeric `polling` does NOT await an async predicate (the returned Promise is
  truthy → resolves on the first poll). Poll from Node via `page.evaluate(async…)`
  instead. Cost us two premature-exit re-runs before the products were confirmed
  fine.
- Prepare step-list generalizes to other models: add a `prepare` block. klein9b /
  zimage are DATA-ONLY today (their Stage-B bins unbuilt) → honest 422.
- LoRA is a structurally-valid re-loadable krea2 PEFT; a live load in the
  inference UI (serenity-server) is the one optional check not yet run.
- Inference UI (serenity-server, mojodiffusion repo) Part I wiring shipped in
  parallel — detail in `serenity-server/UI_WIRING_2026-07-11.md`.
