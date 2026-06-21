# Ideogram-4 — full session handoff (2026-06-20)

Cross-repo handoff for the Ideogram-4 work: the **Rust trainer** (EriDiffusion-v2),
the **Mojo inference** (mojodiffusion/serenitymojo), and the **Mojo trainer**
(serenity-trainer — the current focus). Everything below is MEASURED in-session
(RTX 3090 Ti, 24 GB). Where a claim is not yet measured it is labeled HYPOTHESIS.

---

## 0. TL;DR — what changed this session

1. **Rust trainer proven end-to-end** — real 1000-step LoRA on the full 70-image
   gigerver3 set; produces a working ai-toolkit LoRA that demonstrably imparts the
   Giger style (clean with/without test). Two real bugs fixed (prep OOM, F16 save).
2. **Mojo inference quality fixed** — the sampler used the wrong preset's noise std
   (1.5 = 48-step QUALITY) on 20-step runs; corrected to 1.75 (V4_DEFAULT_20).
   Verified sharper; a photographic prompt now matches the Rust path.
3. **Mojo trainer telemetry fixed (committed)** — `grad_norm` was a hardcoded fake
   `0.0000` (in `Ideogram4LoRATrainer.mojo`) AND `apply_ideogram4_lora_grads`
   returned stub `grad_b_l1=0.0`/`adapter_b_l1=0.0` (in `Ideogram4StackTrain.mojo`).
   The trainer DOES learn (LoRA-B 0→nonzero, measured), but every learning signal it
   printed was fake. The stub also made the **10-step gate RED** (it asserts
   `result.grad_b_l1 > 0`). Both files fixed: real LoRA-B gradient L1 on the progress
   line, and `apply_ideogram4_lora_grads` now returns the real grad-B L1 (pre-step)
   and param-B L1 (post-step). **Gate now GREEN** (loss byte-identical → only
   telemetry changed). Committed this session.
4. **Mojo compile-OOM switch documented** — `--num-threads`/`-j` caps the
   threads×per-unit memory multiplier; robust recipe = `--optimization-level 2 -j 4`.

---

## 1. The three Ideogram-4 implementations (orientation)

| Layer | Repo | Entry point | Status |
|---|---|---|---|
| **Rust trainer** | `/home/alex/EriDiffusion/EriDiffusion-v2` | `train_ideogram` (bin) | PROVEN end-to-end this session |
| **Mojo trainer** | `/home/alex/serenity-trainer` | `target/serenity_ideogram4_live_trainer` | runs + learns; telemetry just fixed; not yet proven via inference |
| **Mojo inference** | `/home/alex/mojodiffusion/serenitymojo` | `pipeline/ideogram4_generate{,_lora}.mojo` | works; std fix landed + pushed |
| **OneTrainer (oracle)** | `/home/alex/OneTrainer-dxqb-ideogram` | `scripts/train.py` | baseline established (loss 0.811) |

Model weights:
- fp8 (Mojo + Rust): `/home/alex/.serenity/models/ideogram-4-fp8/`
- bf16 diffusers (OneTrainer only — OT can't load fp8): `/home/alex/.serenity/models/ideogram-4-bf16-diffusers`

Dataset: `/home/alex/1/datasets/gigerver3_json/` — 70 `.jpg` + 70 `.json` (structured
Ideogram-4 captions, "gigerver3" Giger-biomechanical style).

---

## 2. BASELINES — measured loss / speed / artifact (for cross-testing)

All three train the SAME 70-image gigerver3 set, rank 16, alpha 16. **Read the
caveats — the headline numbers are NOT apples-to-apples.**

| Trainer | model | res | batch | optim | lr | loss (smooth) | s/step | LoRA file | grad_norm shown |
|---|---|---|---|---|---|---|---|---|---|
| **OneTrainer** | bf16 | 512 | 2 | AdamW | 5e-5 | **0.811** @99 steps (3 ep) | 2.31 s/it (≈1.16/sample) | 95 MB | **no** (not displayed) |
| **Rust `train_ideogram`** | fp8 | 256 | 1 | AdamW | 1e-4 | **0.815** @1000 steps (0.927→0.815) | 2.30 s/step | 105 MB F16 | ~0.10–0.21 (**L2, all grads, clip-by-norm**) |
| **Mojo live trainer** | fp8 | 512 | 1 | AdamW | 1e-4 | ~1.0–1.12 @5 steps (UNTRAINED) | 5.4–5.7 s/step | 105 MB bf16 | 46–134 (**L1, B-grads only**) |

### Caveats (critical — do not over-read the table)
- **grad_norm is three different metrics.** Mojo = **L1 norm of LoRA-B gradients
  only** (sum of |g| over ~20M B elements). Rust = **L2 norm of ALL LoRA grads,
  after clip-by-norm**. OneTrainer = **not reported at all**. So Mojo's `108` vs
  Rust's `0.15` is NOT "700× bigger gradients" — it is L1-of-millions vs L2-clipped.
  They cannot be compared as-is. To compare honestly you must compute the SAME norm
  (e.g. global L2 of all grads, no clip) in each — none of the three currently does.
- **Loss is at different training stages.** Mojo's ~1.0–1.12 is a 5-step UNTRAINED
  smoke (the base-model loss on the style; no learning yet). OT's 0.811 and Rust's
  0.815 are AFTER ~100–1000 steps of training. The untrained Mojo loss matches the
  gated base loss (0.961 class) and the Rust step-1 loss (~0.85–1.12) — consistent.
  A fair Mojo-vs-Rust-vs-OT loss comparison needs equal steps on the same res/batch.
- **Speed gap is real and expected.** Per sample: OT ≈1.16 s (fp-bf16 + torch.compile,
  batch 2), Rust 3.90 s @512 / 2.30 @256 (fp8 dequant + flame), Mojo 5.4 s @512
  (fp8 dequant + MAX/cuDNN, no torch.compile equivalent). Mojo is the newest port;
  the gap is an optimization lever (resident-fp8 + kernel fusion), not a correctness
  issue. **MEASURE before optimizing.**

### How to reproduce each baseline

**OneTrainer** (the oracle; needs the isolated venv — do NOT touch `/home/alex/OneTrainer/venv`):
```bash
cd /home/alex/OneTrainer-dxqb-ideogram
/home/alex/ot-ideogram-venv/bin/python scripts/train.py \
  --config-path /tmp/ideogram_giger_config.json
# config: bf16 model, rank16/alpha16, res512, batch2, AdamW lr5e-5, 3 epochs.
# dataset concept: /tmp/giger_concept.json -> gigerver3_json.
# needs diffusers @9a0aaba36 + bitsandbytes==0.49.1 (in the isolated venv).
# result: smooth loss 0.811, ~2.31 s/it, 99 steps, 6:49 total, 95MB LoRA.
```

**Rust `train_ideogram`** (proven):
```bash
cd /home/alex/EriDiffusion/EriDiffusion-v2
# 1) cache (no_grad fix REQUIRED — without it caching OOMs after ~14 imgs):
LIBTORCH=/home/alex/libs/libtorch LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib \
  ./target/release/prepare_ideogram \
  --input-dir /home/alex/1/datasets/gigerver3_json --output-dir /tmp/ideo_real_cache \
  --vae-ckpt   /home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors \
  --text-encoder /home/alex/.serenity/models/ideogram-4-fp8/text_encoder/model.safetensors \
  --tokenizer-path /home/alex/.serenity/models/ideogram-4-fp8/tokenizer/tokenizer.json \
  --resolution 256
# 2) train (1000 steps -> ai-toolkit F16 LoRA):
LIBTORCH=/home/alex/libs/libtorch LD_LIBRARY_PATH=/home/alex/libs/libtorch/lib \
  ./target/release/train_ideogram \
  --model /home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors \
  --cache-dir /tmp/ideo_real_cache --steps 1000 --rank 16 --lora-alpha 16 --lr 1e-4 \
  --warmup-steps 50 --output-dir /tmp/ideo_real_lora
# output: /tmp/ideo_real_lora/ideogram4_lora.safetensors (F16, 105MB, 408 tensors)
```

**Mojo live trainer** (runs + learns; see §4 for build/run):
```bash
cd /home/alex/serenity-trainer
pixi run ./target/serenity_ideogram4_live_trainer \
  /tmp/mojo_ideo_progress.log -  \
  /home/alex/trainings/ideogram4_giger_cache/cache.safetensors \
  /tmp/mojo_ideo_lora  20 16 16 1e-4 20
# positional argv: progress transformer(-=default) cache output steps rank alpha lr save_every
# cache is the 1.9GB 70-sample giger cache @512 (NT=256), built 06-11.
```

---

## 3. Rust trainer (EriDiffusion-v2) — DONE this session

### Bugs fixed (both committed + pushed to github.com/CodeAlexx)
1. **`prepare_ideogram` OOM (EDv2 `9d863ca`)** — full-dataset caching died after
   ~14 images. ROOT CAUSE (measured): no `no_grad` guard, so the VAE + Qwen3-VL
   encodes accumulated on the global **autograd tape** (saved activations across
   images; the mempool `clear_pool_cache` did NOT help — it's the TAPE not the pool).
   FIX = `AutogradContext::no_grad()` before the loop (mirrors prepare_klein) +
   per-image `drop`+`clear_pool_cache`. Now caches 70/70.
2. **F16 save (flame-core `493b27f`)** — `save_tensors_safetensors` hardcoded
   `"F32"` + wrote 4 bytes/elem via `to_vec()`, so F16/BF16 tensors saved at 2× size.
   Now dtype-aware (F16→half::f16 bytes, BF16→bf16, else F32; round f32→native,
   lossless; F32 path unchanged). LoRA now F16 105MB (was 211MB).

### Proven
- Real run: loss 0.927→0.815 over 1000 steps, 2.30 s/step, fit 21.6GB.
- **Clean LoRA-works test** (the decisive proof): the giger prompt with style words
  ("gigerver3"/"H.R. Giger"/"biomechanical") REMOVED, same seed, base vs +lora →
  base = generic smooth red demon, +lora = learned Giger biomechanical texture.
  Style comes from the LoRA, not the prompt.
- Artifacts in `/home/alex/EriTrainer/output/`: `ideogram4_giger_lora.safetensors`
  (the trained LoRA), `ideogram4_clean_{nolora,withlora}.png`, `ideogram4_photo_snowleopard.png`.

---

## 4. Mojo trainer (serenity-trainer) — CURRENT FOCUS

### Current state (measured)
- The train STEP is fully gated (see `docs/PARITY_GATES.md`): data path (VAE mean +
  Qwen3-VL 13-tap), forward split == monolithic at B=0 (velocity cos 0.9999373),
  loss 0.96100146 (≈ torch 0.961231), backward, AdamW, ai-toolkit save.
- The **driver exists**: `train_ideogram4_lora_from_cache` in
  `src/serenity_trainer/trainer/Ideogram4LoRATrainer.mojo` loads the transformer
  ONCE, streams one cached sample at a time, loops `steps`, saves LoRA + Adam state.
  (The pipeline doc's "DataLoader/train-loop not yet wired" note is STALE.)
- It **runs + learns**: 20-step run @512 = 5.4 s/step; after 20 steps LoRA-B is
  nonzero on **204/204** adapters (B inits to 0 → grads flowed → learning happened).
- LoRA save format: 612 tensors (A/B/alpha × 204), BF16, 105MB (no F32 bug — Mojo's
  `save_safetensors` already writes 2-byte).
- Scope: **204 block adapters only** (6 targets/block × 34 = 204). Same scope as the
  Rust trainer. Global-target LoRA (7 embed/final targets) is unwired in BOTH.

### Bug fixed this session: FAKE telemetry (COMMITTED)
`apply_ideogram4_lora_grads` (in `Ideogram4StackTrain.mojo:69`) returned
`Ideogram4StackTrainResult(..., grad_b_l1=0.0, adapter_b_l1=Float32(0.0))` — both
**stubs** (the adamw_step calls DO apply grads, so B learns, but the returned metrics
were 0). And the progress line in `Ideogram4LoRATrainer.mojo` hardcoded the literal
string `" | grad_norm 0.0000 | "`. Net effect: the trainer looked dead (grad_norm 0,
no learning signal) while actually learning.

**FIX (in `Ideogram4LoRATrainer.mojo`, uncommitted):** compute the real LoRA-B
gradient L1 directly from `grads.d_b` in the step loop, BEFORE the optimizer consumes
`grads` (works for both the default-AdamW and levers paths), and thread it to the
progress line:
- `~line 300`: `var step_grad_l1` loop summing `|grads.d_b[gi][].to_host(ctx)|`
  (`to_host` returns `List[Float32]`, no cast needed).
- `~line 351`: pass `step_grad_l1` to `_append_ideogram4_live_progress`.
- `~line 527`: new `grad_norm: Float32` param.
- `~line 545`: `" | grad_norm " + String(grad_norm) + " | "` (was hardcoded "0.0000").

**VERIFIED** (5-step run, `--optimization-level 2` rebuild): grad_norm now
`108.70, 50.79, 86.84, 46.56, 133.86` — real, varying per-step values; loss values
byte-identical to before (1.12493, …) proving only telemetry changed, not the math.

> **NOTE on the metric:** `step_grad_l1` is the **L1 norm of the LoRA-B gradients
> only**. It is honest and nonzero, but it is NOT the same norm as Rust's grad_norm
> (L2 of all grads, post-clip). If cross-trainer grad comparison matters, add a
> global L2 (A+B, no clip) to all three.
>
> **UPDATE (committed):** `apply_ideogram4_lora_grads`'s stub returns are now real —
> `grad_b_l1` = Σ|d_b| over all adapters (pre-step), `adapter_b_l1` = Σ|b| over all
> adapters (post-step, mirrors the levers path at `:291-304`). This was REQUIRED, not
> optional: the 10-step gate (`test_ideogram4_lora_block_10step.mojo`) asserts
> `result.grad_b_l1 > 0` and was failing (`IDEOGRAM4 LORA 10STEP FAIL: final LoRA-B
> gradient was zero`). Now passes: `grad_b_l1` 16.23→15.36, `b_l1` 0.77→7.18, loss
> byte-identical (1.6162775→1.4668901). The gate's own `_total_b_l1`=7.1778255 equals
> the returned `adapter_b_l1` exactly, cross-confirming the param-L1.

### Build (use the OOM-safe recipe)
```bash
cd /home/alex/serenity-trainer
pixi run mojo build --optimization-level 2 -j 4 -I . -I src -I /home/alex/mojodiffusion \
  -Xlinker -lm src/serenity_trainer/trainer/Ideogram4LiveTrainer.mojo \
  -o target/serenity_ideogram4_live_trainer
# ~1 min at -O2, ~6GB RAM. The pixi task `ideogram4-live-trainer-build` does NOT
# pin -O2/-j — prefer this explicit command (or fix the task).
```

### Run / cache
- Binary: `target/serenity_ideogram4_live_trainer` (rebuilt 2026-06-20 with the fix).
- Cache: `/home/alex/trainings/ideogram4_giger_cache/cache.safetensors` (1.9GB, 70
  samples @512, NT=256, built 06-11 by `smoke/ideogram4_prepare_cache.mojo`).
- argv (positional, `-` = default): `progress transformer cache output steps rank alpha lr save_every` (+ argv10 caption_dropout, argv11 levers JSON).
- Progress line on stdout AND `progress_file`: `[Ideogram4-lora] ... | loss X | smooth_loss X | grad_norm X | Ns/step | elapsed | ETA`.

---

## 5. Mojo inference (mojodiffusion) — std fix DONE + pushed

Pushed to `github.com/CodeAlexx/mojodiffusion` branch `training-port-5models-lora`
(`d121610..35c5e30`): `fb07782` (std fix), `3b7fec6` (comments), `35c5e30` (docs).

- **Bug:** sampler used logit-normal **std=1.5** on a 20-step run, but 1.5 is the
  48-step **V4_QUALITY** preset's std; 20-step **V4_DEFAULT_20** uses **1.75**
  (V4_TURBO_12 also 1.75). Source of truth = inference-flame `scheduler.rs`
  `SamplerParameters`. Wrong std softened output ("rust did better").
- **Fix:** `pipeline/ideogram4_generate{,_lora}.mojo` std 1.5→1.75, guidance polish
  `step<3`→`step<2`; `serve/ideogram4_backend.mojo` preset-aware
  `std = 1.75 if steps<=20 else 1.5`. Daemon build 0 errors.
- **Verified:** same prompt+seed+lora std=1.75 is visibly sharper; a *photographic*
  prompt (snow leopard) renders sharp/photorealistic, on par with the Rust path. The
  giger demon's painterly look is PROMPT-driven (`medium=painting`), not a bug.
- LoRA inference path: `pipeline/ideogram4_generate_lora.mojo` + `load_lora` reads
  ai-toolkit keys (`diffusion_model.layers.N.<module>.lora_A/B.weight`, 204 adapters).
  cuDNN shim MUST be linked at run: `-Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker
  -lserenity_cudnn_sdpa -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib`.

---

## 6. Mojo compile-OOM switch (learned this session)

- `--num-threads <N>` / `-j` (from `mojo build --help`): default `0` = all threads;
  Mojo compiles one unit per thread, each holding its own IPO/inlining working set →
  **peak ≈ threads × per-unit**. Orthogonal to `-O` (which sets per-unit cost).
- `-O3` (default) ≈ 48GB/unit → OOMs even at `-j 1`; `-O2` ≈ 2GB/unit.
- **Robust recipe: `--optimization-level 2 -j 4`.** `-j` lets you keep a higher `-O`
  within RAM by trading build speed. Recorded in memory `project-mojo-o3-compile-oom`.

---

## 7. Open items / next steps

1. ~~**Commit the Mojo trainer grad_norm fix**~~ **DONE.** Both telemetry files
   committed; 10-step gate re-run and GREEN (built `-O2 -j4`); live-trainer binary
   rebuilt with both fixes (compiles, exit 0).
2. ~~**Populate the stub returns**~~ **DONE** (was a prerequisite for #1, not
   optional — the gate reads `result.grad_b_l1`). `grad_b_l1`/`adapter_b_l1` now real.
3. **Prove the Mojo-trained LoRA in inference** — run the live trainer for real
   (e.g. 1000 steps), then load its LoRA into `ideogram4_generate_lora.mojo` and
   generate (like the Rust proof). This closes the Mojo loop.
4. **Fair cross-trainer comparison** — equal steps/res/batch, and the SAME grad norm
   (global L2, no clip) in all three, to actually compare Mojo vs Rust vs OneTrainer.
5. **Speed** — Mojo 5.4 s/step @512 vs Rust 3.90 vs OT ~1.16/sample. Profile (nsys)
   before optimizing; candidates = resident-fp8 reuse, kernel fusion. MEASURE first.
6. **Global-target LoRA (7 targets)** — unwired in both Rust + Mojo (204-block-only).

## Key paths
- Rust: `/home/alex/EriDiffusion/EriDiffusion-v2` (bins in `target/release/`)
- Mojo trainer: `/home/alex/serenity-trainer` (binary `target/serenity_ideogram4_live_trainer`)
- Mojo inference: `/home/alex/mojodiffusion/serenitymojo/pipeline/`
- OneTrainer: `/home/alex/OneTrainer-dxqb-ideogram` + venv `/home/alex/ot-ideogram-venv`
- Dataset: `/home/alex/1/datasets/gigerver3_json/`
- Trained LoRAs + sample images: `/home/alex/EriTrainer/output/`
- Mojo giger cache: `/home/alex/trainings/ideogram4_giger_cache/cache.safetensors`
