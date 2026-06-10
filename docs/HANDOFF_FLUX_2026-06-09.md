# Handoff — FLUX.1-dev first genuine pure-Mojo image (2026-06-09)

**Repo:** `/home/alex/mojodiffusion` · **Branch:** `training-port-5models-lora`
**Pushed:** `ebe07fe` (origin/training-port-5models-lora) · **Hardware:** RTX 3090 Ti 24GB

---

## TL;DR

FLUX.1-dev now produces a **real, pixel-verified 1024² image end-to-end in pure
Mojo** (prompt → CLIP+T5 → DiT → denoise → tiled VAE → PNG). The long-standing
all-white FLUX output was a **T5-XXL fp16 overflow → NaN** bug, not (only) a
prompt-assembly placeholder. Two fixes shipped + docs updated + pushed.

This is the **first UI-dropdown model run truly end-to-end** (others are
compile-only / cached-input / blank).

---

## What was the task vs what was actually wrong

- **Requested:** "Polish FLUX tile seams" (overlap+blend the 1024² VAE tiles).
- **Implemented & verified:** 3×3 overlapping tiled decode + feathered blend
  (builds, runs, no OOM). BUT measuring the output revealed every FLUX PNG —
  512², 1024² non-overlap, and 1024² overlap — was **pure white** (mean 255,
  std 0). A blank 1024² PNG compresses to identical bytes each run, which had
  masked the bug behind a false "FLUX works" claim from a prior session.

## Root cause (MEASURED, layer-by-layer)

Traced NaN backward: latent → DiT `pred` → **T5-XXL hidden = 100% NaN** (CLIP
pooled and noise were clean). Instrumented T5 internals:

```
layer6 max 1344  → layer7 max 21424 → layer9 max 27008
layer10 = ±inf   → layer11 = NaN
```

T5-XXL's residual stream grows to tens of thousands by mid-stack and **overflows
fp16 (max 65504)**. The weights file `t5xxl_fp16.safetensors` was loaded in fp16
(`Tensor.from_view` preserves file dtype), so the whole forward ran in fp16.
This is the classic reason diffusers runs T5 in bf16/fp32.

## Fixes shipped

1. **`serenitymojo/models/text_encoder/t5_encoder.mojo` `load()`** — cast every
   T5 weight to **BF16** at load (`cast_tensor(Tensor.from_view(...), BF16)`).
   bf16 range ~3e38 absorbs the large residual stream; matches the port header's
   stated "BF16 storage, F32 accumulation". After fix: **no NaN through all 24
   layers**, final hidden in `[-7.06, 2.39]`.
2. **`serenitymojo/pipeline/flux_sample_cli.mojo`** — `_tiled_decode` now does a
   **3×3 overlapping** decode (64² latent crops at stride 32 → 256px image
   overlap) with separable feathered cross-fade blend (`_weight_tensor` /
   `_xfade` / `_blend3`). Tiles stay at the proven 64² size and are decoded/freed
   per row → retained-memory peak ≈ the working 2×2 path. (A 2×2/72² overlap
   OOM'd in the tight post-DiT allocator pool — measured `CUDA_ERROR_OUT_OF_MEMORY`.)

## Verification (this session, measured)

- Clean rebuild `/tmp/flux_clean`, run: 20 steps → `[vae] 3x3 overlap+blend` →
  `image shape 1 3 1024 1024`.
- PIL pixel check `/tmp/flux_clean_1024.png`: **mean 66.3, std 49.80,
  frac==255 0.001, RGB 87.2/63.5/48.2** (reddish, matches "red apple on wooden
  table"). Visually confirmed a photorealistic red apple on a wooden table.
- Intermediate stats post-fix: `t5_hidden` no NaN, `latent` no NaN
  (`[-4.91, 5.16]`), decoded floats `[-1.03, 1.12]` (in SIGNED range).

## How to reproduce

```bash
cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
pixi run mojo build -I . -I /home/alex/serenity-trainer/src \
  -Xlinker -lm -Xlinker -lcuda \
  serenitymojo/pipeline/flux_sample_cli.mojo -o /tmp/flux_clean
# args: <config|-> <lora|-> <prompts.json> <id> <out.png>
MODULAR_DEVICE_CONTEXT_SYNC_MODE=true /tmp/flux_clean - - /tmp/flux_prompts.json apple /tmp/out.png
```
`/tmp/flux_prompts.json` = serenity.sample_prompts.v1 with one `apple` prompt.
Offload build needs `-lcuda`, NO prlimit. `MODULAR_DEVICE_CONTEXT_SYNC_MODE=true`
lowers the allocation peak (helps the post-DiT VAE decode fit).

## Commits

- `db55b08` — code: T5 bf16 cast + 3×3 overlap-blend VAE decode.
- `ebe07fe` — docs: `serenitymojo/docs/SDXL_FLUX_KLEIN_PORT_STATUS.md` 2026-06-09
  pass entry + corrected FLUX status/blockers.
- `serenityUI/MODEL_UI_PLUMBING_AUDIT.md` updated locally (different repo, kept
  local per campaign rule).

---

## NOT done / next steps (honest scope)

FLUX produces **real coherent pixels**, NOT verified **quality-parity** vs a
reference. Open items (from the updated status doc):

1. **Quality parity** — diff the Mojo FLUX image against a diffusers/Rust
   reference run (same prompt/seed/steps). Current evidence = "real pixels", not
   numeric parity.
2. **FLUX LoRA at inference** — CLI accepts `<lora>` but ignores it today.
3. **JSON-driven sampler params** — steps/guidance/seed/size are comptime-fixed;
   wire them from the prompt JSON (currently logged as "ignored").
4. **Single-shot 1024² decode still OOMs** the post-DiT pool — tiled VAE is the
   mitigation; a memory-lifetime cleanup around encoder/DiT/VAE scopes could
   remove the need for tiling.
5. **Seam quality** — overlap+blend is structurally correct (256px feathered
   cross-fades) but NOT A/B'd against a non-tiled 1024² reference (that decode
   OOMs). Eyeball the delivered image for residual seams.

## Cross-model implication (worth checking)

The T5 fp16→NaN bug likely affects **any model loading `t5xxl_fp16.safetensors`
through `T5Encoder`** — i.e. the planned `chroma_encode_runtime`, `sd3_encode_runtime`,
and Anima T5 paths. The bf16-at-load fix is in `T5Encoder.load`, so they inherit
it, but confirm none of them re-load T5 in fp16 by another route.

## UI-plumbing audit status (Ideogram4 bar)

Per `serenityUI/MODEL_UI_PLUMBING_AUDIT.md`: FLUX is now the **first** model at
the bar (FULL conditioning, RUN, pixel-verified). Next cheapest wins:
- Run the already-FULL/PRECACHE models on GPU (Z-Image, Qwen-Image, Klein, ERNIE)
  — all compile-only, never run.
- Build `*_encode_runtime` for SDXL (dual CLIP) and Chroma (T5-only) — smallest
  conditioning lift, and Chroma reuses the now-fixed T5 encoder.
