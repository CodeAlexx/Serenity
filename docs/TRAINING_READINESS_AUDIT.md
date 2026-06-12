# Training-Readiness Audit — serenity-trainer (CLI, no UI)

> **CORRECTIONS 2026-06-12 (today's work supersedes specific claims below):**
> 1. **Klein**: "Klein has a CLI `def main` (KleinLiveTrainer.mojo:53)" is now a
>    LEGACY path — measured (aa0e2cf): it was hardwired MSE/AdamW, 11 positional
>    argv, zero lever support. The UI (and the pixi klein-live-trainer-build
>    target) now launch serenitymojo's `train_klein_real` as the 7th
>    config-driven runner; the `_force_constant_timestep` fidelity caveat does
>    not apply to that path.
> 2. **HiDream**: "STUB → cannot train · T3" is superseded for HiDream-O1 — it
>    trains via serenitymojo `training/train_hidream_o1_real.mojo` (~1.0 s/step,
>    30-step gate green) launched from this repo's NEW
>    `target/serenity_hidream_live_trainer` runner (eaa88f1). The in-repo
>    OneTrainer-port `BaseHiDreamSetup` stubs themselves remain stubs.
> 3. **Ideogram4** (excluded from this audit) now trains for REAL: the prior
>    run was a one-sample parity fixture at fixed t=0.7 (loss 1.3e-4, grad 0 —
>    measured); fixed in 166882f (real giger cache + per-step logit-normal t).
> 4. Lever capability (06-12, measured by consumption grep): klein/zimage/
>    hidream/ideogram4 consume runtime levers; chroma/ernie/anima/sdxl/l2p
>    consume none — UI warns loudly pre-launch (aa0e2cf).

**Status: RECONCILED — auditor + skeptic + reviewer.** Reviewer personally opened the disputed
files (Z-Image/Lens smokes, the 6 stubs, predict-only setups, Klein driver, factory) and confirmed
every claim below with file:line evidence. No per-model verdict changed during reconciliation; the
only edits vs. the original auditor draft are (a) Flux2-dev "needs config" → "needs weights +
remaining diffusers assets + cache", (b) explicit three-tier effort classification, (c) Z-Image/Lens
full-loop-incl-save confirmed in-process.

Question: for each model (excl. ideogram4), what does it still need to **start training via CLI**
(load ckpt → encode/tokenize OR read cache → batch → forward+activation-save → flow-match loss →
hand-chained backward (all arms) → LoRA-grad + AdamW (or full-FT) → save)? Validation sampler tracked
separately as nice-to-have.

Verdict driver = whether a model's LoRASetup **Spec has BOTH a real `predict` and a real
hand-chained `backward`** (the only thing the shared train recipe can call), plus a runnable driver.

### Effort tiers (assigned per model below)
- **T1 — thin wrapper**: real predict+backward+AdamW+save loop already runs in-process; gap is only
  argparse CLI + an externally pre-built latent/text cache. → Z-Image, Lens. (Klein = T1-complete: it
  already has the CLI; its remaining gaps are fidelity, not plumbing.)
- **T2 — build backward**: real `predict` exists, but `backward` is absent (or a no-op string); needs
  hand-chained backward (all arms) + LoRA-grad gather + AdamW Spec + driver/CLI + cache. → SDXL,
  Qwen, Ernie, Anima, Chroma, Flux.1, SD3.
- **T3 — build from scratch**: bare 2-line `# TODO: port` directory mirror; needs the entire vertical.
  → SD1.5, HiDream, HunyuanVideo, PixArtAlpha, Sana, Wuerstchen.

---

## SHARED INFRASTRUCTURE (frames everything)

REAL and load-bearing — NOT the blocker for any model:
- **Shared step recipe** `trainer/train_step.mojo:114-239`: genuine `predict → tape MSE →
  autograd backward → grad-scale/accum → clip_grad_norm → AdamW(SR/WD) → zero_grad`. Generic over
  `trait ModelSpec` (`modelSetup/BaseModelSetup.mojo:40`). Faithful Serenity step. (skeptic VERIFIED)
- **Generic loop** `trainer/GenericTrainer.mojo:177,286,469` invokes the shared step
  (`train/train_from/train_resume/train_with_progress_file/train_zimage_cadence`).
- **Autograd/optimizer** imported from proven serenitymojo (`backward`, `adamw_step`,
  `clip_grad_norm`, `mse_backward`) — `train_step.mojo:24,35`. Real.
- **Cache reader** `dataLoader/CacheReader.mojo` (328L) — VERIFIED reads real latent+text cache,
  batches, reproducible shuffle.
- **Config/concept readers**, **TrainState** save/load round-trip — VERIFIED.
- **Per-vertical LoRA savers** real (Klein/Z-Image/Lens key parity verified).

REAL-but-UNVERIFIED / cross-cutting GAPs (limit many models at once):
- **No in-Mojo prepare path proven.** `dataLoader/Prepare.mojo` (image→VAE-latent +
  caption→text-embed) is UNVERIFIED (no gate). Consequence: every trainable vertical today requires
  an **externally pre-generated latent+text cache**. "read cache" works; "encode from raw images
  in-Mojo" is unproven. **This is the single biggest shared unblocker** — a verified Prepare gate
  removes the external-cache dependency from all of Klein/Z-Image/Lens (and every T2 model later).
- **No generic CLI / product run layer.** `util/factory.mojo:32,54-94` is **build-only metadata
  dispatch** — `register_*` functions return `FactoryRegistration` structs (`factory.mojo:32`), NOT
  instantiated loader/setup/saver runtime objects (skeptic VERIFIED). `util/create.mojo` likewise.
  The ONLY real CLI `def main` train entrypoints in `trainer/` are `KleinLiveTrainer.mojo:53` and
  `Ideogram4LiveTrainer.mojo` (excluded). Trainable models otherwise reach training by **bypassing
  the factory** via bespoke drivers/smokes.
- **Runnable full-loop entrypoints DO exist as smokes** (correction to auditor's "def main
  scarcity"): `smoke/zimage_realdata_train.mojo:60` and `src/serenity_trainer/smoke/
  lens_train_step_smoke.mojo:72` + `lens_train_gates_smoke.mojo:172` are genuine cache→predict→
  loss→backward→AdamW(→save) `def main`s. The gap to "CLI" is **argparse, not a missing stack.**
- **`trainer/MultiTrainer.mojo` is a 2-line `# TODO: port` stub** (VERIFIED).

Net: the shared spine is real and exercised by 2 of the 3 trainable verticals via `train_step`
(Z-Image `train_zimage_cadence` GenericTrainer:411/469; Lens `LensTrainStep.mojo:42`); only Klein
uses a bespoke loop. The per-model **backward spec** + a **CLI entry** + a **prebuilt cache** are the
real gaps.

---

## PER-MODEL

### Klein (Flux2 family) — **CAN TRAIN TODAY via CLI** (fidelity gaps) · T1-complete
- Spec `modelSetup/Flux2LoRASetup.mojo`: `predict` ✓, `backward` ✓ (**17 backward refs**, VERIFIED) — REAL.
- Driver `trainer/KleinLoRATrainer.mojo:114` (REAL): opens cache, loads real 9B ckpt, builds LoRA,
  `spec.predict → mse_backward → spec.backward_lora → klein_lora_adamw_step → save_flux2_lora`.
- CLI `trainer/KleinLiveTrainer.mojo:53` `def main` (argv ckpt/cache/dataset/output/steps/rank/lr/save).
- Verified: forward cos 0.9994; smoke train finite; 144/144 adapters imprinted; loss-only replay matches.
- **To start CLI training, needs (fidelity only):** (1) external latent+text cache — driver
  `KleinLoRATrainer.mojo:143` errors without one; (2) `KleinLoRATrainer.mojo:160` calls
  `_force_constant_timestep` (def `:323`) → pins ONE timestep, **NOT** Serenity LOGIT_NORMAL
  flow-match sampling → trains but **not faithful**; (3) train-numeric parity (grad-norm/AdamW-delta
  vs `parity/klein_train_ref_*`) still UNPROVEN; (4) it bypasses the shared `train_step` with a
  bespoke loop. Sampler: helper-gate only, not denoise/decode parity.

### Z-Image — **REAL FULL-LOOP TRAIN PATH, smoke-only (no CLI)** · T1
- Spec `modelSetup/ZImageLoRASetup.mojo:149`: `predict:185` ✓, `backward_lora:288` ✓
  (**20 backward refs**, VERIFIED) — REAL.
- REVIEWER-CONFIRMED full loop: `smoke/zimage_realdata_train.mojo:60-135` runs, per step,
  `spec.predict → mse_backward → spec.backward_lora → adamw_step` per adapter on real cached data,
  asserts loss in Serenity range [0.2,0.8] and LoRA-B 0→nonzero, nonfinite=0. **Save proven** in a
  separate gate `smoke/zimage_lora_save_gate.mojo:9` via the REAL `save_zimage_lora` (630 keys ==
  Serenity). Also routes through the shared recipe via `GenericTrainer.train_zimage_cadence:411`.
- **To start CLI training, needs (ordered):** (1) a CLI `def main` (argparse) — no `ZImageLiveTrainer`
  exists; (2) an external cache (`dataLoader/ZImageBaseDataLoader.mojo` is a stub; relies on
  CacheReader/Prepare). Most behaviorally complete vertical.

### Lens — **REAL FULL-LOOP TRAIN PATH incl. in-process SAVE/RESUME, smoke-only (no CLI)** · T1
- Spec `modelSetup/LensLoRASetup.mojo`: `predict` ✓, `backward` ✓ (**14 backward refs**, VERIFIED) —
  REAL. Backward core `model/lens/lens_backward.mojo` (612L); DiT `model/LensDiT.mojo` (1019L) real.
- REVIEWER-CONFIRMED full loop AND save: `lens_train_step_smoke.mojo:117-139`
  (predict→loss→`backward_lora`→`lens_lora_adamw_step`); `lens_train_gates_smoke.mojo` runs three
  gates — **A** AdamW parity vs torch (max|Δ|≤2e-3), **B** 4-step real-data train (loss in [0.30,0.75],
  476/480 adapters move — the 4 last-block txt-post adapters are *architecturally* zero-grad on one
  step), **C** save→reload via the REAL `save_lens_lora`/`load_lens_lora` + a post-resume train step.
  Lens proves the entire predict→backward→AdamW→**save→resume** loop in-process. Uses shared step via
  `LensTrainStep.mojo:42`.
- **To start CLI training, needs (ordered):** (1) a CLI `def main` (argparse) — driven by smokes today
  (`lens_gen_cli.mojo` is a SAMPLER CLI, not train); (2) external cache.

### Flux2 dev — **CODE could train (reuses real Flux2 backward), blocked on missing assets** · T1 (asset-blocked)
- Reuses real `Flux2LoRASetup` backward. Only a structural branch gate
  (`smoke/flux2_dev_branch_check.mojo`) exists.
- **To start CLI training, needs:** dev **weights** + remaining diffusers assets (model_index/
  scheduler/VAE/tokenizer/text_encoder; transformer config.json present locally
  `parity/flux2_dev_train_ref_blockers.json` heads=48) + a built cache + `Flux2Model.is_dev()`
  runtime wiring into the Klein driver (heads==48 branch) + CUDA ref tensors. Then == Klein status.

### SDXL — **predict only, NO backward → cannot train** · T2
- `modelSetup/BaseStableDiffusionXLSetup.mojo`: `predict` ✓, **0 backward refs** (VERIFIED);
  `StableDiffusionXLLoRASetup.mojo` no Spec.
- Has only forward-side gates: LoRA file parity (2382 keys), loss-only replay, sampler-helper,
  adapter-delta contract, 100-step baseline — these prove NOTHING about backward/optimizer.
- **Needs (ordered):** (1) hand-chained backward arms for UNet + dual-text path; (2) LoRA-grad
  gather + AdamW wiring into a `ModelSpec`-conforming Spec; (3) driver/CLI; (4) cache. **Backward is
  the hard blocker.**

### Qwen — **predict only, NO backward → cannot train** · T2
- `modelSetup/BaseQwenSetup.mojo`: `predict` ✓, **0 backward** (VERIFIED); `QwenLoRASetup.mojo` no spec.
- Has LoRA file parity (2160 keys), one-step dump + loss-only replay, sampler-helper. NOTE
  `model/QwenTextEncoder.mojo` is the Z-Image Qwen3 helper, NOT the Qwen-Image text path.
- **Needs:** transformer hand-chained backward, Qwen2.5-VL cached-text branch wiring, LoRA-grad+AdamW
  Spec, driver/CLI, cache. Backward is the blocker.

### Ernie — **predict only, NO backward → cannot train** · T2
- `modelSetup/BaseErnieSetup.mojo`: `predict` ✓, **0 backward** (VERIFIED); `ErnieLoRASetup.mojo` no spec.
- Has LoRA file parity (756 keys), real CUDA one-step dump + loss-only replay, sampler-helper.
- **Needs:** backward arms, LoRA-grad+AdamW Spec, driver/CLI, cache.

### Anima — **predict only, NO backward → cannot train** · T2
- `modelSetup/BaseAnimaSetup.mojo`: `predict` ✓, **0 backward** (VERIFIED); `AnimaLoRASetup.mojo` no spec.
  Reference tree `/home/alex/Serenity-anima-ref`.
- Has LoRA file parity (840 keys), one-step dump + loss-only replay, sampler-helper.
- **Needs:** backward arms, LoRA-grad+AdamW Spec, driver/CLI, cache.

### Chroma — **predict only, backward is a NO-OP string → cannot train** · T2
- `modelSetup/BaseChromaSetup.mojo`: `predict` ✓; "backward" is a single no-op —
  `BaseChromaSetup.mojo:232` appends the string `"ChromaTransformer2DModel forward/backward is not
  implemented here"` (VERIFIED — not a chained backward). `ChromaLoRASetup.mojo` no Spec.
- Has model/setup/loader/saver/LoRA-conversion contracts, LoRA file parity (912 keys), one-step
  dump + loss-only replay, sampler-helper, 100-step baseline.
- **Needs:** real transformer hand-chained backward, LoRA-grad+AdamW Spec, driver/CLI, cache.

### Flux.1 dev — **predict only, NO backward → cannot train** · T2
- `modelSetup/BaseFluxSetup.mojo`: `predict` ✓, **0 backward** (VERIFIED); `FluxLoRASetup.mojo` no spec.
  Distinct from Flux2/Klein — does NOT share the Flux2 backward.
- Has LoRA file parity (1512 keys), sampler-helper, 100-step baseline.
- **Needs:** backward arms, LoRA-grad+AdamW Spec, driver/CLI, cache.

### SD3 / SD3.5 — **predict only, NO backward + blocked references → cannot train** · T2
- `modelSetup/BaseStableDiffusion3Setup.mojo`: `predict` ✓, **0 backward** (VERIFIED);
  `StableDiffusion3LoRASetup.mojo` no spec.
- Has model/sampler/LoRA-inventory contracts only. **No numeric baseline** — blocked:
  `data_loader_registered:false`, `text_encoder_key_count:0`, data-loader registered for
  `STABLE_DIFFUSION_35` while config uses `STABLE_DIFFUSION_3`, missing cache
  (`parity/sd3_train_ref_blockers.json`).
- **Needs:** backward arms + Spec + driver, PLUS TE weights + dataloader registration + cache before
  even a reference can exist.

### SD1.5 (StableDiffusion) — **STUB → cannot train** · T3
- `modelSetup/BaseStableDiffusionSetup.mojo` & `StableDiffusionLoRASetup.mojo` = 2-line `# TODO: port`
  (VERIFIED). Model + dataLoader also stubs.
- **Needs:** entire vertical — model forward+activation-save, predict, backward, LoRA-grad+AdamW Spec,
  loader, dataLoader, saver, driver/CLI. Essentially nothing built.

### HiDream — **STUB → cannot train** · T3
- `BaseHiDreamSetup.mojo` & `HiDreamLoRASetup.mojo` = 2-line stubs (VERIFIED). **Needs:** entire vertical.

### HunyuanVideo — **STUB → cannot train** · T3
- `BaseHunyuanVideoSetup.mojo` & `HunyuanVideoLoRASetup.mojo` = 2-line stubs (VERIFIED). **Needs:**
  entire vertical + video/temporal-latent data path.

### PixArtAlpha — **STUB → cannot train** · T3
- `BasePixArtAlphaSetup.mojo` & `PixArtAlphaLoRASetup.mojo` = 2-line stubs (VERIFIED). **Needs:** entire vertical.

### Sana — **STUB → cannot train** · T3
- `BaseSanaSetup.mojo` & `SanaLoRASetup.mojo` = 2-line stubs (VERIFIED). **Needs:** entire vertical.

### Wuerstchen — **STUB → cannot train** · T3
- `BaseWuerstchenSetup.mojo` & `WuerstchenLoRASetup.mojo` = 2-line stubs (VERIFIED). **Needs:** entire
  vertical + multi-stage (Stage A/B/C) machinery.

---

## RANKING (closest-to-trainable first)

| # | Model | Tier | Verdict | Single biggest blocker |
|---|-------|------|---------|------------------------|
| 1 | **Klein/Flux2** | T1-complete | Trainable TODAY via CLI | Fidelity: constant-timestep (`KleinLoRATrainer.mojo:160`) + unproven grad/AdamW parity; needs external cache |
| 2 | **Z-Image** | T1 | Real full loop+save, smoke-only | No CLI `def main` (argparse) |
| 3 | **Lens** | T1 | Real full loop+save+resume, smoke-only | No CLI `def main` (argparse) |
| 4 | **Flux2 dev** | T1 (asset-blocked) | Code reuses real Flux2 backward | Missing dev weights + diffusers assets + cache + is_dev wiring |
| 5 | **SDXL** | T2 | predict only, no backward | Hand-chained backward + LoRA-grad/AdamW Spec |
| 6 | **Qwen** | T2 | predict only, no backward | Backward (+ Qwen2.5-VL text branch) |
| 7 | **Ernie** | T2 | predict only, no backward | Backward |
| 8 | **Anima** | T2 | predict only, no backward | Backward |
| 9 | **Chroma** | T2 | predict only, backward=no-op (`:232`) | Real backward + Spec |
| 10 | **Flux.1 dev** | T2 | predict only, no backward | Backward |
| 11 | **SD3/3.5** | T2 | predict only, no backward + blocked refs | Backward + TE weights/cache + DL-registration mismatch |
| 12 | **SD1.5** | T3 | stub | Entire vertical (setup is 2L TODO) |
| 13 | **HiDream** | T3 | stub | Entire vertical |
| 14 | **HunyuanVideo** | T3 | stub | Entire vertical + video data path |
| 15 | **PixArtAlpha** | T3 | stub | Entire vertical |
| 16 | **Sana** | T3 | stub | Entire vertical |
| 17 | **Wuerstchen** | T3 | stub | Entire vertical + multi-stage |

---

## Bottom line

- **Models that can run a real train loop today: 3** — Klein/Flux2, Z-Image, Lens. All three have a
  real `predict` + real hand-chained `backward` + AdamW + LoRA-save, VERIFIED end-to-end in-process
  (Lens even proves save→resume in one gate). Only **Klein has a CLI `def main`**
  (`KleinLiveTrainer.mojo:53`); Z-Image and Lens are **T1 thin wrappers** — they need only an
  argparse CLI + a prebuilt cache, no new training stack.

- **What the 3 trainable verticals still need for FAITHFUL (not just "a loop runs") training:**
  - **Klein**: replace the pinned `_force_constant_timestep` (`KleinLoRATrainer.mojo:160/323`) with
    Serenity's LOGIT_NORMAL flow-match timestep sampling, and prove train-numeric parity
    (grad-norm / AdamW-delta vs `parity/klein_train_ref_*`). Until then it trains but is not faithful.
  - **Z-Image & Lens**: an argparse `def main` (the only thing between smoke and CLI), and a cache.
    Their loss/grad/AdamW/save are already gated against Serenity ranges and key parity.
  - **All three**: a **verified in-Mojo Prepare path** so they don't depend on an externally
    pre-generated latent+text cache (`Prepare.mojo` is currently UNVERIFIED).

- **Shared work that unblocks the most models at once (do these first):**
  1. **A verified Prepare/encode→cache gate** — removes the external-cache dependency from all of
     Klein/Z-Image/Lens immediately, and is reused by every later model.
  2. **A generic argparse CLI runtime** (real object dispatch, not `factory.mojo`'s build-only
     `FactoryRegistration` strings) — converts Z-Image and Lens from smoke-only to CLI for free, and
     is the host every future vertical plugs into.

- **The other 8 models are genuinely far:**
  - **7 are T2 "build backward"** (SDXL, Qwen, Ernie, Anima, Chroma, Flux.1, SD3; Chroma's
    `backward` is a no-op string `BaseChromaSetup.mojo:232`). They have a real `predict` and rich
    forward-side gates (LoRA key parity, loss-only replay, sampler-helpers) — but those prove
    **nothing** about backward/optimizer. There are only **2 `*_backward.mojo` files in the entire
    tree** (Klein/Flux2 + Lens); **no unwired shared backward arms are sitting available** — each T2
    model needs its backward hand-chained from scratch, then LoRA-grad+AdamW Spec + driver + cache.
    SD3 additionally needs TE weights, a cache, and a data-loader registration fix.
  - **6 are T3 bare stubs** (SD1.5, HiDream, HunyuanVideo, PixArtAlpha, Sana, Wuerstchen) — 2-line
    `# TODO: port` directory mirrors with nothing built; each needs the entire vertical.
