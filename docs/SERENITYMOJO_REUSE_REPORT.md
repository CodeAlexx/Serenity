# serenitymojo → serenity-trainer Reuse Report (Skeptic, read-only)

Date: 2026-06-08. Method: source inspection only (no build — cache-corruption/OOM risk).
Scope verified: `/home/alex/mojodiffusion/serenitymojo`. Every verdict below cites `file:line`.

---

## Bottom line

- **The auditor's central thesis is VERIFIED.** serenitymojo contains a real, hand-chained
  LoRA **backward + AdamW + cache→predict→backward→optim→save** driver for **all 7 T2
  models** (SDXL, Qwen-Image, Ernie, Anima, Chroma, Flux.1, SD3.5), each as
  `models/<model>/<model>_stack_lora.mojo` (gradient math) + `training/train_<model>_real.mojo`
  (`def main`, argv, real cache). None are stubs or forward-only probes. This directly fills
  serenity-trainer's headline gap ("only 2 `*_backward.mojo` exist; each T2 model needs
  backward from scratch").
- **Gaps cheaply closable by porting:** the **per-model LoRA backward+optimizer math for all 7
  T2 models** (the single biggest win — this is the hardest, highest-value code and it is
  done and recipe-cited), plus the **Klein logit-normal timestep drop-in** and the
  **pure-Mojo VAE image→latent encode**.
- **Single biggest win:** the 7 T2 backward kernels. They are not skeletal — they are
  block-by-block reverse-mode chains with offload variants (Chroma/Flux/SD3.5/Qwen stream
  blocks in reverse).
- **Single biggest risk — RE-CHARACTERIZED, smaller than the auditor feared.** serenitymojo's
  reference is uniformly **EriDiffusion-v2 `*.rs` + OneTrainer cache**, never Serenity
  (106 reference lines across the 12 drivers; zero Serenity *math* references). BUT the LoRA
  **key-naming divergence is small**: serenity-trainer's cited key counts are
  OneTrainer-convention counts (adapters × 3), and serenitymojo's `save_lora_onetrainer`
  emits exactly that. Qwen is an exact match (720 adapters × 3 = **2160** =
  serenity-trainer's cited Qwen 2160). The real residual risk is **per-model adapted-module-set
  parity**, **dtype seams**, and the fact that **these trainers are single-sample
  loss-monotonicity smoke harnesses, not dataset loops** — there is no multi-file dataLoader.

---

## Verification table

| # | Auditor claim | Verdict | Evidence |
|---|---|---|---|
| 1 | 7 T2 `*_stack_lora.mojo` + `train_*_real.mojo` exist, real, under `models/<model>/` | **VERIFIED** | all 11 stack_lora + 12 train_real present (`models/*/`, `training/`); line counts 477–2332 / 390–1321 |
| 2 | SDXL bwd `:256/:328/:433/:577`, adamw `:632`, driver `:361 def main` | **VERIFIED** (±a few lines) | `sdxl_unet_stack_lora.mojo:262 _attn_lora_backward`, `:332 _geglu_lora_backward`, `:437 _basic_block_lora_backward`, `:577 sdxl_st_lora_backward`, `:632 sdxl_lora_adamw_step`; `train_sdxl_real.mojo:361 def main` |
| 3 | SDXL adamw is a real AdamW | **VERIFIED** | `sdxl_lora_adamw_step` → `_lora_adamw` (m/v buffers, β1/β2/eps/wd) from `training/train_step.mojo` (`:70` import, `:641` call) |
| 4 | Qwen bwd `:642`, driver `:395`; Qwen2.5-VL text = placeholder zeros | **VERIFIED** | `qwenimage_stack_lora.mojo:642 qwenimage_stack_lora_backward_offload`; `train_qwenimage_real.mojo:395 def main`, `:42` header: "txt_ch=3584 … cache dir uses placeholder zeros" |
| 5 | Ernie `:844` / driver `:567` | **VERIFIED** | `ernie_stack_lora.mojo:844 ernie_stack_lora_backward_resident_device`; `train_ernie_real.mojo:567 def main` |
| 6 | Anima `:271` / driver `:511` | **VERIFIED** | `anima_stack_lora.mojo:271 anima_stack_lora_backward`; `train_anima_real.mojo:511 def main` |
| 7 | Chroma `:715` / driver `:394`; replaces no-op `BaseChromaSetup.mojo:232` | **VERIFIED** | `chroma_stack_lora.mojo:715 chroma_stack_lora_backward_offload`; `train_chroma_real.mojo:394 def main`, `:568` call, `:579 flux_lora_adamw_step` |
| 8 | Flux `:344/:932` / driver `:395` | **VERIFIED** | `flux_stack_lora.mojo:344 flux_stack_lora_backward` (resident), `:932 flux_stack_lora_backward_offload`; `train_flux_real.mojo:395 def main`. Caveat: `:667/:669` raise "lora_te1/te2 save not implemented" (text-encoder LoRA save absent; **DiT LoRA save is fine**) |
| 9 | SD3.5 `:706` / driver `:582` | **VERIFIED** | `sd35_stack_lora.mojo:706 sd35_stack_lora_backward_offload`; `train_sd35_real.mojo:582 def main`, `:767 sd35_lora_adamw_step` |
| 10 | Klein `schedule.mojo:253 sample_timestep_logit_normal(seed,shift)` is a drop-in | **VERIFIED** | `training/schedule.mojo:253` — `t=sigmoid(N(0,1))` then qwen-shift `shift*t/(1+(shift-1)*t)`, clamp `[1/1000,1]`. Real, self-contained, ~40 lines. Klein bwd `klein_stack_lora.mojo:1318` real |
| 11 | Lens ABSENT; `models/lens/` is contract-only; `train_l2p` is a different model | **PARTIALLY WRONG → net-correct** | `models/lens/` has `lens_contract.mojo` **and a 75 KB `lens_dit_math.mojo` (real forward math)** — NOT contract-only. But there is **no Lens backward/trainer** (no `lens_*_stack_lora`, no `train_lens_real`), so Lens **training** is genuinely absent. `train_l2p_real.mojo:1` is **Z-Image L2P pixel-space**, not Lens — VERIFIED |
| 12 | HunyuanVideo PARTIAL + identity risk (may be Hunyuan-IMAGE not video) | **identity REFUTED; PARTIAL VERIFIED** | `models/dit/hunyuan15_dit.mojo:1–11` — explicitly "HunyuanVideo-1.5 DiT (video transformer)", refs `hunyuan15_dit.rs` + HunyuanVideo-1.5 repo, T2V 480p, `patch_size=(1,1,1)`, 3D RoPE. It IS video. Inference-only (no backward) |
| 13 | VAE encode is real pure-Mojo (`ldm_encoder.mojo:504 encode`) | **VERIFIED** | `models/vae/ldm_encoder.mojo:504 def encode` → `encode_moments` → mu/logvar split → `diag_gaussian_sample` (real reparam). `pipeline/flux_prepare.mojo:13–15`: real Mojo Flux-VAE encode, gated cos 0.9999985 |
| 14 | Prepare text embeds still sourced from Rust cache | **VERIFIED** | `flux_prepare.mojo:20–23`: raw-caption T5 encode "BLOCKED on a Unigram-tokenizer port … text keys SOURCED from an existing REAL flux cache (Rust prepare_flux output)" |
| 15 | CLIs: argv `def main` but hardcoded `comptime CACHE_DIR/LORA_DIR` (per-model, not generic) | **VERIFIED** | e.g. `train_sdxl_real.mojo:119 comptime CACHE_DIR=".../EriDiffusion-v2/cache/eri2_sdxl_512_smoke"`; `train_ernie_real.mojo:135`; `train_sd35_real.mojo:153`; `train_qwenimage_real.mojo:145 DEFAULT_CACHE_DIR` |
| 16 | BIG RISK: reference is EriDiffusion/OneTrainer, NOT Serenity; LoRA key counts/dtype may differ | **VERIFIED (reference) / RE-SCOPED (key-count risk smaller)** | see Reference-divergence section |

Additional honesty findings (not in the auditor's map):

| Finding | Verdict | Evidence |
|---|---|---|
| The `train_*_real.mojo` are **single-sample smoke harnesses**, not dataset loops | **TRUE — material gap** | `train_sdxl_real.mojo:114 comptime FIXED_SMOKE = True`, `:443 var sample_path = files[0]` (only file[0] ever loaded; even the `FIXED_SMOKE=False` branch only varies timestep `:501`+noise `:507`, **never iterates files**). Header `:30–33`: "backward MUST drive loss DOWN monotonically (trainer-correctness gate)". **No multi-sample dataLoader** |
| L2P uses a simplified final-linear in place of the full `local_decoder` | **TRUE (documented approximation)** | `train_l2p_real.mojo:25–30` |
| wan22 / ltx2 trainers are NOT production-ready (video, outside the 7 T2) | **TRUE** | `train_wan22_real.mojo:222,265–271` placeholder RoPE; `train_ltx2_real.mojo:231` "production AV trainer not implemented here" |

---

## Corrected reuse map (per model)

Effort tiers after copy-in: **T-low** = adapt paths/keys, re-run gates; **T-med** = also reconcile
adapted-module set or conditioning; **T-high** = significant rebuild.

| serenity-trainer gap | serenitymojo source | Effort after reuse | Serenity gates to re-verify post-port |
|---|---|---|---|
| SDXL backward/optim | `models/sdxl/sdxl_unet_stack_lora.mojo` (bwd `:262/:332/:437/:577`, adamw `:632`) + `training/train_sdxl_real.mojo` | T-low | LoRA key-set parity vs 2382; eps-pred schedule; dataLoader (smoke→dataset) |
| Qwen backward/optim | `models/qwenimage/qwenimage_stack_lora.mojo:642` + `train_qwenimage_real.mojo:395` | T-med | **Qwen2.5-VL text is placeholder zeros** — wire real text embeds; key-set (2160 — likely exact, see below) |
| Ernie backward/optim | `models/ernie/ernie_stack_lora.mojo:844` + `train_ernie_real.mojo:567` | T-low | key-set vs 756; logit-normal/shift identity |
| Anima backward/optim | `models/anima/anima_stack_lora.mojo:271` + `train_anima_real.mojo:511` | T-low | key-set vs 840; rectified-flow shift |
| Chroma backward/optim — replaces no-op `BaseChromaSetup.mojo:232` | `models/chroma/chroma_stack_lora.mojo:715` + `train_chroma_real.mojo:394` | T-low | key-set vs 912; frozen distilled-guidance scope (mod-vec grads discarded) |
| Flux.1 backward/optim | `models/flux/flux_stack_lora.mojo:344/:932` + `train_flux_real.mojo:395` | T-low | key-set vs 1512; **TE-LoRA save absent** if Serenity trains text encoders |
| SD3.5 backward/optim | `models/sd35/sd35_stack_lora.mojo:706` + `train_sd35_real.mojo:582` | T-med | OneTrainer split CLIP/T5 cache schema vs Serenity's; key-set |
| Klein constant-timestep → logit-normal | `training/schedule.mojo:253 sample_timestep_logit_normal` | T-low | confirm Serenity flow-match uses sigmoid-logit-normal + same shift convention |
| Z-Image CLI | `training/train_zimage_real.mojo` (`def main` + argv) | T-low | local cache paths |
| VAE image→latent (Prepare) | `models/vae/ldm_encoder.mojo:504 encode`; `pipeline/{flux,anima,zimage,giger3}_prepare.mojo` | T-low | per-model VAE shift/scale; **text embeds still from Rust cache** |
| Lens training | **NONE** (forward math only in `lens_dit_math.mojo`; no backward) | T-high | serenity-trainer's Lens is more complete — do not port |
| HiDream / HunyuanVideo / PixArt / Sana / Wuerstchen | forward-only or absent | T-high | out of scope for cheap reuse |

---

## Reference-divergence risk (the load-bearing section)

**Reference is EriDiffusion-v2 + OneTrainer, never Serenity — confirmed.**
Every driver header cites Rust + OneTrainer recipe lines, e.g.
`train_sdxl_real.mojo:13` "translated from train_sdxl.rs main loop";
`train_chroma_real.mojo:3` "TRANSLATION of EriDiffusion-v2 chroma.rs";
`train_ernie_real.mojo:5` "EriDiffusion-v2/.../train_ernie.rs", recipe scalars cite
`train_ernie.rs:904/949/956/1072`;
`train_flux_real.mojo:8` "TRANSLATION of … train_flux.rs", math cites `train_flux.rs:736/767/797/802`;
`train_sd35_real.mojo:5,38` "real OneTrainer cache … OneTrainer SD3.5 LoRA preset".
106 lines reference `.rs`/OneTrainer across the 12 drivers; **zero reference Serenity for math**.
The only `.serenity/...` hits are **checkpoint file locations on disk**
(`train_chroma_real.mojo:145 CKPT=.../.serenity/models/checkpoints/...`) and a progress-board
writer (`train_klein_real.mojo:597 SerenityBoardWriter`) — not algorithmic references.

**But the LoRA key-naming risk is smaller than the auditor assumed.**
`training/lora_save.mojo:135 save_lora_onetrainer` emits, per adapter, exactly three tensors:
`<prefix>.alpha`, `<prefix>.lora_down.weight`, `<prefix>.lora_up.weight` (`:171–175`), the
OneTrainer/kohya convention. It also offers PEFT keys (`:91 save_lora_peft` →
`.lora_A.weight`/`.lora_B.weight`). serenity-trainer's cited counts are **OneTrainer-convention
counts (adapters × 3)**:
- **Qwen: 12 slots × 60 blocks = 720 adapters** (`qwenimage_stack_lora.mojo:61 DBL_SLOTS=12`,
  `:640` "12 slots x 60 blocks = 720 adapters") **× 3 = 2160 = serenity-trainer's cited 2160. Exact.**
- **SDXL: 10 slots/block** (`sdxl_unet_stack_lora.mojo:28` `{a1.to_q,k,v,to_out.0, a2.to_q,k,v,to_out.0,
  ff.net.0.proj, ff.net.2}`) × num_blocks; **2382 / 3 = 794 adapters** — consistent with the
  diffusers SDXL attn+ff module set. **HYPOTHESIS:** SDXL also matches; confirm by counting
  `Σ sdxl_st_depth(i)·10` once.

So convention parity is high. The genuine residual risks per model, to re-verify after copy-in:
1. **Adapted-module-set parity** — does serenitymojo adapt the *same layers* Serenity's recipe
   does? serenitymojo is "LoRA-on-projection; base proj_in/out, group_norm, LayerNorms NOT
   adapted" (`sdxl_unet_stack_lora.mojo:32`). If Serenity also LoRAs resnet/time-embed/conv
   modules, counts and keys diverge.
2. **Dtype seams** — saves are BF16 (`lora_save.mojo:98,141`); host step math upcasts to F32
   (`train_sdxl_real.mojo:335 _load_cache_preserving_dtype`, `_host_f32_for_step_math`).
   Confirm Serenity's expected adapter dtype.
3. **Timestep identity** — Klein/flow-match drop-in is logit-normal + qwen-shift; verify
   Serenity uses the same distribution and shift default (shift=1 ⇒ identity remap).
4. **Cache schema/keys** — drivers read OneTrainer keys (`latent`,`pooled`,`text_embedding`,
   `time_ids` for SDXL; `latent_image`+split CLIP/T5 for SD3.5). Map to Serenity's cache keys.

---

## Ranked port plan (cheapest high-value first)

1. **Klein timestep drop-in** (`schedule.mojo:253`) — ~40 lines, replaces
   `_force_constant_timestep`. Smallest change, immediate fidelity win. Re-verify only the
   distribution/shift convention.
2. **Chroma backward** (`chroma_stack_lora.mojo:715` + `train_chroma_real.mojo:394`) — directly
   replaces serenity-trainer's no-op `BaseChromaSetup.mojo:232`; key count likely OneTrainer-clean.
3. **SDXL + Ernie + Anima backward/optim** — T-low, eps-pred (SDXL) / flow-match (Ernie/Anima)
   recipes already encoded; re-verify key-set + dataLoader.
4. **Flux.1 + SD3.5 backward/optim** — T-low/med; Flux watch the absent TE-LoRA save; SD3.5
   watch the OneTrainer split-text cache schema.
5. **Qwen backward/optim** — defer until the **Qwen2.5-VL text placeholder-zeros**
   (`train_qwenimage_real.mojo:42`) is replaced with real text embeds; key count is an exact 2160 match.
6. **VAE encode + prepare drivers** — reuse `ldm_encoder.mojo:504` and `*_prepare.mojo`; cheap,
   but **text embeds still come from the Rust cache** (T5 Unigram tokenizer unported), so this
   closes image→latent only, not text conditioning.

**Cross-cutting, NOT free from any copy-in (build it once on top):** a real **multi-sample
dataLoader** and **generic CLI**. Every `train_*_real.mojo` is a `FIXED_SMOKE` single-sample
loss-monotonicity gate (`train_sdxl_real.mojo:114,443`) with a hardcoded `comptime CACHE_DIR`.
The forward/backward/optimizer/save spine is reusable; the dataset-iteration and config-driven
CLI layer that serenity-trainer needs for production is still absent and must be written.

**Do NOT port:** Lens (serenity-trainer's is more complete), HiDream/HunyuanVideo/PixArt/Sana/
Wuerstchen (forward-only or absent).
