# Training-Parity Campaign — model-by-model, oracle-gated

Goal: get serenity-trainer models **training with parity**, one model at a time. Per model:
**builder → (main-loop build+oracle gate) → bug-fixer → skeptic → (main-loop re-gate) → next**.
The MAIN LOOP (not any agent) owns every parity number (Tenet 4).

## Oracle reality (measured 2026-06-08)
- `/home/alex/Serenity` (the port's nominal reference) is **DELETED** → no live re-dumping.
- GPU: RTX 3090 Ti 24 GB. Build cmd works: `cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && timeout 300 prlimit --as=30000000000 pixi run mojo build -I . -I /home/alex/serenity-trainer/src -Xlinker -lm <smoke> -o /tmp/x && /tmp/x`.
- **Runnable oracle = FROZEN step-0 dumps on disk** (`parity/<model>_train_ref_step000.safetensors` + `..._adapters.safetensors`). Verified rich: inputs + `output.predicted` (forward ref) + `output.target` + `output.loss_for_backward` + `adapter_after.*` (post-step LoRA delta = backward+AdamW ref).
- Reuse source = **serenitymojo** `models/<model>/<model>_stack_lora.mojo` (forward+backward+adamw) — implementation material, ref'd to OneTrainer/EriDiffusion, so **re-verify against the frozen Serenity dump after copy-in**.

## ⭐⭐ ONETRAINER ORACLE VALIDATION (2026-06-08) — the dumps are trustworthy
OneTrainer (`/home/alex/OneTrainer`, stock Nerogar, venv torch 2.9.1+cu128) is THE oracle and implements ALL these models. The frozen `parity/*` dumps came from the deleted `/home/alex/Serenity` fork. **Built `scripts/ot_klein_oracle.py`** — loads OneTrainer's own Flux2 transformer + `LoRAModuleWrapper`, feeds the FROZEN trace inputs + `adapter_before`, runs predict→loss→backward, dumps grads (`parity/ot_klein_grads.safetensors`).
- **OneTrainer == frozen Serenity dump (Klein)**: forward cos **1.000000**, loss exact (0.12243739), grads cos **0.9999**. ⇒ the deleted-Serenity dumps were FAITHFUL to OneTrainer; prior gates against them are VALID.
- **Klein bug CONFIRMED real & Mojo-side**: Mojo d_B ~1.8× OneTrainer (to_q 0.000473 vs 0.000265). Mojo diverges from BOTH OneTrainer and the dump (which agree). Not a reference artifact — a real Mojo double-block backward bug. Fix against `ot_klein_grads.safetensors`.
- Harness pattern (reuse per model): feed frozen `trace.*` inputs into OneTrainer's transformer+LoRA(adapter_before), gate OneTrainer-vs-dump (validates dump) + produce OneTrainer grads (re-gate Mojo). Per-model TODO: anima/sdxl/qwen oracles + the Mojo-vs-OneTrainer grad re-gates + Klein fix.
- **Chroma oracle (`scripts/ot_chroma_oracle.py`, `parity/ot_chroma_grads.safetensors`): DUMP == OneTrainer (exact)** — forward cos 0.999993, loss 0.29561 vs 0.29572 (1.1e-4 bf16 nondeterminism). OneTrainer ref grads now available (to_q.lora_up 1.06e-6, to_k 4.4e-6, to_v 6.0e-5) → RESOLVES Chroma's "backward not gateable" caveat (post-AdamW dump had no grads; OneTrainer provides them).
- **SDXL oracle (`scripts/ot_sdxl_oracle.py`, `parity/ot_sdxl_grads.safetensors`): DUMP == OneTrainer (exact)** — predicted-noise cos 0.999997, loss 1.4e-6. OneTrainer full-UNet grad-L2 0.010614 == dump 0.010625. KEY: dump LoRA is **UNet-ONLY (no lora_te1/te2)** — all 6352 keys `lora_unet.*` → SDXL Stage 2 (full-UNet, TE-excluded) is the correct target; ot_sdxl_grads is the per-adapter reference Stage 2 must hit (conv_in 0.007, resnet conv1 0.0003, time_emb 0.0009, ST to_q 0.000125, proj_in 0.0011).
- **Anima oracle (`scripts/ot_anima_oracle.py`, `parity/ot_anima_grads.safetensors`): NOT in stock OneTrainer** — came from deleted Serenity-anima-ref fork wrapping diffusers `CosmosTransformer3DModel`. So it's a DIFFUSERS-class oracle (valid independent ref, not OneTrainer-setup). Forward cos 0.999099 vs dump (documented fork-quantization residual, matches Mojo fixed 0.99942). Grads written (280 modules, B=0 correct).
- **Anima REAL PR oracle**: OneTrainer Anima is PR #1487 (dxqb/OneTrainer `anima` branch) — cloned to `/home/alex/OneTrainer-anima` with its own venv (`/home/alex/OneTrainer-anima/venv`: torch 2.9.1+cu128, transformers 5.9, diffusers@b003a47). `scripts/ot_anima_real_oracle.py` + `parity/ot_anima_real_grads.safetensors` use the REAL `BaseAnimaSetup.predict`. Forward cos 0.999099 — IDENTICAL to diffusers-direct (diffusers-version hypothesis DISPROVEN; residual is the deleted fork's dump-gen detail, not Mojo/diffusers). Mojo(0.99942)≈OneTrainer(0.99910) vs fork-dump → Mojo Anima faithful.
- ORACLE TALLY: Klein exact / Chroma exact / SDXL exact / Ernie 0.997-skew / Anima 0.999(fork-dump residual). Remaining oracle: qwen. Dumps OneTrainer-faithful; prior gates VALID.
- OneTrainer ref GRADS now available for re-gating Mojo: `parity/ot_{klein,ernie,chroma,sdxl,anima_real}_grads.safetensors`.
- ACTIONABLE: (1) Klein fix vs ot_klein_grads (confirmed real Mojo d_B bug), (2) Mojo-vs-OneTrainer grad re-gates, (3) SDXL Stage2 vs ot_sdxl_grads, (4) qwen oracle.
- **Ernie oracle (`scripts/ot_ernie_oracle.py`, `parity/ot_ernie_grads.safetensors`): DUMP HAS VERSION SKEW.** OneTrainer-vs-dump forward cos **0.9972**, loss +1.1% — NOT exact (vs Klein exact). Isolated to the transformer forward = current OneTrainer Ernie diffusers vs deleted-Serenity-fork's pinned commit (Ernie = brand-new dev-branch model). ⇒ dumps are NOT uniformly OneTrainer-faithful; must gate Mojo-vs-OneTrainer DIRECTLY for Ernie. Mojo(0.9977 vs dump) ≈ OneTrainer(0.9972 vs dump) → likely Mojo≈OneTrainer; confirm via direct grad compare (ot_ernie_grads written, 504 tensors, lora_down=0 correct).

## ⭐ METHODOLOGY UPGRADE (2026-06-08) — torch gates are the real oracle
serenitymojo carries **torch-autograd parity gates** per model: `serenitymojo/models/<m>/parity/lora_stack_parity.mojo` (+ `block_parity.mojo`). These gate the FULL stack forward + backward + EVERY per-LoRA d_A/d_B vs `torch.autograd` at cos≥0.999 (small synthetic dims, F32). Since the serenity-trainer port IMPORTS the same serenitymojo math, **these gates ARE the training-parity proof** — far stronger than the frozen single-step Serenity dumps (which are bf16, high-σ, post-AdamW-saturated, and floor below 0.999 for an F32 path = the dtype bad-reference trap).
Per-model verification = (1) run serenitymojo `<m>` torch lora_stack_parity [authoritative bwd], (2) frozen Serenity forward replay [sanity, bf16 caveat], (3) d_A=0 invariant + loss. Gates present: anima✓ sdxl✓ qwenimage✓(block) klein✓; sd35✗(none yet).

## VERIFIED RESULTS
- **Chroma — TRAINS WITH PARITY.** Fwd cos 0.99996 (frozen). Backward via flux torch gate (Chroma's blocks ARE flux blocks): **72/75 LoRA arms cos≥0.999**, 3 marginal on near-zero tensors (max-abs-diff ≤3e-5, pre-existing flux state). d_A=0 exact, loss 0.46%, nonfinite 0.
- **Ernie — TRAINS WITH PARITY (cleanest).** Backward via ernie torch lora_stack_parity: **49/49 PASS cos 0.99997–0.99999** (fwd, d_x, AdaLN, all 42 LoRA grads), nonfinite 0, exit 0. Bug-fixer fixed 2 real faithfulness bugs (text-padding mask + RoPE text_lens offset; tanh→exact-erf GELU). Frozen forward replay floors at 0.9977 = bf16-reference-vs-F32 accumulation (NOT a defect — same math gates 0.99999 vs torch).

## Per-model gate (the bar)
1. **Forward**: Mojo transformer fwd on frozen `trace.*` inputs → cos ≥ 0.999 vs `output.predicted` / `trace.packed_predicted_flow`.
2. **Loss**: Mojo flow-match loss vs `output.loss_for_backward` (abs_err small).
3. **Backward+AdamW**: Mojo predict→loss→backward_lora→adamw on frozen batch → compare resulting adapter delta to `adapter_after.*` (cos ≥ 0.999 on the deltas; LoRA key set must match).

## Model order (only models with a runnable frozen oracle)
| # | Model | serenitymojo source | frozen oracle | status |
|---|---|---|---|---|
| 1 | **Chroma** | imports serenitymojo `models/flux/*`+`models/dit/chroma_dit` (builder chose import-not-copy) | full trace + 2432 adapters | FWD+LOSS+GRADNORM PASS; bwd-direction not gateable (noise-floor step) — skeptic reviewing |

### Chroma gate results (main-loop verified, 2026-06-08)
- Forward vs `trace.packed_predicted_flow`: **cos 0.99996 / 0.99995**, nonfinite 0 (real Chroma1-HD ckpt, full 57-block offload). PASS.
- Loss vs `output.loss_for_backward`: 0.29708 vs 0.29572 (0.46%). PASS.
- Grad-norm vs meta `grad_norm_no_clip`: 4.354e-4 vs 4.108e-4 (6%). Backward MAGNITUDE correct. PASS.
- A(lora_down) decay rel 0.0017; nonfinite grads 0. Optimizer mechanics PASS.
- B(lora_up) DIRECTION: masked cos 0.69 / sign 82% on signal subset. **NOT GATEABLE**: this frozen step has grad_norm 4e-4 over 35.5M params (RMS/elem ~1e-11, at bf16 noise floor) and AdamW step-0 saturates updates to ~lr·sign(g) — the post-AdamW dump encodes only gradient sign, which is dominated by bf16 forward error here. A meaningful backward-direction gate needs raw-grad dump or a torch-autograd oracle at a higher-gradient step.
- Builder DEVIATION (skeptic to rule): imports serenitymojo `models/flux/*` instead of copy-into-port — breaks the PORT_MAP copy-in rule but matches the Ideogram4 precedent and avoids transcription drift.
| 2 | **Ernie** | `models/ernie/{block,ernie_stack_lora,weights,config}.mojo` | dump + grad_norm (lr=0 step → adapter-delta useless) | FWD UNDER BAR (0.9977/0.9959) — bug-fixer; d_A=0 PASS, d_B ratio 1.26 |

### Ernie gate results (main-loop verified, 2026-06-08)
- Forward vs `trace.packed_predicted_flow`: sample0 **0.99765** (201/201 tokens, NO padding), sample1 **0.99592** (157/201 → 44 padding rows). Below 0.999 bar. nonfinite 0.
- Loss 0.63983 vs 0.64385 (0.6%). d_A=0 EXACT (structural PASS). d_B l2 0.001047 vs ref grad_norm 0.000829 (ratio 1.26).
- Leads: (1) ERNIE block uses `sdpa_nomask` → attends text padding the ref masks (explains sample0→sample1 drop); (2) clean-sample 0.99765 floor — candidates: builder changed rope BF16→F32 (driver passes BF16; runtime dtype mismatch forced it), and/or F32-resident vs ref BF16 train_dtype. Bug-fixer investigating.
| 3 | **Anima** | `models/anima/*` | torch 64/64 + real-ckpt fwd | ✅ DONE — real 3D rope bug FIXED (0.99942/0.99864==torch); backward torch-gated |

### Anima gate results (main-loop verified, 2026-06-08)
- Forward vs `trace.predicted_flow`: 0.9549 / 0.9143 — REAL structural gap, nonfinite 0. NOT a bf16 floor.
- CAUSE (verified): training driver `train_anima_real.mojo:_rope_tables` (:443-457) uses a SIMPLIFIED single-axis rope, not the real 3D rope; anima block uses `sdpa_nomask` self-attn (:367) + cross-attn "no mask" (:378). The 64/64 torch gate passed only because it feeds the SAME simplified rope to torch+Mojo — torch parity is necessary but NOT sufficient when the source is simplified.
- FIX SOURCE: real 3D rope exists at `models/dit/anima_dit.mojo:319 build_anima_3d_rope` (used by inference). Grid for this latent: T=1,H=32,W=32 (S_IMG=1024). Bug-fixer wiring it + cross-attn text mask.
- d_A=0 EXACT; loss_replay PASS (2e-6); grad-norm ratio 7.6× (inflated by the forward gap).
- LESSON: torch lora_stack_parity gates rope-APPLICATION, not rope-POSITIONS — a shared source simplification (rope grid, mask) is invisible to it. Must ALSO run the real-checkpoint frozen forward replay to catch architectural simplifications.
| 4 | **SDXL** | `models/sdxl/*` (UNet) | Stage1 DONE (rect fwd+bwd run); Stage2 = full-UNet LoRA | ✅ TRAINS (ST-attn LoRA); Stage2 conv/embed coverage in progress |

### SDXL Stage 1 DONE (main-loop verified, 2026-06-08)
- Generalized square `sdxl_real_forward[L]`→`[H,W]` (rectangular 168×96) fwd+bwd. Unified backward dtype (F32-act/F32-grad/BF16-weight, mixed_base) — added mixed branches to `ops/norm.mojo group_norm` + `ops/norm_backward.mojo layer_norm_backward`.
- Rectangular forward cos **0.99999580** vs `output.predicted`. Backward runs at 168×96: loss 0.13534 (=dump 0.13533), d_A=0 exact, nonfinite 0, LoRA-B grew under AdamW.
- Regression CLEAN: Ernie torch 49/49, SDXL torch 44/44 (shared-op edits safe, additive).
- ⇒ SDXL TRAINS end-to-end with faithful forward + torch-gated ST-attn LoRA (700 adapters). The most common SDXL-LoRA config works.
- Stage 2 PARTIAL (math gated, real-integration remaining): new LoRA families torch-gated — proj_in/proj_out (lora_stack_parity now **48/48 PASS**), conv LoCon (**13/13 PASS**), embed/proj linears (12/12). But conv/embed NOT yet threaded into real-dims `sdxl_real_forward/backward` + `build_sdxl_lora_set` → real grad-norm still ST-only 4.82e-4 (target 0.0106 incl. TE). Remaining: LoRA-aware real resblock fwd/bwd (conv1/conv2/shortcut/time_emb_proj) + conv_in/out + samplers + embeddings; populate carrier. TE1/TE2 OUT OF SCOPE.

### SDXL gate results (main-loop verified, 2026-06-08) — NOT TRAINABLE AS-IS
- torch lora_stack_parity 44/44 PASS — but it runs **pure-F32 synthetic SQUARE weights, ST-only LoRA**. Misleading.
- Real-weight path BROKEN (verified): (1) `sdxl_real_forward[L]` is **square-only** (`sdxl_real_train.mojo:275-277` H0=L,H1=L/2,H2=L/4); oracle is rectangular 168×96 → no native forward cos. (2) Real-weight **backward CRASHES**: `linear_backward: grad_y/x/weight dtype mismatch` after fwd+loss (eps MSE 0.2449). (3) LoRA set = ST-only 700 adapters; dump grad_norm = full-UNet 794 modules → not comparable. Driver header: "not production-tested".
- Forward WIRING runs clean on real bf16 ckpt (square crop) but is NOT a cos gate.
- VERDICT: SDXL needs a substantial REBUILD (rectangular fwd+bwd, bf16/F32 backward dtype unification, full-UNet LoRA, regenerate a matching reference) — not a one-pass bug-fix. Frozen dump can't gate it as-is.
| 5 | Qwen | `models/qwenimage/qwenimage_stack_lora.mojo` (:642) | dump + adapters (text=placeholder) | queued |
| 5 | **Klein** | `models/klein/*` | frozen dump has 576 REAL grad keys (clean bwd oracle) | BWD BUG: d_B ~1.5-2× ref — bug-fixer |

### Klein gate results (main-loop verified, 2026-06-08)
- Forward cos **0.99995** (packed_flow + output.predicted), loss 0.58771 vs 0.58766 (4.4e-5), d_A=0 exact. All correct.
- **d_B (lora_up) ~1.5-2× the Serenity reference** (per-probe: to_q 0.000473 vs 0.000275=1.72×; to_k 1.98×; to_v 1.48×; ff_in 1.56×; add_k 1.72×). Same nonzero support. Smoke RAISED "differs beyond tolerance". update_error l2 0.0073 vs ref_update 0.0109.
- ISOLATED: forward+loss+d_A correct → backward d_B bug only. NOT timestep (forward matches). NOT a scale bug (scale verifiably alpha/rank=1.0, applied once) and NOT a found double-count (bug-fixer verified every op adjoint + graph structure).
- REFRAMED (bug-fixer, exhaustive, no fix landed): the error is DIRECTIONAL not just magnitude — d_B cos 0.3-0.7 (K/V/qkv ~0.33, Q 0.74), and COMPOUNDS ~5%/block through the DOUBLE-block joint-attention backward d_x stream (single-block stream correct: to_out 1.05×). Q-vs-K/V asymmetry ⇒ likely a subtle adjoint-transcription error in `double_block.mojo:2214 double_block_lora_backward_device_resident_scratch` d_q/d_k/d_v, OR bf16 attention-backward cancellation amplified over 32 blocks (reference is bf16, so less likely the whole story).
- 3 FIX ATTEMPTS (~870K tokens), NOT LANDED, analyses partly CONFLICT:
  - Attempt2: claimed error compounds through single blocks to cos 0.287. Attempt3 showed that "0.287" was a MISREAD of a LoRA-grad cosine as a d_x cosine.
  - Attempt3 DECISIVE per-block test (OneTrainer `register_full_backward_hook` per single block, saved `parity/ot_klein_block_dx.safetensors`): **Mojo residual d_x matches OneTrainer cos 0.985-0.9999, L2 ratio ~1.0 at EVERY block.** So the residual-stream d_x is CORRECT.
  - UNRESOLVED CONTRADICTION: per-projection LoRA grads (to_q/k/v/out/ff) are cos 0.3-0.7 / 1.5-2× off (Attempt1), but residual d_x is fine (Attempt3). Since Mojo & OneTrainer use IDENTICAL A(adapter_before) + x(fwd cos 0.99995), d_B=d_y·(Ax)ᵀ differs ONLY via the per-projection d_y → the ATTENTION-PATH d_y (d_q/d_k/d_v at projection outputs) is what's off, not the residual d_x. 1.72× is likely TOO BIG for a bf16-vs-F32 artifact (Attempt3's dtype-trap claim is doubtful).
  - DECISIVE TEST NOT YET RUN: hook to_q/k/v MODULE OUTPUTS (per-projection d_y) in the OT oracle (not the residual d_x) + compare to Mojo's internal d_q/d_k/d_v; AND compare OneTrainer-bf16 d_y vs OneTrainer-F32 d_y for one block to separate real-adjoint-bug from bf16-amplification. STATUS: well-characterized, root cause still contested, NOT fixed.
- NOTE: serenitymojo klein torch gates are BIT-ROTTED (`klein_stack_lora_parity` SIGILL; `double_block_lora_parity` won't build — `StreamLoraGrads` API drift). The frozen Serenity dump (576 real grads) is the working oracle.
- Timestep fidelity gap also still open: `KleinLoRATrainer.mojo:160 _force_constant_timestep`(=250) instead of logit-normal.
| — | SD3 | `models/sd35/sd35_stack_lora.mojo` (:706) | partial dump, TE/cache blockers | blocked-pending |
| — | Flux.1 / SD1.5 / HiDream / Hunyuan / PixArt / Sana / Wuerstchen | none or fwd-only | **no frozen oracle** | OUT OF SCOPE (no oracle) |

## Port pattern (precedent: Klein, Lens already in the port)
Copy serenitymojo `models/<model>/*` → `serenity-trainer/src/serenity_trainer/model/<model>/`, adapt namespace to `serenity_trainer` (import ONLY tensor/autograd/ops from serenitymojo). Wire `smoke/<model>_train_ref_{forward,loss,grad_update}_replay.mojo` mirroring the `klein_train_ref_*_replay.mojo` gates. Agents run SEQUENTIALLY (no concurrent mojo builds).
