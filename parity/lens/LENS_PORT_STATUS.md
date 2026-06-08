# Microsoft Lens LoRA — port status (live, measured)

Target: `/home/alex/serenity-trainer/src/serenity_trainer/` (mirrors Serenity module tree).
Reference: **Serenity ONLY** — PR #1510 (fetched at `/home/alex/Serenity` branch `pr-1510`)
+ the `lens` python package (`/home/alex/vendor-refs/Lens/lens/*.py`) which Serenity's
`LensModel.py` imports (so it IS the 1:1 forward-math source). **No Rust / EriDiffusion ever.**

Borrow: import `serenitymojo.{tensor,autograd,ops,io}`; copied Lens forward/encoder/VAE
from serenitymojo into the port (namespace `serenity_trainer`).

## Oracle (Serenity-derived, in `parity/lens/`)
- `lens_oracle.py` — dumps real-weights Lens DiT forward (lens/transformer.py) on fixed seeded
  inputs (img 8x8=64 tok, txt 16, seed 1234). Run with `/home/alex/Serenity/venv/bin/python`.
- `dit_fwd_in_*.safetensors` (hidden, txt_0..3, mask, timestep) + `dit_fwd_out.safetensors` (ref output).
- `vae_bn.safetensors` (AutoencoderKLFlux2 bn running_mean/var) for scale_latents.
- `scheduler_config.json` + `predict_ref.json` (FlowMatchEulerDiscreteScheduler config + timestep_shift refs).
- `meta.json` (config, shapes, out stats: mean -0.065439 std 1.721513 absmax 7.333670).

## Gates (main loop measured, not agent self-report)
| # | Gate | Smoke | Result |
|---|------|-------|--------|
| 1 | predict scalars | `smoke/lens_predict_scalars_smoke.mojo` | ✅ scale_latents finite; timestep_shift(8,8)=1.5830842 == OT ref 1.58308425 |
| 2 | DiT forward parity (LoRA B=0) | `smoke/lens_forward_parity_smoke.mojo` | ✅ cos(Mojo, oracle)=**0.9998504** ≥ 0.999, 1264 tensors loaded |
| 3 | train step (loss finite + LoRA-B 0→nonzero) | `smoke/lens_train_step_smoke.mojo` | ✅ loss 1.097, 476/480 B move, 0 nonfinite |
| 4 | backward grad parity (dtype-aware) | `smoke/lens_backward_parity_smoke.mojo` | ✅ 476/476 within BF16 ceiling, 4 zeros exact (see finding) |
| 5 | save/load LoRA keys (REAL saver) | `smoke/lens_saveload_smoke.mojo` | ✅ 1440 keys, 0 missing/extra/shape, round-trip Δ=0, alpha preserved |

### Saver bug found + fixed (real, not papered over)
`modelSaver/lens/LensLoRASaver.mojo` was a ZImage copy that didn't compile against the real model
(imported `LensLoraSet` from wrong module; used `set.ad`/`set.n_blocks`/`.alpha` vs real
`set.block`/`set.rank`/host-list `a,b,scale`; stale "6 top-level" path). Rewired to the real
`LensLoraSet`(model/lens/lens_backward.mojo)/`LoraAdapter`(module/LensLoRAModule.mojo); smoke now
drives the REAL `save_lens_lora`→`load_lens_lora` end-to-end. ⇒ other ZImage-copied plumbing
(loader/trainer-glue/dataLoader/sampler) under compile-inventory.

### Integration compile-inventory (2026-06-07)
| File | Compiles | Note |
|------|----------|------|
| modelLoader/LensModelLoader.mojo | ✅ | as-is |
| trainer/LensTrainStep.mojo | ✅ | train-loop glue; needed `prog^`, `LensLoRASpec:ModelSpec` conformance, GenericTrainer copy fix |
| dataLoader/LensBaseDataLoader.mojo | ✅ | as-is (name/plan contract) |
| modelSetup/LensFineTuneSetup.mojo | ✅ | comments-only stub (full-finetune backward NOT implemented — documented) |
| modelSaver/lens/LensModelSaver.mojo | ✅ | full-transformer saver |
| modelSampler/LensSampler.mojo | ✅ | FIXED: ported `sampling/lens_flowmatch.mojo` into the package (compute_empirical_mu verified == Serenity lens/pipeline.py:38, byte-for-byte; Rust comments stripped); fixed the decode double-unscale → tail = `unpack→vae.decode` (LensVAE/KleinVaeDecoder fuses unscale+unpatchify+decode), 1:1 with LensSampler.py:139-148; fixed VAE generic param to packed dims. |

⇒ LoRA TRAINING path (loader→dataLoader→LoRA setup→train-step glue→backward→AdamW→save) compiles end-to-end and the computation is gate-verified. SAMPLER compiles; schedule + decode verified.

### Sampler gates (main loop verified)
| Gate | Smoke | Result |
|------|-------|--------|
| schedule parity (mu/sigmas/timesteps, steps=20) | `smoke/lens_sampler_smoke.mojo` GATE A | ✅ max |Δσ|=6e-8, |Δt|=9e-5 vs diffusers `set_timesteps` |
| VAE decode-tail (unpack→vae.decode) | `smoke/lens_sampler_smoke.mojo` GATE B | ✅ **PSNR 67.42 dB**, mean-abs-diff 6e-4 vs Serenity decode (double-unscale would tank this) |
Oracle: `lens_sampler_oracle.py` → `sampler_schedule_ref.json`, `sampler_tail_in/out.safetensors`.
Sampler denoise loop = verified DiT infer forward + verified schedule + verified decode. The only
un-runnable front-end is prompt text-encode (GPT-OSS MXFP4 — same blocker as real-data loss).

### GPT-OSS text encoder — VERIFIED (the "MXFP4 blocker" was not a blocker)
The encoder runs in `/home/alex/ai-toolkit/venv` (transformers 5.8.1, diffusers 0.38, torch 2.7+cu126);
MXFP4 dequants to bf16 on load (triton 3.3<3.4 so no fast path — fine). Oracle: `lens_gptoss_oracle.py`
→ `gptoss_ref.safetensors` (input_ids[160] + hidden_layer_05/11/17/23). Run on GPU via device_map="auto".
| Gate | Smoke | Result |
|------|-------|--------|
| GPT-OSS 4-layer features (S=160, sliding window exercised) | `smoke/lens_gptoss_parity_smoke.mojo` | ✅ cos per layer 0.99962/0.99994/0.99968/0.99994, **all ≥0.999** |
Validates: MXFP4 dequant, YaRN RoPE (theta150000/factor32), sliding(even)/full(odd) causal masks,
attention sinks, GQA, MoE top-4. Run the v5 env on GPU (device_map=auto) — NOT CPU.

### CORE GATE — real-data predict→loss parity (Serenity vs our trainer) ✅
Oracle `lens_loss_oracle.py` (replicates BaseLensSetup.predict on a real image+caption,
deterministic t=499) → `loss_ref.safetensors` (packed_in, feat_0..3, target) + `loss_ref_meta.json`.
| Gate | Smoke | Result |
|------|-------|--------|
| predict→loss vs Serenity (identical inputs) | `smoke/lens_loss_parity_smoke.mojo` | ✅ mojo **0.507432** vs OT **0.508090**, rel **0.13%** (bar 2%); in OT range |

### Pure-Mojo end-to-end gen
512×512 works → `parity/lens/gen/lens_mojo.png` (coherent owl+woman portrait). Tokenizer: Mojo o200k
splitter matched only 25/298 ids → used reference ids (Mojo tokenizer NOT yet parity).
1024×1024 OOMs in sample_lens on 24GB (cause undiagnosed — needs instrumentation; not faked).

### Remaining (not done)
- 1024 OOM fix (chunked/flash SDPA or stream blocks).
- Mojo o200k tokenizer parity (pre-tokenizer split).
- Denoise-trajectory parity vs Serenity (cos) + measured denoise speed.
- Full fine-tune backward (documented stub).

### Backward parity — key finding (measured)
- d_B vs F32 oracle: img-stream adapters cos 0.998–0.9995 (BF16-exact); txt-stream 0.975–0.990; 4 last-block txt-post adapters exactly 0.0 (matches torch autograd → mapping/formula/transpose all correct).
- The txt gap is the **BF16 ceiling, not a bug**: torch's OWN BF16 d_B vs its F32 d_B drops txt_qkv→0.981, txt_w1→0.957 (min 0.897). The Mojo txt d_B (0.975–0.990) is **as good as or better than torch BF16**. Low-magnitude txt gradient (txt influences the img-only loss only via attention coupling) is ill-conditioned under BF16.
- ⇒ This was the dtype bad-reference trap. The backward is correct to BF16 precision. Gate re-baselined: each adapter must be ≥ torch's per-adapter BF16 ceiling (`backward_grad_bf16_meta.json`) − 0.01, OR ≥0.999. (Avoided dispatching agents to "fix" correct code — measured first.)
- Oracle: `lens_backward_oracle.py` (F32 d_B ref) + `lens_backward_bf16_ceiling.py` (BF16 ceiling). Refs: `backward_grad_ref.safetensors`, `backward_grad_ref_bf16.safetensors`, `backward_grad_bf16_meta.json`.

## Architecture (config.json confirmed)
Double-stream MM-DiT: 48 blocks, inner_dim 1536, 24 heads × 64, enc_hidden_dim 2880,
selected text layers [5,11,17,23] (per-layer RMSNorm eps1e-5 → concat → txt_in 11520→1536),
complex-valued RoPE axes [8,28,28], QK RMSNorm, joint SDPA additive mask, SwiGLU GateMLP
(hidden 4096), AdaLayerNormContinuous out, proj_out 1536→128. VAE = AutoencoderKLFlux2 (32ch,
batch-norm latent scaling, eps 1e-4). Text encoder = GPT-OSS (MXFP4, YaRN, on-demand).

## Training design (binding): HAND-CHAINED, not shared-tape
Mirrors the Klein LoRA trainer. `lens_forward_full_lora` (model/lens/lens_stack_lora.mojo) runs
the 48-block forward, applies host-list LoRA deltas (module/LensLoRAModule), saves the
`LensFullSaved` activation bundle; `model/lens/lens_backward.mojo` hand-chains the backward to
LoRA A/B grads; `lens_lora_adamw_step` updates host-list adapters. **480 LoRA adapters**
(48 blocks × 10 = attn{img_qkv,txt_qkv,to_out.0,to_add_out} + mlp{img/txt w1,w2,w3}).
TO VERIFY vs Serenity: default `layer_filter`/preset (is it attn-mlp? are top-level
img_in/txt_in/proj_out wrapped?) — adapter-count fidelity check still open.

## Known open numeric items
- FIXME-NUMERIC: last block's 4 txt-post adapters (to_add_out, txt_mlp w1/w2/w3) get zero grad
  on a single step (output head reads image stream only). Gate asserts ≥476/480 move. Confirm
  with torch autograd (gate 5).
- Slot-order reconciliation in lens_stack_lora (W1=4,W2=5,W3=6 per lensLoraTargets/lens_backward,
  NOT LensDiT's W1=4,W3=5,W2=6). Validated by gate 5 (backward parity), not yet by a number.
- predict→loss numeric parity on a REAL cache sample (loss in Serenity range) — not yet built.
