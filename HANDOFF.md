# Serenity → pure-Mojo PORT — HANDOFF (2026-06-05)

This is a complete pickup doc. Read it fully before touching anything. Pair it with
`PORT_MAP.md` (live status) and the two skills (below).

## 2026-06-05 UPDATE — USER DIRECTIVE
The immediate goal changed from "just get Klein training" to:
1. Audit Serenity as the **only behavioral reference**.
2. Add the missing Serenity training-runtime work to the port plan.
3. Verify every claim with the `serenity-trainer-port` / `numeric-parity-testing`
   method: Serenity reference run -> byte-identical dumped inputs -> Mojo run ->
   metric gate. Existing Mojo/serenity/mojodiffusion code is implementation
   material only; it is not proof.
   **CPU PyTorch is not numeric parity evidence.** CPU/dry-run paths may prove
   config, cache, file presence, source routing, and deterministic helper
   formulas only. Loss/grad/speed parity must come from Serenity's CUDA path
   and the Mojo gate must compare against those dumped CUDA-reference tensors.
4. Required model breadth now includes **Qwen, Ernie, Anima, SD3, SDXL,
   Flux.1 dev, Flux2/Klein, Flux2 dev, and Chroma**. Qwen is no longer
   dropped. Klein is the Flux2 class/family in Serenity, not a separate
   architecture; Flux2 dev and Klein share `ModelType.FLUX_2` and split at
   runtime through `Flux2Model.is_dev()`. Chroma is `ModelType.CHROMA_1`.
   Anima's Serenity-family reference is `/home/alex/Serenity-anima-ref`;
   the other listed models use `/home/alex/Serenity`.
5. **Full finetuning is in scope for every required model where Serenity
   registers `TrainingMethod.FINE_TUNE`**. Code audit found FINE_TUNE setup
   registrations for Qwen, Ernie, SD3/3.5, SDXL, Flux.1 dev, Flux2, Chroma,
   and Z-Image. Surface stubs are not enough; full-weight train/save/load/resume
   parity gates are required for those paths.

Hard correction for current and future agents: **port-facing files must use the
Serenity file names and module roles.** Status must be written as
`Serenity .py -> serenity-trainer .mojo mirror -> parity gate`. Code under
`/home/alex/mojodiffusion/serenitymojo/*` is implementation material only unless
it has been copied into the matching Serenity-named
`/home/alex/serenity-trainer/src/serenity_trainer/.../*.mojo` file and verified
against Serenity on byte-identical inputs.
Use `TRACEABILITY.md` as the per-model map before touching code.

`PORT_MAP.md` now has a new "Serenity Audit (2026-06-05)" section with the
missing product-run-layer, data-path, trainer-loop, sampling, loader/saver,
Klein, required model-breadth, and optimizer/scheduler gates. Follow that plan
before architecture refactors. A narrow Klein loader edit was made to keep large
checkpoint tensors in stored dtype (BF16) and a build-only check for
`smoke/klein_load_only.mojo` passed. The load smoke was then run and reached
`ALL DONE: 8 double + 24 single`, so the measured Klein base-load OOM is fixed.
Klein real-weight forward/backward now runs without nonfinite values, and the
inference forward parity gate passes vs the Serenity/diffusers reference. Klein
real-data train loss/AdamW parity is still missing.

## 2026-06-05 UPDATE — REQUIRED MODEL BASELINES + SURFACES
Fresh 100-step Serenity baselines now exist for Ernie, Qwen, Anima, SDXL,
Flux.1 dev, and Chroma. Each records requested/global steps, loss count, grad
norm count, step intervals, sampled GPU memory, and torch CUDA allocation.
SD3/SD3.5 is
blocked as a reference baseline because the local SD3.5 single-file checkpoints
contain diffusion+VAE weights but no CLIP/T5 text encoder weights; the HF cache
does not contain the full diffusers snapshot.

The Mojo port now has build-only model/setup/loader/saver/sampler/factory
surfaces passing for **Qwen, Ernie, Anima, SD3, SDXL, Flux.1 dev, Chroma
model/setup contracts, and Flux2/Klein factory/loader/saver contracts**.
Flux.1 shared `util/factory.mojo` / `util/create.mojo` dispatch was added and
verified. Flux2/Klein and Chroma mirrors already exist under Serenity file
names, but code audit shows several loader/saver/runtime paths are still
TODO/surface-only and must be promoted to real LoRA + full-finetune + sampler
parity.
Chroma's sampler TODO has now been replaced with a helper-only mirror plus
generated fixture/gate source:
`src/serenity_trainer/modelSampler/ChromaSampler.mojo`,
`parity/gen_chroma_sampler_helper_ref.py`,
`parity/chroma_sampler_helper_ref.json`, and
`smoke/chroma_sampler_helper_gate.mojo`. The Chroma Mojo gate was built/run by
the main loop and passes as a helper-only sampler contract. This is not
end-to-end sampler parity.
Chroma's `model/ChromaModel.py`, `modelSetup/BaseChromaSetup.py`,
`modelSetup/ChromaFineTuneSetup.py`, and `modelSetup/ChromaLoRASetup.py` stubs
are now replaced with build-only Serenity contract mirrors, verified by
`smoke/chroma_model_setup_contract_check.mojo` (`text seq ragged = 32`, packed
`4608x64`, FINE_TUNE params `3`, LoRA params `3`, `create te = True`). This is
still not runtime transformer/backward/optimizer parity.
This is not numeric parity. Each model still needs byte-identical Serenity
cache/input dumps, predict/loss/optimizer parity, sampler parity, PEFT key parity,
and speed gates before it can be called done.

---

## 0. THE TWO SKILLS — read these FIRST
Both live under `~/.claude/skills/` and load by name. They encode the hard-won method.

### `serenity-trainer-port`  (`~/.claude/skills/serenity-trainer-port/SKILL.md`)
How to port a model from Serenity (Python) to pure Mojo 1:1. Key contents:
- **THE PORTING RULE (mandatory):** for every function, FIRST find + paste the EXACT source
  (mojodiffusion/serenitymojo for model forwards/VAE/encoders/sampler cores; Serenity `.py`
  for setup/trainer/loader/saver/scheduler), then translate API/namespace ONLY — no deviations,
  no "improvements". If no source exists, **STOP and ASK**. You are PORTING a working pipeline,
  not designing one.
- **Tenet 4:** never say "works/fixed/parity/done" without a tool result in-session; agent
  self-reports are NEVER the gate — the main loop re-runs every gate.
- **Borrow boundary:** IMPORT serenitymojo.{tensor,autograd,ops,io}; COPY model code into the
  port (namespace `serenity_trainer`). bf16 storage / f32 compute, no persistent f32. MGDS dropped.
- **CODE SEPARATION (a 1:1 requirement):** training forward saves activations for backward;
  inference/sampling forward = no-grad, no saved activations, no recompute checkpoints. TWO
  distinct functions; the sampler MUST use the inference one.
- **Verification protocol:** the 6 gates (predict fns exact / forward cos≥0.999 / real-DATA loss
  match / train gate loss-in-range+LoRA-imprint / sampler scheduler-exact+latent-cos+image-PSNR /
  PEFT keys==real OT file). The dtype bad-reference trap (compute the reference in the impl's dtype).
- **Serenity baselines come from its OWN tfevents** (`workspace/<run>/tensorboard/.../events.out.tfevents*`)
  via `EventAccumulator` — NOT remembered numbers. Better: RUN Serenity fresh.
- **Filename traceability:** every model item must name the Serenity `.py`
  source, the same-named Mojo mirror under `src/serenity_trainer`, and the parity
  artifact/gate. `serenitymojo/*` names are helper/core implementation material,
  not the port surface.
- Mojo 1.0.0b1 gotchas + build/checkpoint/cache pointers.

### `numeric-parity-testing`  (`~/.claude/skills/numeric-parity-testing/SKILL.md`)
General (any port/rewrite/kernel/optimization vs a reference):
- Core loop: reference from the REFERENCE's own runtime → byte-identical inputs → compare.
- Metric/bar table: bit-exact / cosine≥0.999 / PSNR>40 / loss-in-range / RNG formula-parity.
- The **dtype bad-reference trap** (bf16 0.318359 vs f64 0.319 — re-derive at the impl's precision).
- **Isolate divergence by stage** (the double-unscale catch).
- Measure-don't-assert; per-function before per-system.

---

## 1. WHAT THIS IS
1:1 port of Serenity (`/home/alex/Serenity`, Python) → pure Mojo in `/home/alex/serenity-trainer`
(456 `.mojo` files / 57 pkgs mirroring `modules/`). Reuse serenitymojo (`/home/alex/mojodiffusion`)
for autograd/tensors/ops by import; borrow model forwards/VAE/encoders by copy-into-port. No MGDS.
First two models remain **Z-Image, then Flux2/Klein** because the runtime needs
a working product path and a large-model gate. Required breadth after that is
**Qwen, Ernie, Anima, SD3, SDXL, Flux.1 dev, Flux2 dev, and Chroma**.
Serenity is the SOLE reference (never EriDiffusion/Rust); Anima's reference tree is
`/home/alex/Serenity-anima-ref`.

## 2. BUILD / RUN (the one command)
```
cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
  timeout 180 prlimit --as=24000000000 \
    pixi run mojo build -I . -I /home/alex/serenity-trainer/src -Xlinker -lm <smoke.mojo> -o /tmp/x && /tmp/x
```
- `-Xlinker -lm` REQUIRED (libm sinf in RoPE).
- Only ONE build at a time (mojopkg race). Workflow agents are told NOT to build; the MAIN LOOP builds + runs + verifies.
- Mojo compile gates should be run with a hard process cap from VSCode:
  `timeout 180 prlimit --as=24000000000 pixi run mojo build ...`. On
  2026-06-05 the old monolithic `smoke/sdxl_setup_surface_check.mojo` compile
  reached about **62 GiB RSS** (`mojo` pid 58957, total-vm 102555300 kB,
  anon-rss 62290732 kB) and the kernel OOM killer killed it inside the VSCode
  snap scope, crashing VSCode. That file is now a tiny manifest; use the split
  `sdxl_setup_*_contract_check.mojo` gates instead.
- Static dtype guard: `python3 scripts/check_flux_family_dtype_contract.py`.
  It reads Serenity sampler/config refs and enforces the narrow F32 contract:
  Flux/Flux2/Z-Image/Chroma sampler Euler latents are F32 because Serenity
  creates them with `dtype=torch.float32`, transformer inputs are cast to BF16
  train dtype, and persistent Klein/Flux2 checkpoint weights must load through
  dtype-preserving `Tensor.from_view`, not F32 host casts. Current run: PASS.
- Checkpoints: Z-Image `/home/alex/.serenity/models/zimage_base/{transformer,vae}` (diffusers dirs).
  Klein 9B `/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors` (18GB, original key layout)
  + VAE `/home/alex/.serenity/models/vaes/flux2-vae.safetensors` + diffusers dir
  `~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/snapshots/<hash>/`.
- Cache: `Serenity/workspace-cache/<run>` (.pt torch-pickle — NOT pure-Mojo readable; dump to safetensors with the OT venv for Mojo gates).
- OT venv for references: `/home/alex/Serenity/venv/bin/python` (torch 2.9.1+cu128).

## 3. STATUS — VERIFIED vs BLOCKED vs UNVERIFIED
Trust only what's marked VERIFIED (a tool result this session showed it).

### VERIFIED (measured by me)
| item | result | smoke / artifact |
|---|---|---|
| AdamW / LR cosine / masked_loss (Phase 1) | bit-exact / exact / exact | parity/{adamw,lr,masked_loss}_ref.json |
| Z-Image forward parity (LoRA B=0) vs diffusers | **cos 0.99939** | smoke/zimage_forward_parity.mojo, parity/zi_fwd.safetensors |
| Z-Image predict fns each vs Serenity | exact (scale_latents 0.318359375 bf16) | smoke/predict_fn_parity.mojo, parity/predict_fn_ref.json |
| Z-Image real-DATA loss | **Mojo 0.4690 = OT 0.4692**, vel cos 0.99996 | smoke/zimage_realdata_loss.mojo, parity/zi_realdata.safetensors |
| Z-Image real-data train (16 steps) | mean **0.5047** (in OT range) | smoke/zimage_realdata_train.mojo, parity/zi_realclean.safetensors |
| Z-Image scheduler+denoise | sigmas exact, latent **cos 0.99897** @1024 | smoke/zimage_sampler_parity.mojo, parity/zi_sampler_ref.safetensors |
| Z-Image VAE decode | **PSNR 54.8** vs OT (same latent) | smoke/zimage_decode_parity.mojo, parity/zi_OT_1024.png |
| Z-Image **conv im2col speedup** | decode **149.9s→1.07s**, PSNR preserved | serenitymojo/ops/conv.mojo (commit **bd5633f** in mojodiffusion, branch training-port-5models-lora) |
| Z-Image **1024 sampler speed** | denoise avg **3.876141s/step**, total **116.28423s**, latent cos vs Serenity **0.9992005027602735** | smoke/zimage_gen1024.mojo, parity/zi_gen1024_ref.safetensors, parity/zi_MOJO_1024_latent.safetensors |
| Z-Image 1024 decode artifact | fresh-process VAE decode **1.162677724s**, image **1024x1024** | smoke/zimage_decode_saved1024.mojo, parity/zi_MOJO_1024.png |
| Z-Image same-process VAE lifetime | transformer+VAE in one singleton CUDA context still OOMs after denoise; current product gate saves latent then decodes in a fresh process/context | smoke/zimage_gen1024.mojo + smoke/zimage_decode_saved1024.mojo |
| Z-Image **2e PEFT save** | **630 keys == Serenity** exactly | smoke/zimage_lora_save_gate.mojo, modelSaver/zImage/ZImageLoRASaver.mojo |
| **1b preset reader** | alina preset exact (lr 3e-4/batch2/epochs100/seed42/LOGIT_NORMAL) | smoke/preset_reader_parity.mojo, util/config/TrainConfigReader.mojo |
| **concept reader** | alina concept exact (name/path/enabled) | smoke/concept_reader_parity.mojo, util/config/ConceptConfig.mojo |
| **Phase 3 CacheReader** | real cache→[16,72,56]/[224,2560], batches, shuffle | smoke/cache_reader_smoke.mojo, dataLoader/CacheReader.mojo |
| **Phase 5 TrainState save/load** | global_step/opt_step/n_slots round-trip | smoke/train_state_roundtrip.mojo, trainer/TrainState.mojo |
| **Klein base-load OOM fix** | checkpoint loads all 8 double + 24 single blocks; `ALL DONE` | smoke/klein_load_only.mojo |
| **Klein forward parity** | **cos 0.9994858031641485**, max_abs_diff 0.08984375, nonfinite 0 | smoke/klein_forward_parity.mojo, parity/klein_fwd.safetensors |
| **Klein real-weight forward/backward smoke** | forward n=2048 mean -0.06562518 var 0.26505914 nonfinite 0; backward 288 grad groups / 43,515,904 elems / abs_sum 700.71155 / nonfinite 0; speed at HL=4/WL=4/NTXT=8: predict ~43.6-43.9 ms, backward ~185.7-203.7 ms, step_no_optim ~229.6-247.5 ms, setup/load ~4.40-4.46 s | smoke/klein_real_forward.mojo |
| **Klein Serenity train reference dump** | real cached-data step 0 dumped: loss **0.12243738770484924**, grad norm **0.005975008010864258**, lr **0.0 -> 3e-6**, elapsed **310.75666852100403s**; 22 step tensors + 1728 adapter tensors | scripts/klein_dump_train_ref.py, parity/klein_train_ref_meta.json, parity/klein_train_ref_step000.safetensors, parity/klein_train_ref_step000_adapters.safetensors |
| **Klein train-ref loss replay** | Mojo computes MSE from Serenity dumped `output.predicted`/`output.target`: Mojo loss **0.12241738** vs dump **0.12243739**, abs err **2.0004809e-05**, loss kernel **0.107232224s**, total **0.268313362s**. This uses a Klein-specific F32 reduction-order tolerance and is loss-only replay, not backward/AdamW train parity. | smoke/klein_train_ref_loss_replay.mojo |
| **Klein adapter grad/delta oracle** | Python gate over the Serenity adapter dump passes with `--check`: **1728 keys** = six phases × **288 FP32 LoRA tensors** / **43,515,904 elems**. Pre/post clip grads are identical and finite: L2 **0.0059750078751807986** vs Serenity grad norm **0.005975008010864258**, abs sum **18.8721539509502**, max **0.00014209747314453125**, nonzero **27,262,171**. Optimizer state advances from 0 to **288 parameter entries / 864 tensors / 87,032,096 elems**, but `adapter_after - adapter_before` is exactly zero because this captured step has `lr_before=[0.0]` and `lr_after=[3e-6]`. A new two-step Serenity CUDA dump in `/tmp` also passes `--expect-update` on step 1: loss **0.5876612663269043**, grad norm **0.008699021302163601**, lr **3e-6 -> 6e-6**, grad L2 **0.008699021115427636**, grad abs sum **25.369888222906162**, update nonzero elems **43,515,904**, update abs sum **51.57613097811465**, update L2 **0.010862200286706213**, max update **3.0034029805392493e-06**. These are Serenity oracles for future Mojo backward/optimizer gates, not Mojo train parity. | scripts/klein_adapter_delta_contract.py (`python3 ... --check`; `python3 ... --meta /tmp/klein_train_ref_2step_meta.json --step-index 1 --expect-update`), parity/klein_train_ref_step000_adapters.safetensors, parity/klein_train_ref_meta.json, /tmp/klein_train_ref_2step_meta.json, /tmp/klein_train_ref_2step_step001_adapters.safetensors |
| **Klein train-path smoke** | finite one-step `Flux2LoRASpec.predict -> loss -> backward_lora -> AdamW`; loss **1.5305071**, grad norm **0.16702795893468853**, LoRA-B imprinted **144/144**, nonfinite 0; latest speed: load **149.778842062s**, predict **0.234450175s**, backward **0.667401176s**, AdamW **4.480295034s**, step **5.506434476s** | smoke/klein_train_parity_smoke.mojo |
| **Klein Flux2 LoRA key+load gate** | tiny state dict emits Serenity raw `transformer.*.lora_down/up/alpha` keys and loads back into **12 double + 2 single** slots | smoke/klein_lora_key_parity.mojo, modelLoader/Flux2ModelLoader.mojo, modelSaver/flux2/Flux2LoRASaver.mojo |
| **Klein sampler helper gate** | bounded helper check passes: 4 scheduler steps, CFG combine `cfg[1]=5.0` | smoke/klein_sampler_helper_gate.mojo |
| **Flux2 dev branch/scaffold gate** | Serenity dev branch is `Flux2Model.is_dev() == transformer.config.num_attention_heads == 48`; local partial FLUX.2-dev transformer config has `num_attention_heads=48`, `num_layers=8`, `num_single_layers=48`, `joint_attention_dim=15360`. Main-loop Mojo structural gate passes: `FLUX2 DEV BRANCH/SAMPLER HELPER GATE OK`, `seq_len=4096`, `steps=28`, `mu=2.1514432`, `sigma1=0.9957105`. Train scaffold/contract exists and blocks correctly on missing Flux2 dev Serenity config and missing CUDA reference tensors. **No Flux2 dev numeric train/sampler parity claim** | smoke/flux2_dev_branch_check.mojo, scripts/flux2_dev_dump_train_ref.py, parity/flux2_dev_train_ref_contract.json, parity/flux2_dev_train_ref_blockers.json |
| **Flux2/Klein fine-tune setup contract** | Serenity `Flux2FineTuneSetup.py` mirror now build-runs as a setup-only gate: `model type = 18`, `param groups = 1` (`transformer`), frozen parts `2` (`text_encoder`, `vae`), cached text on train `False`, uncached text on train `True`, dtype caveats `4`. **No Flux2/Klein full-weight train/save/load/resume parity claim** | smoke/flux2_finetune_setup_contract_check.mojo, src/serenity_trainer/modelSetup/Flux2FineTuneSetup.mojo |
| **Flux2/Klein factory + loader/saver contracts** | Main-loop verified source-contract gates pass: `FLUX2 FACTORY CONTRACT OK`, `FLUX2 SURFACE LOADER CONTRACT OK`, `FLUX2 SURFACE SAVER CONTRACT OK`, and `FLUX2 RUNTIME LOADER KEY CONTRACT OK`; static dtype guard also passes. They mirror Serenity `ModelType.FLUX_2` dispatch, `flux_2.0-lora.json`/`flux_2.0.json` wrapper specs, Dev/Klein `num_attention_heads == 48` split, BF16 transformer override defaults, unsupported single-file Flux2 error, fine-tune saver routes, LoRA raw `transformer.<diffusers module>.lora_down/up/alpha` state sources, and internal LoRA destination. **No Flux2/Klein runtime load/save, full-weight save/load/resume, train, sampler, or numeric parity claim** | smoke/flux2_factory_contract_check.mojo, smoke/flux2_surface_loader_contract_check.mojo, smoke/flux2_surface_saver_contract_check.mojo, smoke/flux2_runtime_loader_key_contract_check.mojo, src/serenity_trainer/util/factory.mojo, src/serenity_trainer/util/create.mojo, src/serenity_trainer/modelLoader/Flux2ModelLoader.mojo, src/serenity_trainer/modelLoader/Flux2RuntimeLoader.mojo, src/serenity_trainer/modelSaver/Flux2FineTuneModelSaver.mojo, src/serenity_trainer/modelSaver/Flux2LoRAModelSaver.mojo, src/serenity_trainer/modelSaver/flux2/Flux2ModelSaver.mojo, src/serenity_trainer/modelSaver/flux2/Flux2LoRASaver.mojo, scripts/check_flux_family_dtype_contract.py |
| **Qwen surfaces + helper/file/train refs** | model-core, Z-Image Qwen3 text encoder compile, loader/saver/sampler/factory surfaces build and run; real Serenity one-step dump, loss-only replay, real LoRA file parity, and sampler helper parity exist below. **No Qwen transformer/backward/AdamW train parity, full-finetune parity, text branch parity, VAE parity, or end-to-end sampler parity claim** | smoke/qwen_model_compile_check.mojo, smoke/qwen_text_encoder_compile_check.mojo, smoke/qwen_surface_check.mojo |
| **Qwen Serenity train reference dump** | real cached-data step 0 dumped: loss **0.08948630839586258**, grad norm **0.00014067116717342287**, lr **0.0 -> 1.5e-6**, elapsed **612.1598391110019s**; trainable tensors **1440**, params **106,168,320**, 29 step tensors + 5760 adapter tensors | scripts/qwen_dump_train_ref.py, parity/qwen_train_ref_meta.json, parity/qwen_train_ref_step000.safetensors, parity/qwen_train_ref_step000_adapters.safetensors |
| **Qwen Serenity profile-only step** | same dumped-step loss **0.08948630839586258** without safetensor/adapter writes; harness elapsed **465.4570777180197s** split into load/setup **295.3092309619824s**, cache epoch start **153.70464057300705s**, predict **12.852268689981429s**, backward **2.742950047017075s**, optimizer **0.4419737099960912s**. This is first-step oracle overhead, not steady-state training speed. | scripts/qwen_dump_train_ref.py --profile-only, parity/qwen_train_profile_meta.json |
| **Qwen train-ref loss replay** | Mojo computes MSE from Serenity dumped `output.predicted`/`output.target`: Mojo loss **0.0894845** vs dump **0.08948631**, abs err **1.8104911e-06**, loss kernel **0.133140851s**, total **0.291906782s**. This is loss-only replay, not Qwen transformer parity. | smoke/qwen_train_ref_loss_replay.mojo |
| **Qwen LoRA key+load gate** | bounded smoke still passes; real Serenity baseline file now also passes full key/shape/dtype gate: **2160 keys / 720 adapters / rank 16 / BF16 / alpha 1.0** from `/home/alex/Serenity/output/qwen_100step_baseline/lora.safetensors` | smoke/qwen_lora_key_parity.mojo, smoke/qwen_real_lora_file_parity.mojo, modelSetup/qwenLoraTargets.mojo, modelSaver/qwen/QwenLoRASaver.mojo, modelLoader/qwen/QwenLoRALoader.mojo |
| **Qwen sampler helper gate** | bounded helper math passes in **0.02s**: 1024x1024 -> latent 128x128, packed 4096x64, CFG batch 2, shift 3.1581929, sigma1 0.86349744, timestep1 863.49744 | smoke/qwen_sampler_helper_gate.mojo, modelSampler/QwenSampler.mojo |
| **Ernie surfaces + LoRA file + train refs** | model-core and loader/saver/sampler/factory surfaces build and run; real Serenity baseline LoRA file key/shape/dtype gate passes: **756 keys / 252 adapters / rank 16 / BF16 / alpha 1.0** from `/home/alex/Serenity/output/ernie_eri2_100step_baseline/lora.safetensors`; real CUDA one-step dump exists: loss **0.643847644329071**, grad norm **0.000828770047519356**, lr **0.0003 -> 0.0003**, elapsed **97.59743943800095s**, trainable tensors **504**, params **47,185,920**; Mojo loss-only replay passes with loss **0.6438152** vs dump **0.64384764**, abs err **3.2424927e-05** under an explicit Ernie F32 reduction tolerance, loss kernel **0.238489161s**, total **0.717235302s**. Sampler helper gate passes: plan **1024x1024**, latent **128x128**, patch contract **64x32x128**, CFG batch **2**, sigma1 **0.75**, timestep1 **750.0**, Euler **0.125**. **No Ernie transformer/backward/AdamW train parity, full-finetune parity, or end-to-end sampler parity claim yet** | smoke/ernie_model_compile_check.mojo, smoke/ernie_surface_check.mojo, smoke/ernie_real_lora_file_parity.mojo, scripts/ernie_dump_train_ref.py, parity/ernie_train_ref_meta.json, parity/ernie_train_ref_step000.safetensors, parity/ernie_train_ref_step000_adapters.safetensors, smoke/ernie_train_ref_loss_replay.mojo, smoke/ernie_sampler_helper_gate.mojo |
| **Anima build-only surfaces + helper gates** | model-core, setup/data, loader/saver/sampler/factory surfaces build and run; shared `ModelType` now includes `ANIMA` before `Z_IMAGE` to avoid collision; real Serenity Anima LoRA file key/shape/dtype gate passes: **840 keys / 280 adapters / rank 16 / BF16 / alpha 1.0** from `/home/alex/Serenity-anima-ref/output/anima_100step_baseline/lora.safetensors`; sampler helper gate passes (`1024x1152`, latent `128x144`, CFG batch 2, sigma1 `0.9`, timestep1 `900.0`); train-reference dry-run is structural-only and unblocked, and the separate CUDA Serenity one-step dump exists below. **No Anima full train/sampler numeric parity claim** | smoke/anima_model_compile_check.mojo, smoke/anima_setup_surface_check.mojo, smoke/anima_surface_check.mojo, smoke/anima_real_lora_file_parity.mojo, parity/gen_anima_sampler_helper_ref.py, parity/anima_sampler_helper_ref.json, smoke/anima_sampler_helper_gate.mojo, scripts/anima_dump_train_ref.py, parity/anima_train_ref_blockers.json |
| **Anima Serenity train reference dump** | real cached-data step 0 dumped: loss **0.0667838305234909**, grad norm **0.0014594600070267916**, lr **0.0 -> 1.5e-7**, elapsed **44.37211945699528s**; trainable tensors **560**, params **22,937,600**, 28 step tensors + 2240 adapter tensors | scripts/anima_dump_train_ref.py, parity/anima_train_ref_meta.json, parity/anima_train_ref_step000.safetensors, parity/anima_train_ref_step000_adapters.safetensors |
| **Anima train-ref loss replay** | Mojo computes MSE from Serenity dumped `output.predicted`/`output.target`: Mojo loss **0.06678182** vs dump **0.06678383**, abs err **2.0116568e-06**, loss kernel **0.135911962s**, total **0.31113741s**. This is loss-only replay, not Anima transformer parity. | smoke/anima_train_ref_loss_replay.mojo |
| **SD3 build-only surfaces + structural helper gates** | model-core, setup/data, loader/saver/sampler/factory surfaces build and run; CUDA retry outside the sandbox passes `smoke/sd3_model_compile_check.mojo` and expanded Linear LoRA inventory contract (`linear_targets=96 entries=288 rank=2 dtype=BF16 alpha=4.0`); bounded LoRA raw key gate previously passes targets=9/entries=27; sampler helper contract passes against generated Serenity/diffusers ref (`plan 1024x1056`, latent `128x132`, `cfg_batch=2`, `sigma1=0.85769236`, `sigma2=0.60215056`, `timestep1=857.6924`, latest helper timing `7.0001e-05s`); CUDA-visible dry-run writes structured blockers and explicitly marks CPU PyTorch/dry-run as structural only. **No SD3 train/sampler numeric parity claim** | smoke/sd3_model_compile_check.mojo, smoke/sd3_setup_surface_check.mojo, smoke/sd3_surface_check.mojo, smoke/sd3_lora_key_parity.mojo, smoke/sd3_lora_inventory_contract.mojo, parity/gen_sd3_sampler_helper_ref.py, parity/sd3_sampler_helper_ref.json, smoke/sd3_sampler_helper_gate.mojo, scripts/sd3_dump_train_ref.py, parity/sd3_train_ref_contract.json, parity/sd3_train_ref_blockers.json |
| **SDXL surfaces + helper/file/train refs** | model-core metadata/shape gate passes in the sandbox, and a separate tiny CUDA tensor gate verifies BF16 VAE-scale round-trip; split setup/data contracts and split loader/saver/sampler/factory surfaces build and run; the old monolithic setup and surface gates are disabled as manifests after verified compiler OOMs, and the split gates pass under a 24 GB `prlimit`; BF16 final-output LoRA file gate passes **2382 keys / 794 adapters / rank 16 / alpha 16.0**; sampler helper contract gate passes and explicitly scopes itself to helper-only, no denoise/decode/image/end-to-end sampler parity (`1024x1152`, latent `128x144`, CFG batch 2, inpaint channels 9, timesteps 3-4, scheduler `EulerAncestralDiscreteScheduler`, latest helper timing `0.000143672s`); refreshed Serenity CUDA one-step dump exists: loss **0.13533265888690948**, grad norm **0.010624590329825878**, lr **1e-4 -> 1e-4**, elapsed **61.35357101200134s**; Mojo loss-only replay passes with loss **0.13533124** vs dump **0.13533266**, abs err **1.4156103e-06**, loss kernel **0.054720012s**, total **0.220353558s**; CPU-only adapter-delta contract passes over the Serenity adapter dump: **6352 keys**, four phases of **1588 tensors / 49,412,736 elems**, reference dtype **FLOAT_32**, unchanged before/pre/post phases, after-step deltas in **1588 tensors / 49,412,638 elems**, abs sum **1993.9059369890892**, max **9.999924077419564e-05**, runtime **1.2808432990004803s**. **No SDXL transformer/backward/AdamW train parity, full-finetune parity, or end-to-end sampler parity claim** | smoke/sdxl_model_compile_check.mojo, smoke/sdxl_model_tensor_contract_check.mojo, smoke/sdxl_setup_surface_check.mojo, smoke/sdxl_setup_base_contract_check.mojo, smoke/sdxl_setup_method_contract_check.mojo, smoke/sdxl_setup_dataloader_contract_check.mojo, smoke/sdxl_surface_check.mojo, smoke/sdxl_surface_loader_contract_check.mojo, smoke/sdxl_surface_sampler_contract_check.mojo, smoke/sdxl_surface_saver_contract_check.mojo, smoke/sdxl_surface_factory_contract_check.mojo, smoke/sdxl_real_lora_file_parity.mojo, parity/gen_sdxl_sampler_helper_ref.py, parity/sdxl_sampler_helper_ref.json, smoke/sdxl_sampler_helper_gate.mojo, scripts/sdxl_dump_train_ref.py, parity/sdxl_train_ref_meta.json, parity/sdxl_train_ref_step000_adapters.safetensors, smoke/sdxl_train_ref_loss_replay.mojo, scripts/sdxl_adapter_delta_contract.py |
| **Chroma model/setup + loader/saver + helper/file/train refs** | Serenity 100-step baseline completes: mean step **3.2088356397171727s**, min/max **2.755967059987597/6.3993451290007215s**, last loss **0.3658050298690796**, smooth **0.4213832823932175**, grad norm **0.001406678231433034**, peak sampled VRAM **9913 MiB**, CUDA max allocated **7024 MiB**. Model/setup contract gate passes: `model type = 22`, predict outputs `4`, dtype caveats `6`, LoRA opt parts `5`, FINE_TUNE params `3`, LoRA params `3`, `create te = True`, ragged text seq `32`, packed `4608x64`, `model_t(500)=0.5`, `sigma(499)=0.5`. Loader/saver split contract gates pass: `CHROMA SURFACE LOADER CONTRACT OK`, `CHROMA SURFACE SAVER CONTRACT OK`, `CHROMA LORA CONVERSION CONTRACT OK`. Sampler helper gate passes: `1024x1152`, latent `128x144`, packed `4608x64`, CFG batch 2, FlowMatch shift `3.0`, sigma1 `0.85769236`, timestep1 `857.6924`, runtime **0.00010915s**. Real Serenity LoRA file gate passes: **912 keys / 304 adapters / rank 16 / BF16 / alpha keys present / bundle_keys 0**. One-step dump exists: loss **0.2957186698913574**, grad norm **0.00041076153866015375**, lr **0.0003 -> 0.0003**, elapsed **169.69719535199692s**; Mojo loss-only replay passes with loss **0.29572487** vs dump **0.29571867**, abs err **6.198883e-06**, loss kernel **0.108526602s**, total **0.267413886s**. **No Chroma transformer/backward/AdamW train parity, full-finetune runtime/save/load/resume parity, or end-to-end sampler parity claim** | smoke/chroma_model_setup_contract_check.mojo, src/serenity_trainer/model/ChromaModel.mojo, src/serenity_trainer/modelSetup/BaseChromaSetup.mojo, src/serenity_trainer/modelSetup/ChromaFineTuneSetup.mojo, src/serenity_trainer/modelSetup/ChromaLoRASetup.mojo, smoke/chroma_surface_loader_contract_check.mojo, smoke/chroma_surface_saver_contract_check.mojo, smoke/chroma_lora_conversion_contract_check.mojo, src/serenity_trainer/modelLoader/ChromaFineTuneModelLoader.mojo, src/serenity_trainer/modelLoader/ChromaLoRAModelLoader.mojo, src/serenity_trainer/modelLoader/ChromaEmbeddingModelLoader.mojo, src/serenity_trainer/modelLoader/chroma/ChromaModelLoader.mojo, src/serenity_trainer/modelLoader/chroma/ChromaLoRALoader.mojo, src/serenity_trainer/modelLoader/chroma/ChromaEmbeddingLoader.mojo, src/serenity_trainer/modelSaver/ChromaFineTuneModelSaver.mojo, src/serenity_trainer/modelSaver/ChromaLoRAModelSaver.mojo, src/serenity_trainer/modelSaver/ChromaEmbeddingModelSaver.mojo, src/serenity_trainer/modelSaver/chroma/ChromaModelSaver.mojo, src/serenity_trainer/modelSaver/chroma/ChromaLoRASaver.mojo, src/serenity_trainer/modelSaver/chroma/ChromaEmbeddingSaver.mojo, src/serenity_trainer/util/convert/lora/convert_chroma_lora.mojo, smoke/chroma_sampler_helper_gate.mojo, parity/chroma_sampler_helper_ref.json, smoke/chroma_real_lora_file_parity.mojo, scripts/chroma_dump_train_ref.py, parity/chroma_train_ref_meta.json, parity/chroma_train_ref_step000.safetensors, parity/chroma_train_ref_step000_adapters.safetensors, smoke/chroma_train_ref_loss_replay.mojo |
| **Flux.1 dev surfaces + sampler/file gates** | model-core, setup/data, loader/saver/sampler/factory surfaces build and run for dev/fill; shared Flux.1 dispatch verified; sampler helper gate passes: plan **1024x1024**, latent **128x128**, packed **4096x64**, shift **3.1581929**, mu **1.15**, model timestep **0.5**, fill mask channels **256**. Real Serenity LoRA file gate passes against `/home/alex/Serenity/output/flux1_100step_baseline/lora_last.safetensors`: **1512 keys / 504 adapters / rank 16 / BF16 / scalar alpha / bundle_keys 0**. This is a key/shape/dtype gate only; the Mojo gate does not read alpha values. **No Flux train numeric parity, full-finetune parity, or end-to-end sampler parity claim** | smoke/flux_model_compile_check.mojo, smoke/flux_setup_surface_check.mojo, smoke/flux_surface_check.mojo, smoke/flux_sampler_helper_gate.mojo, smoke/flux_real_lora_file_parity.mojo, src/serenity_trainer/util/convert/lora/convert_flux_lora.mojo |

### MEASURED BASELINES (Serenity's own, this session)
- **Z-Image** (`workspace/alina_zimage_OTpreset_100_baseline` tfevents, 117 steps): loss mean **0.461**,
  smooth ~0.46, min 0.309; **2.19 s/step @512**, lr 3e-4. (grad_norm + lr/transformer also logged.)
- **Klein 9B** (RAN FRESH this session, 102 steps, `klein9b_alina_baseline` config 2026-05-14): loss mean
  **0.6320**, min 0.2177, smooth **0.6320**; **~3.2 s/step @512**, lr 3e-5, batch 2, bf16. ← the gate the
  Mojo Klein TRAIN path must match.
- **Ernie** (`/home/alex/Serenity/output/ernie_eri2_100step_baseline/metrics_nocompile.json`):
  100/100 steps, losses=100, grad_norms=100; mean step **3.210763874979797s**,
  min/max 2.4881598510000913/13.119598707000478s; last loss
  **0.7464916110038757**, smooth **0.6612294220924374**, grad norm
  **0.0014650028897449374**; peak sampled VRAM **13083 MiB**.
  One-step dump now exists: loss **0.643847644329071**, grad norm
  **0.000828770047519356**, lr **0.0003 -> 0.0003**, elapsed
  **97.59743943800095s**; 27M step safetensors + 721M adapter safetensors.
  Mojo loss-only replay is **0.6438152** vs dump **0.64384764**.
- **Qwen** (`/home/alex/Serenity/output/qwen_100step_baseline/metrics.json`):
  100/100 steps, losses=100, grad_norms=100; mean step **3.9018539588785535s**,
  min/max 3.2750983249861747/6.007235787983518s; last loss
  **0.08166131377220154**, smooth **0.10073877252638333**, grad norm
  **0.0011983560398221016**; peak sampled VRAM **15227 MiB**.
  One-step dump for Mojo replay:
  `parity/qwen_train_ref_step000.safetensors` + adapters; step loss
  **0.08948630839586258**, grad norm **0.00014067116717342287**, lr
  **0.0 -> 1.5e-6**, elapsed **612.1598391110019s**. Do not compare that
  dump elapsed to Serenity's 100-step throughput: a profile-only rerun split
  first-step overhead into load/setup **295.3092309619824s**, cache epoch start
  **153.70464057300705s**, predict **12.852268689981429s**, backward
  **2.742950047017075s**, optimizer **0.4419737099960912s**. Mojo Qwen has only
  loss-only replay so far: **0.0894845** vs dump **0.08948631**.
- **Anima** (`/home/alex/Serenity-anima-ref/output/anima_100step_baseline/metrics_with_grad.json`):
  100/100 steps, losses=100, grad_norms=100 after adding the same pre-clip
  `grad_norm` tensorboard scalar instrumentation used by the main Serenity ref;
  mean step **1.0924571535151424s**, min/max
  0.9320262139954139/6.304523999016965s; last loss
  **0.11683385074138641**, smooth **0.10126834750175476**, grad norm
  **0.0030804856214672327**; peak sampled VRAM **5351 MiB**.
  One-step dump now exists: loss **0.0667838305234909**, grad norm
  **0.0014594600070267916**, lr **0.0 -> 1.5e-7**, elapsed
  **44.37211945699528s**; Mojo loss-only replay is **0.06678182** vs dump
  **0.06678383**.
- **SDXL** (`/home/alex/Serenity/output/sdxl_100step_baseline/metrics.json`):
  100/100 steps, losses=100, grad_norms=100; mean step **0.9730077266059298s**,
  min/max 0.8891557799943257/1.2480809040134773s; last loss
  **0.018004747107625008**, smooth **0.11401765364687881**, grad norm
  **0.0029590092599391937**; peak sampled VRAM **8822 MiB**.
  Refreshed CUDA one-step dump now exists: loss **0.13533265888690948**,
  grad norm **0.010624590329825878**, lr **0.0001 -> 0.0001**, elapsed
  **61.35357101200134s**; 31 step tensors + 6352 adapter tensors.
- **Chroma** (`/home/alex/Serenity/output/chroma_100step_baseline/metrics.json`):
  100/100 steps, losses=100, grad_norms=100; mean step
  **3.2088356397171727s**, min/max
  **2.755967059987597/6.3993451290007215s**; last loss
  **0.3658050298690796**, smooth **0.4213832823932175**, grad norm
  **0.001406678231433034**; peak sampled VRAM **9913 MiB**, CUDA max allocated
  **7024 MiB**. One-step dump now exists: loss **0.2957186698913574**,
  grad norm **0.00041076153866015375**, lr **0.0003 -> 0.0003**, elapsed
  **169.69719535199692s**; 33 step tensors + 608 trainable tensors
  (35,487,744 params) across adapter before/pre/post/after phases. Mojo
  loss-only replay is **0.29572487** vs dump **0.29571867**.
- **Flux.1 dev** (`/home/alex/Serenity/output/flux1_100step_baseline/metrics_full_snapshot.json`):
  100/100 steps, losses=100, grad_norms=100; mean step **2.8278740769089907s**,
  min/max 2.804770661983639/3.843770764011424s; last loss
  **0.5231022238731384**, smooth **0.4240351316332816**, grad norm
  **0.049582332372665405**; peak sampled VRAM **11672 MiB**.
- **SD3/SD3.5** baseline is blocked structurally before numeric parity:
  `configs/sd35m_100step_baseline.json` uses
  `ModelType.STABLE_DIFFUSION_3`, but Serenity's current
  `StableDiffusion3BaseDataLoader.py` registration is for
  `ModelType.STABLE_DIFFUSION_35`; the required
  `/home/alex/Serenity/workspace-cache/sd35m_100step_baseline/{image,text}`
  cache is absent; local single-file SD3.5 checkpoints such as
  `/home/alex/.serenity/models/checkpoints/stablediffusion35_medium.safetensors`
  have diffusion+VAE keys only, not inspectable CLIP/T5 text encoder weights.
  CUDA is visible outside the sandbox (`cuda_available=true`, one RTX 3090 Ti),
  but a non-dry-run probe still stops before model load on those three blockers.
  The dry-run artifact is structural only and is not CPU numeric parity.

### Klein (Phase 4) — load/forward/backward viable, train parity still partial
- Klein full forward COMPILES (after fixing the 2nd-pass agents' dropout-code bugs: `out` used as a
  param name = reserved keyword; LoraDropout/StreamLoraDropout/DoubleBlockLoraDropout needed
  `ImplicitlyCopyable`). 144 adapters (8×12 double + 24×2 single, separate q/k/v) confirmed faithful.
- **9B load OOM fixed:** after the BF16-storage loader edit,
  `smoke/klein_load_only.mojo` reaches `ALL DONE: 8 double + 24 single`.
- **Forward parity fixed:** `smoke/klein_forward_parity.mojo` passes vs
  Serenity/diffusers FLUX.2-klein-base-9B with cosine 0.9994858031641485,
  max_abs_diff 0.08984375, and nonfinite 0.
- **Training-path forward/backward smoke fixed:** `smoke/klein_real_forward.mojo`
  now runs `Flux2LoRASpec.predict` and `Flux2LoRASpec.backward_lora` on real 9B
  weights. Forward stats: n=2048, mean=-0.06562518, var=0.26505914, nonfinite=0.
  Backward stats: 288 LoRA grad groups, 43,515,904 elems, abs_sum=700.71155,
  nonfinite=0. Warm-run speed at HL=4/WL=4/NTXT=8: predict ~43.6-43.9 ms,
  backward ~185.7-203.7 ms, step_no_optim ~229.6-247.5 ms, setup/load
  ~4.40-4.46 s.
- **Serenity train reference now exists:** `scripts/klein_dump_train_ref.py`
  produced `parity/klein_train_ref_step000.safetensors` and
  `parity/klein_train_ref_step000_adapters.safetensors` from the real
  `/home/alex/Serenity/configs/AB50_klein9b_ot.json` cached-data path. Step 0
  reference: loss 0.12243738770484924, grad norm 0.005975008010864258, lr
  0.0 -> 3e-6.
- **Loss-only replay fixed:** `smoke/klein_train_ref_loss_replay.mojo` consumes
  the real Serenity dump and computes Mojo MSE 0.12241738 vs dump
  0.12243739, abs error 2.0004809e-05, with loss kernel 0.107232224s and total
  0.268313362s. The gate carries a Klein-specific 2.5e-5 F32 reduction-order
  tolerance; it is not a backward/AdamW parity claim.
- **Train-path smoke fixed, not numeric parity:** `smoke/klein_train_parity_smoke.mojo`
  runs predict/loss/backward/AdamW finite and imprints all 144 adapters, but it
  still uses the older `parity/klein_fwd.safetensors` fixture and Mojo RNG path.
  Latest measured speed: load 149.778842062s, predict 0.234450175s, backward
  0.667401176s, AdamW 4.480295034s, step 5.506434476s.
- **PEFT key/load fixed for the block carrier:** `smoke/klein_lora_key_parity.mojo`
  now writes a real safetensors file and loads it back into 12 double + 2 single
  slots. Remaining PEFT gaps are real-file comparison, preloaded extra-key
  preservation, and full/default layer-filter coverage beyond the block set.
- **Full-finetune setup contract added:** `smoke/flux2_finetune_setup_contract_check.mojo`
  passes for Serenity's transformer-only `Flux2FineTuneSetup.py`: one
  `transformer` parameter group, text encoder and VAE frozen, cached text/VAE on
  temp device, uncached text/VAE on train device, and explicit dtype caveats for
  the Serenity `latent_image.float()` compute boundary.
- **Adapter dump oracle added:** `scripts/klein_adapter_delta_contract.py --check`
  now validates the Serenity adapter dump phases, grad norm aggregate, and
  optimizer-state transition. It also exposes why this dump cannot prove AdamW
  parameter-update parity: step 0 has `lr_before=[0.0]`, so `adapter_after -
  adapter_before` is exactly zero even though gradients and optimizer state are
  nonzero. A two-step Serenity CUDA dump was produced under `/tmp`, and
  `scripts/klein_adapter_delta_contract.py --meta /tmp/klein_train_ref_2step_meta.json
  --step-index 1 --expect-update` passes, proving a nonzero-LR adapter update
  oracle exists for step 1.
- **Factory/loader/saver contracts added:** Flux2/Klein now has build-run
  source-contract coverage for `util/create.py`/`factory.py`, `Flux2ModelLoader.py`,
  `Flux2FineTuneModelSaver.py`, `Flux2LoRAModelSaver.py`,
  `flux2/Flux2ModelSaver.py`, and `flux2/Flux2LoRASaver.py`. These gates
  passed in the main loop, plus `scripts/check_flux_family_dtype_contract.py`.
  The pre-existing full `smoke/klein_lora_key_parity.mojo` roundtrip was not
  rebuilt here because this sandbox reports an unknown GPU architecture when it
  instantiates `DeviceContext()`.
- **Still missing:** Mojo replay of the real Serenity dump through
  byte-identical `predict -> backward_lora -> AdamW` with numeric comparison for
  grad norm, adapter update delta, phase speed, sampler trajectory, and saved
  PEFT keys.

### UNVERIFIED (agent-written, NOT re-run by me)
- Phase 3 `Prepare.mojo` (images→VAE/text encode→safetensors cache) + `Bucketing.mojo` — own smokes pending.
- Phase 5 `SampleCadence.mojo` + `SaveBackupCadence.mojo` — cadence-trigger + full save→resume smoke pending.
- Qwen now has a Serenity one-step dump, profile-only phase timings, a narrow
  Mojo loss replay from dumped tensors, plus bounded LoRA key/load and sampler
  helper gates. Qwen also has real Serenity LoRA file key/shape/dtype parity
  for 2160 keys / 720 adapters / rank 16 / BF16 / alpha 1.0, but still lacks Mojo transformer forward/backward train replay
  and end-to-end sampler parity. Ernie has build surfaces, a 100-step Serenity
  baseline, real LoRA file key/shape/dtype parity, a real CUDA one-step dump,
  and a loss-only replay, but still lacks Mojo transformer/backward/AdamW train
  parity, full-finetune parity, and end-to-end sampler parity. Anima has build
  surfaces plus real LoRA file parity, sampler helper parity, a one-step dump,
  and loss-only replay; Flux.1 dev remains build-only surface verified with a
  100-step Serenity baseline; SDXL now additionally has BF16 final-output
  LoRA file parity, sampler helper-contract parity, and a one-step Serenity train
  reference dump, but still lacks Mojo train replay and end-to-end sampler
  parity. SD3 CUDA visibility is fixed outside the sandbox, but numeric
  reference generation is still blocked by a data-loader registration mismatch,
  missing cache, and missing inspectable CLIP/T5 weights in the local
  single-file checkpoint; CPU/dry-run output is structural only.
  Chroma now has
  build-only model/setup/LoRA-setup plus loader/saver/LoRA-conversion
  Serenity contracts, a main-loop verified helper-only sampler gate, real
  LoRA file parity, a one-step Serenity dump, and a loss-only replay, but no
  transformer/backward/AdamW, full-finetune runtime/save/load/resume, or
  end-to-end sampler numeric parity. Flux2 dev now has only a
  structural branch/scaffold gate: local partial transformer config proves the
  Serenity `num_attention_heads == 48` dev condition, but there is no
  Serenity Flux2 dev config/baseline or CUDA tensor dump yet. Each still needs
  predict/loss/optimizer/sampler/PEFT numeric gates and Mojo speed measurements.
- Z-Image transformer inference @1024 now measures **~3.88s/step** after using
  matmul-backed SDPA for the 4320-token sequence and caching prompt/RoPE state.
  This is a real sampler speed gate, not comparable to OT's 2.19s/step training
  @512. Same-process VAE decode after denoise still OOMs because the singleton
  CUDA context retains transformer allocations; the current verified artifact
  path saves the latent and decodes in a fresh process.

## 4. THE 7 PHASES (see PORT_MAP.md for the live table)
0 skeleton ✅ · 1 spine ✅ (+1b preset reader ✅, concept reader ✅) · 2 Z-Image ✅ (train+sample+save
verified; only transformer-train-speed open) · 3 dataLoader 🔄 (CacheReader ✅; Prepare/bucketing
unverified) · 4 Klein 🔄 (load/forward/backward/ref-dump viable; Mojo real train replay missing) · 5 cadence/save-resume 🔄
(TrainState ✅; cadence unverified) · 6 required breadth ⬜
(Qwen/Ernie/Anima/SD3/SDXL/Flux.1 dev/Flux2 dev/Chroma + remaining target
optimizers and all Serenity-supported full-finetune paths). Qwen is REQUIRED
again; Anima comes from `/home/alex/Serenity-anima-ref`.

### CONFIG/RUN LAYER (was missing from the original plan — track it)
preset reader ✅ · concept reader ✅ · sample-prompt reader ⬜ · **terminal-UI/CLI entry ⬜** (NOT
Tkinter — the `ui/` stubs are Tkinter and Mojo has no GUI toolkit; build a TERMINAL UI = CLI that loads
preset+concepts → runs GenericTrainer → prints live step/loss/lr + sample/save/backup cadence. Build it
AFTER Phase 3/5 since it orchestrates them).

## 5. NEXT ACTIONS (in order)
1. **Audit follow-through:** use `PORT_MAP.md` "Serenity Audit (2026-06-05)"
   as the active plan. The first parity target is a Serenity-style pure-Mojo
   training runtime, not a collection of smoke-local model assemblies.
2. **Product run layer:** implement the first real CLI entry (`scripts/train.py`
   equivalent) plus `util/create.py` / `util/factory.py` dispatch for Z-Image
   LoRA and Flux2/Klein LoRA. Gate it by running a preset through the CLI and
   comparing to Serenity outputs.
3. **Config completeness:** extend `TrainConfig` / `TrainConfigReader` for the
   fields consumed by Serenity's `scripts/train.py -> GenericTrainer` path:
   paths, model names, model type, training method, dtype, optimizer,
   cache/workspace/output, sample/save/backup, validation, EMA/prior/masked flags.
4. **Klein train gate:** load, forward parity, Serenity one-step reference
   dump, loss-only replay, LoRA key/load, sampler helpers, and smoke train path
   are viable. Next required gate is Mojo replay of
   `parity/klein_train_ref_step000*.safetensors` with grad/update numbers
   matching the Serenity dump, then longer loss range comparison against the
   **0.632** Klein baseline.
5. **Data path gates:** verify Prepare/Bucketing against Serenity, then run
   `Prepare -> CacheReader -> predict` with first-batch order and tensors matching
   Serenity for the same seed.
6. **Save/resume/cadence gates:** full save→resume at accum=1 and accum>1,
   sample-prompt reader, and in-training sample cadence with Serenity sampler
   parity.
7. **Required model breadth:** once the product run layer, data path, and
   save/resume/cadence gates can host more than smoke-local assemblies, add
   Qwen, Ernie, Anima, SD3, SDXL, Flux.1 dev, Flux2 dev, and Chroma. Each model
   needs loader/setup/data-loader/sampler/saver gates against Serenity for
   LoRA and for full finetuning wherever Serenity registers
   `TrainingMethod.FINE_TUNE`. Anima uses `/home/alex/Serenity-anima-ref`;
   the rest use `/home/alex/Serenity`. Current state: the earlier six required
   model surfaces have build-only Mojo smoke gates; Qwen additionally has a
   Serenity step dump plus bounded LoRA/sampler helper gates; Anima additionally
   has real LoRA file parity, sampler helper parity, a one-step dump, and a
   loss-only replay; SDXL now has BF16 final-output LoRA file parity, sampler
   helper-contract parity, and a one-step dump. Ernie additionally has a real CUDA
   one-step dump and loss-only replay. Flux2 dev additionally has a structural branch/scaffold
   gate proving the Serenity `num_attention_heads == 48` dev condition from a
   partial local transformer config, but it still lacks a usable Serenity dev
   config, baseline, and CUDA reference tensors. Ernie/Qwen/Anima/SDXL/Flux.1/Chroma have fresh
   100-step Serenity baselines. None of these required breadth models have full
   Mojo numeric train/sampler parity yet; SD3 still needs a CUDA Serenity
   reference path plus a matching data-loader registration, cache/text encoders,
   and full local reference snapshot before its Serenity baseline can be run.
   Chroma has
   model/setup/LoRA-setup/loader/saver/LoRA-conversion/helper/file/train-reference
   gates but still lacks full train/sampler parity; Flux2 dev still needs a
   baseline, and both Flux2 dev and Chroma still need full parity gates.
8. **Only after Serenity parity:** refactor/mold the architecture into the new
   trainer. Correctness gates come before speed passes.

## 6. KEY MOJO 1.0.0b1 GOTCHAS (cost real time this session)
- `out` and `ref` are RESERVED — never a param/var NAME (field `out` is fine; param `out:` is not).
- By-value struct copy needs `ImplicitlyCopyable` OR explicit `.copy()`; move-only `Tensor` → `ArcPointer`.
- Tuple subscript can't be moved (`tup[0]^` fails; `var (a,b)=f()` COPIES move-only elems). For
  `Tuple[Tensor,Tensor]` returns, prefer changing the fn to a named struct, or `ref x = tup[0]` if a
  borrow suffices (can't be `^`-moved into a consuming call).
- `len()` needs the struct to declare `Sized`.
- `save_safetensors` needs the destination DIR to pre-exist (`mkdir -p`).
- NEVER partial-move a field out of a destructor-bearing struct; move the whole struct or clone the field.

## 7. PARITY ARTIFACTS (`serenity-trainer/parity/`)
adamw_ref, lr_ref, masked_loss_ref, predict_ref, predict_fn_ref, zi_fwd, zi_realdata, zi_realclean,
zi_sampler_ref, zi_gen1024_ref, zi_OT_1024.png, zi_MOJO_decode.png, zi_MOJO_1024_latent.safetensors,
zi_MOJO_1024.png, klein_fwd.safetensors +
klein_fwd_meta.json (diffusers FLUX.2-9B forward ref),
klein_train_ref_meta.json, klein_train_ref_step000.safetensors,
klein_train_ref_step000_adapters.safetensors, qwen_train_ref_meta.json,
qwen_train_ref_step000.safetensors, qwen_train_ref_step000_adapters.safetensors
+ gen_*.py / dump scripts (run with the OT venv).

## 8. PLAN / MEMORY
Plan: `~/.claude/plans/polymorphic-yawning-hinton.md`. Memory: `serenity-trainer-port-{progress,constraints}`.
Live status: `PORT_MAP.md`. mojodiffusion conv change committed on branch `training-port-5models-lora` (bd5633f).
