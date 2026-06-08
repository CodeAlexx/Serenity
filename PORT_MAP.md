# Serenity → Mojo — PORT STATUS (live)

> The single source of "what's going on". 1:1 port of Serenity (Python) → pure Mojo in
> `/home/alex/serenity-trainer`, mirroring Serenity's `modules/` tree (456 Serenity
> Python modules; 542 Mojo files including native model/kernels/smokes).

## Rules (binding)
- **Serenity is the ONE AND ONLY reference.** Never EriDiffusion/Rust/flame-core.
- Existing Mojo/serenity/mojodiffusion code may be used only as implementation material.
  It is not evidence. A behavior is trusted only after an in-session Serenity
  reference run and a Mojo comparison on byte-identical inputs.
- **PORTING RULE**: for every behavior, find the EXACT Serenity source path and
  reference invocation first, paste/cite it, then translate API/namespace only.
  Existing Mojo/serenity/mojodiffusion code can be copied as implementation
  material, but it must be checked against Serenity before it counts. No
  deviations, no "improvements". If no Serenity source or runtime reference
  exists, ASK.
- **FILENAME TRACEABILITY RULE**: port-facing files must mirror Serenity file
  names and module locations. Every model status entry must read:
  `Serenity .py -> serenity-trainer .mojo mirror -> parity artifact/gate`.
  `serenitymojo/*` or `mojodiffusion/*` files are implementation material only
  and must never be listed as the port surface or as proof of Serenity parity.
- **Reuse serenitymojo ONLY for autograd+tensors+ops** (import). **BORROW** model code by COPY-into-port.
- **BF16 storage, F32 compute. No persistent F32.** MGDS dropped.
- Static dtype guard: `python3 scripts/check_flux_family_dtype_contract.py`.
  It reads Serenity sampler/config refs and checks that allowed F32 boundaries
  are Serenity-backed (`dtype=torch.float32` sampler latents and FLOAT_32 LoRA
  weight configs) while persistent Klein/Flux2 checkpoint weights stay
  dtype-preserving through `Tensor.from_view`.
- **Verify EVERY phase against Serenity** — run Serenity's own Python on fixed inputs, compare numbers. **Main loop owns every gate** (build + GPU run + numeric parity); agent self-reports are never the gate.
- **CPU PyTorch is structural only, never numeric parity.** CPU/dry-run checks may
  validate config/cache/source routing or helper formulas; loss, grad, optimizer,
  sampler trajectory, image, and speed gates must compare Mojo against
  Serenity CUDA-reference dumps.
- Skills: `serenity-trainer-port` + `numeric-parity-testing` carry the method.
- Traceability map: `TRACEABILITY.md` lists each required model as
  `Serenity source -> Mojo mirror -> existing gate -> missing gate`. Update it
  before touching code for that model.

Build/run: `cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && timeout 180 prlimit --as=24000000000 pixi run mojo build -I . -I /home/alex/serenity-trainer/src -Xlinker -lm <smoke> -o /tmp/x && /tmp/x`

The memory cap is part of the standard command. On 2026-06-05, broad Mojo
smoke compiles reached about 61-62 GiB RSS and were killed by the kernel inside
the VSCode snap scope. Do not run broad Mojo build gates uncapped from VSCode.

## Serenity Audit (2026-06-05): What Is Still Missing

This audit is for the first useful parity milestone: **a pure-Mojo training
runtime that can run Serenity-style jobs and match Serenity's numbers**. It
does not require cloning Tkinter UI, cloud launchers, ZLUDA installer, captioning
UI, masking UI, or model conversion tools unless they become part of that runtime
path.

Inventory:
- Serenity has **456** Python module files under `modules/`.
- The Mojo port has **542** `.mojo` files because it also carries native kernels,
  smokes, and model-unit files.
- Only one Serenity path is intentionally absent as a mirror:
  `dataLoader/mixin/DataLoaderMgdsMixin.py` (MGDS is dropped).
- **408 Mojo files are still `# TODO: port` stubs**; **383** are tiny skeleton
  files (<=200 bytes, excluding `__init__`). The directory mirror exists; the
  functional port does not.

Parity rule for every item below:
1. Generate the reference with `/home/alex/Serenity/venv/bin/python` or
   Serenity's own training run/tfevents on fixed inputs.
2. Dump the exact inputs plus reference outputs to `parity/`.
3. Run Mojo on the dumped inputs.
4. Compare with the correct gate: exact scalar/schedule equality, cosine >= 0.999
   for tensors, PSNR > 40 dB for decoded images, and loss matching Serenity's
   own range for training gates.
5. Do not mark an item VERIFIED from compile success, a smoke artifact, or an
   implementation claim.

### P0 - Product Run Layer (missing)
Serenity's product entry is `scripts/train.py -> util/create.py -> factory ->
GenericTrainer.start/train/end`. The Mojo port currently has a train-step driver
and some verified smokes, but no equivalent product entry.

Missing:
- `scripts/train.py` equivalent: CLI reads preset, secrets, concepts, sample
  prompts, builds the trainer, runs `start -> train -> end`.
- `util/create.py` / `util/factory.py`: build-only registration surfaces now
  exist for the required model breadth, but the product runtime dispatch is still
  incomplete. Need model-type + training-method dispatch through real loader/
  setup/saver/sampler/data-loader instances.
- `util/args/TrainArgs.py`, callbacks, commands, and secrets handling are TODO
  stubs. For the first runtime, implement terminal-safe equivalents rather than
  Tkinter.
- `TrainConfig` is a narrow scalar subset. It must cover the fields consumed by
  the production path: model names, workspace/cache/output paths, train/temp
  device, dtype/weight dtype, model type/training method, optimizer object,
  sample/save/backup cadence, validation flags, EMA flags, prior/masked training
  flags, and latent caching.

Gate:
- Run a Z-Image preset through the Mojo CLI, with no smoke-only hardcoding, and
  match Serenity loss/log/cadence behavior on the same cached inputs.

### P1 - Data Path (partially verified)
Current verified: `CacheReader` reads a real safetensors cache and batches it.
Current unverified: `Prepare.mojo`, `Bucketing.mojo`, model-specific data-loader
classes, concept weighting/repeats, validation concepts, prior-preservation
concepts, and cache generation.

Missing:
- Pure-Mojo replacement for Serenity's dataset semantics: concept JSON,
  repeats, crop/resize/bucket assignment, batch construction, shuffle order, and
  validation/prior concept labeling.
- Real prepare path: image -> VAE latent cache, caption -> text embedding cache,
  cache metadata, and reader round-trip. `.pt` torch-pickle caches are not a Mojo
  runtime input; conversion/dump gates are allowed only as reference tooling.
- Model-specific loaders for Flux2/Klein and later models are mostly TODO stubs.

Gate:
- `Prepare -> CacheReader -> predict` on a real concept set, compared against
  Serenity's cached tensors and first-batch order for the same seed.

### P2 - Trainer Loop Features (partial)
Current verified pieces: AdamW/LR/masked-loss smokes, train-step loop basics,
`TrainState` round-trip. Current driver intentionally bypasses
`GenericTrainer.start` and real `BaseDataLoader`.

Missing:
- `GenericTrainer.start`: clear cache, continue-last-backup, model load, setup
  optimizations/device/model, data-loader creation, saver creation, sampler
  creation, parameter collection.
- `GenericTrainer.train`: command queue, save/sample/backup decision timing,
  validation loop, prior prediction branch, masked prior preservation branch,
  EMA update/apply, tensorboard scalar behavior, grad-scaler path, multi-GPU
  reduction hooks, schedule-free optimizer mode, final `end()` save.
- Full save/resume gate at `gradient_accumulation_steps=1` and `>1` with a real
  model, not just `TrainState` standalone.

Gate:
- Run Serenity and Mojo for the same preset to a fixed step count; compare loss,
  LR, grad norm, optimizer step count, saved LoRA keys, and resume continuation
  loss after backup restore.

### P3 - Sampling During Training (partial)
Current verified: Z-Image scheduler/denoise/VAE decode parity in isolation.
Missing:
- `SampleConfig` reader and `training_samples/*.json` reader.
- `BaseModelSampler` product behavior: sample path naming, sample cadence, custom
  samples, image/video/audio format handling, no-grad model path, EMA/no-EMA
  sampling branches.
- A **model-specific sampler for every supported vertical**. Samplers are not
  interchangeable: each model owns its prompt/text encode path, scheduler config,
  latent packing/unpacking, VAE scale/decode behavior, CFG branch, and no-grad
  model path. A vertical is not parity-complete without its sampler gate.

Gate:
- During a Mojo training run, cadence-trigger a sample using the same sample JSON
  Serenity uses, then compare scheduler values, final latent, and decoded image
  against Serenity for a fixed latent/noise path.

Sampler coverage matrix:

| model | Serenity reference | Mojo status | required sampler gate |
|---|---|---|---|
| Z-Image | `/home/alex/Serenity/modules/modelSampler/ZImageSampler.py` | `ZImageSampler.mojo` verified in isolation; 1024 denoise speed gate fixed | keep scheduler/latent/decode parity wired into product cadence |
| Klein / Flux2 | `/home/alex/Serenity/modules/modelSampler/Flux2Sampler.py` | `Flux2Sampler.mojo` non-stub, parity not complete | scheduler values, packed latent flow, VAE decode, image parity |
| Qwen | `/home/alex/Serenity/modules/modelSampler/QwenSampler.py` | `QwenSampler.mojo` build-only surface; helper gate passes in `smoke/qwen_sampler_helper_gate.mojo`; no denoise/decode runtime | 64px quantization, CFG batch behavior, latent/decode parity |
| Ernie | `/home/alex/Serenity/modules/modelSampler/ErnieSampler.py` | `ErnieSampler.mojo` build-only surface; `smoke/ernie_surface_check.mojo` passes | full sampler runtime + scheduler/latent/decode parity |
| Anima | `/home/alex/Serenity-anima-ref/modules/modelSampler/AnimaSampler.py` | `AnimaSampler.mojo` build-only surface; `smoke/anima_surface_check.mojo` passes; helper gate passes against `parity/anima_sampler_helper_ref.json` | full denoise trajectory, text encode, transformer forward, VAE decode/image parity |
| SD3 / SD3.5 | `/home/alex/Serenity/modules/modelSampler/StableDiffusion3Sampler.py` | `StableDiffusion3Sampler.mojo` build-only surface; `smoke/sd3_surface_check.mojo` passes; `smoke/sd3_sampler_helper_gate.mojo` passes structural helper contract only | text encoder routing, scheduler/latent/decode parity |
| SDXL | `/home/alex/Serenity/modules/modelSampler/StableDiffusionXLSampler.py` | `StableDiffusionXLSampler.mojo` build-only surface; `smoke/sdxl_surface_check.mojo` is now a manifest after compiler OOM, while `smoke/sdxl_surface_sampler_contract_check.mojo` and `smoke/sdxl_sampler_helper_gate.mojo` pass helper/contract-only scope (`1024x1152`, latent `128x144`, CFG batch 2, inpaint channels 9), not denoise/decode/image parity | dual text encoder, refiner/inpaint flags if preset uses them, latent/decode parity |
| Flux.1 dev | `/home/alex/Serenity/modules/modelSampler/FluxSampler.py` | `FluxSampler.mojo` surface plus `smoke/flux_sampler_helper_gate.mojo` passes (`1024x1024`, latent `128x128`, packed `4096x64`, shift `3.1581929`, mu `1.15`); real Serenity LoRA file gate passes (`1512` keys / `504` adapters / rank `16` / BF16) | full no-grad denoise, T5/CLIP text encode, transformer forward, VAE decode/image parity |
| Flux2 dev | `/home/alex/Serenity/modules/modelSampler/Flux2Sampler.py` | same `Flux2Sampler.mojo` mirror as Klein plus `smoke/flux2_dev_branch_check.mojo` structural/helper gate passes (`dev_heads=48`, `seq_len=4096`, `steps=28`, `mu=2.1514432`, `sigma1=0.9957105`); not denoise/decode parity | dev checkpoint scheduler/guidance/packed latent/decode parity |
| Chroma | `/home/alex/Serenity/modules/modelSampler/ChromaSampler.py` | `ChromaSampler.mojo` helper slice plus `parity/chroma_sampler_helper_ref.json` and `smoke/chroma_sampler_helper_gate.mojo`; main-loop gate passes (`1024x1152`, latent `128x144`, packed `4608x64`, CFG batch 2, shift `3.0`, sigma1 `0.85769236`, runtime `0.00010915s`) | full no-grad denoise, text encode, transformer forward, VAE decode/image parity |

### P4 - Model Loader/Saver/Internal State (partial)
Current verified: Z-Image LoRA save key/shape parity; Flux.1 real Serenity
LoRA file key/shape/dtype metadata gate; Flux2/Klein tiny LoRA save/load
roundtrip for the block carrier; `TrainState` save/load round-trip.
Current missing/stubbed surface is broad: 78 modelSaver files and 74 modelLoader
files are TODO stubs.

Missing:
- Product model loaders for the first supported verticals: Z-Image and Flux2/Klein
  need full checkpoint/LoRA/internal-backup loading through the same run layer,
  not smoke-local assembly.
- Flux2/Klein LoRA saver and loader gates against real Serenity files, including
  extra-key preservation and full/default layer-filter coverage beyond the block
  carrier.
- Internal backup layout parity: model weights + optimizer state + EMA state +
  progress + config/concepts/samples copied under backup.
- Fine-tune savers are deferred unless fine-tune training becomes a target.

Gate:
- For each supported model/training method, compare saved key set, shapes, dtypes,
  progress counters, optimizer moments, and resume behavior against Serenity.

### P5 - Model Verticals
Current:
- Z-Image LoRA is the only mostly-verified vertical.
- Klein/Flux2 is the priority incomplete vertical: load OOM was traced to
  persistent F32 loader storage. The BF16-storage loader now passes
  `smoke/klein_load_only.mojo` to `ALL DONE: 8 double + 24 single`; inference
  forward parity passes with cosine 0.9994858031641485; the real-weight
  `Flux2LoRASpec.predict -> backward_lora` smoke runs finite; a one-step
  Serenity cached-data reference dump now exists; tiny Flux2 LoRA key/load and
  sampler-helper gates pass. Flux2/Klein factory/create dispatch plus loader
  and saver Serenity-name source contracts now build-run as main-loop gates.
  The Mojo cached-data train/backward/AdamW replay gate does not yet exist.
- Required breadth is now in scope after the product run layer can host it:
  **Qwen, Ernie, Anima, SD3, SDXL, Flux.1 dev, Flux2 dev, and Chroma**.
  These are not deferred.
- Qwen, Ernie, Anima, SD3, SDXL, and Flux.1 dev now have **build-only**
  model/setup/data-loader/loader/saver/sampler/factory surfaces with local smoke
  gates. Flux.1 dev also has a real Serenity LoRA file key/shape/dtype gate.
  This is not train or sampler behavioral parity.
- Ernie, Qwen, Anima, SDXL, Flux.1 dev, and Chroma have fresh 100-step
  Serenity baselines with loss, grad norm, speed, and memory metrics.
  SD3/SD3.5 has no numeric baseline yet: CUDA is visible outside the sandbox,
  but the current Serenity config uses `ModelType.STABLE_DIFFUSION_3` while
  the audited data-loader registration is `ModelType.STABLE_DIFFUSION_35`, the
  required image/text cache is absent, and the available local single-file
  checkpoints contain diffusion+VAE weights but no inspectable CLIP/T5 text
  encoders. The SD3 dry-run blocker artifact is structural only, never CPU
  numeric parity.
- Flux2 dev has a structural branch/scaffold gate only:
  `smoke/flux2_dev_branch_check.mojo` passes (`dev_heads=48`, `seq_len=4096`,
  `steps=28`, `mu=2.1514432`, `sigma1=0.9957105`). The train scaffold
  `scripts/flux2_dev_dump_train_ref.py` and
  `parity/flux2_dev_train_ref_{contract,blockers}.json` record that the local
  FLUX.2-dev cache is only a partial transformer config (`num_attention_heads=48`,
  `num_layers=8`, `num_single_layers=48`, `joint_attention_dim=15360`) and that
  no Flux2 dev Serenity config, baseline, CUDA reference tensors, full
  checkpoint, VAE, tokenizer, text encoder, or scheduler are present. This is
  not numeric train/sampler parity.
- Chroma now has a helper-only sampler fixture generated from Serenity source
  paths and the local Chroma1-HD scheduler/VAE configs:
  `parity/gen_chroma_sampler_helper_ref.py`,
  `parity/chroma_sampler_helper_ref.json`, and
  `smoke/chroma_sampler_helper_gate.mojo`. The Mojo gate is main-loop verified
  as a helper-only sampler contract; it is not end-to-end sampler parity.
  Chroma now also has build-only Serenity mirrors for
  `model/ChromaModel.py`, `modelSetup/BaseChromaSetup.py`, and
  `modelSetup/ChromaFineTuneSetup.py`, and `modelSetup/ChromaLoRASetup.py`;
  `smoke/chroma_model_setup_contract_check.mojo` passes with predict outputs
  `4`, dtype caveats `6`, FINE_TUNE params `3`, LoRA params `3`,
  `create te = True`, ragged text seq `32`, and packed latents `4608x64`.
  Chroma loader/saver/LoRA-conversion Serenity-name mirrors also pass as
  split build-only gates:
  `smoke/chroma_surface_loader_contract_check.mojo`,
  `smoke/chroma_surface_saver_contract_check.mojo`, and
  `smoke/chroma_lora_conversion_contract_check.mojo`. They cover wrapper spec
  filenames, leaf route ordering, BF16 transformer override default,
  embedding keys, LoRA bundle keys, and conversion namespaces/counts, but not
  runtime load/save parity.
  Chroma also has a real
  Serenity 100-step baseline, a BF16 real-file LoRA gate, a one-step train
  dump, and a Mojo loss-only replay. Verified
  baseline: mean step `3.2088356397171727s`, last loss
  `0.3658050298690796`, grad norm `0.001406678231433034`, peak sampled VRAM
  `9913 MiB`. Real LoRA gate: `912` keys / `304` adapters / rank `16` /
  BF16. Train dump: loss `0.2957186698913574`, grad norm
  `0.00041076153866015375`, lr `0.0003 -> 0.0003`. Mojo loss replay:
  `0.29572487` vs dump `0.29571867`, abs err `6.198883e-06`.
  These are still surfaces/contracts, so Chroma transformer/backward/AdamW train
  parity, full-finetune runtime save/load/resume parity, and end-to-end sampler
  denoise/decode parity remain missing.

Missing for Klein:
- Full Mojo replay of `parity/klein_train_ref_step000*.safetensors`:
  `Flux2LoRASpec.predict -> backward_lora -> AdamW` over 144 adapters on
  byte-identical Serenity cached tensors/noise/timestep.
- Numeric comparison to the dump: step-0 grad norm 0.005975008010864258, lr
  0.0 -> 3e-6, adapter before/after deltas, and phase speed. Loss-only replay
  now passes separately.
- Longer loss comparison to Serenity's own baseline:
  `klein9b_alina_baseline`, loss mean ~0.632 over 102 steps, batch 2, lr 3e-5,
  bf16.
- Full sampler trajectory, VAE decode/image parity, real Serenity LoRA-file key
  comparison, extra-key preservation, and full/default layer-filter coverage.

Verified Klein sub-gates:
- `smoke/klein_load_only.mojo`: `ALL DONE: 8 double + 24 single`.
- `smoke/klein_forward_parity.mojo`: n=32768, cosine=0.9994858031641485,
  max_abs_diff=0.08984375, nonfinite=0.
- `smoke/klein_real_forward.mojo`: forward n=2048, mean=-0.06562518,
  var=0.26505914, nonfinite=0; backward 288 grad groups, 43,515,904 elements,
  abs_sum=700.71155, nonfinite=0. Warm-run speed at HL=4/WL=4/NTXT=8:
  predict ~43.6-43.9 ms, backward ~185.7-203.7 ms, step_no_optim
  ~229.6-247.5 ms, setup/load ~4.40-4.46 s.
- `scripts/klein_dump_train_ref.py`: real Serenity cached-data step dump with
  loss=0.12243738770484924, grad_norm=0.005975008010864258, lr 0.0 -> 3e-6,
  elapsed=310.75666852100403s, 22 step tensors, and 1728 adapter tensors.
- `scripts/klein_adapter_delta_contract.py --check`:
  Serenity adapter-dump oracle passes. It checks six adapter phases over 288
  FP32 tensors / 43,515,904 elems, pre/post clip grad L2
  0.0059750078751807986, abs_sum 18.8721539509502, max_abs
  0.00014209747314453125, nonfinite 0, and optimizer state advancing to 288
  parameter entries. It also records that `adapter_after - adapter_before` is
  zero because the captured step has lr_before=0.0; this is not Mojo train
  parity and not an AdamW update-delta gate.
- `/tmp/klein_train_ref_2step_meta.json` plus
  `/tmp/klein_train_ref_2step_step001_adapters.safetensors`: two-step
  Serenity CUDA dump completed. `scripts/klein_adapter_delta_contract.py
  --meta /tmp/klein_train_ref_2step_meta.json --step-index 1 --expect-update`
  passes with loss=0.5876612663269043, grad_norm=0.008699021302163601,
  lr 2.9999999999999997e-06 -> 5.999999999999999e-06, grad L2
  0.008699021115427636, update nonzero elems=43,515,904, update abs_sum
  51.57613097811465, update L2=0.010862200286706213, max update
  3.0034029805392493e-06. This is a Serenity update oracle, not Mojo AdamW
  parity.
- `smoke/klein_train_ref_loss_replay.mojo`: loss-only replay over the real
  Serenity dump passes. Mojo loss=0.12241738 vs dump=0.12243739,
  abs_err=2.0004809e-05, loss kernel=0.107232224s, total=0.268313362s. The
  gate uses a Klein-specific 2.5e-5 F32 reduction-order tolerance and does not
  claim backward/AdamW parity.
- `smoke/klein_train_parity_smoke.mojo`: smoke-only train path passes with
  loss=1.5305071, grad_norm=0.16702795893468853, LoRA-B 144/144, nonfinite=0;
  latest speed load=149.778842062s, predict=0.234450175s,
  backward=0.667401176s, AdamW=4.480295034s, step=5.506434476s.
- `smoke/klein_lora_key_parity.mojo`: tiny safetensors save/load gate passes for
  Serenity raw Flux2 keys and 12 double + 2 single adapter slots.
- `smoke/klein_sampler_helper_gate.mojo`: helper gate passes for four scheduler
  steps and CFG combine (`cfg[1]=5.0`).
- `smoke/flux2_finetune_setup_contract_check.mojo`: setup-only full-finetune
  contract passes for Serenity `Flux2FineTuneSetup.py`: one `transformer`
  parameter group, frozen `text_encoder`/`vae`, cached text/VAE on temp device,
  uncached text/VAE on train device, ModuleFilter use, and four dtype caveats.
- `smoke/flux2_factory_contract_check.mojo`: factory/create dispatch source
  contract passes for Serenity `ModelType.FLUX_2`, LoRA and fine-tune loader
  specs, LoRA and fine-tune savers, and sampler fallback by model type.
- `smoke/flux2_surface_loader_contract_check.mojo`: loader surface contract
  passes for internal -> diffusers -> safetensors order, Dev/Klein
  `num_attention_heads == 48` branch, BF16 transformer override defaults,
  subfolder/class names, no embedding loader, and unsupported single-file error.
- `smoke/flux2_surface_saver_contract_check.mojo`: saver surface contract
  passes for fine-tune full-model route plans and raw Flux2 LoRA saver state
  sources/keys/internal destination. This is not runtime save/load parity.
- `smoke/flux2_runtime_loader_key_contract_check.mojo`: non-GPU import/key
  helper gate passes after splitting runtime tensor helpers out of
  `Flux2ModelLoader.mojo`.

Gate:
- Same as model vertical protocol: predict functions exact, forward cosine >=
  0.999, real-data loss match, train loss in Serenity range with measured
  seconds/step and all adapter imprint, sampler latent/image parity, PEFT keys
  equal a real OT file.

Traceability format for every required-breadth model:
- **Serenity source**: the exact `/home/alex/Serenity/modules/.../*.py` path
  (or `/home/alex/Serenity-anima-ref/modules/.../*.py` for Anima).
- **Mojo mirror**: the matching `/home/alex/serenity-trainer/src/serenity_trainer/.../*.mojo`
  file with the same Serenity basename and module role.
- **Borrowed core**: optional implementation-only code copied into the Mojo
  mirror after checking it against Serenity. A `serenitymojo/*` path here is
  not a port file and not a parity result.
- **Gate**: the exact `parity/*` reference plus `smoke/*` Mojo comparison and
  measured metric.

Required breadth targets:
- **Qwen** (`ModelType.QWEN`): use `/home/alex/Serenity` refs
  `QwenModel.py`, `QwenBaseDataLoader.py`, `QwenLoRASetup.py`,
  `QwenFineTuneSetup.py`, `QwenSampler.py`, `Qwen*ModelLoader.py`, and
  `Qwen*ModelSaver.py`. The port-facing mirrors are
  `src/serenity_trainer/model/QwenModel.mojo`,
  `src/serenity_trainer/dataLoader/QwenBaseDataLoader.mojo`,
  `src/serenity_trainer/modelSetup/BaseQwenSetup.mojo`,
  `src/serenity_trainer/modelSetup/QwenLoRASetup.mojo`,
  `src/serenity_trainer/modelSetup/QwenFineTuneSetup.mojo`,
  `src/serenity_trainer/modelSampler/QwenSampler.mojo`,
  `src/serenity_trainer/modelLoader/QwenFineTuneModelLoader.mojo`,
  `src/serenity_trainer/modelLoader/QwenLoRAModelLoader.mojo`,
  `src/serenity_trainer/modelLoader/qwen/QwenModelLoader.mojo`,
  `src/serenity_trainer/modelLoader/qwen/QwenLoRALoader.mojo`,
  `src/serenity_trainer/modelSaver/QwenFineTuneModelSaver.mojo`,
  `src/serenity_trainer/modelSaver/QwenLoRAModelSaver.mojo`,
  `src/serenity_trainer/modelSaver/qwen/QwenModelSaver.mojo`, and
  `src/serenity_trainer/modelSaver/qwen/QwenLoRASaver.mojo`.
  Any `serenitymojo/models/qwenimage/*` code is implementation material only;
  it must be copied into the Serenity-named mirrors before it counts as this
  port. Build gates passing: `smoke/qwen_model_compile_check.mojo`,
  `smoke/qwen_text_encoder_compile_check.mojo` (currently only a build surface),
  and `smoke/qwen_surface_check.mojo`. Serenity baseline:
  `/home/alex/Serenity/output/qwen_100step_baseline/metrics.json`
  (100 steps, mean step 3.9018539588785535s, last loss 0.08166131377220154,
  last grad norm 0.0011983560398221016). One-step Serenity dump exists:
  `parity/qwen_train_ref_step000.safetensors`, loss=0.08948630839586258,
  grad_norm=0.00014067116717342287, lr 0.0 -> 1.5e-6. The dump harness elapsed
  612.1598391110019s, but profile-only split that first-step overhead into
  load/setup=295.3092309619824s, cache epoch start=153.70464057300705s,
  predict=12.852268689981429s, backward=2.742950047017075s,
  optimizer=0.4419737099960912s; Serenity steady 100-step speed remains the
  apples-to-apples baseline. Mojo currently has a loss-only replay gate:
  `smoke/qwen_train_ref_loss_replay.mojo` computes 0.0894845 vs dump 0.08948631
  (abs err 1.8104911e-06, loss kernel 0.133140851s). Real Serenity Qwen LoRA
  file gate now passes: `smoke/qwen_real_lora_file_parity.mojo` loads
  `/home/alex/Serenity/output/qwen_100step_baseline/lora.safetensors` and
  verifies 2160 keys / 720 adapters / rank 16 / BF16 / alpha 1.0. Missing:
  Qwen Image transformer forward/backward, Qwen2.5-VL text encode, VAE
  encode/decode, LoRA grad/AdamW replay, and sampler denoise/decode parity.
- **Ernie** (`ModelType.ERNIE`): use `/home/alex/Serenity` refs
  `ErnieModel.py`, `ErnieBaseDataLoader.py`, `BaseErnieSetup.py`,
  `ErnieLoRASetup.py`, `ErnieFineTuneSetup.py`, `ErnieSampler.py`,
  `ErnieModelLoader.py`, and `Ernie*ModelSaver.py`. Build gates passing:
  `smoke/ernie_model_compile_check.mojo` and `smoke/ernie_surface_check.mojo`.
  Real Serenity Ernie LoRA file gate now passes:
  `smoke/ernie_real_lora_file_parity.mojo` loads
  `/home/alex/Serenity/output/ernie_eri2_100step_baseline/lora.safetensors`
  and verifies 756 keys / 252 adapters / rank 16 / BF16 / alpha 1.0.
  A real CUDA one-step train reference now exists:
  `parity/ernie_train_ref_meta.json`,
  `parity/ernie_train_ref_step000.safetensors`, and
  `parity/ernie_train_ref_step000_adapters.safetensors`, loss
  0.643847644329071, grad norm 0.000828770047519356, lr 0.0003 -> 0.0003,
  elapsed 97.59743943800095s, 504 trainable tensors / 47,185,920 params.
  `smoke/ernie_train_ref_loss_replay.mojo` passes loss-only replay:
  0.6438152 vs dump 0.64384764, abs err 3.2424927e-05, loss kernel
  0.238489161s, total 0.717235302s. This uses an explicit Ernie-specific F32
  reduction-order tolerance; PyTorch recomputes the dump loss exactly from the
  same tensors.
  Sampler helper parity now passes in `smoke/ernie_sampler_helper_gate.mojo`
  after extending `modelSampler/ErnieSampler.mojo`: plan 1024x1024, latent
  128x128, patch contract 64x32x128 for the 1024x512 helper contract, CFG batch
  2, sigma1 0.75, timestep1 750.0, Euler 0.125. This is helper-only and not
  denoise/decode/image parity. Missing: transformer/backward/AdamW parity,
  full-finetune parity, and end-to-end sampler parity.
  Serenity baseline:
  `/home/alex/Serenity/output/ernie_eri2_100step_baseline/metrics_nocompile.json`
  (100 steps, mean step 3.210763874979797s, last loss 0.7464916110038757,
  last grad norm 0.0014650028897449374).
- **Anima** (`ModelType.ANIMA`): use `/home/alex/Serenity-anima-ref` refs
  `AnimaModel.py`, `AnimaBaseDataLoader.py`, `BaseAnimaSetup.py`,
  `AnimaLoRASetup.py`, `AnimaFineTuneSetup.py`, `AnimaSampler.py`,
  `AnimaModelLoader.py`, `Anima*ModelSaver.py`, and presets
  `training_presets/#anima LoRA.json` / `#anima Finetune.json`. Build gates
  passing: `smoke/anima_model_compile_check.mojo`,
  `smoke/anima_setup_surface_check.mojo`, and `smoke/anima_surface_check.mojo`.
  Shared `ModelType.mojo` now inserts `ANIMA` before `Z_IMAGE`, matching the
  Anima reference and preventing Anima/Z-Image dispatch collision. Serenity
  Anima LoRA file gate now passes: `smoke/anima_real_lora_file_parity.mojo`
  loads
  `/home/alex/Serenity-anima-ref/output/anima_100step_baseline/lora.safetensors`
  and verifies 840 keys / 280 adapters / rank 16 / BF16 / alpha 1.0.
  Sampler helper gate now passes: `smoke/anima_sampler_helper_gate.mojo`
  compares against `parity/anima_sampler_helper_ref.json` (`1024x1152`,
  latent `128x144`, CFG batch 2, sigma1 0.9, timestep1 900.0). Train-reference
  dry-run is structural-only and unblocked:
  `scripts/anima_dump_train_ref.py --dry-run` writes
  `parity/anima_train_ref_blockers.json` with `blocked=false`; it is not a CPU
  numeric parity gate.
  Serenity one-step train reference now exists:
  `parity/anima_train_ref_step000.safetensors` and
  `parity/anima_train_ref_step000_adapters.safetensors`, loss
  0.0667838305234909, grad norm 0.0014594600070267916, lr 0.0 -> 1.5e-7,
  elapsed 44.37211945699528s, 28 step tensors, 2240 adapter tensors.
  Mojo loss-only replay passes: `smoke/anima_train_ref_loss_replay.mojo`
  computes 0.06678182 vs dump 0.06678383 (abs err 2.0116568e-06).
  Serenity
  baseline:
  `/home/alex/Serenity-anima-ref/output/anima_100step_baseline/metrics_with_grad.json`
  (100 steps, mean step 1.0924571535151424s, last loss 0.11683385074138641,
  last grad norm 0.0030804856214672327).
- **SD3** (`ModelType.STABLE_DIFFUSION_3` and `STABLE_DIFFUSION_35`): use
  `/home/alex/Serenity` refs `StableDiffusion3Model.py`,
  `StableDiffusion3BaseDataLoader.py`, `BaseStableDiffusion3Setup.py`,
  `StableDiffusion3*Setup.py`, `StableDiffusion3Sampler.py`,
  `StableDiffusion3*ModelLoader.py`, and `StableDiffusion3*ModelSaver.py`.
  Build gates passing: `smoke/sd3_model_compile_check.mojo`,
  `smoke/sd3_setup_surface_check.mojo`, and `smoke/sd3_surface_check.mojo`;
  `smoke/sd3_model_compile_check.mojo` was rerun with CUDA visible outside the
  sandbox and passed. Helper/key gates: `smoke/sd3_lora_key_parity.mojo`
  previously passes bounded raw targets=9/entries=27;
  `smoke/sd3_lora_inventory_contract.mojo` now also runs with CUDA visible and
  passes (`linear_targets=96 entries=288 rank=2 dtype=BF16 alpha=4.0`);
  `smoke/sd3_sampler_helper_gate.mojo` passes against generated
  `parity/sd3_sampler_helper_ref.json` (plan 1024x1056, latent 128x132,
  cfg_batch 2, sigma1 0.85769236, sigma2 0.60215056, timestep1 857.6924,
  latest helper timing 7.0001e-05s). These are helper gates, not SD3 train/sampler numeric
  parity.
  Serenity baseline blockers are reproducible with
  `/home/alex/Serenity/venv/bin/python scripts/sd3_dump_train_ref.py --dry-run --train-device cuda --temp-device cpu`
  and recorded in `parity/sd3_train_ref_blockers.json`: CPU/dry-run is
  structural only, CUDA is visible (`cuda_available=true`, `cuda_device_count=1`),
  no Serenity data loader is registered for `STABLE_DIFFUSION_3`, the
  image/text cache is empty/missing, and the local single-file checkpoint has no
  inspectable CLIP/T5 text encoder keys. A non-dry-run CUDA probe stops before
  model load on those blockers and produces no numeric reference.
  `parity/sd3_train_ref_contract.json` also records that
  LoRA and full-finetune train references are in scope where Serenity registers
  those paths.
- **SDXL** (`ModelType.STABLE_DIFFUSION_XL_10_BASE` plus inpainting only if the
  target preset uses it): use `/home/alex/Serenity` refs
  `StableDiffusionXLModel.py`, `StableDiffusionXLBaseDataLoader.py`,
  `BaseStableDiffusionXLSetup.py`, `StableDiffusionXL*Setup.py`,
  `StableDiffusionXLSampler.py`, `StableDiffusionXL*ModelLoader.py`, and
  `StableDiffusionXL*ModelSaver.py`. Build gates passing:
  `smoke/sdxl_model_compile_check.mojo`,
  `smoke/sdxl_model_tensor_contract_check.mojo` (tiny CUDA-only BF16 tensor
  boundary gate; `DeviceContext` is not sandbox-visible), manifest
  `smoke/sdxl_setup_surface_check.mojo`, split setup gates
  `smoke/sdxl_setup_base_contract_check.mojo`,
  `smoke/sdxl_setup_method_contract_check.mojo`,
  `smoke/sdxl_setup_dataloader_contract_check.mojo`, and
  manifest `smoke/sdxl_surface_check.mojo`, split surface gates
  `smoke/sdxl_surface_loader_contract_check.mojo`,
  `smoke/sdxl_surface_sampler_contract_check.mojo`,
  `smoke/sdxl_surface_saver_contract_check.mojo`, and
  `smoke/sdxl_surface_factory_contract_check.mojo`. The old monolithic setup
  gate was disabled after a verified compiler OOM (`mojo` pid 58957, about
  62 GiB RSS, killed by the kernel inside the VSCode snap scope), and the old
  monolithic surface gate also hit compiler OOM under the 24 GB cap; use
  `prlimit --as=24000000000` for high-risk Mojo compile gates. Sampler helper
  contract now passes in the
  main loop with explicit non-parity scope:
  `SDXL SAMPLER HELPER CONTRACT GATE OK`, `scope = helper/contract only; no
  denoise, decode, image, or end-to-end sampler parity`, plan `1024x1152`,
  latent `128x144`, CFG batch `2`, inpaint channels `9`, timestep contract
  `3 - 4`, scheduler `EulerAncestralDiscreteScheduler`, latest helper timing
  `0.000143672s`. Serenity baseline:
  `/home/alex/Serenity/output/sdxl_100step_baseline/metrics.json`
  (100 steps, mean step 0.9730077266059298s, last loss 0.018004747107625008,
  last grad norm 0.0029590092599391937). CPU-only adapter-delta contract now
  passes in `scripts/sdxl_adapter_delta_contract.py`: 6352 keys, four phases of
  1588 tensors / 49,412,736 elems, reference `FLOAT_32` training adapter dtype,
  unchanged before/pre/post phases, after-step deltas in all 1588 tensors,
  49,412,638 nonzero delta elems, abs sum 1993.9059369890892, max
  9.999924077419564e-05. This is not gradient or AdamW numeric parity because
  the default SDXL step dump does not include per-tensor gradients.
- **Flux.1 dev** (`ModelType.FLUX_DEV_1`): use `/home/alex/Serenity` refs
  `FluxModel.py`, `FluxBaseDataLoader.py`, `BaseFluxSetup.py`,
  `FluxLoRASetup.py`, `FluxFineTuneSetup.py`, `FluxSampler.py`,
  `Flux*ModelLoader.py`, and `Flux*ModelSaver.py`. Build gates passing:
  `smoke/flux_model_compile_check.mojo`, `smoke/flux_setup_surface_check.mojo`,
  `smoke/flux_surface_check.mojo`, and `smoke/flux_sampler_helper_gate.mojo`.
  Shared Flux.1 factory/create dispatch is verified for dev and fill. Sampler
  helper gate passes: plan 1024x1024, latent 128x128, packed 4096x64, shift
  3.1581929, mu 1.15, model timestep 0.5, fill mask channels 256. The real
  Serenity LoRA file gate passes against
  `/home/alex/Serenity/output/flux1_100step_baseline/lora_last.safetensors`:
  1512 keys / 504 adapters / rank 16 / BF16 / scalar alpha / bundle_keys 0.
  This is helper/file metadata only and not denoise/decode/image or train
  parity. Serenity baseline:
  `/home/alex/Serenity/output/flux1_100step_baseline/metrics_full_snapshot.json`
  (100 steps, mean step 2.8278740769089907s, last loss 0.5231022238731384,
  last grad norm 0.049582332372665405).

Gate for each required model:
1. Register model type/training method in the Mojo factory path.
2. Load the Serenity preset and concepts without smoke hardcoding.
3. Match model loader dtype/device/checkpoint behavior.
4. Match text/latent cache generation or consume byte-identical dumped
   Serenity cache tensors.
5. Match `Base*Setup.predict`, loss, and one optimizer update on fixed inputs.
6. Run a short real-data train gate with loss in Serenity's own range and LoRA
   imprint on every expected adapter.
7. Match the model's own sampler scheduler/latent/decode behavior against
   Serenity on the same sample prompt/noise/seed; sampler parity is required
   for every model, not shared by family.
8. Match saved PEFT keys against a real Serenity file.

### P6 - Optimizer/Scheduler Breadth
Current verified: AdamW, cosine LR, masked loss for early gates. Other optimizer
files exist but are not product-dispatched or fully parity-gated.

Missing:
- `util/create.py::create_optimizer` equivalent for the supported first runtime.
- Optimizer enum/config reader coverage for at least the presets we intend to run.
- Gates for Adam, Adafactor, CAME, schedule-free, 8-bit/paged, Muon, and fused
  back pass only when those become product targets.

Gate:
- Per-optimizer fixed-input update parity against Serenity's optimizer object
  and then a short real training run using that optimizer.

### P7 - Explicitly Deferred From First Parity Runtime
These are mirrored but should not block the first training-runtime parity unless
the product scope changes: Tkinter UI, cloud/remote trainers, ZLUDA installer,
caption/mask generation tools, model converters, and quantized loaders not used
by the target presets. Qwen, Ernie, Anima, SD3, SDXL, and Flux.1 dev are no
longer deferred; they are required breadth targets after the runtime gates.

## THE 7 PHASES

| # | Phase | State |
|---|---|---|
| **0** | 1:1 skeleton + core migrated to Serenity paths | ✅ DONE — compiles + runs |
| **1** | Spine: optimizer/loss/lr/enum, module/{LoRA,EMA,DoRA,…}, trainer/{Base,Generic} | ✅ DONE — **AdamW bit-exact, LR cosine exact, masked_loss exact** vs Serenity |
| **1b** | Preset JSON → `TrainConfig` reader (`util/config/TrainConfigReader.mojo`) | ✅ DONE — ported from mojodiffusion verbatim + Serenity key map. Verified: alina preset → lr 3e-4, batch 2, epochs 100, seed 42, LOGIT_NORMAL. PASS |
| **2** | **Z-Image vertical (MODEL #1)** | 🔄 **~complete** — see sub-table below |
| **3** | dataLoader (read pre-encoded latents/captions; NO MGDS) | 🔄 **CacheReader VERIFIED** (reads real cache → latent [16,72,56]/cap [224,2560], batches, reproducible shuffle. PASS). Prepare + bucketing written, own smokes pending. |
| **4** | **Klein (Flux2) vertical (MODEL #2)** | 🔄 load/forward/backward/ref-dump viable: `smoke/klein_load_only.mojo` reaches `ALL DONE: 8 double + 24 single`; forward parity cos 0.9994858031641485; smoke train path finite; Serenity step dump exists; tiny LoRA key/load and sampler-helper gates pass. Mojo cached-data loss/AdamW replay is still missing. |
| **5** | Sampler cadence / backup / save-resume | 🔄 **TrainState save/load VERIFIED** (round-trip: global_step/opt_step/n_slots exact, PASS; incl. accum>1 opt-step fix). SampleCadence + SaveBackupCadence written. Full save→resume byte-exact smoke pending. |
| **6** | Required breadth: Qwen, Ernie, Anima, SD3, SDXL, Flux.1 dev, Flux2 dev, Chroma + remaining target optimizers + full-finetune paths | 🔄 IN PROGRESS — Qwen, Ernie, Anima, SD3, SDXL, Flux.1 dev, and Flux2/Klein have partial helper/file/train-reference gates. Flux2 dev now has a branch/scaffold structural gate only; Chroma has model/setup/LoRA-setup/loader/saver/LoRA-conversion/helper/file/train-reference gates. All models still need full train/sampler parity, and every model with a Serenity `FINE_TUNE` registration needs full-weight train/save/load/resume parity. |

Required breadth comes after the Serenity-style run layer, data path, cadence,
and save/resume gates are usable, because those gates are the harness every
model must pass.

## CONFIG / RUN LAYER (was MISSING from the original plan — now tracked)
The original plan covered models + dataLoader + cadence but NOT the layer that feeds them from Serenity's preset+concept JSON, nor the operator entry. This is that layer:
| piece | state |
|---|---|
| **preset reader** `util/config/TrainConfigReader.mojo` (preset JSON → TrainConfig) | ✅ verified (alina preset exact) |
| **concept reader** `util/config/ConceptConfig.mojo` (concept JSON → dataset def: path/captions/repeats) | ✅ verified (alina concept: name/path/enabled match) |
| **sample-prompt reader** (sample_prompts JSON → prompt list for sample cadence) | ⬜ |
| **terminal UI / CLI entry** — load preset+concepts → run GenericTrainer → live terminal progress (step/loss/lr + sample/save/backup cadence events). NOT Tkinter (Mojo has no GUI toolkit); the `ui/` stubs are Tkinter and stay deferred. | ⬜ build after Phase 3/5 (it orchestrates them) |

## Phase 2 — Z-Image (sub-status, all verified by me)
| item | result |
|---|---|
| forward wrapper `model/ZImageDiT.mojo` on real weights | ✅ finite velocity; fixed SwiGLU FF=10240 |
| forward parity (LoRA B=0) vs Serenity-loaded `ZImageTransformer2DModel` | ✅ **cos 0.99939** |
| predict fns (sigma/calc_shift/scale_latents/det timestep) each vs Serenity | ✅ exact (scale_latents 0.318359375 bf16 bit-exact) |
| real-DATA loss vs Serenity | ✅ **Mojo 0.4690 = OT 0.4692**, velocity cos 0.99996 |
| real-data train (faithful predict, 16 steps) | ✅ **mean 0.5047 ≈ OT 0.47**, LoRA-B 210/210, nonfinite 0 |
| scheduler + denoise (`FlowMatchEulerDiscreteScheduler`, `ZImageSampler`) | ✅ sigmas exact; small cached gate latent **cos 0.9999247**; 1024 latent **cos 0.9992005027602735** |
| VAE decode parity | ✅ **PSNR 54.8** (decode OT's `latent_final`). decode folds unscale internally → pass RAW latent |
| 1024 VAE decode SPEED | ✅ **149.9s→1.07s** (im2col+gemm conv, serenitymojo `ops/conv.mojo`, commit bd5633f), PSNR preserved |
| full 1024 sample artifact | ✅ denoise saves `zi_MOJO_1024_latent.safetensors`; fresh-process VAE decode writes `zi_MOJO_1024.png` |
| same-process transformer+VAE lifetime | 🔄 after full denoise, loading VAE in the same singleton CUDA context still OOMs; split-process decode is the verified current path |
| **transformer inference speed** | ✅ **3.876141s/step @1024**, denoise total **116.28423s**; fixed by cached prompt/RoPE state plus matmul-backed SDPA for the 4320-token sequence |
| **2e PEFT LoRA save** | ✅ **630 keys == Serenity exactly**, all shapes match (`transformer.…lora_down/lora_up/.alpha`, alpha 0-dim). PASS. Fixed lora_A/B→lora_down/up + `transformer.` prefix + alpha 0-dim. |

## Phase 4 — Klein (Flux2) current
Klein compiles and the 144-adapter block carrier is 1:1 with Serenity's block
LoRA scope (8×12 double + 24×2 single, separate q/k/v). The loader reaches
`ALL DONE: 8 double + 24 single`, forward parity passes, the smoke train path is
finite, a real Serenity one-step cached-data dump exists, loss-only replay
passes, and Flux2 full-finetune setup is now covered by a build-only
transformer-only contract. Flux2 factory, loader, saver, and runtime-key helper
source contracts also pass. The Serenity adapter-dump oracle now has a checked
grad contract and the `/tmp` two-step dump has a checked nonzero update oracle.
**Still not full numeric train parity**: Mojo must replay
`parity/klein_train_ref_step000*.safetensors` through backward/AdamW and match
grad norm and update deltas. Full sampler trajectory and real-file PEFT parity
are also still open.

## Gate inputs (Serenity's own)
- checkpoint `/home/alex/.serenity/models/zimage_base` (transformer/ + vae/)
- cache `/home/alex/Serenity/workspace-cache/alina_zimage_OTpreset_100_baseline`
- preset `configs/alina_zimage_OTpreset_100_baseline.json` (lr 3e-4, baseline loss ≈0.5)
- real OT LoRA (key ref) `workspace/alina_zimage_OTpreset_2000/save/*.safetensors`
- reference gen: `/home/alex/Serenity/venv/bin/python`

## Parity refs (`serenity-trainer/parity/`)
adamw_ref, lr_ref, masked_loss_ref, predict_ref, predict_fn_ref, zi_fwd,
zi_realdata, zi_realclean, zi_sampler_ref, zi_gen1024_ref, zi_OT_1024.png,
zi_MOJO_decode.png, zi_MOJO_1024_latent.safetensors, zi_MOJO_1024.png,
klein_fwd.safetensors, klein_train_ref_meta.json,
klein_train_ref_step000.safetensors, klein_train_ref_step000_adapters.safetensors
(+ gen_*.py / dump scripts).

## NEXT (in order)
1. **Product run layer audit follow-through** — implement the first real Mojo CLI
   entry (`scripts/train` equivalent) plus `create/factory` dispatch for the
   supported target set (Z-Image LoRA and Flux2/Klein LoRA first). Gate by running
   a preset through the CLI, not by a smoke-local hardcoded assembly.
2. **Config completeness for target presets** — extend `TrainConfig` and
   `TrainConfigReader` to cover every field consumed by Serenity's
   `scripts/train.py -> GenericTrainer.start/train/end` path for the target
   presets: paths, model names, model type, training method, dtype, optimizer,
   cache/workspace/output, sample/save/backup, validation, EMA/prior/masked flags.
3. **Klein train gate** — extend the Mojo replay of
   `parity/klein_train_ref_step000*.safetensors` from loss-only into
   `Flux2LoRASpec.predict -> backward_lora -> AdamW`; compare step-0 grad norm,
   adapter deltas, and speed to the Serenity dump, then compare longer loss
   range to `klein9b_alina_baseline` (~0.632 mean).
4. **Data path gates** — verify `Prepare.mojo` and `Bucketing.mojo` against
   Serenity on real concept data; then run `Prepare -> CacheReader -> predict`
   with first-batch order and tensors matching Serenity for the same seed.
5. **Save/resume/cadence product gates** — run full save/resume at accum=1 and
   accum>1, compare continuation loss/optimizer step count, then add sample prompt
   reader and in-training sample cadence with Serenity sampler parity.
6. **Flux2/Klein saver/loader parity** — tiny block-carrier key/load parity
   passes. Next compare Flux2 LoRA saved keys/shapes/dtype against a real
   Serenity LoRA file, preserve preloaded extra keys, cover full/default layer
   filters, and load through the product run layer.
7. **Required model breadth** — add Qwen, Ernie, Anima, SD3, SDXL, Flux.1
   dev, Flux2 dev, and Chroma through the same product path. Anima's Serenity
   reference is `/home/alex/Serenity-anima-ref`; the others use
   `/home/alex/Serenity`. Klein is the Flux2 class/family, and Flux2 dev is
   the other `ModelType.FLUX_2` runtime variant (`Flux2Model.is_dev()` checks
   `transformer.config.num_attention_heads == 48`). Chroma is
   `ModelType.CHROMA_1`. Each model needs loader/setup/data-loader/model-specific
   sampler/saver parity gates before it can be marked present.
   Full finetuning is not optional where Serenity registers it: Qwen, Ernie,
   SD3/3.5, SDXL, Flux.1 dev, Flux2, Chroma, and Z-Image all have
   `TrainingMethod.FINE_TUNE` setup/saver/loader support in Serenity and need
   full-weight train/save/load/resume gates.
8. **Only after OT parity** — reshape the architecture into the new trainer.
   Refactors are allowed only after the Serenity gates above are reproducible.
9. **Perf work** — Z-Image/Klein train/inference speed passes come after
   correctness gates; measure against Serenity's own step times for comparable
   resolution/batch/dtype.
