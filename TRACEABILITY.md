# Serenity Mirror Traceability

This file answers "what is what" for the port.

Every model must be tracked as:

`Serenity source .py -> serenity-trainer mirror .mojo -> parity artifact/gate`

`/home/alex/mojodiffusion/serenitymojo/*` is implementation material only. It is
not the port surface and is not proof of Serenity parity unless the code has
been copied into the matching `/home/alex/serenity-trainer/src/serenity_trainer/*`
mirror and verified against Serenity on byte-identical inputs.

CPU PyTorch / dry-run checks are structural evidence only. They can validate
configuration, cache presence, source routing, and deterministic helper formulas,
but they do not count as numeric parity for loss, gradients, optimizer deltas,
sampler trajectories, images, or speed. Numeric gates must use Serenity CUDA
reference dumps.

## Qwen

Reference root: `/home/alex/Serenity/modules`

| Serenity source | Mojo mirror |
|---|---|
| `model/QwenModel.py` | `src/serenity_trainer/model/QwenModel.mojo` |
| `dataLoader/QwenBaseDataLoader.py` | `src/serenity_trainer/dataLoader/QwenBaseDataLoader.mojo` |
| `modelSetup/BaseQwenSetup.py` | `src/serenity_trainer/modelSetup/BaseQwenSetup.mojo` |
| `modelSetup/QwenLoRASetup.py` | `src/serenity_trainer/modelSetup/QwenLoRASetup.mojo` |
| `modelSetup/QwenFineTuneSetup.py` | `src/serenity_trainer/modelSetup/QwenFineTuneSetup.mojo` |
| `modelSampler/QwenSampler.py` | `src/serenity_trainer/modelSampler/QwenSampler.mojo` |
| `modelLoader/QwenFineTuneModelLoader.py` | `src/serenity_trainer/modelLoader/QwenFineTuneModelLoader.mojo` |
| `modelLoader/QwenLoRAModelLoader.py` | `src/serenity_trainer/modelLoader/QwenLoRAModelLoader.mojo` |
| `modelLoader/qwen/QwenModelLoader.py` | `src/serenity_trainer/modelLoader/qwen/QwenModelLoader.mojo` |
| `modelLoader/qwen/QwenLoRALoader.py` | `src/serenity_trainer/modelLoader/qwen/QwenLoRALoader.mojo` |
| `modelSaver/QwenFineTuneModelSaver.py` | `src/serenity_trainer/modelSaver/QwenFineTuneModelSaver.mojo` |
| `modelSaver/QwenLoRAModelSaver.py` | `src/serenity_trainer/modelSaver/QwenLoRAModelSaver.mojo` |
| `modelSaver/qwen/QwenModelSaver.py` | `src/serenity_trainer/modelSaver/qwen/QwenModelSaver.mojo` |
| `modelSaver/qwen/QwenLoRASaver.py` | `src/serenity_trainer/modelSaver/qwen/QwenLoRASaver.mojo` |

Current gates: `parity/qwen_train_ref_step000.safetensors`,
`parity/qwen_train_ref_meta.json`, `parity/qwen_train_profile_meta.json`,
`smoke/qwen_train_ref_loss_replay.mojo`, `smoke/qwen_lora_key_parity.mojo`,
`smoke/qwen_real_lora_file_parity.mojo`,
`smoke/qwen_sampler_helper_gate.mojo`.

Status: loss-only replay matches the dumped Serenity loss. Qwen transformer
forward/backward, Qwen2.5-VL cached text branch, VAE encode/decode, LoRA
grad/AdamW, and sampler denoise/decode parity are still missing. Real
Serenity Qwen LoRA file key/shape/dtype parity now passes:
`keys=2160 adapters=720 rank=16 dtype=BF16 alpha=1.0`.

Naming traps:
- `src/serenity_trainer/model/QwenTextEncoder.mojo` is a Z-Image Qwen3 text
  encoder helper, not the Qwen Image `QwenModel.py` text path. Its compile smoke
  must not be counted as Qwen Image parity.
- `src/serenity_trainer/modelSetup/qwenLoraTargets.mojo` is implementation
  metadata and has no Serenity `.py` mirror. Its bounded smoke is not the full
  Qwen adapter inventory gate.
- A real Serenity Qwen LoRA output exists at
  `/home/alex/Serenity/output/qwen_100step_baseline/lora.safetensors` with
  2160 keys. `smoke/qwen_real_lora_file_parity.mojo` verifies that file against
  the full 60-block transformer attn-mlp target inventory.

## Klein / Flux2

Reference root: `/home/alex/Serenity/modules`

Klein is Serenity's Flux2 class/family. The Serenity mirror anchor is
Flux2. Flux2 dev and Klein share `ModelType.FLUX_2`; the runtime split is
`Flux2Model.is_dev()` (`transformer.config.num_attention_heads == 48`) vs
`is_klein()`. `KleinModel.mojo` and `model/klein/*` are copied transformer-core
files inside the port namespace; they are not replacements for the Serenity
file-name mapping below.

| Serenity source | Mojo mirror |
|---|---|
| `model/Flux2Model.py` | `src/serenity_trainer/model/Flux2Model.mojo` |
| `dataLoader/Flux2BaseDataLoader.py` | `src/serenity_trainer/dataLoader/Flux2BaseDataLoader.mojo` |
| `modelSetup/BaseFlux2Setup.py` | `src/serenity_trainer/modelSetup/BaseFlux2Setup.mojo` |
| `modelSetup/Flux2LoRASetup.py` | `src/serenity_trainer/modelSetup/Flux2LoRASetup.mojo` |
| `modelSetup/Flux2FineTuneSetup.py` | `src/serenity_trainer/modelSetup/Flux2FineTuneSetup.mojo` |
| `modelSampler/Flux2Sampler.py` | `src/serenity_trainer/modelSampler/Flux2Sampler.mojo` |
| `modelLoader/Flux2ModelLoader.py` | `src/serenity_trainer/modelLoader/Flux2ModelLoader.mojo` |
| `modelSaver/Flux2LoRAModelSaver.py` | `src/serenity_trainer/modelSaver/Flux2LoRAModelSaver.mojo` |
| `modelSaver/Flux2FineTuneModelSaver.py` | `src/serenity_trainer/modelSaver/Flux2FineTuneModelSaver.mojo` |
| `modelSaver/flux2/Flux2ModelSaver.py` | `src/serenity_trainer/modelSaver/flux2/Flux2ModelSaver.mojo` |
| `modelSaver/flux2/Flux2LoRASaver.py` | `src/serenity_trainer/modelSaver/flux2/Flux2LoRASaver.mojo` |

Current gates: `parity/klein_fwd.safetensors`,
`smoke/klein_forward_parity.mojo`, `smoke/klein_load_only.mojo`,
`parity/klein_train_ref_step000.safetensors`,
`parity/klein_train_ref_step000_adapters.safetensors`,
`scripts/klein_adapter_delta_contract.py`,
`smoke/klein_train_ref_forward_replay.mojo`,
`smoke/klein_train_ref_loss_replay.mojo`,
`smoke/klein_train_parity_smoke.mojo`, `smoke/klein_lora_key_parity.mojo`,
`smoke/klein_sampler_helper_gate.mojo`,
`smoke/flux2_finetune_setup_contract_check.mojo`,
`smoke/flux2_factory_contract_check.mojo`,
`smoke/flux2_surface_loader_contract_check.mojo`,
`smoke/flux2_surface_saver_contract_check.mojo`,
`smoke/flux2_runtime_loader_key_contract_check.mojo`,
`smoke/flux2_dev_branch_check.mojo`,
`scripts/flux2_dev_dump_train_ref.py`,
`parity/flux2_dev_train_ref_contract.json`, and
`parity/flux2_dev_train_ref_blockers.json`.

Status: load/forward/backward smokes are viable, and real Serenity train-dump
loss-only replay now passes. The full-finetune setup mirror is build-verified:
Serenity `Flux2FineTuneSetup.py` trains only `transformer`, freezes
`text_encoder` and `vae`, uses ModuleFilter for the transformer, puts text/VAE on
the temp device when latent caching is on, and reapplies requires-grad after the
optimizer step. Flux2/Klein factory/create dispatch plus loader/saver
Serenity-name mirrors are also build-run verified as source contracts only:
LoRA/fine-tune loader specs, Dev/Klein branch metadata, BF16 override defaults,
unsupported single-file error, full-model saver route plans, raw Flux2 LoRA
state sources, and the runtime-loader key helper split. The checked adapter-dump
oracle now verifies Serenity step-0 gradients: 288 FP32 tensors, 43,515,904
elements, grad L2 `0.0059750078751807986`, no clipping delta, and optimizer
state advancing to 288 parameter entries. The original step-0 dump has zero
adapter parameter delta because `lr_before=[0.0]`. A new two-step Serenity
CUDA dump in `/tmp` now gives a nonzero-LR step-1 update oracle:
`lr_before=[2.9999999999999997e-06]`, loss `0.5876612663269043`, grad norm
`0.008699021302163601`, update nonzero elems `43515904`, update abs sum
`51.57613097811465`, update L2 `0.010862200286706213`, and max update
`3.0034029805392493e-06`. Real Mojo train replay through grad norm, AdamW
adapter update, runtime load/save/resume, sampler denoise/decode, and real
PEFT-file key parity are still missing.
Flux2 dev is not covered by the Klein 9B gates. It now has only a structural
branch/scaffold gate: Serenity dev is
`transformer.config.num_attention_heads == 48`, the local partial FLUX.2-dev
transformer config has `num_attention_heads=48`, `num_layers=8`,
`num_single_layers=48`, and `joint_attention_dim=15360`, and the main-loop
Mojo helper gate passes (`seq_len=4096`, `steps=28`, `mu=2.1514432`,
`sigma1=0.9957105`). No Flux2 dev Serenity config, baseline, CUDA reference
tensors, full checkpoint, VAE, tokenizer, text encoder, or scheduler are present,
so Flux2 dev numeric train/sampler parity is still absent.

Parity-skill verdict:
- `smoke/klein_forward_parity.mojo` has the right reference/gate shape and a
  recorded cosine above the `>=0.999` bar.
- `smoke/klein_train_parity_smoke.mojo` does not satisfy numeric train parity; it
  uses the older `klein_fwd` fixture and Mojo RNG, not the real Serenity train
  dump.
- `smoke/klein_train_ref_loss_replay.mojo` consumes the real dump and passes
  loss-only replay: Mojo loss `0.12241738` vs dump `0.12243739`, abs error
  `2.0004809e-05`, loss kernel `0.107232224s`, total `0.268313362s`. Required
  remaining comparison is byte-identical `predict -> backward_lora -> AdamW`
  against step-0 grad norm `0.005975008010864258`, LR `0.0 -> 3e-6`, and
  adapter deltas.
- `smoke/klein_sampler_helper_gate.mojo` does not satisfy sampler parity. It is a
  helper check only; missing are full no-grad denoise, exact scheduler
  sigmas/timesteps, latent cosine `>=0.999`, VAE decode, valid image, PSNR, and
  mean-abs-diff.

## SD3 / SD3.5

Reference root: `/home/alex/Serenity/modules`

| Serenity source | Mojo mirror |
|---|---|
| `model/StableDiffusion3Model.py` | `src/serenity_trainer/model/StableDiffusion3Model.mojo` |
| `dataLoader/StableDiffusion3BaseDataLoader.py` | `src/serenity_trainer/dataLoader/StableDiffusion3BaseDataLoader.mojo` |
| `modelSetup/BaseStableDiffusion3Setup.py` | `src/serenity_trainer/modelSetup/BaseStableDiffusion3Setup.mojo` |
| `modelSetup/StableDiffusion3EmbeddingSetup.py` | `src/serenity_trainer/modelSetup/StableDiffusion3EmbeddingSetup.mojo` |
| `modelSetup/StableDiffusion3FineTuneSetup.py` | `src/serenity_trainer/modelSetup/StableDiffusion3FineTuneSetup.mojo` |
| `modelSetup/StableDiffusion3LoRASetup.py` | `src/serenity_trainer/modelSetup/StableDiffusion3LoRASetup.mojo` |
| `modelSampler/StableDiffusion3Sampler.py` | `src/serenity_trainer/modelSampler/StableDiffusion3Sampler.mojo` |
| `modelLoader/StableDiffusion3EmbeddingModelLoader.py` | `src/serenity_trainer/modelLoader/StableDiffusion3EmbeddingModelLoader.mojo` |
| `modelLoader/StableDiffusion3FineTuneModelLoader.py` | `src/serenity_trainer/modelLoader/StableDiffusion3FineTuneModelLoader.mojo` |
| `modelLoader/StableDiffusion3LoRAModelLoader.py` | `src/serenity_trainer/modelLoader/StableDiffusion3LoRAModelLoader.mojo` |
| `modelLoader/stableDiffusion3/StableDiffusion3ModelLoader.py` | `src/serenity_trainer/modelLoader/stableDiffusion3/StableDiffusion3ModelLoader.mojo` |
| `modelSaver/StableDiffusion3EmbeddingModelSaver.py` | `src/serenity_trainer/modelSaver/StableDiffusion3EmbeddingModelSaver.mojo` |
| `modelSaver/StableDiffusion3FineTuneModelSaver.py` | `src/serenity_trainer/modelSaver/StableDiffusion3FineTuneModelSaver.mojo` |
| `modelSaver/StableDiffusion3LoRAModelSaver.py` | `src/serenity_trainer/modelSaver/StableDiffusion3LoRAModelSaver.mojo` |
| `modelSaver/stableDiffusion3/StableDiffusion3ModelSaver.py` | `src/serenity_trainer/modelSaver/stableDiffusion3/StableDiffusion3ModelSaver.mojo` |

Current gates: `smoke/sd3_model_compile_check.mojo`,
`smoke/sd3_setup_surface_check.mojo`, `smoke/sd3_surface_check.mojo`,
`smoke/sd3_lora_key_parity.mojo`, `smoke/sd3_lora_inventory_contract.mojo`,
`parity/gen_sd3_sampler_helper_ref.py`, `parity/sd3_sampler_helper_ref.json`,
`smoke/sd3_sampler_helper_gate.mojo`, `scripts/sd3_dump_train_ref.py`,
`parity/sd3_train_ref_contract.json`, `parity/sd3_train_ref_blockers.json`.

Status: build/helper surfaces only. The current verified helper evidence is:
- bounded raw SD3 LoRA key/load previously passed: `targets=9 entries=27`;
- CUDA retry outside the sandbox passes `smoke/sd3_model_compile_check.mojo`;
- expanded SD3 Linear LoRA inventory contract passes with CUDA visible:
  `linear_targets=96 entries=288 rank=2 dtype=BF16 alpha=4.0`;
- sampler helper gate against generated Serenity/diffusers reference passes:
  `plan=1024x1056`, latent `128x132`, `cfg_batch=2`, `sigma1=0.85769236`,
  `sigma2=0.60215056`, `timestep1=857.6924`, latest helper timing
  `7.0001e-05s`;
- dry-run blocker artifact from Serenity's own config path. It explicitly
  marks CPU PyTorch / dry-run checks as structural only.

A real Serenity train baseline remains blocked by missing full local SD3
text-encoder weights/cache and data-loader registration. CUDA is visible outside
the sandbox, and the non-dry-run CUDA probe stops before model load on those
blockers. CPU PyTorch is not numeric parity evidence.

Blocked reference artifacts:
- `parity/sd3_train_ref_contract.json` exists.
- `parity/sd3_train_ref_blockers.json` exists and is regenerated by
  `/home/alex/Serenity/venv/bin/python scripts/sd3_dump_train_ref.py --dry-run --train-device cuda --temp-device cpu`.
  The contract records LoRA and full-finetune train-reference scope where
  Serenity registers those paths.
- `parity/sd3_train_ref_meta.json`,
  `parity/sd3_train_ref_step000.safetensors`, and
  `parity/sd3_train_ref_step000_adapters.safetensors` are absent.

Current baseline blockers:
- No registered Serenity data loader for `STABLE_DIFFUSION_3` in the current
  dry-run path.
- Missing cache at `/home/alex/Serenity/workspace-cache/sd35m_100step_baseline`.
- `/home/alex/.serenity/models/checkpoints/stablediffusion35_medium.safetensors`
  has diffusion/VAE keys but no inspectable CLIP/T5 text-encoder keys.

## SDXL

Reference root: `/home/alex/Serenity/modules`

| Serenity source | Mojo mirror |
|---|---|
| `model/StableDiffusionXLModel.py` | `src/serenity_trainer/model/StableDiffusionXLModel.mojo` |
| `dataLoader/StableDiffusionXLBaseDataLoader.py` | `src/serenity_trainer/dataLoader/StableDiffusionXLBaseDataLoader.mojo` |
| `modelSetup/BaseStableDiffusionXLSetup.py` | `src/serenity_trainer/modelSetup/BaseStableDiffusionXLSetup.mojo` |
| `modelSetup/StableDiffusionXLEmbeddingSetup.py` | `src/serenity_trainer/modelSetup/StableDiffusionXLEmbeddingSetup.mojo` |
| `modelSetup/StableDiffusionXLFineTuneSetup.py` | `src/serenity_trainer/modelSetup/StableDiffusionXLFineTuneSetup.mojo` |
| `modelSetup/StableDiffusionXLLoRASetup.py` | `src/serenity_trainer/modelSetup/StableDiffusionXLLoRASetup.mojo` |
| `modelSampler/StableDiffusionXLSampler.py` | `src/serenity_trainer/modelSampler/StableDiffusionXLSampler.mojo` |
| `modelLoader/StableDiffusionXLEmbeddingModelLoader.py` | `src/serenity_trainer/modelLoader/StableDiffusionXLEmbeddingModelLoader.mojo` |
| `modelLoader/StableDiffusionXLFineTuneModelLoader.py` | `src/serenity_trainer/modelLoader/StableDiffusionXLFineTuneModelLoader.mojo` |
| `modelLoader/StableDiffusionXLLoRAModelLoader.py` | `src/serenity_trainer/modelLoader/StableDiffusionXLLoRAModelLoader.mojo` |
| `modelLoader/stableDiffusionXL/StableDiffusionXLModelLoader.py` | `src/serenity_trainer/modelLoader/stableDiffusionXL/StableDiffusionXLModelLoader.mojo` |
| `modelSaver/StableDiffusionXLEmbeddingModelSaver.py` | `src/serenity_trainer/modelSaver/StableDiffusionXLEmbeddingModelSaver.mojo` |
| `modelSaver/StableDiffusionXLFineTuneModelSaver.py` | `src/serenity_trainer/modelSaver/StableDiffusionXLFineTuneModelSaver.mojo` |
| `modelSaver/StableDiffusionXLLoRAModelSaver.py` | `src/serenity_trainer/modelSaver/StableDiffusionXLLoRAModelSaver.mojo` |
| `modelSaver/stableDiffusionXL/StableDiffusionXLModelSaver.py` | `src/serenity_trainer/modelSaver/stableDiffusionXL/StableDiffusionXLModelSaver.mojo` |

Current gates: `smoke/sdxl_model_compile_check.mojo`,
`smoke/sdxl_model_tensor_contract_check.mojo`,
`smoke/sdxl_setup_surface_check.mojo` (manifest only after compiler OOM),
`smoke/sdxl_setup_base_contract_check.mojo`,
`smoke/sdxl_setup_method_contract_check.mojo`,
`smoke/sdxl_setup_dataloader_contract_check.mojo`,
`smoke/sdxl_surface_check.mojo` (manifest only after compiler OOM),
`smoke/sdxl_surface_loader_contract_check.mojo`,
`smoke/sdxl_surface_sampler_contract_check.mojo`,
`smoke/sdxl_surface_saver_contract_check.mojo`,
`smoke/sdxl_surface_factory_contract_check.mojo`,
`smoke/sdxl_real_lora_file_parity.mojo`,
`parity/gen_sdxl_sampler_helper_ref.py`, `parity/sdxl_sampler_helper_ref.json`,
`smoke/sdxl_sampler_helper_gate.mojo`,
`smoke/sdxl_train_ref_loss_replay.mojo`,
`scripts/sdxl_adapter_delta_contract.py`,
`parity/sdxl_train_ref_step000.safetensors`,
`parity/sdxl_train_ref_step000_adapters.safetensors`,
`/home/alex/Serenity/output/sdxl_100step_baseline/metrics.json`.

Status: build surfaces plus Serenity 100-step baseline. The model-core gate
is now split into sandbox-safe metadata/shape coverage
(`sdxl_model_compile_check`) and a tiny CUDA-only BF16 tensor round-trip
(`sdxl_model_tensor_contract_check`) because `DeviceContext` cannot detect the
GPU architecture inside the current sandbox. The old monolithic
`smoke/sdxl_setup_surface_check.mojo` was replaced with a tiny manifest after a
verified kernel OOM during compile (`mojo` pid 58957 at about 62 GiB RSS inside
the VSCode snap scope). Its coverage is now split across
`sdxl_setup_base_contract_check`, `sdxl_setup_method_contract_check`, and
`sdxl_setup_dataloader_contract_check`, all built and run under
`prlimit --as=24000000000`. The old monolithic
`smoke/sdxl_surface_check.mojo` also hit the compiler OOM path under that cap,
so loader/conversion, sampler, saver, and factory coverage now lives in the
four `sdxl_surface_*_contract_check.mojo` split gates. BF16 final-output LoRA
file parity passes
(`2382` keys / `794` adapters / rank `16` / alpha `16.0`),
sampler helper contract parity passes and explicitly says it is not
denoise/decode/image/end-to-end sampler parity (`1024x1152`, latent `128x144`,
CFG batch `2`, inpaint channels `9`, timestep contract `3 - 4`, scheduler
`EulerAncestralDiscreteScheduler`, latest helper timing `0.000143672s`), and
loss-only replay passes: Mojo loss `0.13533124` vs Serenity dump
`0.13533266`, abs err `1.4156103e-06`, loss kernel `0.054720012s`, total
`0.220353558s`. CPU-only adapter-delta contract passes over the Serenity
adapter dump: 6352 keys, four phases of 1588 tensors / 49,412,736 elems,
reference `FLOAT_32` training adapter dtype, unchanged before/pre/post phases,
after-step deltas in 1588 tensors / 49,412,638 elems, abs sum
`1993.9059369890892`, max `9.999924077419564e-05`, runtime
`1.2808432990004803s`. This is static phase/delta evidence only; the dump does
not include per-tensor gradients, so Transformer forward/backward, LoRA
grad/AdamW numeric parity, full-finetune save/load/resume, and end-to-end
sampler denoise/decode parity are still missing.

## Chroma

Reference root: `/home/alex/Serenity/modules`

Serenity code audit: Chroma is `ModelType.CHROMA_1`. It has LoRA,
full-finetune, and embedding setup registrations; the required scope includes
LoRA parity, full-weight finetune parity, sampler parity, loader/saver parity,
and Chroma-specific T5 attention-mask / VAE shift-scale behavior.

| Serenity source | Mojo mirror |
|---|---|
| `model/ChromaModel.py` | `src/serenity_trainer/model/ChromaModel.mojo` |
| `dataLoader/ChromaBaseDataLoader.py` | `src/serenity_trainer/dataLoader/ChromaBaseDataLoader.mojo` |
| `modelSetup/BaseChromaSetup.py` | `src/serenity_trainer/modelSetup/BaseChromaSetup.mojo` |
| `modelSetup/ChromaLoRASetup.py` | `src/serenity_trainer/modelSetup/ChromaLoRASetup.mojo` |
| `modelSetup/ChromaFineTuneSetup.py` | `src/serenity_trainer/modelSetup/ChromaFineTuneSetup.mojo` |
| `modelSetup/ChromaEmbeddingSetup.py` | `src/serenity_trainer/modelSetup/ChromaEmbeddingSetup.mojo` |
| `modelSampler/ChromaSampler.py` | `src/serenity_trainer/modelSampler/ChromaSampler.mojo` |
| `modelLoader/ChromaLoRAModelLoader.py` | `src/serenity_trainer/modelLoader/ChromaLoRAModelLoader.mojo` |
| `modelLoader/ChromaFineTuneModelLoader.py` | `src/serenity_trainer/modelLoader/ChromaFineTuneModelLoader.mojo` |
| `modelLoader/ChromaEmbeddingModelLoader.py` | `src/serenity_trainer/modelLoader/ChromaEmbeddingModelLoader.mojo` |
| `modelLoader/chroma/ChromaModelLoader.py` | `src/serenity_trainer/modelLoader/chroma/ChromaModelLoader.mojo` |
| `modelLoader/chroma/ChromaLoRALoader.py` | `src/serenity_trainer/modelLoader/chroma/ChromaLoRALoader.mojo` |
| `modelLoader/chroma/ChromaEmbeddingLoader.py` | `src/serenity_trainer/modelLoader/chroma/ChromaEmbeddingLoader.mojo` |
| `modelSaver/ChromaLoRAModelSaver.py` | `src/serenity_trainer/modelSaver/ChromaLoRAModelSaver.mojo` |
| `modelSaver/ChromaFineTuneModelSaver.py` | `src/serenity_trainer/modelSaver/ChromaFineTuneModelSaver.mojo` |
| `modelSaver/ChromaEmbeddingModelSaver.py` | `src/serenity_trainer/modelSaver/ChromaEmbeddingModelSaver.mojo` |
| `modelSaver/chroma/ChromaModelSaver.py` | `src/serenity_trainer/modelSaver/chroma/ChromaModelSaver.mojo` |
| `modelSaver/chroma/ChromaLoRASaver.py` | `src/serenity_trainer/modelSaver/chroma/ChromaLoRASaver.mojo` |
| `modelSaver/chroma/ChromaEmbeddingSaver.py` | `src/serenity_trainer/modelSaver/chroma/ChromaEmbeddingSaver.mojo` |
| `util/convert/lora/convert_chroma_lora.py` | `src/serenity_trainer/util/convert/lora/convert_chroma_lora.mojo` |
| `util/convert/convert_chroma_diffusers_to_ckpt.py` | `src/serenity_trainer/util/convert/convert_chroma_diffusers_to_ckpt.mojo` |

Current gates/artifacts: `parity/gen_chroma_sampler_helper_ref.py`,
`parity/chroma_sampler_helper_ref.json`,
`smoke/chroma_sampler_helper_gate.mojo`,
`smoke/chroma_model_setup_contract_check.mojo`,
`smoke/chroma_surface_loader_contract_check.mojo`,
`smoke/chroma_surface_saver_contract_check.mojo`,
`smoke/chroma_lora_conversion_contract_check.mojo`,
`smoke/chroma_real_lora_file_parity.mojo`,
`/home/alex/Serenity/output/chroma_100step_baseline/metrics.json`,
`/home/alex/Serenity/output/chroma_100step_baseline/lora.safetensors`,
`parity/chroma_train_ref_step000.safetensors`,
`parity/chroma_train_ref_step000_adapters.safetensors`,
`parity/chroma_train_ref_meta.json`, and
`smoke/chroma_train_ref_loss_replay.mojo`. The helper-only sampler slice covers:
64px quantization, latent/packed shape, CFG combine, T5 mask/text-id/image-id
contracts, FlowMatch scheduler metadata from the local Chroma1-HD scheduler
config, VAE scale/shift decode formula, and image output metadata.

Status: required, code-audited, and partially verified for model/setup
contracts plus helper/file/loss-only artifacts. `src/serenity_trainer/model/ChromaModel.mojo`,
`src/serenity_trainer/modelSetup/BaseChromaSetup.mojo`, and
`src/serenity_trainer/modelSetup/ChromaFineTuneSetup.mojo`, and
`src/serenity_trainer/modelSetup/ChromaLoRASetup.mojo` now contain build-only
Serenity contract mirrors, not runtime implementations. The model/setup smoke
passes: `model type = 22`, predict outputs `4`, dtype caveats `6`, LoRA opt
parts `5`, FINE_TUNE params `3`, LoRA params `3`, `create te = True`, ragged
text seq `32`, packed `4608x64`, `model_t(500)=0.5`, `sigma(499)=0.5`.
Chroma loader/saver/LoRA-conversion Serenity-name mirrors also build-run as
split gates: `CHROMA SURFACE LOADER CONTRACT OK`, `CHROMA SURFACE SAVER
CONTRACT OK`, and `CHROMA LORA CONVERSION CONTRACT OK`. These check wrapper
spec filenames, internal/diffusers/safetensors route order, BF16 transformer
override default, T5/FlowMatch/VAE/transformer class names, embedding keys
`t5`/`t5_out`, LoRA bundle keys `bundle_emb.{placeholder}.t5`/`t5_out`, and
Chroma LoRA conversion namespaces/counts. The Chroma sampler helper
gate is main-loop verified: `1024x1152`, latent `128x144`, packed
`4608x64`, CFG batch `2`, FlowMatch shift `3.0`, sigma1 `0.85769236`,
timestep1 `857.6924`, runtime `0.00010915s`. Serenity 100-step baseline now exists:
mean step `3.2088356397171727s`, last loss `0.3658050298690796`, smooth
`0.4213832823932175`, grad norm `0.001406678231433034`, peak sampled VRAM
`9913 MiB`, CUDA max allocated `7024 MiB`. Real Serenity LoRA file parity
passes: `912` keys / `304` adapters / rank `16` / BF16 / alpha keys present.
One-step dump exists: loss `0.2957186698913574`, grad norm
`0.00041076153866015375`, lr `0.0003 -> 0.0003`, elapsed
`169.69719535199692s`; `608` trainable tensors / `35,487,744` params. Mojo
loss-only replay passes: `0.29572487` vs dump `0.29571867`, abs err
`6.198883e-06`. Chroma transformer forward/backward, LoRA grad/AdamW update
parity, full-finetune runtime train/save/load/resume parity, and end-to-end
sampler denoise/decode parity are still missing.

## Flux.1 Dev

Reference root: `/home/alex/Serenity/modules`

| Serenity source | Mojo mirror |
|---|---|
| `model/FluxModel.py` | `src/serenity_trainer/model/FluxModel.mojo` |
| `dataLoader/FluxBaseDataLoader.py` | `src/serenity_trainer/dataLoader/FluxBaseDataLoader.mojo` |
| `modelSetup/BaseFluxSetup.py` | `src/serenity_trainer/modelSetup/BaseFluxSetup.mojo` |
| `modelSetup/FluxLoRASetup.py` | `src/serenity_trainer/modelSetup/FluxLoRASetup.mojo` |
| `modelSetup/FluxFineTuneSetup.py` | `src/serenity_trainer/modelSetup/FluxFineTuneSetup.mojo` |
| `modelSampler/FluxSampler.py` | `src/serenity_trainer/modelSampler/FluxSampler.mojo` |
| `modelLoader/FluxLoRAModelLoader.py` | `src/serenity_trainer/modelLoader/FluxLoRAModelLoader.mojo` |
| `modelLoader/FluxFineTuneModelLoader.py` | `src/serenity_trainer/modelLoader/FluxFineTuneModelLoader.mojo` |
| `modelLoader/flux/FluxModelLoader.py` | `src/serenity_trainer/modelLoader/flux/FluxModelLoader.mojo` |
| `modelLoader/flux/FluxLoRALoader.py` | `src/serenity_trainer/modelLoader/flux/FluxLoRALoader.mojo` |
| `modelSaver/FluxLoRAModelSaver.py` | `src/serenity_trainer/modelSaver/FluxLoRAModelSaver.mojo` |
| `modelSaver/FluxFineTuneModelSaver.py` | `src/serenity_trainer/modelSaver/FluxFineTuneModelSaver.mojo` |
| `modelSaver/flux/FluxModelSaver.py` | `src/serenity_trainer/modelSaver/flux/FluxModelSaver.mojo` |
| `modelSaver/flux/FluxLoRASaver.py` | `src/serenity_trainer/modelSaver/flux/FluxLoRASaver.mojo` |

Current gates: `smoke/flux_model_compile_check.mojo`,
`smoke/flux_setup_surface_check.mojo`, `smoke/flux_surface_check.mojo`,
`smoke/flux_sampler_helper_gate.mojo`,
`smoke/flux_real_lora_file_parity.mojo`,
`src/serenity_trainer/util/convert/lora/convert_flux_lora.mojo`,
`/home/alex/Serenity/output/flux1_100step_baseline/metrics_full_snapshot.json`.

Status: build surfaces plus Serenity 100-step baseline. Sampler helper parity
now passes: plan 1024x1024, latent 128x128, packed 4096x64, shift 3.1581929,
mu 1.15, model timestep 0.5, fill mask channels 256. This is helper-only and
not denoise/decode/image parity. Real Serenity LoRA file metadata gate passes
against `/home/alex/Serenity/output/flux1_100step_baseline/lora_last.safetensors`:
1512 keys, 504 complete adapters, rank 16, BF16 tensors, scalar alpha tensors,
and 0 bundled embedding keys. The Mojo gate checks keys/shapes/dtypes only; it
does not read alpha numeric values. Mojo train/full sampler/full-finetune numeric
parity is missing.

## Ernie

Reference root: `/home/alex/Serenity/modules`

| Serenity source | Mojo mirror |
|---|---|
| `model/ErnieModel.py` | `src/serenity_trainer/model/ErnieModel.mojo` |
| `dataLoader/ErnieBaseDataLoader.py` | `src/serenity_trainer/dataLoader/ErnieBaseDataLoader.mojo` |
| `modelSetup/BaseErnieSetup.py` | `src/serenity_trainer/modelSetup/BaseErnieSetup.mojo` |
| `modelSetup/ErnieLoRASetup.py` | `src/serenity_trainer/modelSetup/ErnieLoRASetup.mojo` |
| `modelSetup/ErnieFineTuneSetup.py` | `src/serenity_trainer/modelSetup/ErnieFineTuneSetup.mojo` |
| `modelSampler/ErnieSampler.py` | `src/serenity_trainer/modelSampler/ErnieSampler.mojo` |
| `modelLoader/ErnieModelLoader.py` | `src/serenity_trainer/modelLoader/ErnieModelLoader.mojo` |
| `modelSaver/ErnieLoRAModelSaver.py` | `src/serenity_trainer/modelSaver/ErnieLoRAModelSaver.mojo` |
| `modelSaver/ErnieFineTuneModelSaver.py` | `src/serenity_trainer/modelSaver/ErnieFineTuneModelSaver.mojo` |
| `modelSaver/ernie/ErnieModelSaver.py` | `src/serenity_trainer/modelSaver/ernie/ErnieModelSaver.mojo` |
| `modelSaver/ernie/ErnieLoRASaver.py` | `src/serenity_trainer/modelSaver/ernie/ErnieLoRASaver.mojo` |

Current gates: `smoke/ernie_model_compile_check.mojo`,
`smoke/ernie_surface_check.mojo`,
`smoke/ernie_real_lora_file_parity.mojo`,
`scripts/ernie_dump_train_ref.py`,
`parity/ernie_train_ref_contract.json`,
`parity/ernie_train_ref_blockers.json`,
`parity/ernie_train_ref_meta.json`,
`parity/ernie_train_ref_step000.safetensors`,
`parity/ernie_train_ref_step000_adapters.safetensors`,
`smoke/ernie_train_ref_loss_replay.mojo`,
`smoke/ernie_sampler_helper_gate.mojo`,
`/home/alex/Serenity/output/ernie_eri2_100step_baseline/metrics_nocompile.json`.

Status: build surfaces plus Serenity 100-step baseline. Real Serenity Ernie
LoRA file key/shape/dtype parity now passes:
`keys=756 adapters=252 rank=16 dtype=BF16 alpha=1.0`. A real CUDA one-step
train reference now exists: loss `0.643847644329071`, grad norm
`0.000828770047519356`, lr `0.0003 -> 0.0003`, elapsed
`97.59743943800095s`, 504 trainable tensors / 47,185,920 params. Mojo
loss-only replay passes: `0.6438152` vs dump `0.64384764`, abs err
`3.2424927e-05`, loss kernel `0.238489161s`, total `0.717235302s`. The gate
uses explicit Ernie-specific F32 reduction tolerance; PyTorch recomputes the
dump loss exactly from the same tensors. Sampler helper parity passes: plan
1024x1024, latent 128x128, patch contract 64x32x128, CFG batch 2, sigma1 0.75,
timestep1 750.0, Euler 0.125. This is helper-only and not denoise/decode/image
parity. Mojo transformer/backward/AdamW train parity, full-finetune parity, and
end-to-end sampler parity are still missing.

## Anima

Reference root: `/home/alex/Serenity-anima-ref/modules`

| Serenity source | Mojo mirror |
|---|---|
| `model/AnimaModel.py` | `src/serenity_trainer/model/AnimaModel.mojo` |
| `dataLoader/AnimaBaseDataLoader.py` | `src/serenity_trainer/dataLoader/AnimaBaseDataLoader.mojo` |
| `modelSetup/BaseAnimaSetup.py` | `src/serenity_trainer/modelSetup/BaseAnimaSetup.mojo` |
| `modelSetup/AnimaLoRASetup.py` | `src/serenity_trainer/modelSetup/AnimaLoRASetup.mojo` |
| `modelSetup/AnimaFineTuneSetup.py` | `src/serenity_trainer/modelSetup/AnimaFineTuneSetup.mojo` |
| `modelSampler/AnimaSampler.py` | `src/serenity_trainer/modelSampler/AnimaSampler.mojo` |
| `modelLoader/AnimaModelLoader.py` | `src/serenity_trainer/modelLoader/AnimaModelLoader.mojo` |
| `modelSaver/AnimaLoRAModelSaver.py` | `src/serenity_trainer/modelSaver/AnimaLoRAModelSaver.mojo` |
| `modelSaver/AnimaFineTuneModelSaver.py` | `src/serenity_trainer/modelSaver/AnimaFineTuneModelSaver.mojo` |
| `modelSaver/anima/AnimaModelSaver.py` | `src/serenity_trainer/modelSaver/anima/AnimaModelSaver.mojo` |
| `modelSaver/anima/AnimaLoRASaver.py` | `src/serenity_trainer/modelSaver/anima/AnimaLoRASaver.mojo` |

Current gates: `smoke/anima_model_compile_check.mojo`,
`smoke/anima_setup_surface_check.mojo`, `smoke/anima_surface_check.mojo`,
`smoke/anima_real_lora_file_parity.mojo`,
`parity/gen_anima_sampler_helper_ref.py`,
`parity/anima_sampler_helper_ref.json`,
`smoke/anima_sampler_helper_gate.mojo`,
`scripts/anima_dump_train_ref.py`,
`parity/anima_train_ref_contract.json`,
`parity/anima_train_ref_blockers.json`,
`parity/anima_train_ref_meta.json`,
`parity/anima_train_ref_step000.safetensors`,
`parity/anima_train_ref_step000_adapters.safetensors`,
`smoke/anima_train_ref_loss_replay.mojo`,
`/home/alex/Serenity-anima-ref/output/anima_100step_baseline/metrics_with_grad.json`.

Status: build surfaces plus Serenity-family 100-step baseline. Real
Serenity Anima LoRA file key/shape/dtype gate passes: 840 keys / 280
adapters / rank 16 / BF16 / alpha 1.0 from
`/home/alex/Serenity-anima-ref/output/anima_100step_baseline/lora.safetensors`.
Sampler helper gate passes against generated Serenity/diffusers reference:
plan `1024x1152`, latent `128x144`, CFG batch 2, sigma1 `0.9`, timestep1
`900.0`. Train-reference dry-run is structural-only and unblocked
(`blocked=false`) with local baseline/cache/weight evidence; it is not a CPU
numeric parity gate.
Serenity one-step train reference exists: loss `0.0667838305234909`, grad norm
`0.0014594600070267916`, lr `0.0 -> 1.5e-7`, elapsed `44.37211945699528s`.
Mojo loss-only replay passes: `0.06678182` vs dump `0.06678383`, abs err
`2.0116568e-06`.
Mojo train/sampler numeric parity is missing.
