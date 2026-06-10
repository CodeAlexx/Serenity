# Handoff — FLUX.1-dev parity verification + 3 follow-ups (2026-06-09 pass 2)

Continues `HANDOFF_FLUX_2026-06-09.md`. Repo `/home/alex/mojodiffusion`,
branch `training-port-5models-lora`.

## Current objective + model target
FLUX.1-dev pure-Mojo inference. This pass: (1) numeric quality-parity of every
pipeline stage vs reference, then (2) the three open handoff items — tile-seam
A/B, JSON sampler params, runtime LoRA. All DONE + measured.

## Accepted evidence level
Per-stage numeric parity on byte-identical inputs (oracle = reference's own
runtime at the Mojo's dtype fp16→bf16→fp32). Cosine ≥0.99 (bf16-compute floor)
for forwards, PSNR for pixels, exact for the schedule. Measured this session.

## Results (all PASS)
| Gate | metric | result |
|------|--------|--------|
| VAE decode (per-tile) | PSNR | 88.7 dB |
| DiT forward (full 57-block stack, real wts) | cos | 0.99942 |
| sigma schedule | max-abs-diff | 0.0 (bit-exact) |
| T5-XXL encoder | cos | 0.99920 (no NaN) |
| CLIP-L pooled | cos | 0.99999 |
| full 20-step denoise | cos (final latent) | 0.99969 |
| tile-seam (3×3 overlap vs seamless 1024²) | PSNR | 58.6 dB, no seam spike |
| DiT + Kohya LoRA overlay | cos | 0.99937 |

## Files changed (mine; disjoint from the pre-existing M set)
Commit `842d34d` (PUSHED): parity gates
- serenitymojo/vae/parity/flux_vae_decode_{oracle.py,parity.mojo}
- serenitymojo/models/flux/parity/flux_{dit,sched,t5,clip,denoise}_{oracle.py,parity.mojo}
- serenitymojo/docs/FLUX_PARITY_VERIFICATION_2026-06-09.md

Commit `0be3970` (LOCAL, NOT pushed):
- serenitymojo/pipeline/flux_tiled_decode.mojo  (NEW shared module)
- serenitymojo/pipeline/flux_sample_cli.mojo    (refactor + JSON params + LoRA)
- serenitymojo/vae/parity/flux_tiled_decode_parity.mojo (NEW gate)
- serenitymojo/models/flux/flux_lora_overlay.mojo (NEW)
- serenitymojo/models/dit/flux1_dit.mojo  (LoRA field + load_with_lora + _block_model_lora)
- serenitymojo/models/flux/parity/flux_dit_lora_{oracle.py,parity.mojo} (NEW gate)

## OT/reference → Mojo map
- BFL `flux/model.py Flux.forward` → `models/dit/flux1_dit.mojo Flux1Offloaded.forward` (cos 0.99942)
- BFL `flux/sampling.py get_schedule` → `sampling/flux1_dev.mojo build_flux1_sigma_schedule` (bit-exact)
- BFL AE decoder (`ae.safetensors decoder.*`) → `models/vae/ldm_decoder.mojo LdmVaeDecoder.decode` (88.7 dB)
- HF `T5EncoderModel` → `models/text_encoder/t5_encoder.mojo T5Encoder.encode` (0.99920)
- HF `CLIPTextModel.pooler_output` → `models/text_encoder/clip_encoder.mojo ClipEncoder.encode_sdxl[..][1]` (0.99999)
- Kohya `lora_unet_double_blocks_*` → `models/flux/flux_lora_overlay.mojo` overlay onto BFL weights (0.99937)

## Commands run (exit 0 unless noted)
- python3 serenitymojo/vae/parity/flux_vae_decode_oracle.py 64 64
- pixi run mojo run -I . serenitymojo/vae/parity/flux_vae_decode_parity.mojo
- python3 serenitymojo/models/flux/parity/flux_dit_oracle.py 4 4 16   (CPU, ~1-2 min, 48GB fp32)
- pixi run mojo run -I . -Xlinker -lcuda serenitymojo/models/flux/parity/flux_dit_parity.mojo
- python3 serenitymojo/models/flux/parity/flux_sched_oracle.py 20 4096 ; …flux_sched_parity.mojo
- python3 .../flux_t5_oracle.py ; …flux_t5_parity.mojo
- python3 .../flux_clip_oracle.py ; …flux_clip_parity.mojo
- python3 .../flux_denoise_oracle.py 4 4 16 20 ; …flux_denoise_parity.mojo
- python3 serenitymojo/vae/parity/flux_vae_decode_oracle.py 128 128 ; …flux_tiled_decode_parity.mojo
- python3 .../flux_dit_lora_oracle.py ; …flux_dit_lora_parity.mojo
- CLI build: pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/pipeline/flux_sample_cli.mojo -o /tmp/flux_cli_lora  (exit 0)
- CLI run base vs LoRA (steps=3): A/B image abs-diff mean 14.9/255, 90% pixels changed

## Generated artifacts (NOT committed — reproducible via oracles)
- *.bin under serenitymojo/{vae,models/flux}/parity/  (refs)
- /tmp/flux_{base,lora}_out.png  (A/B images, 1024²)
- /tmp/flux_cli_lora  (built CLI binary)

## Behaviors now live in flux_sample_cli
- argv[2] LoRA: Kohya/BFL format applied as additive overlay @ multiplier 1.0;
  diffusers-format → fail-loud.
- steps/guidance/seed: honored from prompt JSON (defaults 20/3.5/42 when unset).
- width/height: comptime 1024²; non-1024 request → fail-loud.

## Exact blockers / NOT done
- Diffusers-format FLUX LoRA (transformer.*.lora_A/B, q/k/v split) — not mapped.
- LoRA multiplier fixed at 1.0 in the CLI (overlay supports an arg; not surfaced).
- Production 1024²-grid DiT parity not run (verified at tiny 4×4 grid; DiT is
  resolution-agnostic so this is sound, but not directly measured at full res).
- Tokenizers fed identical ids in gates (not re-checked here; claimed bit-exact).

## Next command to run
DECISION PENDING: push `0be3970` →
  `git push origin training-port-5models-lora`
(842d34d already on origin; 0be3970 is local-only.)

## GPU / process state
GPU FREE (939 MiB / 24564 used, no compute). No mojo/flux processes running;
stray poll-loop wrappers killed.
