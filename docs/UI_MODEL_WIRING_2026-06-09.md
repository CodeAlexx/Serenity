# Trainer-UI model wiring — Chroma / Ernie / Anima / SDXL / Z-Image / L2P (+video surface) (2026-06-09)

The native trainer UI now launches the four campaign-verified trainable models in
addition to Klein and Ideogram4. The trainers themselves are the parity-campaign
serenitymojo runners (`/home/alex/mojodiffusion/serenitymojo/training/train_<m>_real.mojo`)
built unchanged into `target/serenity_<m>_live_trainer` — reuse-not-copy, same
precedent as the Chroma campaign builder deviation.

## Seam (Tenet 8: decoupled protocol)
- UI → runner: `target/serenity_<target>_live_trainer <target/serenity_<target>_train_config.json> <steps>`
  via `scripts/serenity_terminal_launcher.sh`. The config JSON is written at
  launch by the UI (`trainer_ui_runner_train_config_json`) in the serenitymojo
  TrainConfig schema: arch dims verbatim from `serenitymojo/configs/<m>.json`,
  recipe (lr/rank/alpha/steps/save_every/cache_dir/checkpoint) from the UI.
  The runner re-validates every dim and pinned recipe value at startup — drift
  fails loud before any GPU work.
- runner → UI: stdout is teed into the polled progress file
  (`target/serenity_trainer_progress.log`); the runners stream the shared
  `print_trainer_progress` line shape, which `trainer_ui_apply_progress_line`
  parses (legacy progress-line bridge).
- Bridge scanners (`_find_token`/`_find_char`/`_token_end`) were rewritten to
  RAW-BYTE scanning: trainer stdout contains multi-byte UTF-8 ("—" banner) and
  String codepoint indexing at arbitrary byte offsets asserts mid-codepoint.

## Pinned recipe constants (fail-loud at runner startup)
- **SDXL**: rank=16, alpha=16, lr=1e-4, clip=1.0 (comptime).
- **Chroma**: rank=16, alpha=16, lr=1e-4, shift=1.15, clip=1.0 (comptime).
- Ernie/Anima take runtime rank/alpha/lr. UI presets default to the pinned values.

## Gates run (main loop, 2026-06-09)
| Gate | Result |
|---|---|
| `smoke/runner_train_config_gate.mojo` — UI JSON → real `read_model_config`, 4 targets | PASS (45 checks) |
| `smoke/runner_progress_line_gate.mojo` — real runner stdout → bridge live stats | PASS |
| `pixi run trainer-ui-runtime-test` | PASS |
| Anima end-to-end: UI config, 2 steps | **PASS** — loss 0.9748→0.9641, 280 adapters saved |
| SDXL launch+load: UI config, 0 steps | PASS (ckpt+700 adapters+cache loaded) |
| SDXL 1 step | **TRAINER BUG** `elementwise: a/b dtype mismatch BF16/F32 [1,1280]` — reproduces on canonical `configs/sdxl.json` (Stage-2 in-flight on `training-port-5models-lora`) |
| Ernie launch+load: UI config | PASS (36 blocks resident, 22 cached samples) |
| Ernie 1 step | **TRAINER BUG** `rope_halfsplit_full: x/cos/sin dtype mismatch` in step-0 baseline sampler — reproduces on canonical `configs/ernie_image.json` (campaign rope bugfix in-flight) |
| Chroma 1 step: UI config | **TRAIN STEP PASS** — loss 0.4155, grad_norm 0.0014, LoRA-B 0→2372.8, 57-block swap (271s/step, h2d 17.9 GiB); progress line parsed shape. Save then raised `failed to open for write` (missing `/home/alex/mojodiffusion/output/chroma_boxjana`) |

Save-dir fix: the config-driven runners save to fixed per-model output dirs and
don't create them. The UI launch command now `mkdir -p`s all four
(chroma_boxjana, alina_sdxl, serenitymojo/output, output) before exec.
**VERIFIED**: chroma 1-step re-run with the dir present saved
`output/chroma_boxjana/chroma_lora_step1.safetensors` (131 MB) + `.state` (654 MB),
identical loss/grad (0.4155349 / 0.0014), exit 0.

## Round 2 (same day): Z-Image, L2P, video surface
- **Z-Image** (`zimage`, runner from `train_zimage_real.mojo`; pinned rank=16
  alpha=1.0 lr=3e-4): wired + binary built + config seam gated. Launch loads
  blocks (2 NR + 2 CR + 30 main, 210 trainable adapters) but raises
  `unsupported Z-Image production bucket` on BOTH existing EriDiffusion caches
  (boxjana/alina are 64x64 latent / seq-512; trainer is comptime-shaped on
  72x56 & 88x48 cap224/cap256). Blocked on a bucket-compatible prepare run at
  `/home/alex/mojodiffusion/output/alina_zimage_cache` (preset default).
- **Z-Image L2P** (`l2p`, from `train_l2p_real.mojo`; pixel-space, VAE-less,
  Z-Image DiT body verbatim; pinned rank=16 alpha=16 lr=3e-4 shift=3.0):
  wired + built. Preflight fails loud `cache does not exist`
  (`output/alina_l2p_cache`) before the 19 GB ckpt load — no prepared pixel
  cache exists yet. NOTE the trainer's final layer is a DOCUMENTED PROXY
  (x_embedder^T) for the frozen local_decoder ConvNet.
- **Video (LTX-2 AV, Wan2.2 T2V 14B)**: dropdown entries added, routed to
  UNWIRED backends (`ltx2`, `wan22`) that refuse to launch — measured reality:
  `train_ltx2_real` is fail-closed legacy (its own header), `train_ltx2_av` is
  a readiness contract not a loop, and `train_wan22_real` has PLACEHOLDER rope
  tables ("TODO: replace with wan22_build_rope for real training") + no config
  parsing. Wiring those as live would silently run unfaithful training. The
  runner config schema now carries `frames` so real AV trainers plug straight in.
- Lens (microsoft_lens): NOT wired — only one-step synthetic-latent parity
  smokes exist (no real-data multi-step loop, no cache); a UI entry would
  launch a fake train run.

## Per-model launch matrix (measured 2026-06-09)
| Model | Launch+load | Train step | Save |
|---|---|---|---|
| Anima | PASS | PASS (loss falls 2-step) | PASS (280 adapters) |
| Chroma | PASS | PASS (loss 0.4155) | PASS (131 MB LoRA + state) |
| Ernie | PASS | trainer-side rope dtype bug (in-flight) | — |
| SDXL | PASS | trainer-side BF16/F32 dtype bug (in-flight) | — |
| Z-Image | PASS (blocks resident) | blocked: no bucket-compatible cache | — |
| L2P | fail-loud preflight (no pixel cache) | — | — |
| LTX-2 / Wan2.2 | unwired backends, launch refused by design | — | — |

Zero training-time overhead: all wiring is launch-time only (config JSON write,
mkdir -p, process spawn); trainers run as separate processes; the only UI-side
recurring cost is the pre-existing progress-file poll, unchanged.

The SDXL/Ernie step failures are trainer-side regressions on the current
mojodiffusion working tree (shared-op dtype edits in flight), NOT UI-seam bugs:
identical failures occur with the trainers' own canonical configs. When the
campaign loop lands those fixes, rebuilding the runner binaries
(`pixi run sdxl-live-trainer-build` etc.) picks them up with no UI change.

## Selector mapping
model_type: IDEOGRAM_4, FLUX_2, STABLE_DIFFUSION_XL_10_BASE, STABLE_DIFFUSION_35,
CHROMA_1, ERNIE_IMAGE, ANIMA. STABLE_DIFFUSION_35 routes to backend `sd35`
which has no runner — launch fails loudly ("No trainer wired") instead of
silently training the previous selection.
