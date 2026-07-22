# UI/Trainer Production-Wiring Campaign — 2026-07-22

Source: two exhaustive audits (UI-side + runner-side; full texts in session task
outputs a7d4aef / a40c8756). Goal: every model production-drivable from a config
file through the webui. Ranked by leverage.

## P0 — Broken builds (blocks everything)
9 drivers import dead pre-scrub modules (`serenitymojo.training.onetrainer_train_loop_policy`,
`onetrainer_cache_preflight` → renamed `serenity_trainer_*`): klein9b, chroma, ernie,
anima, flux, qwenimage, sd35, sdxl, zimage (each 2 dead imports; only klein4b migrated).
Mechanical rename + rebuild each + re-smoke the roster. NOTHING else matters first.

## P1 — Emit-path correctness bugs (silent wrongness today)
1. EMA: UI emits `ema_enabled` bool; reader/seam key on string `ema` → EMA silently
   dead everywhere. Fix index.html:644 (emit `ema`:"EMA"/"OFF" + interval) .
2. `caption_dropout` (UI) vs `caption_dropout_prob` (reader:1157). Rename emit.
3. Optimizer dropdown enums reader-rejected: ADAMW8BIT→ADAMW_8BIT; CAME/MUON not in
   whitelist (reader fail-loud ADAMW/ADAMW_8BIT/ADAFACTOR/SCHEDULE_FREE_ADAMW/AUTOMAGIC3).
4. Launch path never runs validate_config_enums (only offline smoke) → run it in
   main.rs launch → 422 in-UI.

## P2 — ideogram4: merged config never delivered
main.rs:563-582 omits cfg_path; resolution hardcoded 512, sampler 20/4.5, argv11
levers "-" though the Mojo seam HAS trainer_ui_ideogram4_levers_path_or_skip.
Deliver config (or levers-JSON argv11) + unhardcode.

## P3 — Env-only knobs → config keys (reproducibility)
wan21/22: WAN21_MODEL/WAN22_DATA_CACHE/WAN22_DIT*/WAN22_DUAL_EXPERT/boundary etc.
mageflow: MAGEFLOW_DATA_CACHE. ltx2_av flag-CLI (ic_lora_strategy, reference_cache_dir,
resident_blocks — reader ALREADY parses these at 949-964; driver re-reads argv).
krea2 modes env-gated (KREA2_FULL_FT/EDIT/BUCKETED...). Route cfg-first, env override.

## P4 — Levers unreachable on 10 legacy-block drivers
chroma/ernie/anima/l2p/sd35/sdxl(partial)/flux/qwenimage/wan22/ltx2_real ignore
optimizer/loss_fn/min_snr/caption_dropout_prob (non-ADAMW tag → SILENT AdamW).
Wire levers seams (start sdxl/sd35). mageflow: loss-lever only today.

## P5 — Per-model capability layer in the UI
applyPreset (index.html:620) sets values only; no show/hide/disable per backend.
Mirror TrainerConfigModel.mojo:1192-1222 capability table (and ADD wave-3 backends
to it — ltx2/wan/mageflow/flux/klein4b absent). Surface config_merge.rs:72 sampling
strip (sd35/hidream/ideogram4) as in-form warning. Resume slot table (main.rs:630)
covers 4/13 → extend or grey the tab per backend.

## P6 — Deeper production gaps
- Levers-optimizer resume sidecar: non-ADAMW can't resume (k≠1 fail-loud) except
  ltx2_av's sidecar — port levers_optimizer_sidecar_* generic into levers.mojo.
- Masked loss: implemented+gated, wired NOWHERE (ideogram4 private path only).
- Training-cache builders missing for ~11 backends (klein9b/4b, zimage, sdxl, sd35,
  hidream, l2p, ideogram4, ltx2, ernie latents, flux native) — off-box cache contract.
- Comptime geometry locks ignore cfg.resolution (mageflow 256², klein N_IMG 1024).
- Gate holes: krea2 (idx12) ungated; sd35 template missing (TrainerConfigModel:1617).
- flux1dev: host-OOM load refit (TurboPlannedLoader pins 23.8G host-side) + P0 debt.
- Dead keys parsed-by-reader, consumed-nowhere: masked_* family (all backends),
  audio_* (non-ltx2), controlnet_* (non-zimage), debiased/multires/offset-noise
  (0-levers backends).
- Duplicate #sec-cloud DOM ids (index.html:488/519) — fix before per-id wiring.
