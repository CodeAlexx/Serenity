# HANDOFF 2026-07-06 — fleet fully device-class + the product wave

Session arc: 2026-07-04 → 07-06 (this doc written mid-wave, 3 agents still in
flight — see IN FLIGHT). Successor: read this + TODO.md row 18 + the ledger
(eng-knowledge/KNOWN_ISSUES.jsonl MJ-1065..MJ-1081) before acting.

## THE HEADLINE STATE

**Fleet (all measured s/step, all device-class, all run-gated):**
hidream 1.0 · anima 1.1 (b2 2.0/pair) · zimage 2.x · ideogram4 ~2.1/sample ·
krea2 2.4 · ernie 3.1 · klein 3.3 · chroma 3.6-4.0 (was 139) · **flux 5.6-6.2
(was 135, landed 238c089 this morning — bit-identical, every digit)** · qwen
~192 (works; device-optimizer port queued). Zero broken, zero unverified.
SDXL + SD3.5 UNFROZEN 07-06 (gates in flight); lens + BOFT stay dead.

**Product (no Python/PyTorch/X11 in the shipped path):**
- Web trainer (Rust axum, :8188, systemd `serenity-web-trainer`): all Mojo-UI
  parameters (census-complete, honest [not wired] dims), presets-as-data,
  4 argv shapes, .state full-resume w/ wrong-file 422, dry-run preview,
  working Qwen3-VL captioner, dataset grid w/ caption coverage + editor,
  structured sample-prompt editor (1024-min server-enforced), 10 serenityUI
  themes, FULL-form server persistence (/api/ui/state), run re-attach on load,
  step-echo in the log pane.
- SerenityBoard BUILT IN at /board (rusqlite, live hook + CLI workspace tailer,
  PK-dedupe). Live-gated with real runs. **TB-PARITY GATED late 07-06
  (tests/board/, 296be1d): values f32-exact vs OneTrainer tfevents, smoothing
  BIT-IDENTICAL to TB 2.20's bundled algorithm, cross-trainer overlay (OT runs
  import via scripts/board_import_tfevents.py and chart beside Mojo runs).**
- Konva inference canvas: serenity-server (:7811) ComfyUI Tier-A adapters →
  Mojo zimage worker, click-through e2e green, Playwright console-clean.
- RESILIENCE (MJ-1080): children write their own log files (no pipe),
  supervisor TAILS; KillMode=process — a service restart mid-run was
  REPRODUCED ON PURPOSE and the trainer survived + completed.

**Save/resume (audited vs OneTrainer/SimpleTuner/ai-toolkit/musubi —
docs/SAVE_RESUME_AUDIT_2026-07-05.md):** atomic tmp+rename writes fleet-wide;
.state auto-probe + LOUD warm banner (MJ-1077 closed); fast-arm moment restore
GATED 9x tighter than the measured GPU determinism floor; __meta__ seed/
dataset guards; save_max_keep rolling retention (live prune incl. .state
siblings). Open: A3 optimizer-state serialization (MJ-1081).

## IN FLIGHT (3 agents, this wave)
1. **sdxl-sd35-gates**: deferred gates for SDXL + SD3.5-Medium (b1/accum/b2/
   parity). Both binaries built; runs rotating GPU windows. On green → wire
   webui presets (lead does presets.json).
2. **fp8-cache**: krea2 launch sidecar (12GB fp8cache.safetensors WRITTEN next
   to Krea-2-Raw). Pending: byte-identity gate vs fresh quantize + speed gate
   (expect 3-4min → ~30s launches).
3. **board-complete**: /board Artifacts tab (workspace PNGs), real HParams at
   launch, LoRA analytics from server-side checkpoint reads. board.rs+main.rs
   mid-edit — DO NOT commit serenity-trainer/webui until it reports.

## QUEUE AFTER THE WAVE
qwen device-resident optimizer (~192s/step, fused_lora_adamw_plain_step_resident
exists; beware MJ-1070's unmapped-pinned-buffer hazard) · sd35-Large device port
(MJ-1069, now has flux/chroma templates) · MJ-1081 A3 resume state · MJ-1079
flux b2 config gap · 768px flux fp8 VRAM measure (needs a 768 cache) ·
konva Tier-A leftovers (multipart /upload = img2img/inpaint) · anima 1024
sampling · l2p full-depth fit · zimage b2 cost (needs user dataset).

## FRESH LoRAs (user artifacts, do not delete)
output/krea2_boxjana_lora_adamw (2000 steps, box1jana, 4 ckpts + states +
turbo renders) · output/krea2_giger3v2_lora_adamw (Gigerverse30) · eri2 runs.

## GOTCHAS FOR THE SUCCESSOR
- One GPU (3090, 24GB), ONE training/whole-DiT load at a time (MJ-1066);
  62GB host RAM — check RSS before big arms.
- pkill -f self-match kills your own shell (use pattern[ ] bracket trick).
- Firefox session-restore lazy tabs render HTML without JS (test in FRESH
  windows; prefer Playwright at output/konva_wire/pw/).
- Agents stall at turn boundaries: check DISK EVIDENCE (mtimes/logs) before
  believing an idle notification means done or dead — and check for
  90-second-window write collisions before assuming a lost write (the
  param-parity lesson: BOTH implementations had landed).
- Judge s/step from 100+ step windows (first ~20 post-load are warm-up);
  0x4 in nvidia-smi throttle = SwPowerCap = normal, NOT thermal.
- Trainer stdout is line-buffered via stdbuf -oL for file tailing; the fp8
  quantize + step-1 warmup phases PRINT progress now (silence = real problem).
