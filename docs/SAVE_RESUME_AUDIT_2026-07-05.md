# Save / Resume Audit — OURS vs OneTrainer, SimpleTuner, ai-toolkit, musubi-tuner
Date: 2026-07-05. Read-only code audit. Every claim cites file:line. "OURS" = serenity-trainer / serenitymojo (pure-Mojo).

## TL;DR
We sit in the **hand-rolled, moments+step tier** with OneTrainer and ai-toolkit: we persist optimizer moments + step and restore them byte-exact, but we do **not** save RNG state, dataloader position, or LR-scheduler state. The two **accelerate-based** references (SimpleTuner, musubi-tuner) restore a strictly broader state bundle (optimizer + scheduler + RNG for free via `accelerator.save_state`). Where we genuinely lose: the **krea2 device-fast arm silently warm-restarts** (moments reset) and the **warm-resume guard is web-UI-only** (the CLI silently warm-resumes a wrong path). Where we genuinely win: **optimizer state is plain safetensors not a pickle**, **save-before-sample** is explicit, **prune happens after the new save** (SimpleTuner prunes before), and the **web UI is the only tool here that outright rejects the wrong resume artifact**.

One correction to the going-in assumptions: **we DO persist EMA** — `save_klein_lora_ema` writes a `*_ema.safetensors` sibling (`serenitymojo/training/lora_ema.mojo:255`; gated by `serenitymojo/training/ema_save_smoke.mojo`), scoped to the klein path. It is EMA *shadow weights*, not an EMA optimizer object.

---

## Comparative table (dimension × trainer)

| Dimension | OURS (serenity) | OneTrainer | SimpleTuner | ai-toolkit | musubi-tuner |
|---|---|---|---|---|---|
| **Adapter/model weights** | ✅ PEFT keys `save_lora_peft` `lora_save.mojo:91`; kohya keys `save_lora_onetrainer:135` | ✅ OMI safetensors `LoRASaverMixin.py:77` | ✅ PEFT/diffusers via hook `save_hooks.py:504` | ✅ kohya default, PEFT for flux `network_mixins.py:567-624` | ✅ kohya `lora_unet_*` `networks/lora.py:893` |
| **Optimizer moments** | ✅ safetensors: full-FT `param./adam_m/adam_v` `loop.mojo:179-200`; LoRA per-adapter `save_lora_train_state:180` | ✅ pickle `optimizer.pt` `InternalModelSaverMixin.py:21-26` | ✅ accelerate `save_state` `trainer.py:5122` | ✅ pickle `optimizer.pt` (single, unversioned) `BaseSDTrainProcess.py:690-696` | ✅ accelerate `save_state` → `-state/optimizer.bin` `train_utils.py:192` |
| **LR-scheduler state** | ❌ none; warmup recomputed from opt step | ❌ reconstructed from `global_step` `create.py:1224` | ✅ accelerate `scheduler.bin` `trainer.py:5122` | ❌ rebuilt fresh `BaseSDTrainProcess.py:2049` | ✅ accelerate `scheduler.bin` `train_utils.py:192` |
| **RNG state (py/torch/cuda)** | ❌ none (noise = seed+step-derived) | ❌ none saved | ✅ accelerate `random_states_*.pkl` | ❌ none in train path | ✅ accelerate `random_states_*.pkl` |
| **Dataloader position** | ❌ none (order re-derived deterministically) | ✅ exact epoch+sample `DataLoaderMgdsMixin.py:48-49` | ~ coarse `seen_images` set `sampler.py:119-152` | ❌ fresh `iter()` `BaseSDTrainProcess.py:2296` | ~ naive batch-drop `hv_train_network.py:3226-3228` |
| **EMA weights** | ✅ klein only, shadow sibling `lora_ema.mojo:255` | ✅ `ema/ema.pt` `InternalModelSaverMixin.py:29-31` | ✅ separate `ema_model.pt` `save_hooks.py:785-790` | ❌ folded into weights, no file `BaseSDTrainProcess.py:495-497` | ❌ no weight-EMA (only loss/teacher EMA) |
| **Global step** | ✅ `__meta__[t_step]` `loop.mojo:195`; argv step `train_krea2.mojo:2858` | ✅ `meta.json.train_progress` `InternalModelSaverMixin.py:36-41` | ✅ `training_state.json` `state_tracker.py:307` | ✅ safetensors metadata `BaseSDTrainProcess.py:399` | ✅ `resume_metadata.json` `train_utils.py:29` |
| **Loss / EMA-loss history** | ❌ none | ❌ loop-local only `GenericTrainer.py:633-634` | ❌ none `state_tracker.py:306-312` | ❌ none | ~ loss *avg* only `hv_train_network.py:3682-3685` |
| **Resume fidelity** | Full `.state` = **bit-exact masters + moments + step** `loop.mojo:205-207`; LoRA train_state restores m/v `lora_save.mojo:326`; **PEFT path = WARM (moments zeroed)** `lora_save.mojo:306-316` | Moments-preserving warm resume; scheduler reconstructed; **no bit-identical claim** (no RNG snapshot) `InternalModelLoaderMixin.py:33-42` | optimizer+scheduler+RNG restored; dataloader coarse → **warm-approximate** `trainer.py:4010` | Moments reloaded **if `optimizer.pt` present**, else silent warm-restart `BaseSDTrainProcess.py:2004-2028` | load_state restores opt+sched+RNG; dataloader approximate; **weights-only = cold** `hv_train_network.py:960,2336-2339` |
| **Cadence** | `save_every` N steps `train_krea2.mojo:3550` | `backup_after` (def 30 min) `TrainConfig.py:547` | `checkpoint_step_interval` (def 500) `logging_fields.py:71` | `save_every` (def 1000) `config_modules.py:25` | `save_every_n_steps/epochs` `hv_train_network.py:3657` |
| **Retention / prune** | rolling keep-N `train_krea2.mojo:365,3557-3561`; **prune AFTER save** | `rolling_backup_count` (def 3) `GenericTrainer.py:182-198` | `checkpoints_total_limit`; **prune BEFORE save** `trainer.py:4544→4552` | `max_step_saves_to_keep` (def 5) `BaseSDTrainProcess.py:404-479` | `save_last_n_steps(_state)` `train_utils.py:227-237`; prune after save |
| **Save-before-risky-op** | ✅ **save-before-sample** explicit `train_krea2.mojo:3547-3550` | ~ backup-before-final-save `GenericTrainer.py:987`; **no save-before-sampling** | ✅ save before eval `trainer.py:5805` | ✅ save before sample `BaseSDTrainProcess.py:2504-2519` | ❌ no crash/signal save |
| **Atomicity** | ❌ `O_WRONLY\|O_CREAT\|O_TRUNC` direct-to-final `safetensors_writer.mojo:215-216` | ❌ direct-to-final; exception-only `rmtree` `InternalModelSaverMixin.py:26,31` | ~ **opt-in** tempdir+`os.rename` (default OFF) `logging_fields.py:127-141` | ❌ direct-to-final; `optimizer.pt` clobbered `network_mixins.py:624` | ❌ direct-to-final `networks/lora.py:893` |
| **Interop (weights)** | ✅ PEFT/kohya, cross-tool verified `lora_save.mojo:91,135` | ✅ OMI/kohya | ✅ PEFT/diffusers | ✅ kohya/PEFT | ✅ kohya + optional `.comfy` |
| **Interop (opt state)** | ✅ **plain safetensors** (inspectable) `loop.mojo:179` | ❌ torch pickle | ❌ accelerate-internal | ❌ torch pickle | ❌ accelerate-internal |
| **UX guard (wrong-artifact resume)** | ✅ **web UI 422-rejects PEFT-as-resume** `webui/src/main.rs:270`; ❌ **CLI silently warm-resumes** `train_krea2.mojo:2119` | ~ bool, silent both directions `GenericTrainer.py:99,112` | ~ latest auto-detect; local resume skips validation `trainer.py:3956-4010` | ~ auto-latest; silent warm-restart fallback `BaseSDTrainProcess.py:2026-2028` | ~ no `--resume`+`--network_weights` guard; non-strict load |

Legend: ✅ full · ~ partial/opt-in/coarse · ❌ absent.

---

## Where we LOSE (ranked by user impact)

1. **krea2 device-fast arm silently warm-restarts (moments reset) on resume.** Only the host arm has full-moment restore; the *default fast* arm resets AdamW moments while presenting as a full resume. This is our worst gap because it is silent and on the fast path. (Known gap, going-in fact; contrast ai-toolkit which at least reloads `optimizer.pt` when present, `BaseSDTrainProcess.py:2004-2023`.)

2. **The warm-resume guard is web-UI-only.** The web UI correctly 422-rejects a PEFT file passed as a full resume with an explanation (`serenity-trainer/webui/src/main.rs:270`, index.html:511). But the **CLI/argv path just prints "WARM start (A/B only, moments zeroed)" and proceeds** (`serenitymojo/models/krea2/train_krea2.mojo:2119,2158`), and `serenitymojo/MAP.md:225-226` records that the 07-02 turbo wrapper *always* warm-resumed, with a code follow-up still open (probe `path+'.state'`). A CLI user who points at the wrong artifact silently loses their moments. OneTrainer (`GenericTrainer.py:99,112`) and ai-toolkit (`BaseSDTrainProcess.py:2026-2028`) are also silent here, so we are not uniquely bad — but our own web layer proves we *know* the right behavior and haven't pushed it into the engine.

3. **No RNG-state snapshot.** SimpleTuner and musubi persist full python/numpy/torch/cuda RNG via accelerate (`random_states_*.pkl`), so any randomness — dropout, augmentation, shuffle draws — replays exactly. We save none. **Mitigating truth:** our training noise is derived from `seed+global_step`, so the *noise stream itself* is reproducible without an RNG file (same design as OneTrainer, `BaseStableDiffusionSetup.py:154`). The loss is real only for randomness *not* keyed to step. Impact: medium, and narrower than it looks.

4. **No dataloader-position save.** OneTrainer restores the exact within-epoch sample index (`DataLoaderMgdsMixin.py:48-49`); SimpleTuner/musubi track it coarsely. We re-derive order deterministically, which is correct only as long as the order derivation is stable across the code/config in effect at resume. Tied with ai-toolkit (which also starts a fresh iterator). Impact: medium — risk of re-showing or skipping samples if the derivation changes.

5. **Not atomic (truncate-in-place).** `save_safetensors` opens the final path with `O_TRUNC` and writes in place (`serenitymojo/io/safetensors_writer.mojo:215-216`) — a crash mid-write leaves a truncated, corrupt checkpoint. This **confirms the previously-unverified gap: we do not temp+rename.** Everyone here defaults non-atomic too (OneTrainer, ai-toolkit, musubi all direct-to-final), so we are tied with most — but **SimpleTuner has an opt-in `checkpointing_use_tempdir` that does `os.rename`** (`trainer.py:5168-5169`) which we lack entirely. Impact: medium.

6. **No LR-scheduler state file.** Warmup is recounted from the optimizer step. Identical posture to OneTrainer (`create.py:1224`) and ai-toolkit (`BaseSDTrainProcess.py:2049`); behind SimpleTuner/musubi (`scheduler.bin`). Correct for stateless warmup/lambda schedules; wrong only for stateful schedulers. Impact: low.

7. **No loss/EMA-loss history.** Resume loses the loss curve. musubi restores a loss *average* (`hv_train_network.py:3682-3685`); everyone else drops it too. Impact: low (logging only).

---

## Where we WIN (each with file:line)

1. **Optimizer state is plain safetensors, not a pickle.** Full-FT `.state` is `param.<i>/adam_m.<i>/adam_v.<i>/__meta__` safetensors (`serenitymojo/training/loop.mojo:179-200`) and LoRA moments are per-adapter `adam_m/adam_v` safetensors (`serenitymojo/training/lora_save.mojo:204-211`). OneTrainer and ai-toolkit write `optimizer.pt` **torch pickles** (`InternalModelSaverMixin.py:21-26`; `BaseSDTrainProcess.py:690-696`); the accelerate tools write internal `optimizer.bin`. Ours is inspectable, portable, and not a pickle-deserialization surface.

2. **Full `.state` resume is a documented bit-for-bit continuation.** `load_checkpoint` rebuilds F32 masters by byte copy and restores m/v/t/accum_count so "a resumed `apply_step` continues exactly where the saved run left off" (`serenitymojo/training/loop.mojo:205-207`), proven eri2 500→2000. Because our noise is `seed+step`-derived, this reproduces the same noise stream **without** needing an RNG snapshot — a determinism-by-construction edge over ai-toolkit.

3. **Save-before-sample is explicit and load-bearing.** "a sampler OOM must never cost the checkpoint" — save precedes sampling at `serenitymojo/models/krea2/train_krea2.mojo:3547-3550` (and :3726, :4090). Matches ai-toolkit's ordering, **better than OneTrainer** (no save-before-sampling; sampling merely deferred, `GenericTrainer.py:714`) and **better than SimpleTuner's prune-before-save** window.

4. **Prune runs AFTER the new save.** Old checkpoints are removed relative to the just-written step (`train_krea2.mojo:3557-3561`), so a prior good checkpoint is never deleted before its replacement exists. **SimpleTuner deletes the old checkpoint before the new one is durable** (`trainer.py:4544` runs before `:4552`) — a real data-loss window we don't have. Tied with musubi (delete-after-save).

5. **The web UI is the only tool here that outright rejects the wrong resume artifact.** A PEFT file passed where a `.state` is required returns HTTP 422 with a plain-language reason (`serenity-trainer/webui/src/main.rs:270`, main.rs:91, static/index.html:511). None of OneTrainer / SimpleTuner-local / ai-toolkit / musubi surface an equivalent "you passed the wrong file for a full resume" error — they silently warm/cold-start. (Caveat: this guard is UI-only; see loss #2.)

6. **EMA shadow weights are persisted (klein).** `save_klein_lora_ema` writes a `*_ema.safetensors` sibling (`serenitymojo/training/lora_ema.mojo:255`, gated by `serenitymojo/training/ema_save_smoke.mojo`) — on par with SimpleTuner's separate `ema_model.pt`, ahead of ai-toolkit (no EMA file) and musubi (no weight EMA). Scoped to the klein path.

7. **Refuse-to-write-empty guards.** Every writer raises rather than clobbering a good checkpoint with an empty tensor set (`lora_save.mojo:106,149,190`; `safetensors_writer.mojo:208`). No reference has an explicit equivalent.

---

## Honest framing
The fairest one-line placement: **we are at the OneTrainer/ai-toolkit tier** (hand-rolled state: moments + step, byte-exact; no RNG / scheduler / dataloader), **behind the accelerate-based SimpleTuner and musubi on breadth of restored state** (they get optimizer + scheduler + RNG in one well-tested bundle), and **ahead of all four on a few specific safety/UX behaviors** (safetensors optimizer state, save-before-sample, prune-after-save, the UI wrong-artifact rejection). The single most user-damaging gap is ours alone: the **krea2 fast-arm silent warm-restart**, compounded by the **CLI-level warm-resume guard being absent** while the web UI has it.
