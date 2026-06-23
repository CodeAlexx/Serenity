# Ideogram4LoRATrainer.mojo — staged full-block LoRA train driver.
#
# This is trainer-owned orchestration around the verified one-step math:
#   stage cache metadata -> load ONE transformer weight set -> stream one sample
#   -> ideogram4_lora_train_step -> save LoRA + Adam state.
#
# It intentionally does not accept List[Tensor] batches. Samples are materialised
# one at a time from Ideogram4TrainCache so the activation-heavy train step does
# not compete with a resident dataset tensor pile.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.os import makedirs
from std.time import perf_counter

from serenitymojo.tensor import Tensor
from serenitymojo.training.levers import (
    caption_dropout_pick,
    levers_optimizer_active,
    levers_optimizer_validate,
    levers_optimizer_eval_for_save,
    levers_optimizer_train_after_save,
)
from serenitymojo.training.lora_ema import (
    LoraEmaState,
    lora_ema_track,
    ema_update,
    ema_shadow_a_bf16,
    ema_shadow_b_bf16,
    ema_path_for_lora,
)
from serenitymojo.training.train_config import TrainConfig as LeversConfig
from serenitymojo.training.schedule import sample_timestep_logit_normal_scaled
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ideogram4_resident import Ideogram4Weights

from serenity_trainer.dataLoader.Ideogram4CacheReader import Ideogram4TrainCache
from serenity_trainer.model.Ideogram4LoRABlock import (
    Ideogram4LoraSet,
    build_ideogram4_native_lora_set,
)
from serenity_trainer.modelLoader.Ideogram4LoRALoader import (
    load_ideogram4_block_stack_lora,
)
from serenity_trainer.modelSaver.Ideogram4LoRAModelSaver import (
    Ideogram4LoRAModelSaver,
)
from serenity_trainer.module.LoRAModule import LoraAdapter
from serenity_trainer.trainer.Ideogram4LoRATrainStep import (
    Ideogram4LoRATrainResult,
    ideogram4_lora_train_step_resident,
    ideogram4_lora_train_compute_resident,
)
from serenity_trainer.trainer.Ideogram4StackTrain import (
    Ideogram4LoraAdamState,
    Ideogram4LeversBridge,
    apply_ideogram4_lora_grads,
    ideogram4_levers_mirrors_init,
    ideogram4_levers_refresh_mirrors,
    ideogram4_levers_optimizer_step,
    make_ideogram4_lora_adam_state,
)
from serenity_trainer.trainer.TrainState import TrainProgress
from serenity_trainer.trainer.cadence.SampleCadence import SampleCadence
from serenity_trainer.trainer.Ideogram4SampleResident import (
    ideogram4_sample_resident,
    ideogram4_decode_latent_to_png,
)
from serenity_trainer.util.enum.TimeUnit import TU_STEP
from serenity_trainer.util.config.TrainConfig import TrainConfig

from serenitymojo.ops.random import randn


comptime TArc = ArcPointer[Tensor]
comptime _I4_STATE_FILE = "ideogram4_train_state.safetensors"


@fieldwise_init
struct Ideogram4LoRATrainRunConfig(Copyable, Movable):
    var transformer_path: String
    var cache_path: String
    var output_dir: String
    var resume_lora_path: String
    var resume_state_dir: String
    var steps: Int
    var default_t_flow: Float32
    var save_every_steps: Int
    var checkpoint_every_steps: Int
    var noise_seed: UInt64
    var lora_seed: UInt64
    var progress_file_path: String
    # T1.D caption dropout probability (default-off 0.0; serenity-trainer's
    # TrainConfig does not carry caption_dropout_prob, so the run config owns
    # it). When the pick fires, the step trains on cache.uncond[NT] (the
    # llm_uncond empty-caption features) instead of the sample's llm features.
    # 0.0 here falls back to levers.caption_dropout_prob (one knob wins).
    var caption_dropout_prob: Float32
    # T1 levers carrier (serenitymojo TrainConfig): loss_fn/huber_delta/
    # smooth_l1_beta/min_snr_gamma_flow (T1.A), ema_* (T1.B), optimizer/
    # optimizer_* (T1.C), caption_dropout_prob fallback (T1.D). Defaults are
    # ALL default-off (C13) == the pre-lever trainer byte-for-byte. The shared
    # recipe scalars (lr/beta1/beta2/eps/weight_decay/rank/alpha) are SYNCED
    # from the serenity-trainer TrainConfig at run start — that struct stays
    # the single source of truth for them.
    var levers: LeversConfig
    # ── sample-during-training (v1, SampleCadence-wired) ──────────────────────
    # sample_every_steps: TU_STEP cadence interval (0 = disabled, default-off so
    #   the pre-sampling trainer is byte-for-byte unchanged). On fire, for each
    #   prompt index the trainer denoises a sample from the CURRENT resident base
    #   + live LoRA and writes <output_dir>/samples/step_<N>_<promptidx>.png.
    # sample_steps / sample_cfg: the denoise loop length + CFG scale (inference
    #   defaults 8 / 7.0 — ideogram4_pipeline.mojo:35-36).
    # sample_seed: base RNG seed for the t=1 init noise (per-prompt offset added).
    # V1 CONDITIONING (flagged): the prompt list is carried for forward-compat but
    #   each sample is CONDITIONED ON A CACHED CAPTION's llm_features (the cache
    #   has no arbitrary-prompt encoder wired). prompt index i -> cache sample
    #   (i % cache.len()). See Ideogram4SampleResident.mojo header for the why.
    var sample_every_steps: Int
    var sample_steps: Int
    var sample_cfg: Float32
    var sample_seed: UInt64
    var sample_prompts: List[String]

    @staticmethod
    def defaults(
        transformer_path: String,
        cache_path: String,
        output_dir: String,
    ) -> Ideogram4LoRATrainRunConfig:
        return Ideogram4LoRATrainRunConfig(
            transformer_path=transformer_path,
            cache_path=cache_path,
            output_dir=output_dir,
            resume_lora_path=String(""),
            resume_state_dir=String(""),
            steps=10,
            default_t_flow=Float32(0.7),
            save_every_steps=0,
            checkpoint_every_steps=0,
            noise_seed=UInt64(0x1D3A_4A11),
            lora_seed=UInt64(0x1D3A_4000),
            progress_file_path=String(""),
            caption_dropout_prob=Float32(0.0),
            levers=LeversConfig.default(),
            sample_every_steps=0,          # default-off: no sampling-in-training
            sample_steps=8,                # inference default (pipeline STEPS)
            sample_cfg=Float32(7.0),       # inference default (pipeline CFG)
            sample_seed=UInt64(0x1D3A_5A91),
            sample_prompts=List[String](),
        )


@fieldwise_init
struct Ideogram4LoRATrainSummary(Copyable, Movable):
    var steps_ran: Int
    var cache_samples: Int
    var optimizer_steps: Int
    var last_loss: Float32
    var adapter_b_l1: Float32
    var elapsed_seconds: Float64
    var seconds_per_step: Float64
    var lora_path: String
    var state_dir: String
    var loaded_weight_sets: Int
    var progress: TrainProgress


@fieldwise_init
struct Ideogram4LoadedLoRAState(Copyable, Movable):
    var progress: TrainProgress
    var opt_step: Int


def train_ideogram4_lora_from_cache[NT: Int, GH: Int, GW: Int](
    cfg: TrainConfig,
    run_cfg: Ideogram4LoRATrainRunConfig,
    ctx: DeviceContext,
) raises -> Ideogram4LoRATrainSummary:
    if run_cfg.steps < 1:
        raise Error("train_ideogram4_lora_from_cache: steps must be >= 1")

    # ── T1 levers config (TIER1_PARITY_CAMPAIGN_2026-06-11.md) ────────────────
    # run_cfg.levers carries the LEVER keys; the shared optimizer/LoRA recipe
    # scalars are synced FROM the serenity-trainer cfg (the struct argv/gates
    # already control) so there is exactly one knob per scalar.
    var lcfg = run_cfg.levers.copy()
    lcfg.lr = cfg.learning_rate
    lcfg.beta1 = cfg.beta1
    lcfg.beta2 = cfg.beta2
    lcfg.eps = cfg.eps
    lcfg.weight_decay = cfg.weight_decay
    lcfg.lora_rank = cfg.lora_rank
    lcfg.lora_alpha = cfg.lora_alpha
    if lcfg.masked_training:
        raise Error(
            "train_ideogram4_lora_from_cache: masked_training is set but the"
            " ideogram4 stager emits no masks — masked loss (T1.E) is not"
            " wired for this trainer"
        )
    levers_optimizer_validate(lcfg, String("Ideogram4 trainer"))
    var levers_opt = levers_optimizer_active(lcfg)
    if levers_opt:
        print(
            "[Ideogram4-lora] T1.C levers optimizer active: tag=",
            lcfg.optimizer,
            " (2=ADAFACTOR, 7=SCHEDULE_FREE_ADAMW) warmup=",
            lcfg.optimizer_warmup_steps,
        )
    # effective caption-dropout p: run_cfg owns it (T1.D precedent); 0.0 falls
    # back to the levers key so a config-file value still reaches the pick.
    var drop_p = run_cfg.caption_dropout_prob
    if drop_p <= Float32(0.0):
        drop_p = lcfg.caption_dropout_prob

    makedirs(run_cfg.output_dir, exist_ok=True)
    var final_lora_path = _final_lora_path(run_cfg.output_dir)
    var final_state_dir = _final_state_dir(run_cfg.output_dir)

    # Stage 1: cache metadata only. Samples are loaded inside the loop one at a
    # time; no dataset tensor list is kept resident.
    var cache = Ideogram4TrainCache.open(run_cfg.cache_path)

    # Stage 2: exactly one resident FP8 transformer weight set for training.
    # This matches the fast Ideogram inference path: no per-step safetensors
    # reads, and no per-step H2D transfer of the frozen trunk.
    var weights = Ideogram4Weights.load(
        ShardedSafeTensors.open(run_cfg.transformer_path), ctx
    )

    # Stage 3: LoRA + optimizer state. Resume loads adapter weights first, then
    # rebases Adam moments/progress if a state dir was provided.
    var loras = _load_or_build_loras(
        run_cfg.resume_lora_path,
        cfg.lora_rank,
        cfg.lora_alpha,
        run_cfg.lora_seed,
        ctx,
    )
    var opt = make_ideogram4_lora_adam_state(loras, ctx)
    var progress = TrainProgress()
    var opt_step = 0
    if run_cfg.resume_state_dir.byte_length() > 0:
        var loaded = load_ideogram4_lora_train_state(
            run_cfg.resume_state_dir, opt, ctx
        )
        progress = loaded.progress.copy()
        opt_step = loaded.opt_step

    # T1.C levers bridge (host mirrors + optimizer state; nothing allocated on
    # the default AdamW path unless EMA needs the mirrors) + T1.B EMA shadows.
    # EMA tracks AFTER any resume load (shadow init = clone of current params,
    # SimpleTuner ema.py:123 semantics via training/lora_ema.mojo).
    var bridge = Ideogram4LeversBridge()
    var ema = LoraEmaState(
        lcfg.ema_decay, lcfg.ema_min_decay,
        lcfg.ema_update_after_step, lcfg.ema_update_step_interval,
    )
    if lcfg.ema_enabled or levers_opt:
        ideogram4_levers_mirrors_init(bridge, loras, ctx)
    if lcfg.ema_enabled:
        var ema_base = lora_ema_track(ema, bridge.mirrors, 0, len(bridge.mirrors))
        if ema_base != 0:
            raise Error("train_ideogram4_lora_from_cache: ema shadow base must be 0")
        print(
            "[Ideogram4-lora] T1.B EMA tracking", len(bridge.mirrors),
            "adapters decay=", lcfg.ema_decay,
            " min_decay=", lcfg.ema_min_decay,
            " update_after_step=", lcfg.ema_update_after_step,
            " interval=", lcfg.ema_update_step_interval,
        )

    var last_loss = Float32(0.0)
    var last_b = Float32(0.0)
    var train_start = perf_counter()
    var smooth_loss = Float32(0.0)
    var smooth_inited = False

    # ── sample-during-training cadence (default-off when sample_every_steps==0) ─
    # SampleCadence drives "should we sample now?" on the SAME TrainProgress that
    # threads through the loop. TU_STEP + start_at_zero=True: fires when
    # global_step % sample_every_steps == 0. We check it AFTER progress.next_step
    # so the just-completed optimizer step is reflected in the sampled LoRA state.
    var sample_enabled = run_cfg.sample_every_steps > 0
    var samples_dir = run_cfg.output_dir + String("/samples")
    if sample_enabled:
        makedirs(samples_dir, exist_ok=True)
    # prompt list: carried for forward-compat. If the caller gave none but enabled
    # sampling, default to ONE prompt so the path still fires (its conditioning is
    # cache sample 0 — see the V1 CONDITIONING note in the run-config fields).
    var sample_prompt_list = run_cfg.sample_prompts.copy()
    if sample_enabled and len(sample_prompt_list) == 0:
        sample_prompt_list.append(String("(cached caption 0)"))
    var cadence = SampleCadence(
        Float64(run_cfg.sample_every_steps),
        TU_STEP,
        Float64(0.0),                      # sample_skip_first: no delay
        sample_prompt_list^,
    )

    for local_step in range(run_cfg.steps):
        var sample_index = progress.global_step % cache.len()
        var seed = run_cfg.noise_seed + UInt64(opt_step + local_step)
        # Per-step flow time ~ logit-normal(0, 1.0) = t = sigmoid(N(0,1)),
        # matching BOTH references: OneTrainer (ModelSetupNoiseMixin LOGIT_NORMAL,
        # scale = noising_weight+1.0 = 1.0 with the ideogram preset's defaults) and
        # the Rust train_ideogram (t = sigmoid(u)). Was 1.5 (DiffSynth INFERENCE
        # set_timesteps_ideogram4) — wrong vs both training oracles.
        # The old fixed default_t_flow=0.7 trained EVERY step at one timestep
        # (measured: loss collapsed to 1.3e-4, grad_norm 0 — learned nothing).
        # Separate RNG stream from the noise draw (the zimage *7919 idiom).
        var t_step = sample_timestep_logit_normal_scaled(
            run_cfg.noise_seed * UInt64(7919) + UInt64(opt_step + local_step),
            Float32(1.0),
        )
        var sample = cache.sample[NT, GH, GW](
            sample_index, t_step, seed, ctx
        )

        # T1.D caption dropout (default-off p<=0 never draws): shared levers
        # pick on the noise_seed stream; when it fires, train this step on the
        # cached empty-caption llm_uncond features (fail-loud if the cache
        # predates the --uncond stager).
        var llm_in = sample.llm_features.copy()
        if caption_dropout_pick(
            UInt64(opt_step + local_step),
            run_cfg.noise_seed,
            drop_p,
        ):
            llm_in = ArcPointer[Tensor](cache.uncond[NT](ctx))

        # forward + loss (T1.A lever seam inside) + backward — NO optimizer.
        var step_loss = Float32(0.0)
        var grads = ideogram4_lora_train_compute_resident[NT, GH, GW](
            weights,
            sample.noisy[],
            sample.clean[],
            sample.noise[],
            sample.t_flow,
            llm_in[],
            loras,
            lcfg,
            step_loss,
            ctx,
        )
        # Real gradient L1 for the progress line. The apply/levers telemetry
        # returns were stubbed (apply_ideogram4_lora_grads returned grad_b_l1=0.0,
        # adapter_b_l1=0.0) and the progress line hardcoded "grad_norm 0.0000" —
        # so the trainer looked dead even though LoRA-B is learning. grads.d_b
        # holds the per-adapter LoRA-B gradients; to_host upcasts to Float32.
        # Computed BEFORE the optimizer consumes `grads` (grads^), once for both
        # the default-AdamW and levers paths.
        var step_grad_l1 = Float32(0.0)
        for gi in range(len(grads.d_b)):
            var gh = grads.d_b[gi][].to_host(ctx)
            for gj in range(len(gh)):
                var gv = gh[gj]
                if gv < Float32(0.0):
                    step_grad_l1 -= gv
                else:
                    step_grad_l1 += gv
        # T1.C optimizer seam: levers host path vs the existing literal fused
        # AdamW call (C13: optimizer=ADAMW routes around the levers entirely).
        var k = opt_step + 1
        var step_b_l1 = Float32(0.0)
        if levers_opt:
            step_b_l1 = ideogram4_levers_optimizer_step(
                lcfg, loras, bridge, grads, k,
                cfg.learning_rate, ctx,
            )
            _ = grads^
        else:
            var res = apply_ideogram4_lora_grads(
                loras, opt, grads^, k, cfg, ctx
            )
            step_b_l1 = res.adapter_b_l1
        # T1.B EMA, post-optimizer: the default AdamW stepped the params on
        # device, so refresh the host mirrors first; the levers optimizer
        # keeps the mirrors authoritative already.
        if lcfg.ema_enabled:
            if not levers_opt:
                ideogram4_levers_refresh_mirrors(bridge, loras, ctx)
            ema_update(ema, bridge.mirrors, k)
        last_loss = step_loss
        last_b = step_b_l1
        opt_step += 1
        progress.next_step(cfg.batch_size)
        if not smooth_inited:
            smooth_loss = step_loss
            smooth_inited = True
        else:
            smooth_loss = smooth_loss * Float32(0.99) + step_loss * Float32(0.01)

        if run_cfg.progress_file_path.byte_length() > 0:
            var elapsed = perf_counter() - train_start
            var speed = elapsed / Float64(local_step + 1)
            _append_ideogram4_live_progress(
                run_cfg.progress_file_path,
                progress,
                run_cfg.steps,
                cfg,
                last_loss,
                smooth_loss,
                step_grad_l1,
                Float32(speed),
                elapsed,
            )

        if run_cfg.save_every_steps > 0 and opt_step % run_cfg.save_every_steps == 0:
            # schedule-free save bracket (no-op for every other optimizer —
            # levers.mojo SAVE CONTRACT) around the product save + EMA sibling.
            levers_optimizer_eval_for_save(lcfg, bridge.opt_st)
            var step_path = _step_lora_path(run_cfg.output_dir, opt_step)
            _save_lora(loras, step_path, ctx)
            if lcfg.ema_enabled:
                _save_lora_ema(ema, loras, step_path, ctx)
            levers_optimizer_train_after_save(lcfg, bridge.opt_st)

        if (
            run_cfg.checkpoint_every_steps > 0
            and opt_step % run_cfg.checkpoint_every_steps == 0
        ):
            levers_optimizer_eval_for_save(lcfg, bridge.opt_st)
            var ckpt_dir = _step_state_dir(run_cfg.output_dir, opt_step)
            save_ideogram4_lora_train_state(ckpt_dir, opt, progress, opt_step, ctx)
            levers_optimizer_train_after_save(lcfg, bridge.opt_st)

        # ── sample-during-training (fail-loud; default-off) ───────────────────
        # Checked AFTER next_step so global_step reflects the just-completed step.
        # cadence.should_sample mutates the cadence clock; only fires when
        # sample_every_steps>0 (sample_enabled) AND the TU_STEP interval is hit.
        if sample_enabled and cadence.should_sample(progress):
            for pi in range(cadence.num_prompts()):
                _ideogram4_run_sample[NT, GH, GW](
                    cache, weights, loras, run_cfg, samples_dir,
                    progress.global_step, pi, ctx,
                )

    var train_elapsed = perf_counter() - train_start
    var seconds_per_step = train_elapsed / Float64(run_cfg.steps)

    levers_optimizer_eval_for_save(lcfg, bridge.opt_st)
    _save_lora(loras, final_lora_path, ctx)
    if lcfg.ema_enabled:
        _save_lora_ema(ema, loras, final_lora_path, ctx)
    save_ideogram4_lora_train_state(final_state_dir, opt, progress, opt_step, ctx)
    levers_optimizer_train_after_save(lcfg, bridge.opt_st)

    return Ideogram4LoRATrainSummary(
        run_cfg.steps,
        cache.len(),
        opt_step,
        last_loss,
        last_b,
        train_elapsed,
        seconds_per_step,
        final_lora_path,
        final_state_dir,
        1,
        progress.copy(),
    )


def save_ideogram4_lora_train_state(
    dir: String,
    opt: Ideogram4LoraAdamState,
    progress: TrainProgress,
    opt_step: Int,
    ctx: DeviceContext,
) raises:
    makedirs(dir, exist_ok=True)
    var names = List[String]()
    var tensors = List[TArc]()

    if (
        len(opt.m_a) != len(opt.v_a)
        or len(opt.m_a) != len(opt.m_b)
        or len(opt.m_a) != len(opt.v_b)
    ):
        raise Error("save_ideogram4_lora_train_state: Adam state list mismatch")

    for i in range(len(opt.m_a)):
        names.append(_state_key(i, String("m_a")))
        tensors.append(TArc(opt.m_a[i][].clone(ctx)))
        names.append(_state_key(i, String("v_a")))
        tensors.append(TArc(opt.v_a[i][].clone(ctx)))
        names.append(_state_key(i, String("m_b")))
        tensors.append(TArc(opt.m_b[i][].clone(ctx)))
        names.append(_state_key(i, String("v_b")))
        tensors.append(TArc(opt.v_b[i][].clone(ctx)))

    var meta_vals = List[Float32]()
    meta_vals.append(Float32(progress.epoch))
    meta_vals.append(Float32(progress.epoch_step))
    meta_vals.append(Float32(progress.epoch_sample))
    meta_vals.append(Float32(progress.global_step))
    meta_vals.append(Float32(opt_step))
    var meta = Tensor.from_host(meta_vals^, [5], STDtype.F32, ctx)
    names.append(String("train_progress"))
    tensors.append(TArc(meta^))

    save_safetensors(names^, tensors^, _state_file(dir), ctx)


def load_ideogram4_lora_train_state(
    dir: String,
    mut opt: Ideogram4LoraAdamState,
    ctx: DeviceContext,
) raises -> Ideogram4LoadedLoRAState:
    var src = ShardedSafeTensors.open(_state_file(dir))
    var meta = Tensor.from_view(src.tensor_view(String("train_progress")), ctx).to_host(ctx)
    if len(meta) != 5:
        raise Error("load_ideogram4_lora_train_state: malformed train_progress meta")

    var expected = len(opt.m_a)
    var n = 0
    while _has_state_slot(src, n):
        n += 1
    if n != expected:
        raise Error(
            String("load_ideogram4_lora_train_state: slot mismatch have ")
            + String(expected) + String(" checkpoint ") + String(n)
        )

    for i in range(expected):
        opt.m_a[i] = TArc(Tensor.from_view(src.tensor_view(_state_key(i, String("m_a"))), ctx))
        opt.v_a[i] = TArc(Tensor.from_view(src.tensor_view(_state_key(i, String("v_a"))), ctx))
        opt.m_b[i] = TArc(Tensor.from_view(src.tensor_view(_state_key(i, String("m_b"))), ctx))
        opt.v_b[i] = TArc(Tensor.from_view(src.tensor_view(_state_key(i, String("v_b"))), ctx))

    var progress = TrainProgress(
        Int(meta[0]), Int(meta[1]), Int(meta[2]), Int(meta[3])
    )
    return Ideogram4LoadedLoRAState(progress.copy(), Int(meta[4]))


# ──────────────────────────────────────────────────────────────────────────────
# _ideogram4_run_sample — one sample-during-training image.
#   cond conditioning : cached caption (prompt_idx % cache.len()) llm_features
#                       (v1 — see the run-config sample_* field docs).
#   uncond conditioning: a zeroed [1,NT,53248] bf16 tensor (CFG empty cond, same
#                       as the inference pipeline's neg_llm).
#   init noise        : randn [1,128,GH,GW] F32, seed = sample_seed + step*1000 +
#                       prompt_idx (deterministic per (step,prompt)).
#   denoise           : ideogram4_sample_resident (resident base + live LoRA).
#   decode + write    : ideogram4_decode_latent_to_png ->
#                       <samples_dir>/step_<N>_<promptidx>.png.
# Fail-loud: any raise propagates (no silent skip), per the build request.
def _ideogram4_run_sample[NT: Int, GH: Int, GW: Int](
    cache: Ideogram4TrainCache,
    weights: Ideogram4Weights,
    loras: Ideogram4LoraSet,
    run_cfg: Ideogram4LoRATrainRunConfig,
    samples_dir: String,
    step: Int,
    prompt_idx: Int,
    ctx: DeviceContext,
) raises:
    comptime PACKED_CH = 128

    # COND conditioning: a cached caption's llm_features [1,NT,53248] bf16. We
    # pull it via cache.sample (the validated accessor); the clean/noise it also
    # loads are unused here (cheap relative to the denoise).
    var cond_index = prompt_idx % cache.len()
    var cond_sample = cache.sample[NT, GH, GW](
        cond_index, Float32(0.0), run_cfg.sample_seed, ctx
    )
    var cond_llm = cond_sample.llm_features[].clone(ctx)   # [1,NT,53248] bf16

    # UNCOND conditioning: zeroed text features (CFG empty cond), same dtype.
    var uncond_llm = zeros_device(
        [1, NT, cond_llm.shape()[2]], cond_llm.dtype(), ctx
    )

    # t=1 init noise [1,128,GH,GW] F32, deterministic per (step, prompt).
    var seed = run_cfg.sample_seed + UInt64(step * 1000 + prompt_idx)
    var init_noise = randn([1, PACKED_CH, GH, GW], seed, STDtype.F32, ctx)

    var latent = ideogram4_sample_resident[NT, GH, GW](
        weights, loras, cond_llm, uncond_llm, init_noise,
        run_cfg.sample_steps, run_cfg.sample_cfg, GH, GW, ctx,
    )

    var out_path = (
        samples_dir + String("/step_") + String(step)
        + String("_") + String(prompt_idx) + String(".png")
    )
    ideogram4_decode_latent_to_png[GH, GW](latent, out_path, ctx)
    print(
        "[Ideogram4-lora] sample step=", step, " prompt=", prompt_idx,
        " cond_cache_idx=", cond_index, " -> ", out_path,
    )


def _load_or_build_loras(
    resume_lora_path: String,
    rank: Int,
    alpha: Float32,
    seed: UInt64,
    ctx: DeviceContext,
) raises -> Ideogram4LoraSet:
    if resume_lora_path.byte_length() > 0:
        return load_ideogram4_block_stack_lora(resume_lora_path, ctx)
    return build_ideogram4_native_lora_set(rank, alpha, ctx, seed=seed)


def _save_lora(loras: Ideogram4LoraSet, path: String, ctx: DeviceContext) raises:
    var saver = Ideogram4LoRAModelSaver()
    saver.save_block_stack_lora(loras, path, ctx)


# T1.B: save the EMA shadow set as the *_ema.safetensors sibling of a
# just-saved LoRA, through the SAME product writer (zimage's
# _save_zimage_lora_ema precedent). Shadows are flat-indexed 1:1 with
# loras.ad (lora_ema_track tracked the full mirror set, base 0); the bf16
# export round is lora_ema.mojo's copy_to cast (ema.py:454).
def _save_lora_ema(
    ema: LoraEmaState, loras: Ideogram4LoraSet, lora_path: String,
    ctx: DeviceContext,
) raises:
    var ad = List[ArcPointer[LoraAdapter]]()
    for i in range(len(loras.ad)):
        var a_t = Tensor.from_host_bf16(
            ema_shadow_a_bf16(ema, i), loras.ad[i][].a.shape(), ctx
        )
        var b_t = Tensor.from_host_bf16(
            ema_shadow_b_bf16(ema, i), loras.ad[i][].b.shape(), ctx
        )
        ad.append(
            ArcPointer[LoraAdapter](
                LoraAdapter(
                    a_t^, b_t^, loras.ad[i][].rank, loras.ad[i][].alpha
                )
            )
        )
    var ema_set = Ideogram4LoraSet(ad^, loras.n_layers, loras.rank)
    var ema_path = ema_path_for_lora(lora_path)
    _save_lora(ema_set, ema_path, ctx)
    print("[Ideogram4-lora] save_ema path=", ema_path)


def _append_ideogram4_live_progress(
    path: String,
    progress: TrainProgress,
    max_steps: Int,
    cfg: TrainConfig,
    loss: Float32,
    smooth_loss: Float32,
    grad_norm: Float32,
    speed_seconds: Float32,
    elapsed_seconds: Float64,
) raises:
    var eta_seconds = Float64(max_steps - progress.epoch_step) * Float64(speed_seconds)
    if eta_seconds < 0.0:
        eta_seconds = 0.0
    var line = (
        String("[Ideogram4-lora] model IDEOGRAM_4 | type LoRA | step ")
        + String(progress.epoch_step)
        + String("/")
        + String(max_steps)
        + String(" | epoch ")
        + String(progress.epoch + 1)
        + String("/1 | loss ")
        + String(loss)
        + String(" | smooth_loss ")
        + String(smooth_loss)
        + String(" | grad_norm ")
        + String(grad_norm)
        + String(" | ")
        + String(speed_seconds)
        + String("s/step | elapsed ")
        + _format_hms(elapsed_seconds)
        + String(" | ETA ")
        + _format_hms(eta_seconds)
    )
    var f = open(path, "a")
    f.write(line)
    f.write("\n")
    f.close()
    print(line)


def _format_hms(seconds_f: Float64) -> String:
    var total = Int(seconds_f)
    if total < 0:
        total = 0
    var hours = total // 3600
    var rem = total - hours * 3600
    var mins = rem // 60
    var secs = rem - mins * 60
    return String(hours) + String(":") + _two_digits(mins) + String(":") + _two_digits(secs)


def _two_digits(v: Int) -> String:
    if v < 10:
        return String("0") + String(v)
    return String(v)


def _state_key(i: Int, part: String) -> String:
    return String("adapter.") + String(i) + String(".") + part


def _has_state_slot(src: ShardedSafeTensors, i: Int) -> Bool:
    return _state_key(i, String("m_a")) in src.name_to_shard


def _state_file(dir: String) -> String:
    return dir + String("/") + String(_I4_STATE_FILE)


def _final_lora_path(output_dir: String) -> String:
    return output_dir + String("/lora_last.safetensors")


def _final_state_dir(output_dir: String) -> String:
    return output_dir + String("/state_last")


def _step_lora_path(output_dir: String, step: Int) -> String:
    return output_dir + String("/lora_step_") + String(step) + String(".safetensors")


def _step_state_dir(output_dir: String, step: Int) -> String:
    return output_dir + String("/state_step_") + String(step)
