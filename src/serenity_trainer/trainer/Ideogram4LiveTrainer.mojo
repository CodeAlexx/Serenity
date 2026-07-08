# Ideogram4LiveTrainer.mojo — UI-launched live trainer process.
#
# argv:
#   1 progress_file
#   2 transformer_safetensors
#   3 cache_safetensors
#   4 output_dir
#   5 steps
#   6 rank
#   7 alpha
#   8 learning_rate
#   9 save_every_steps
#  10 caption_dropout_prob   (T1.D; "-" or absent = 0.0 = never drop)
#  11 levers_config_json     (T1 levers: serenitymojo train-config JSON read by
#                             io/train_config_reader read_model_config — the
#                             same format trainer_ui_runner_train_config_json
#                             emits for the config-driven runners; carries
#                             loss_fn/min_snr_gamma_flow/ema_*/optimizer*/
#                             caption_dropout_prob; "-" or absent = all
#                             default-off). NOTE (gap, documented):
#                             TrainerRuntimeBridge currently launches this
#                             runner with only argv 1-9, so the UI's lever
#                             widgets do NOT reach ideogram4 until the bridge
#                             appends argv 10/11 (bridge is not owned by this
#                             wiring pass). The recipe scalars argv already
#                             carries (lr/rank/alpha/steps/save) keep winning
#                             over the JSON — the JSON contributes ONLY the
#                             lever keys (Ideogram4LoRATrainer syncs the shared
#                             scalars from the argv-built TrainConfig).
#  12 ft_resume_overlay      (FULL-FT arm ONLY, -D IDEOGRAM4_FULL_FT=1: a prior
#                             FT run's saved overlay to resume from; "-" or
#                             absent = fresh run. Parsed ONLY inside the
#                             comptime FT gate — inert on default builds.)
#  13 ft_sample_every        (FULL-FT arm ONLY: inline sample cadence — sample
#                             the LIVE host store every N steps; "-" or absent
#                             = off; must be a positive int. Parsed ONLY inside
#                             the comptime FT gate — inert on default builds.
#                             argv 14 is RESERVED for a sample-prompt/
#                             conditioning path; v1 conditions on the cached
#                             llm features — the model's resident-sampler
#                             contract, no live TE — so it is not wired.)
#
# The UI launches this as a background process. Progress is written as
# Serenity-shaped callback lines so TrainerRuntimeBridge can tail the file.

from std.gpu.host import DeviceContext
from std.os import makedirs
from std.sys import argv

from serenitymojo.io.train_config_reader import read_model_config

from serenity_trainer.trainer.Ideogram4LoRATrainer import (
    Ideogram4LoRATrainRunConfig,
    train_ideogram4_lora_from_cache,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig

# ── FULL FINETUNE arm (-D IDEOGRAM4_FULL_FT=1; FULL_FINETUNE_ROLLOUT_PLAN_
# 2026-07-07 ideogram4 card — the krea2/chroma `ot-mojo-full-finetune`
# blueprint; mirrors train_chroma_real.mojo `_chroma_full_ft_run`). Default 0 =
# every LoRA path below byte-unchanged (C13 gate-don't-fork). ───────────────
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    reshape, slice, permute, mul_scalar, add, zeros_device,
)
from serenitymojo.models.dit.ideogram4_resident import Ideogram4Weights
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope
from serenitymojo.models.ideogram4.ideogram4_full_ft import (
    ideogram4_host_bf16_overlay_resume, ideogram4_ft_state_shapes,
    Ideogram4HostBf16,
    build_ideogram4_host_bf16,
    build_ideogram4_ft_adafactor_states,
    ideogram4_stack_ft_forward_streamed,
    ideogram4_stack_ft_backward_streamed,
    ideogram4_host_bf16_save,
)
# FULL-FT resume sidecar (the fleet helper): adafactor row/col states + t_step
# + seed_base round-trip; sidecar path derived from the overlay path.
from serenitymojo.training.full_ft_sidecar import (
    full_ft_sidecar_save, full_ft_sidecar_load,
    full_ft_sidecar_path_for_overlay,
)
from serenitymojo.training.schedule import sample_timestep_logit_normal_scaled
# FULL-FT inline sampling (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #6):
# denoise from the LIVE pinned-host bf16 store via the FT streamed forward;
# schedule + decode are the LoRA sampler's parity-gated pieces
# (Ideogram4SampleResident / ideogram4_pipeline chunk 9).
from serenitymojo.io.cap_cache import save_tensor_bin
from serenitymojo.ops.random import randn
from serenitymojo.sampling.ideogram4_schedule import (
    ideogram4_logitnormal, ideogram4_schedule_mean, make_step_intervals,
)
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current

from serenity_trainer.dataLoader.Ideogram4CacheReader import Ideogram4TrainCache
from serenity_trainer.model.Ideogram4Predict import (
    ideogram4_build_packed_inputs,
    ideogram4_flow_target,
)
from serenity_trainer.trainer.Ideogram4LoRATrainStep import (
    ideogram4_lora_embed_resident,
    ideogram4_lora_final_forward_resident,
    ideogram4_lora_final_backward,
)
from serenity_trainer.trainer.Ideogram4SampleResident import (
    ideogram4_decode_latent_to_png,
)
from serenity_trainer.modelSampler.Ideogram4Sampler import (
    IDEOGRAM4_NUM_LAYERS,
    IDEOGRAM4_NUM_HEADS,
    IDEOGRAM4_HEAD_DIM,
    IDEOGRAM4_HIDDEN,
    IDEOGRAM4_INTERMEDIATE_SIZE,
    IDEOGRAM4_ADALN_DIM,
    IDEOGRAM4_PACKED_CHANNELS,
    IDEOGRAM4_MROPE_SECTION_0,
    IDEOGRAM4_MROPE_SECTION_1,
    IDEOGRAM4_MROPE_SECTION_2,
    IDEOGRAM4_MROPE_THETA,
)

comptime IDEOGRAM4_FULL_FT = get_defined_int["IDEOGRAM4_FULL_FT", 0]() != 0


comptime DEFAULT_TRANSFORMER = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime DEFAULT_CACHE = "/home/alex/trainings/ideogram4_giger_cache/cache.safetensors"
comptime DEFAULT_OUTPUT = "/home/alex/mojodiffusion/output"
comptime DEFAULT_PROGRESS = "target/serenity_trainer_progress.log"

comptime NT = 256   # giger 512px cache bucket (prepare pads ids to 256)
comptime GH = 32    # 512px -> packed 32x32
comptime GW = 32


def _clear_progress(path: String) raises:
    var f = open(path, "w")
    f.write("")
    f.close()


def _append_status(path: String, text: String) raises:
    var f = open(path, "a")
    f.write(
        String("[Serenity-callback] progress epoch 0/1 | step 0/1 | global_step 0 | loss 0.0 | smooth_loss 0.0 | grad_norm 0.0 | lr 0.0 | status ")
        + text
    )
    f.write("\n")
    f.close()


# ══════════════════════════════════════════════════════════════════════════
# FULL FINETUNE run (self-contained; called from main behind IDEOGRAM4_FULL_FT).
#
# Mirrors _chroma_full_ft_run (train_chroma_real.mojo:524-661): cache fetch →
# ideogram4's OWN aitk sigma policy (t = logit-normal scale 1.0 = sigmoid(N(0,1)),
# the LoRA loop's exact dispatch — Ideogram4LoRATrainer.mojo:315-318, separate
# *7919 RNG stream) → flow-match in the PACKED latent space (cache clean
# [1,128,GH,GW]; noisy = (1-t)*clean + t*noise via the cache reader's
# ideogram4_add_noise; target = noise - clean via ideogram4_flow_target —
# Ideogram4Predict.mojo:244-267, aitk-audited row 8 of the parity ledger) →
# cond glue = the LoRA loop's exact packing+embed (ideogram4_build_packed_inputs
# + mrope + ideogram4_lora_embed_resident — Ideogram4LoRATrainStep.mojo:466-498)
# → streamed FT forward from the pinned-host bf16 store → ideogram4's own MSE
# loss (loss = mean((velocity-target)^2), d = (2/N)(velocity-target) — the
# literal default path of _i4_flow_loss_and_dvel, Ideogram4LoRATrainStep.mojo:
# 601-610, computed host-side like the chroma FT arm) → final-frozen backward
# (ideogram4_lora_final_backward) → streamed FT backward with fused device
# Adafactor (b2d=-0.8, eps2=1e-3, d=1.0, wd=0) + SR + write-back → host-direct
# safetensors overlay save (ORIGINAL layers.N.* key names). Bypasses the
# ENTIRE LoRA machinery (no LoRA set, no AdamW state, no levers).
#
# Frozen GLOBALS (embedders input_proj/t_embedding/adaln_proj/llm_cond_*/
# embed_image_indicator + final_layer) stay DEVICE-RESIDENT as an
# Ideogram4Weights SUBSET (every non-"layers." checkpoint tensor, fp8 kept
# fp8-resident + per-row F32 scale exactly like Ideogram4Weights.load —
# ideogram4_resident.mojo:38-50; the per-step forward dequants through the
# parity-gated _resident_w_fp8 path). The 34 layers' weights live ONLY in the
# pinned-host bf16 store (built fp8→bf16 via load_fp8_dequant).
#
# v1 fail-loud scope: b1 only (this runner has no batch/accum argv — enforced
# by construction), no EMA, no caption dropout, no levers, no grad clip
# (Adafactor d=1.0 update clipping is the OT recipe), save at run end only.
# ══════════════════════════════════════════════════════════════════════════
comptime I4_FT_SEED = UInt64(0x1D3A_4A11)   # the LoRA run-config noise_seed default

# FT inline-sampling denoise recipe = the inference/LoRA-sampler defaults
# (ideogram4_pipeline.mojo STEPS/CFG via Ideogram4LoRATrainRunConfig.defaults).
comptime I4_FT_SAMPLE_STEPS = 8
comptime I4_FT_SAMPLE_CFG = Float32(7.0)


def _ideogram4_load_frozen_globals(
    st: ShardedSafeTensors, ctx: DeviceContext
) raises -> Ideogram4Weights:
    """Ideogram4Weights over the NON-layer checkpoint tensors only (embedders +
    final layer + llm cond + indicator embed, with their fp8 scales) — the
    same dtype routing as Ideogram4Weights.load (ideogram4_resident.mojo:38-50)
    filtered to `not name.startswith("layers.")` so the ~9GB block surface is
    NOT device-resident (it streams from the pinned-host bf16 store)."""
    var d = Dict[String, ArcPointer[Tensor]]()
    var n = 0
    for ref nm in st.names():
        if nm.startswith(String("layers.")):
            continue
        var info = st.tensor_info(nm)
        if info.dtype == STDtype.F8_E4M3:
            d[nm] = ArcPointer(Tensor.from_view_raw(
                from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(nm)), ctx))
        elif info.dtype == STDtype.F32:
            d[nm] = ArcPointer(Tensor.from_view_as_f32(
                from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(nm)), ctx))
        else:
            d[nm] = ArcPointer(Tensor.from_view(st.tensor_view(nm), ctx))
        n += 1
    print("[ideogram4-ft] frozen globals device-resident:", n, "non-layer tensors")
    return Ideogram4Weights(d^)


# ── FULL-FT inline sampling (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #6) ───
# ONE transformer forward of the denoise loop: the FT arm's OWN streamed
# forward (ideogram4_stack_ft_forward_streamed) reading the LIVE pinned-host
# bf16 store — base+updates already merged = the current model, NO LoRA, NO
# reload — so sampling reuses the training memory footprint by construction.
# The cond glue is the FT training step's exact packing+embed+final chain
# (this file's train loop / Ideogram4LoRATrainStep.mojo:466-503): packed
# inputs (LoRA-sampler contract: default text_len = NT) → mrope → frozen embed
# → streamed stack fwd → frozen final layer → toolkit velocity = -(image rows)
# in [1,128,GH,GW] F32. The forward struct's 34 recompute checkpoints are
# dropped immediately (no backward runs on samples).
def _ideogram4_ft_forward_velocity[NT: Int, GH: Int, GW: Int](
    frozen_w: Ideogram4Weights,
    store: Ideogram4HostBf16,
    z: Tensor,             # [1, 128, GH, GW] F32 — current denoise latent
    t_flow: Float32,       # flow time; the embed flips model_t = 1 - t_flow
    llm: Tensor,           # [1, NT, 53248] bf16 conditioning (cond OR uncond)
    ctx: DeviceContext,
) raises -> Tensor:
    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG
    comptime HIDDEN = IDEOGRAM4_HIDDEN
    comptime HEADS = IDEOGRAM4_NUM_HEADS
    comptime DH = IDEOGRAM4_HEAD_DIM
    comptime FF = IDEOGRAM4_INTERMEDIATE_SIZE
    comptime ADALN = IDEOGRAM4_ADALN_DIM

    var packed = ideogram4_build_packed_inputs[NT, GH, GW](z, llm, ctx)
    var tv = List[Float32]()
    tv.append(Float32(1.0) - t_flow)
    var model_t = Tensor.from_host(tv^, [1], STDtype.F32, ctx)
    var sec = List[Int]()
    sec.append(IDEOGRAM4_MROPE_SECTION_0)
    sec.append(IDEOGRAM4_MROPE_SECTION_1)
    sec.append(IDEOGRAM4_MROPE_SECTION_2)
    var cs = build_ideogram4_mrope(
        packed.position_ids, DH, sec, IDEOGRAM4_MROPE_THETA, ctx, STDtype.BF16,
    )
    var cosf = cs[0].clone(ctx)
    var sinf = cs[1].clone(ctx)
    var x_bf = cast_tensor(packed.x, STDtype.BF16, ctx)
    var llm_bf = cast_tensor(packed.llm_full, STDtype.BF16, ctx)
    var emb = ideogram4_lora_embed_resident(
        frozen_w, x_bf, llm_bf, model_t, packed.indicator, HIDDEN, ctx
    )
    var x2d = reshape(emb.x_in, [SEQ, HIDDEN], ctx)
    var adaln2 = reshape(emb.adaln_input, [1, ADALN], ctx)

    # streamed FT forward from the live host store (read-only here).
    var fwd = ideogram4_stack_ft_forward_streamed[SEQ, HIDDEN, HEADS, DH, FF, ADALN](
        x2d, adaln2, cosf, sinf, store, ctx
    )
    var h3 = reshape(fwd.out[], [1, SEQ, HIDDEN], ctx)   # copy; fwd dies here
    var fin = ideogram4_lora_final_forward_resident(frozen_w, h3, emb.adaln_input, ctx)
    var image_velocity = slice(fin.out, 1, NT, NIMG, ctx)
    var iv4 = reshape(image_velocity, [1, GH, GW, IDEOGRAM4_PACKED_CHANNELS], ctx)
    var iv = permute(iv4, [0, 3, 1, 2], ctx)
    var velocity = mul_scalar(iv, Float32(-1.0), ctx)
    ctx.synchronize()   # land the stack's deferred frees before the next pass
    return velocity^


# CFG Euler denoise over the live FT store — schedule/step/CFG math 1:1 with
# the parity-gated LoRA sampler (ideogram4_sample_resident, which is itself
# 1:1 with the gated inference pipeline chunk 9):
#   mean = ideogram4_schedule_mean(16*GH, 16*GW, 0.5)   # resolution-aware mu
#   si   = make_step_intervals(n_steps)
#   for step in range(n_steps-1, -1, -1):
#       t_val/s_val = ideogram4_logitnormal(si[step+1] / si[step], mean)
#       t_flow = 1 - t_val            # embed flips back: model_t = t_val
#       v = cfg*pos_vel + (1-cfg)*neg_vel        # asymmetric CFG, negated vel
#       z = z - v*(s_val - t_val)                # NEGATED-velocity Euler step
# No parity claim: smoke + artifact evidence only (the denoise forward is the
# already-gated training forward). Returns the denoised PATCH-SPACE latent
# [1, 128, GH, GW] F32 (the LoRA sampler's decode contract).
def _ideogram4_ft_sample_store_latent[NT: Int, GH: Int, GW: Int](
    frozen_w: Ideogram4Weights,
    store: Ideogram4HostBf16,
    cond_llm: Tensor,        # [1, NT, 53248] bf16 cached-caption llm features
    uncond_llm: Tensor,      # [1, NT, 53248] bf16 zeros (CFG empty cond)
    init_noise: Tensor,      # [1, 128, GH, GW] F32 t=1 noise (own stream)
    n_steps: Int,
    cfg_scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    if n_steps < 1:
        raise Error("ideogram4 FT inline sampler: steps must be >= 1")
    var mean = ideogram4_schedule_mean(GH * 16, GW * 16, 0.5)
    var si = make_step_intervals(n_steps)
    var z = init_noise.clone(ctx)   # [1,128,GH,GW] F32, evolves each step
    print("[ideogram4-ft-sample] steps=", n_steps, " cfg=", cfg_scale)
    for step in range(n_steps - 1, -1, -1):
        var t_val = ideogram4_logitnormal(Float64(si[step + 1]), mean)
        var s_val = ideogram4_logitnormal(Float64(si[step]), mean)
        var t_flow = Float32(1.0) - t_val

        # COND then UNCOND pass — both the FT streamed forward, live store.
        var pos_vel = _ideogram4_ft_forward_velocity[NT, GH, GW](
            frozen_w, store, z, t_flow, cond_llm, ctx
        )
        var neg_vel = _ideogram4_ft_forward_velocity[NT, GH, GW](
            frozen_w, store, z, t_flow, uncond_llm, ctx
        )
        # asymmetric CFG on the (negated) velocities, then Euler down-step
        # (ideogram4_sample_resident:150-158).
        var v = add(
            mul_scalar(pos_vel, cfg_scale, ctx),
            mul_scalar(neg_vel, Float32(1.0) - cfg_scale, ctx),
            ctx,
        )
        z = add(z, mul_scalar(v, -(s_val - t_val), ctx), ctx)
        ctx.synchronize()
        print("[ideogram4-ft-sample] step", n_steps - step, "/", n_steps,
              " sigma=", t_val)
    return z^


# FT-arm cadence body (called from `_ideogram4_full_ft_run` behind
# IDEOGRAM4_FULL_FT). Conditioning = the model's existing resident-sampler
# contract (Ideogram4SampleResident header / _ideogram4_run_sample): a CACHED
# caption's llm_features [1,NT,53248] as COND (prompt index 0 -> cache sample
# 0 — the training cache is the conditioning source, no live TE) + zeroed
# features as UNCOND. Sample-noise seed = its OWN deterministic stream
# ((I4_FT_SEED ^ "SAMPLE") + step*1000003 + prompt_idx) — DISJOINT from all
# three training streams (I4_FT_SEED+k / *7919+k / *2654435761+k); randn and
# cache.sample are pure functions of their seeds, so sampling consumes NO
# training RNG and the BYTE-EQUAL loss/resume class is untouched.
def _ideogram4_ft_run_inline_samples[NT: Int, GH: Int, GW: Int](
    frozen_w: Ideogram4Weights,
    store: Ideogram4HostBf16,
    cache: Ideogram4TrainCache,
    output: String,
    completed_step: Int,
    ctx: DeviceContext,
) raises:
    var samples_dir = output + String("/samples")
    makedirs(samples_dir, exist_ok=True)

    # OWN sample-noise stream (the shipped krea2/klein/zimage/chroma
    # derivation); prompt idx pi = 0 — ONE image per event, conditioned on
    # cache sample 0 (the LoRA sampler's pi % cache.len() contract).
    var pi = 0
    var sample_seed = (
        (I4_FT_SEED ^ UInt64(0x53414D504C45))          # "SAMPLE"
        + UInt64(completed_step) * UInt64(1000003)
        + UInt64(pi)
    )
    var cond_index = pi % cache.len()
    # cache.sample's clean/noise/noisy byproducts are unused here (cheap
    # relative to the denoise); its noise draw is a pure fn of sample_seed.
    var cond_sample = cache.sample[NT, GH, GW](
        cond_index, Float32(0.0), sample_seed, ctx
    )
    var cond_llm = cond_sample.llm_features[].clone(ctx)   # [1,NT,53248] bf16
    var uncond_llm = zeros_device(
        [1, NT, cond_llm.shape()[2]], cond_llm.dtype(), ctx
    )
    print("[ideogram4-ft-sample] step", completed_step,
          " cond_cache_idx=", cond_index, " steps=", I4_FT_SAMPLE_STEPS,
          " cfg=", I4_FT_SAMPLE_CFG, " seed=", sample_seed)
    var init_noise = randn(
        [1, IDEOGRAM4_PACKED_CHANNELS, GH, GW], sample_seed, STDtype.F32, ctx
    )

    var latent = _ideogram4_ft_sample_store_latent[NT, GH, GW](
        frozen_w, store, cond_llm, uncond_llm, init_noise,
        I4_FT_SAMPLE_STEPS, I4_FT_SAMPLE_CFG, ctx,
    )

    var out_png = (
        samples_dir + String("/ft_sample_step") + String(completed_step)
        + String(".png")
    )
    # Persist the LATENT first so a process-separated decode
    # (models/ideogram4/ideogram4_decode_latent CLI, fresh GPU pool) can
    # ALWAYS produce the PNG, then attempt the in-process 512 decode; its
    # failure is non-fatal.
    var lat_bin = out_png + String(".lat.bin")
    save_tensor_bin(latent, lat_bin, ctx)
    ctx.synchronize()
    cu_mempool_trim_current(0)   # release denoise transients pre-decode
    try:
        ideogram4_decode_latent_to_png[GH, GW](latent, out_png, ctx)
        print("[ideogram4-ft-sample] wrote", out_png,
              " (", GW * 16, "x", GH * 16, ")")
    except e:
        print("[ideogram4-ft-sample] in-process decode failed (latent saved):",
              lat_bin, " err:", String(e))
        print("[ideogram4-ft-sample] decode offline (fresh GPU pool): ",
              "ideogram4_decode_latent ", lat_bin, " - ", out_png)


# resume_overlay ("" = fresh run): a prior FT run's saved overlay; the adafactor
# sidecar is derived from it (full_ft_sidecar_path_for_overlay). Resume = fp8
# base store build (dequant) THEN overlay bytes THEN sidecar states/t_step/seed;
# the loop continues at global step t_step+1 with the SAME I4_FT_SEED, so the
# sigma/noise/SR streams (I4_FT_SEED+k, I4_FT_SEED*7919+k, ...) continue
# exactly. Fail-loud on any seed/shape/count mismatch.
def _ideogram4_full_ft_run[NT: Int, GH: Int, GW: Int](
    progress_file: String,
    transformer: String,
    cache_path: String,
    output: String,
    run_steps: Int,
    lr: Float32,
    caption_dropout_prob: Float32,
    levers_config_path: String,
    resume_overlay: String,
    sample_every: Int,
) raises:
    # ── fail-loud v1 guards (b1 / no accum / no EMA / no dropout / no levers) ─
    if run_steps < 1:
        raise Error("ideogram4 full-FT v1: steps must be >= 1")
    if caption_dropout_prob > Float32(0.0):
        raise Error("ideogram4 full-FT v1: caption dropout not wired (fail-loud)")
    if levers_config_path.byte_length() > 0:
        raise Error(
            "ideogram4 full-FT v1: levers config not wired — the optimizer is "
            "FIXED device-Adafactor+SR (the krea2 full-FT contract)"
        )

    comptime NIMG = GH * GW
    comptime SEQ = NT + NIMG
    comptime HIDDEN = IDEOGRAM4_HIDDEN
    comptime HEADS = IDEOGRAM4_NUM_HEADS
    comptime DH = IDEOGRAM4_HEAD_DIM
    comptime FF = IDEOGRAM4_INTERMEDIATE_SIZE
    comptime ADALN = IDEOGRAM4_ADALN_DIM

    print("==== ideogram4 FULL FINETUNE (v1: 34 layers x 6 matmuls ~8.99B, device adafactor) ====")
    print(
        "lr=", lr, " steps=", run_steps,
        " SR=on  optimizer=torch-adafactor (b2d=-0.8 eps2=1e-3 d=1.0 wd=0)",
    )
    print("  ckpt:", transformer, " (fp8 -> bf16 pinned-host store)")
    print("  cache:", cache_path)

    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(transformer)

    # frozen residents: every non-layer tensor (embedders + final layer).
    var gw = _ideogram4_load_frozen_globals(st, ctx)

    # The pinned-host bf16 store = the live model (~18GB host RAM, CONVERTED
    # from the fp8 checkpoint at build via load_fp8_dequant).
    var store = build_ideogram4_host_bf16(st, IDEOGRAM4_NUM_LAYERS, ctx)
    # Adafactor factored states, device-resident, flat li*6+wi.
    var af_states = build_ideogram4_ft_adafactor_states(store, ctx)

    # ── RESUME (fp8-dequant base store built above; now overlay + sidecar) ───
    var start_k = 1
    if resume_overlay != String(""):
        ideogram4_host_bf16_overlay_resume(store, resume_overlay)
        var exp_rows = List[Int]()
        var exp_cols = List[Int]()
        ideogram4_ft_state_shapes(store, exp_rows, exp_cols)
        var sc = full_ft_sidecar_load(
            full_ft_sidecar_path_for_overlay(resume_overlay),
            exp_rows, exp_cols, ctx,
        )
        if sc.seed_base != I4_FT_SEED:
            raise Error(
                String("ideogram4 full-FT resume: sidecar seed_base ")
                + String(sc.seed_base) + String(" != trainer I4_FT_SEED ")
                + String(I4_FT_SEED)
                + String(" — the sigma/noise streams would not continue")
            )
        if sc.t_step >= run_steps:
            raise Error(
                String("ideogram4 full-FT resume: sidecar t_step ")
                + String(sc.t_step) + String(" >= requested total steps ")
                + String(run_steps) + String(" — nothing to continue")
            )
        start_k = sc.t_step + 1
        af_states = sc^.take_states()
        print("[ideogram4-ft] RESUME from", resume_overlay,
              "| continuing at global step", start_k, "/", run_steps)

    var cache = Ideogram4TrainCache.open(cache_path)
    print("[cache] samples:", cache.len())

    # ── inline sampling (FT arm; FT_INLINE_SAMPLING_PLAN_2026-07-08, #6) ─────
    # Fires after global step k when k % sample_every == 0 (never at k=0; the
    # loop starts at k=1). v1 does NOT save-before-sample (the FT arm saves at
    # run end only — documented delta vs ot_should_save_before_sample).
    # Sampling reads the LIVE pinned-host store (the optimizer writes back
    # into it every block — it IS the current model); conditioning = the
    # model's resident-sampler contract (cached llm cond + zero uncond, no
    # live TE).
    var sample_enabled = sample_every > 0
    if sample_enabled:
        print("[ideogram4-ft-sample] enabled every", sample_every,
              "steps (steps=", I4_FT_SAMPLE_STEPS, " cfg=", I4_FT_SAMPLE_CFG,
              ")")
        print("[ideogram4-ft-sample] v1: LIVE host-store weights, sample res",
              GW * 16, "x", GH * 16, ", cached llm cond (cache idx 0) + zero",
              "uncond,")
        print("[ideogram4-ft-sample]     NO save-before-sample (the FT arm",
              "saves at run end)")

    makedirs(output, exist_ok=True)
    if progress_file.byte_length() > 0:
        _clear_progress(progress_file)
        _append_status(progress_file, String("Staging Ideogram4 FULL-FT trainer"))

    print("")
    print("step  loss  (full-FT; sigma policy = ideogram4's own aitk logit-normal)")
    var train_start = perf_counter_ns()

    for k in range(start_k, run_steps + 1):
        var t0 = perf_counter_ns()
        var sample_index = (k - 1) % cache.len()
        var seed = I4_FT_SEED + UInt64(k)
        # ── ideogram4's OWN sigma policy (Ideogram4LoRATrainer.mojo:315-318):
        # t_flow ~ logit-normal(0,1) scale 1.0 = sigmoid(N(0,1)) — matches BOTH
        # aitk (sigmoid(randn) timestep, parity-ledger row 8) and OT; separate
        # *7919 RNG stream from the noise draw (the zimage idiom).
        var t_step = sample_timestep_logit_normal_scaled(
            I4_FT_SEED * UInt64(7919) + UInt64(k), Float32(1.0)
        )
        # cache fetch: clean [1,128,GH,GW] F32 + llm [1,NT,53248] BF16;
        # noise = randn(seed), noisy = (1-t)*clean + t*noise (reader-computed —
        # Ideogram4CacheReader.mojo:233-267 / Ideogram4Predict.ideogram4_add_noise).
        var sample = cache.sample[NT, GH, GW](sample_index, t_step, seed, ctx)

        # ── cond glue = the LoRA loop's exact packing + embed
        # (Ideogram4LoRATrainStep.mojo ideogram4_lora_train_forward_resident:
        # 466-498): packed inputs (+ text_len pad indicator), mrope, bf16 casts,
        # frozen embed -> x_in [1,SEQ,HIDDEN] + adaln_input [1,1,ADALN].
        var packed = ideogram4_build_packed_inputs[NT, GH, GW](
            sample.noisy[], sample.llm_features[], ctx, sample.text_len
        )
        var tv = List[Float32]()
        tv.append(Float32(1.0) - sample.t_flow)
        var model_t = Tensor.from_host(tv^, [1], STDtype.F32, ctx)
        var sec = List[Int]()
        sec.append(IDEOGRAM4_MROPE_SECTION_0)
        sec.append(IDEOGRAM4_MROPE_SECTION_1)
        sec.append(IDEOGRAM4_MROPE_SECTION_2)
        var cs = build_ideogram4_mrope(
            packed.position_ids, DH, sec, IDEOGRAM4_MROPE_THETA, ctx, STDtype.BF16,
        )
        var cosf = cs[0].clone(ctx)
        var sinf = cs[1].clone(ctx)
        var x_bf = cast_tensor(packed.x, STDtype.BF16, ctx)
        var llm_bf = cast_tensor(packed.llm_full, STDtype.BF16, ctx)
        var emb = ideogram4_lora_embed_resident(
            gw, x_bf, llm_bf, model_t, packed.indicator, HIDDEN, ctx
        )

        # seam to the P1-gate stack shapes (2-D x, [1,Adaln] adaln — see
        # ideogram4_full_ft.mojo header / ideogram4_block_ft_parity.mojo:134-141).
        var x2d = reshape(emb.x_in, [SEQ, HIDDEN], ctx)
        var adaln2 = reshape(emb.adaln_input, [1, ADALN], ctx)

        # ── streamed FT forward from the live host store ──
        var fwd = ideogram4_stack_ft_forward_streamed[SEQ, HIDDEN, HEADS, DH, FF, ADALN](
            x2d, adaln2, cosf, sinf, store, ctx
        )

        # final layer (FROZEN) + velocity (the LoRA loop's exact transform —
        # Ideogram4LoRATrainStep.mojo:497-503).
        var h3 = reshape(fwd.out[], [1, SEQ, HIDDEN], ctx)
        var fin = ideogram4_lora_final_forward_resident(gw, h3, emb.adaln_input, ctx)
        var image_velocity = slice(fin.out, 1, NT, NIMG, ctx)
        var iv4 = reshape(image_velocity, [1, GH, GW, IDEOGRAM4_PACKED_CHANNELS], ctx)
        var iv = permute(iv4, [0, 3, 1, 2], ctx)
        var velocity = mul_scalar(iv, Float32(-1.0), ctx)

        # ── flow target + ideogram4's own MSE loss (literal default path of
        # _i4_flow_loss_and_dvel, Ideogram4LoRATrainStep.mojo:601-610; computed
        # host-side like _chroma_full_ft_run:616-625) ──
        var target = ideogram4_flow_target(sample.noise[], sample.clean[], ctx)
        var v_h = velocity.to_host(ctx)
        var t_h = target.to_host(ctx)
        var nout = len(v_h)
        var d_vel_h = List[Float32]()
        var sse = 0.0
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = v_h[i] - t_h[i]
            sse += Float64(diff) * Float64(diff)
            d_vel_h.append(inv_n * diff)
        var loss = Float32(sse / Float64(nout))
        var d_velocity = Tensor.from_host(d_vel_h^, velocity.shape(), STDtype.F32, ctx)

        # final-frozen backward -> d_h [1,SEQ,HIDDEN] bf16 (the LoRA loop's
        # ideogram4_lora_final_backward, dx-only through final linear/ln).
        var d_h = ideogram4_lora_final_backward[NT, GH, GW](
            d_velocity, h3, fin.fscale, fin.flw, ctx
        )
        var d_h2 = reshape(d_h, [SEQ, HIDDEN], ctx)

        # ── streamed FT backward + fused device Adafactor + SR + write-back ──
        var wrote = ideogram4_stack_ft_backward_streamed[SEQ, HIDDEN, HEADS, DH, FF, ADALN](
            d_h2, adaln2, cosf, sinf, store, fwd,
            af_states,
            k, Float64(lr), Float64(-0.8), Float64(1e-3),
            Float64(1.0), Float64(0.0),
            I4_FT_SEED * UInt64(2654435761) + UInt64(k),
            ctx,
        )
        _ = wrote.grad_count

        var t1 = perf_counter_ns()
        var mi = ctx.get_memory_info()
        var used_gb = Float64(Int(mi[1]) - Int(mi[0])) / (1024.0 * 1024.0 * 1024.0)
        print(
            "[ideogram4-ft] step", k, "/", run_steps, "| loss", loss,
            "| t_flow", sample.t_flow,
            "| s/step", Float64(t1 - t0) / 1.0e9,
            "| avg", (Float64(t1 - train_start) / 1.0e9) / Float64(k - start_k + 1),
            "| vram_used_gb", used_gb,
        )

        # ── inline sampler: sample the LIVE host store, no save/reload ──────
        if sample_enabled and k % sample_every == 0:
            ctx.synchronize()
            cu_mempool_trim_current(0)   # release pool blocks before the render
            try:
                _ideogram4_ft_run_inline_samples[NT, GH, GW](
                    gw, store, cache, output, k, ctx,
                )
            except e:
                print("[ideogram4-ft-sample] sample FAILED (training continues):", e)
            ctx.synchronize()

    # Save the trained surface (host bytes -> safetensors overlay, no GPU).
    var out_path = (
        output + String("/ideogram4_full_ft_") + String(run_steps)
        + String(".safetensors")
    )
    ideogram4_host_bf16_save(store, out_path)
    # Resume sidecar NEXT TO the overlay: adafactor row/col states + t_step
    # (= completed global steps) + I4_FT_SEED (the stream continuity contract).
    full_ft_sidecar_save(
        af_states, run_steps, I4_FT_SEED,
        full_ft_sidecar_path_for_overlay(out_path), ctx,
    )
    if progress_file.byte_length() > 0:
        _append_status(
            progress_file,
            String("Finished Ideogram4 FULL-FT: weights ") + out_path,
        )
    print("[ideogram4-ft] DONE —", run_steps, "full-FT steps; weights:", out_path)
    print("[ideogram4-ft] v1 notes: surface = 34x6 block matmuls (adaln bias/rms")
    print("  scales/norm_q/k/embedders/final layer frozen); RESUME: pass the saved")
    print("  overlay as argv 12 (adafactor sidecar derived from it); inline")
    print("  sampling: argv 13 sample_every > 0 (samples the LIVE store at 512,")
    print("  cached llm cond + zero uncond, no save-before-sample).")


def main() raises:
    var args = argv()

    var progress_file = String(DEFAULT_PROGRESS)
    if len(args) > 1:
        var v = String(args[1])
        if v.byte_length() > 0 and v != String("-"):
            progress_file = v^

    var transformer = String(DEFAULT_TRANSFORMER)
    if len(args) > 2:
        var v = String(args[2])
        if v.byte_length() > 0 and v != String("-"):
            transformer = v^

    var cache = String(DEFAULT_CACHE)
    if len(args) > 3:
        var v = String(args[3])
        if v.byte_length() > 0 and v != String("-"):
            cache = v^

    var output = String(DEFAULT_OUTPUT)
    if len(args) > 4:
        var v = String(args[4])
        if v.byte_length() > 0 and v != String("-"):
            output = v^

    var steps = 3000
    if len(args) > 5:
        var v = String(args[5])
        if v.byte_length() > 0 and v != String("-"):
            steps = atol(v)

    var rank = 16
    if len(args) > 6:
        var v = String(args[6])
        if v.byte_length() > 0 and v != String("-"):
            rank = atol(v)

    var alpha = Float32(rank)
    if len(args) > 7:
        var v = String(args[7])
        if v.byte_length() > 0 and v != String("-"):
            alpha = Float32(atof(v))

    var lr = Float32(4.0e-4)
    if len(args) > 8:
        var v = String(args[8])
        if v.byte_length() > 0 and v != String("-"):
            lr = Float32(atof(v))

    var save_every_steps = 500
    if len(args) > 9:
        var v = String(args[9])
        if v.byte_length() > 0 and v != String("-"):
            save_every_steps = atol(v)

    # T1.D caption dropout (argv 10; default-off 0.0)
    var caption_dropout_prob = Float32(0.0)
    if len(args) > 10:
        var v = String(args[10])
        if v.byte_length() > 0 and v != String("-"):
            caption_dropout_prob = Float32(atof(v))

    # T1 levers config JSON (argv 11; optional — fail loud on a bad file)
    var levers_config_path = String("")
    if len(args) > 11:
        var v = String(args[11])
        if v.byte_length() > 0 and v != String("-"):
            levers_config_path = v^

    if steps < 1:
        steps = 1
    if save_every_steps < 0:
        save_every_steps = 0

    # ── FULL FINETUNE arm (IDEOGRAM4_FULL_FT): its own self-contained loop —
    # bypasses the entire LoRA machinery below (TrainConfig recipe, LoRA set,
    # AdamW, levers). Gate-don't-fork (C13): default builds are byte-unchanged.
    # NOTE argv 8 (learning_rate) drives the FT lr; the LoRA default 4e-4 is
    # NOT a sane full-FT lr — pass it explicitly (smoke: 1e-5).
    comptime if IDEOGRAM4_FULL_FT:
        # argv 12 (optional) = a prior FT run's overlay to RESUME from (the
        # adafactor sidecar path is derived from it).
        var ft_resume = String("")
        if len(args) > 12:
            var v12 = String(args[12])
            if v12.byte_length() > 0 and v12 != String("-"):
                if not v12.endswith(String(".safetensors")):
                    raise Error(
                        String("ideogram4 full-FT resume: expected an overlay ")
                        + String(".safetensors path, got ") + v12
                    )
                ft_resume = v12^
        # argv 13 (optional, parsed ONLY inside this comptime FT gate — inert
        # on default builds) = inline sample cadence: sample the LIVE host
        # store every N steps. "-" or absent = off; must be a positive int.
        # argv 14 is RESERVED for a sample-prompt/conditioning path — NOT
        # wired: v1 conditions on the cached llm features (the model's
        # resident-sampler contract, no live TE).
        var ft_sample_every = 0
        if len(args) > 13:
            var v13 = String(args[13])
            if v13.byte_length() > 0 and v13 != String("-"):
                ft_sample_every = atol(v13)
                if ft_sample_every < 1:
                    raise Error(
                        String("ideogram4 full-FT: argv 13 sample_every must ")
                        + String("be a positive int ('-' or absent = off), got ")
                        + v13
                    )
        _ideogram4_full_ft_run[NT, GH, GW](
            progress_file, transformer, cache, output, steps, lr,
            caption_dropout_prob, levers_config_path, ft_resume,
            ft_sample_every,
        )
        return

    makedirs(output, exist_ok=True)
    _clear_progress(progress_file)
    _append_status(progress_file, String("Staging Ideogram4 trainer"))
    print(
        "[Ideogram4-lora] model IDEOGRAM_4 | type LoRA | base ",
        transformer,
        " | cache ",
        cache,
        " | output ",
        output,
        " | steps ",
        steps,
        " | rank ",
        rank,
        " | lr ",
        lr,
        " | save_every ",
        save_every_steps,
    )

    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.lora_rank = rank
    cfg.lora_alpha = alpha
    cfg.learning_rate = lr
    cfg.batch_size = 1
    cfg.gradient_accumulation_steps = 1
    cfg.stochastic_rounding = False
    cfg.seed = UInt32(777)

    var run_cfg = Ideogram4LoRATrainRunConfig.defaults(transformer, cache, output)
    run_cfg.steps = steps
    run_cfg.save_every_steps = save_every_steps
    run_cfg.checkpoint_every_steps = save_every_steps
    run_cfg.progress_file_path = progress_file.copy()
    run_cfg.caption_dropout_prob = caption_dropout_prob
    if levers_config_path.byte_length() > 0:
        # Lever keys (loss_fn / min_snr_gamma_flow / ema_* / optimizer* /
        # caption_dropout_prob fallback) come from the JSON; the shared recipe
        # scalars stay argv-owned (the trainer syncs them from `cfg` above).
        run_cfg.levers = read_model_config(levers_config_path)
        print(
            "[Ideogram4-lora] levers config ", levers_config_path,
            " | loss_fn ", run_cfg.levers.loss_fn,
            " | min_snr_gamma_flow ", run_cfg.levers.min_snr_gamma_flow,
            " | optimizer ", run_cfg.levers.optimizer,
            " | ema_enabled ", run_cfg.levers.ema_enabled,
            " | caption_dropout_prob ", run_cfg.levers.caption_dropout_prob,
        )

    var ctx = DeviceContext()
    var summary = train_ideogram4_lora_from_cache[NT, GH, GW](cfg, run_cfg, ctx)
    _append_status(
        progress_file,
        String("Finished Ideogram4 LoRA: loss ")
        + String(summary.last_loss)
        + String(", ")
        + String(summary.seconds_per_step)
        + String(" s/step, saved ")
        + summary.lora_path.copy(),
    )
    print(
        "[Ideogram4-lora] model IDEOGRAM_4 | type LoRA | complete | step ",
        summary.optimizer_steps,
        "/",
        summary.steps_ran,
        " | loss ",
        summary.last_loss,
        " | ",
        Float32(summary.seconds_per_step),
        "s/step | saved ",
        summary.lora_path,
    )
