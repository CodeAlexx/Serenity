# train_chroma_real.mojo — Chroma1-HD LoRA training loop (block-swap offload).
#
# TRANSLATION of EriDiffusion-v2 chroma.rs onto the parity-verified Mojo Chroma
# LoRA OFFLOAD stack (models/chroma/chroma_stack_lora.mojo). Real Chroma1-HD base
# weights (streamed block-by-block via TurboPlannedLoader), real prepared cache
# (latent + T5), full 19+38 block depth. No synthetic tensors. Mirrors
# train_flux_real.mojo's loop structure (timing, grad clip, shared progress
# display) and chroma.rs's recipe.
#
# CHROMA vs FLUX (the deltas; see chroma_stack_lora.mojo header):
#   - NO guidance / CLIP-pooled vector. Modulation comes from the FROZEN
#     distilled_guidance_layer APPROXIMATOR (models/dit/chroma_dit.mojo
#     ChromaDitCache.approximator_forward), producing a per-step pooled_temb
#     table [mod_index=344, D=3072]; each block's ModVecs are sliced rows.
#   - Block math IS the proven Flux block (after the loader's separate->fused
#     row-stack), so the per-block LoRA fwd/bwd is REUSED verbatim and the LoRA
#     carrier / AdamW is shared while save uses Chroma's OneTrainer raw-key API.
#
# Per step:
#   1. load cached {latent [1,16,64,64] RAW, t5_embed [1,seq,4096]}
#   2. latent_scaled = (latent - SHIFT) * SCALE  (Chroma VAE shift/scale)
#   3. pack_latents([16,64,64]) -> [N_IMG=1024, 64] channel-major patchify
#   4. sigma_idx = floor(logit_normal_sigma(shift=1.15) * 1000) clamp;
#      sig=(idx+1)/1000 ; t_model=idx/1000
#   5. noisy = noise*sig + latent_packed*(1-sig) ; target = noise - latent_packed
#   6. pooled_temb = approximator(t_model)  (frozen; once per step)
#   7. chroma_stack_lora_forward_offload(noisy, txt, pooled, ...) -> pred [N_IMG,64]
#   8. loss = MSE(pred, target); d_loss = (2/N)(pred - target)
#   9. chroma_stack_lora_backward_offload -> LoRA grads; global-norm clip(1.0)
#  10. flux_lora_adamw_step; print shared progress display
#
# Recipe scalars (configs/chroma.json / chroma.rs): lr=1e-4, rank=16, alpha=16,
#   timestep_shift=1.15, clip_grad_norm=1.0, VAE shift=0.1159 scale=0.3611.
#
# MEMORY: the Chroma transformer is ~8.9B params (BF16 17.8GB on disk). The
# OFFLOAD path streams one block at a time + holds the resident base
# (x_embedder/context_embedder/proj_out ~tiny) + the approximator (~loaded once)
# + LoRA optimizer state. FULL 19+38 depth is the default.
#
# FIXED_SIGMA_SMOKE: every step uses the SAME cache sample AND a fixed
# timestep+noise so a correct LoRA backward MUST drive loss DOWN monotonically
# (the canonical trainer-correctness gate, same probe as train_flux_real).
#
# Run (real smoke):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_chroma_real.mojo -o /tmp/train_chroma_real && \
#     /tmp/train_chroma_real [steps]

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns
from std.os import listdir, makedirs

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts

from serenitymojo.models.chroma.weights import load_chroma_stack_base
from serenitymojo.models.chroma.chroma_stack_lora import (
    ChromaStackBase,
    chroma_stack_lora_forward_offload, chroma_stack_lora_backward_offload,
    save_chroma_lora, save_chroma_lora_state,
)
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxLoraGradSet, build_flux_lora_set,
    flux_lora_adamw_step, total_adapters,
)
from serenitymojo.models.flux.lora_block import DBL_STREAM_SLOTS, SGL_SLOTS
from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.models.dit.chroma_dit import ChromaDitCache
from serenitymojo.offload.plan import build_chroma1_hd_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.sample_prompt_config import (
    SampleCadence, read_sample_cadence_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_STEP, SAMPLE_UNIT_NEVER,
)
from serenitymojo.training.onetrainer_train_loop_policy import (
    OT_GRAD_POLICY_ON_ONLY,
    ot_cache_dir_from_train_config,
    ot_lr_for_optimizer_step,
    ot_output_lora_path_from_train_config,
    ot_sample_cadence_from_train_config,
    ot_sampling_enabled,
    ot_should_save_before_sample,
    ot_should_save_checkpoint,
    ot_step_lora_path,
    validate_ot_gradient_checkpointing_policy,
    validate_ot_train_math_policy,
)
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON,
)
from serenitymojo.training.adapter_algo_policy import require_lora_or_locon_linear
from serenitymojo.training.onetrainer_cache_preflight import (
    create_onetrainer_cache_preflight_plan,
    validate_onetrainer_cache_preflight_plan,
)
from serenitymojo.training.chroma_sample_resident import (
    chroma_sample_resident, chroma_decode_latent_to_png,
)
# FULL-FT inline sampling (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #4):
# denoise from the LIVE pinned-host bf16 store via the FT streamed forward;
# schedule + decode are the LoRA sampler's parity-gated pieces.
from serenitymojo.io.cap_cache import save_tensor_bin
from serenitymojo.sampling.flux1_dev import build_flux1_sigma_schedule
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current

# ── FULL FINETUNE (-D CHROMA_FULL_FT=1; FULL_FINETUNE_ROLLOUT_PLAN_2026-07-07,
# chroma card — the krea2/klein `ot-mojo-full-finetune` blueprint) ────────────
from std.sys.defines import get_defined_int
from serenitymojo.models.chroma.chroma_full_ft import (
    ChromaHostBf16, build_chroma_host_bf16, build_chroma_ft_adafactor_states,
    chroma_stack_ft_forward_streamed, chroma_stack_ft_backward_streamed,
    chroma_host_bf16_save, load_chroma_stack_base_bfl, load_chroma_dit_cache_bfl,
    chroma_host_bf16_overlay_resume, chroma_ft_state_shapes,
)
from serenitymojo.training.adafactor_device import AdafactorDeviceState
# FULL-FT resume sidecar (the fleet helper): adafactor row/col states + t_step
# + seed_base round-trip; sidecar path derived from the overlay path.
from serenitymojo.training.full_ft_sidecar import (
    full_ft_sidecar_save, full_ft_sidecar_load,
    full_ft_sidecar_path_for_overlay,
)
from serenitymojo.training.levers import levers_optimizer_active

# FULL FINETUNE arm (build with -D CHROMA_FULL_FT=1): trains the block matmul
# surface (19 double x 8 mats + 38 single x 2 mats, ~8.6B params, ~17.2GB bf16)
# through the pinned-host bf16 both-ways store + fused device Adafactor + SR.
# Default 0 = every LoRA path below byte-unchanged (C13 gate-don't-fork).
comptime CHROMA_FULL_FT = get_defined_int["CHROMA_FULL_FT", 0]() != 0


# ── arch (chroma1-hd; H/Dh/D fixed comptime, verified vs the checkpoint) ─────
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288          # mlp_hidden = D*4
comptime IN_CH = 64            # patch_dim = 16ch * 2*2
comptime TXT_CH = 4096         # T5-XXL hidden
comptime OUT_CH = 64
comptime NUM_DOUBLE = 19
comptime NUM_SINGLE = 38
comptime MOD_INDEX = 3 * NUM_SINGLE + 2 * 6 * NUM_DOUBLE + 2   # 344
comptime EPS = Float32(1e-06)

# ── resolution (512px): latent [16,64,64] -> pack2 -> 32x32=1024 img tokens ──
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime PATCH = 2
comptime HT = LAT_H // PATCH   # 32
comptime WT = LAT_W // PATCH   # 32
comptime N_IMG = HT * WT       # 1024
comptime N_TXT = 512           # T5 padded length
comptime S = N_TXT + N_IMG     # 1536

# ── recipe (configs/chroma.json) ─────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(1.0e-4)
comptime TIMESTEP_SHIFT = Float32(1.15)
comptime VAE_SHIFT = Float32(0.1159)
comptime VAE_SCALE = Float32(0.3611)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_IDX = 500

comptime CKPT = "/home/alex/.serenity/models/checkpoints/chroma1_hd_bf16.safetensors"
comptime CACHE_DIR = "/home/alex/datasets/boxjana_chroma_edv2_512"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/chroma_boxjana"
comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/chroma.json"
comptime DEFAULT_RUN_STEPS = 5

# ── sample-during-training (v1; chroma_sample_resident) ──────────────────────
# When the existing SampleCadence fires (should_sample_completed_step), denoise a
# sample from the CURRENT frozen base + streamed blocks + LIVE LoRA, decode with
# the FLUX VAE, and write <LORA_DIR>/samples/step_<N>.png. Geometry is the
# trainer's 512px latent (LAT_H=LAT_W=64 -> 8x VAE -> 512x512 image).
#   SAMPLE_STEPS / SAMPLE_CFG : denoise loop length + CFG (sampler defaults 30/4.0
#                               — chroma_sample_cli.mojo NUM_STEPS/GUIDANCE).
#   SAMPLE_SEED               : base RNG seed for the t=1 packed init noise.
# v1 CONDITIONING (flagged): no in-tree T5 tokenizer, so the COND text is the
#   CURRENT step's cached caption T5 embeds (the loop's txt_tokens); UNCOND is a
#   zero vector. See chroma_sample_resident.mojo header for the why + drop-in path.
comptime SAMPLE_STEPS = 30
comptime SAMPLE_CFG = Float32(4.0)
comptime SAMPLE_SEED = UInt64(0xC4_303A_5A91)
comptime SAMPLE_VAE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"


def _is_nonnegative_int(s: String) -> Bool:
    if s.byte_length() == 0:
        return False
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        if bs[i] < 0x30 or bs[i] > 0x39:
            return False
    return True


def _parse_nonnegative_int(s: String) raises -> Int:
    if not _is_nonnegative_int(s):
        raise Error(String("expected non-negative integer, got ") + s)
    var out = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        out = out * 10 + Int(bs[i] - 0x30)
    return out


def _close_f32(a: Float32, b: Float32, tol: Float32 = Float32(1.0e-7)) -> Bool:
    var d = a - b
    if d < Float32(0.0):
        d = -d
    return d <= tol


def validate_chroma_train_config(cfg: TrainConfig) raises:
    require_lora_or_locon_linear(cfg, String("Chroma"))
    if cfg.checkpoint == String(""):
        raise Error("Chroma trainer config must set checkpoint")
    if not cfg.checkpoint.endswith(String(".safetensors")):
        raise Error(
            String("Chroma trainer currently requires a single safetensors checkpoint; ")
            + String("sharded transformer dirs need a dedicated product loader")
        )
    if cfg.n_heads != H:
        raise Error(String("Chroma config n_heads ") + String(cfg.n_heads) + String(" != H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("Chroma config head_dim ") + String(cfg.head_dim) + String(" != Dh ") + String(Dh))
    if cfg.d_model != D:
        raise Error(String("Chroma config d_model ") + String(cfg.d_model) + String(" != D ") + String(D))
    if cfg.in_channels != IN_CH:
        raise Error(String("Chroma config in_channels ") + String(cfg.in_channels) + String(" != IN_CH ") + String(IN_CH))
    if cfg.joint_attention_dim != TXT_CH:
        raise Error(String("Chroma config joint_attention_dim ") + String(cfg.joint_attention_dim) + String(" != TXT_CH ") + String(TXT_CH))
    if cfg.out_channels != OUT_CH:
        raise Error(String("Chroma config out_channels ") + String(cfg.out_channels) + String(" != OUT_CH ") + String(OUT_CH))
    if cfg.num_double != NUM_DOUBLE or cfg.num_single != NUM_SINGLE:
        raise Error(
            String("Chroma trainer requires double=") + String(NUM_DOUBLE)
            + String(" single=") + String(NUM_SINGLE)
            + String("; got double=") + String(cfg.num_double)
            + String(" single=") + String(cfg.num_single)
        )
    if cfg.mlp_hidden != FMLP:
        raise Error(String("Chroma config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != FMLP ") + String(FMLP))
    if cfg.lora_rank != RANK:
        raise Error(
            String("Chroma trainer is compiled for lora_rank=")
            + String(RANK)
            + String("; parsed ")
            + String(cfg.lora_rank)
        )
    if not _close_f32(cfg.lora_alpha, ALPHA):
        raise Error("Chroma trainer lora_alpha does not match compiled constant")
    if not _close_f32(cfg.lr, LR, Float32(1.0e-9)):
        raise Error("Chroma trainer learning_rate does not match compiled constant")
    if not _close_f32(cfg.timestep_shift, TIMESTEP_SHIFT):
        raise Error("Chroma trainer timestep_shift does not match compiled constant")
    if not _close_f32(cfg.max_grad_norm, CLIP_GRAD_NORM):
        raise Error("Chroma trainer max_grad_norm does not match compiled constant")
    validate_ot_train_math_policy(cfg, String("Chroma trainer"))
    validate_ot_gradient_checkpointing_policy(
        cfg, String("Chroma trainer"), OT_GRAD_POLICY_ON_ONLY
    )


def chroma_checkpoint_from_train_config(cfg: TrainConfig) -> String:
    if cfg.checkpoint != String(""):
        return cfg.checkpoint.copy()
    return String(CKPT)


def chroma_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return ot_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def chroma_output_lora_path_from_train_config(cfg: TrainConfig, completed_step: Int) -> String:
    return ot_output_lora_path_from_train_config(
        cfg, String(LORA_DIR), String("chroma_lora"), completed_step
    )


def chroma_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return ot_sample_cadence_from_train_config(cfg_path, cfg)


def chroma_sampling_enabled(cadence: SampleCadence) -> Bool:
    return ot_sampling_enabled(cadence)


def chroma_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return ot_should_save_checkpoint(cfg, completed_step)


def chroma_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return ot_should_save_before_sample(cadence, completed_step, saved_this_step)


def _step_lora_path(base_path: String, step: Int) -> String:
    return ot_step_lora_path(base_path, step)


# ── deterministic host gaussian noise (Box-Muller PCG; per-step seed) ─────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


def _absum(v: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        s += x if x >= 0.0 else -x
    return s


def _global_norm(grads: FluxLoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: FluxLoraGradSet, max_norm: Float32) -> Float64:
    var gn = _global_norm(grads)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("chroma cache: no .safetensors in ") + dir)
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]
            fs[j - 1] = fs[j]
            fs[j] = tmp
            j -= 1
    return fs^


def _load_chroma_cache_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _host_f32_for_step_math(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    """Stage tensors through their stored dtype before host step math."""
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    if t.dtype() == STDtype.F16:
        var hf = t.to_host_f16(ctx)
        var out = List[Float32]()
        for i in range(len(hf)):
            out.append(hf[i].cast[DType.float32]())
        return out^
    return t.to_host(ctx)


# pack_latents: [16,LAT_H,LAT_W] flat -> [N_IMG, 64] channel-major patchify.
def _pack_latents(lat: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for ih in range(HT):
        for iw in range(WT):
            for c in range(LAT_C):
                for ph in range(PATCH):
                    for pw in range(PATCH):
                        var hh = ih * PATCH + ph
                        var ww = iw * PATCH + pw
                        var idx = c * LAT_H * LAT_W + hh * LAT_W + ww
                        out.append(lat[idx])
    return out^


# Build the frozen per-step modulation table [MOD_INDEX*D] as a Tensor.
def _pooled_modulation_tensor(
    approx: ChromaDitCache, t_model: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var approx_in = approx._approximator_input(t_model, ctx)
    return approx.approximator_forward(approx_in, ctx)   # [1, MOD_INDEX, D] BF16


# ── deterministic host gaussian PACKED init noise [N_IMG*IN_CH] ──────────────
# Reuses the same Box-Muller PCG as the train loop's _host_noise (so the sample's
# t=1 latent is drawn the same way the training noise is). seed makes it
# deterministic per sampled step.
def _sample_init_noise(seed: UInt64) -> List[Float32]:
    return _host_noise(N_IMG * IN_CH, seed)


# ── _chroma_run_sample — one sample-during-training image ────────────────────
#   cond text    : the current step's cached caption T5 embeds (v1; see header).
#   uncond text  : a zeroed [N_TXT*TXT_CH] vector (CFG empty cond).
#   init noise   : packed gaussian [N_IMG*IN_CH], seed = SAMPLE_SEED + step.
#   denoise      : chroma_sample_resident (frozen base + streamed blocks + live
#                  LoRA + frozen approximator).
#   decode+write : chroma_decode_latent_to_png -> <samples_dir>/step_<N>.png.
# Fail-loud: any raise propagates (no silent skip), matching the trainer's
# fail-loud cadence contract.
def _chroma_run_sample(
    base: ChromaStackBase,
    approx: ChromaDitCache,
    mut loader: TurboPlannedLoader,
    lora: FluxLoraSet,
    cond_txt: List[Float32],     # [N_TXT*TXT_CH] — the step's cached caption embeds
    cos: List[Float32],
    sin: List[Float32],
    samples_dir: String,
    step: Int,
    ctx: DeviceContext,
) raises:
    # UNCOND: zeroed text features (same dtype/shape as cond_txt).
    var uncond_txt = List[Float32]()
    for _ in range(N_TXT * TXT_CH):
        uncond_txt.append(Float32(0.0))

    var init_noise = _sample_init_noise(SAMPLE_SEED + UInt64(step))

    var latent = chroma_sample_resident[H, Dh, N_IMG, N_TXT, S](
        base, approx, loader, lora,
        cond_txt.copy(), uncond_txt^, init_noise^,
        cos.copy(), sin.copy(),
        SAMPLE_STEPS, SAMPLE_CFG, MOD_INDEX,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    var out_path = samples_dir + String("/step_") + String(step) + String(".png")
    chroma_decode_latent_to_png[LAT_C, LAT_H, LAT_W, HT, WT, PATCH, N_IMG, IN_CH](
        latent, String(SAMPLE_VAE_PATH), out_path, ctx,
    )
    print("[Chroma-lora] sample step=", step, " -> ", out_path)


# ── FULL-FT run (self-contained; called from main behind CHROMA_FULL_FT) ─────
# Mirrors _klein_full_ft_run: cache fetch → chroma's OWN sigma policy (logit-
# normal shift=cfg.timestep_shift, sigma_idx quantization — the LoRA loop's
# exact dispatch) → flow-match in PACKED latent space (VAE shift/scale +
# pack_latents) → frozen distilled-guidance approximator -> pooled_temb mods →
# streamed FT forward from the pinned-host bf16 store → chroma's own MSE loss →
# streamed FT backward with fused device Adafactor + SR + write-back →
# host-direct safetensors overlay save (ORIGINAL BFL key names). Bypasses the
# ENTIRE LoRA machinery (no turbo loader, no LoRA set, no OT-AdamW state).
def _validate_chroma_full_ft_config(cfg: TrainConfig) raises:
    # arch checks (the LoRA validate minus the LoRA/recipe pins — full-FT has
    # no rank/alpha and takes lr from the config).
    if cfg.checkpoint == String(""):
        raise Error("Chroma full-FT config must set checkpoint")
    if not cfg.checkpoint.endswith(String(".safetensors")):
        raise Error("Chroma full-FT requires a single safetensors checkpoint")
    if cfg.n_heads != H or cfg.head_dim != Dh or cfg.d_model != D:
        raise Error("Chroma full-FT config arch mismatch (n_heads/head_dim/d_model)")
    if cfg.in_channels != IN_CH or cfg.joint_attention_dim != TXT_CH or cfg.out_channels != OUT_CH:
        raise Error("Chroma full-FT config arch mismatch (in/txt/out channels)")
    if cfg.num_double != NUM_DOUBLE or cfg.num_single != NUM_SINGLE:
        raise Error("Chroma full-FT requires double=19 single=38")
    if cfg.mlp_hidden != FMLP:
        raise Error("Chroma full-FT config mlp_hidden mismatch")
    # fail-loud guards (v1 scope: b1 only, no accum/EMA/dropout/levers-opt)
    if cfg.batch_size != 1:
        raise Error("chroma full-FT v1: batch_size must be 1")
    if cfg.grad_accum_steps > 1:
        raise Error("chroma full-FT v1: grad_accum_steps must be 1")
    if cfg.ema_enabled:
        raise Error("chroma full-FT v1: EMA shadows not wired")
    if cfg.caption_dropout_prob > Float32(0.0):
        raise Error("chroma full-FT v1: caption dropout not wired")
    if levers_optimizer_active(cfg):
        raise Error(
            "chroma full-FT v1: the optimizer is FIXED device-Adafactor+SR "
            + "(the krea2 full-FT contract); unset the optimizer levers"
        )


# ── FULL-FT inline sampling (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #4) ───
# CFG Euler flow-match denoise whose transformer forward is the FT arm's OWN
# streamed forward (chroma_stack_ft_forward_streamed) reading the LIVE
# pinned-host bf16 store — base+updates already merged = the current model,
# NO LoRA, NO reload — so sampling reuses the training memory footprint by
# construction. Schedule/step/CFG math is 1:1 with the parity-gated LoRA
# sampler (training/chroma_sample_resident.mojo):
#   sigmas = build_flux1_sigma_schedule(n_steps, N_IMG)     # n_steps+1, 1->0
#   per step: frozen approximator at t_curr -> pooled mod table [MOD_INDEX,D],
#   pred = uncond + cfg*(cond - uncond), img += dt*pred (dt<0), host F32 math.
# No parity claim: smoke + artifact evidence only (the denoise forward is the
# already-gated training forward). Returns the denoised PACKED latent
# [N_IMG*IN_CH] host floats in trainer-scaled space (the FLUX VAE decode's
# internal z/scale+shift inverts the trainer's (latent-SHIFT)*SCALE).
def _chroma_ft_sample_store_latent(
    base: ChromaStackBase,
    approx: ChromaDitCache,
    store: ChromaHostBf16,
    cond_txt: List[Float32],      # [N_TXT*TXT_CH] cached caption T5 features
    uncond_txt: List[Float32],    # [N_TXT*TXT_CH] zeros (the model's contract)
    init_noise: List[Float32],    # [N_IMG*IN_CH] t=1 packed noise (own stream)
    cos: List[Float32],
    sin: List[Float32],
    n_steps: Int,
    cfg_scale: Float32,
    ctx: DeviceContext,
) raises -> List[Float32]:
    if n_steps < 1:
        raise Error("chroma FT inline sampler: steps must be >= 1")
    var sigmas = build_flux1_sigma_schedule(n_steps, N_IMG)
    var img = init_noise.copy()   # [N_IMG*IN_CH], evolves in place each step
    var n = len(img)
    print("[chroma-ft-sample] steps=", n_steps, " cfg=", cfg_scale)
    for step in range(n_steps):
        var t_curr = sigmas[step]
        var t_next = sigmas[step + 1]
        var dt = t_next - t_curr   # < 0 (down the schedule)

        # frozen approximator -> per-step modulation table at THIS sigma
        # (== t_model; see chroma_sample_resident.mojo header).
        var pooled_tensor = _pooled_modulation_tensor(approx, t_curr, ctx)
        var pooled = _host_f32_for_step_math(pooled_tensor, ctx)

        # COND pass: streamed FT forward from the LIVE host store.
        var fwd_c = chroma_stack_ft_forward_streamed[H, Dh, N_IMG, N_TXT, S](
            img.copy(), cond_txt.copy(), pooled.copy(), MOD_INDEX,
            base, store, cos.copy(), sin.copy(),
            D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )
        var v = fwd_c.out.copy()
        if len(v) != n:
            raise Error("chroma FT inline sampler: velocity length mismatch")
        # DROP the cond forward struct before the uncond forward — its 57 saved
        # block inputs are dead weight here (no backward runs on samples).
        _ = fwd_c^
        if cfg_scale != Float32(1.0):
            ctx.synchronize()   # land the async frees before the uncond fwd
            var fwd_u = chroma_stack_ft_forward_streamed[H, Dh, N_IMG, N_TXT, S](
                img.copy(), uncond_txt.copy(), pooled.copy(), MOD_INDEX,
                base, store, cos.copy(), sin.copy(),
                D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
            # CFG: pred = uncond + cfg*(cond - uncond), host F32 (the LoRA
            # sampler's combine, chroma_sample_resident.mojo:158).
            for i in range(n):
                v[i] = fwd_u.out[i] + cfg_scale * (v[i] - fwd_u.out[i])
            _ = fwd_u^
        # Euler: img += dt*pred (chroma_sample_resident.mojo:159).
        for i in range(n):
            img[i] = img[i] + dt * v[i]
        ctx.synchronize()
        if step == 0 or step + 1 == n_steps or (step + 1) % 5 == 0:
            print("[chroma-ft-sample] step", step + 1, "/", n_steps,
                  " sigma=", t_curr)
    return img^


# FT-arm cadence body (called from `_chroma_full_ft_run` behind CHROMA_FULL_FT).
# Conditioning = the model's existing resident-sampler contract
# (chroma_sample_resident.mojo header): the CURRENT step's cached caption T5
# features as COND + an all-zero vector as UNCOND — no live text encoder.
# Sample-noise seed = its OWN deterministic stream derived from
# (SEED_BASE, completed_step) — DISJOINT from all three training streams
# (SEED_BASE+k / SEED_BASE*7919+k / SEED_BASE*2654435761+k); _host_noise is a
# pure function of its seed, so sampling consumes NO training RNG and the
# wobble-class resume gate (BYTE-EQUAL sigmas) is untouched.
def _chroma_ft_run_inline_samples(
    base: ChromaStackBase,
    approx: ChromaDitCache,
    store: ChromaHostBf16,
    cond_txt: List[Float32],     # [N_TXT*TXT_CH] this step's cached caption T5
    cos: List[Float32],
    sin: List[Float32],
    cfg: TrainConfig,
    completed_step: Int,
    ctx: DeviceContext,
) raises:
    var samples_dir = String(LORA_DIR) + String("/samples")
    makedirs(samples_dir, exist_ok=True)

    # UNCOND: zeroed text features (the LoRA sampler's CFG empty cond).
    var uncond_txt = List[Float32]()
    for _ in range(N_TXT * TXT_CH):
        uncond_txt.append(Float32(0.0))

    # OWN sample-noise stream (the shipped krea2/klein/zimage derivation):
    # (SEED_BASE XOR "SAMPLE") folded with the global step (+ prompt idx = 0 —
    # chroma samples ONE image per event, the cached-caption conditioning).
    var sample_seed = (
        (SEED_BASE ^ UInt64(0x53414D504C45))          # "SAMPLE"
        + UInt64(completed_step) * UInt64(1000003)
        + UInt64(0)
    )
    print("[chroma-ft-sample] step", completed_step, " steps=", SAMPLE_STEPS,
          " cfg=", SAMPLE_CFG, " seed=", sample_seed)
    var init_noise = _host_noise(N_IMG * IN_CH, sample_seed)

    var latent = _chroma_ft_sample_store_latent(
        base, approx, store, cond_txt.copy(), uncond_txt^, init_noise^,
        cos.copy(), sin.copy(), SAMPLE_STEPS, SAMPLE_CFG, ctx,
    )

    var out_png = (
        samples_dir + String("/ft_sample_step") + String(completed_step)
        + String(".png")
    )
    # Persist the LATENT first so a process-separated decode
    # (chroma_decode_latent CLI, fresh GPU pool) can ALWAYS produce the PNG,
    # then attempt the in-process 512 decode; its failure is non-fatal.
    var lat_bin = out_png + String(".lat.bin")
    var lat_t = Tensor.from_host(
        latent.copy(), [N_IMG, IN_CH], STDtype.F32, ctx
    )
    save_tensor_bin(lat_t, lat_bin, ctx)
    ctx.synchronize()
    cu_mempool_trim_current(0)   # release denoise transients pre-decode
    # FLUX VAE (ae.safetensors) — the LoRA sampler's decode contract; cfg.vae
    # overrides when set (must be the BFL flux VAE layout).
    var vae_path = String(SAMPLE_VAE_PATH)
    if cfg.vae.byte_length() > 0:
        vae_path = cfg.vae.copy()
    try:
        chroma_decode_latent_to_png[
            LAT_C, LAT_H, LAT_W, HT, WT, PATCH, N_IMG, IN_CH
        ](latent, vae_path, out_png, ctx)
        print("[chroma-ft-sample] wrote", out_png,
              " (", LAT_W * 8, "x", LAT_H * 8, ")")
    except e:
        print("[chroma-ft-sample] in-process decode failed (latent saved):",
              lat_bin, " err:", String(e))
        print("[chroma-ft-sample] decode offline (fresh GPU pool): ",
              "chroma_decode_latent ", lat_bin, " ", vae_path, " ", out_png)


# resume_overlay ("" = fresh run): a prior FT run's saved overlay; the adafactor
# sidecar is derived from it (full_ft_sidecar_path_for_overlay). Resume = base
# ckpt store build THEN overlay bytes THEN sidecar states/t_step/seed; the loop
# continues at global step t_step+1 with the SAME SEED_BASE, so the sigma/noise/
# SR streams (SEED_BASE+k, SEED_BASE*7919+k, SEED_BASE*2654435761+k) continue
# exactly. Fail-loud on any seed/shape/count mismatch.
def _chroma_full_ft_run(
    cfg: TrainConfig, run_steps: Int, cache_dir: String, resume_overlay: String
) raises:
    _validate_chroma_full_ft_config(cfg)
    if run_steps < 1:
        raise Error("chroma full-FT v1: run_steps must be >= 1")
    print("==== chroma FULL FINETUNE (v1: 19-double + 38-single matmul surface, device adafactor) ====")
    print(
        "lr=", cfg.lr, " steps=", run_steps,
        " SR=on  optimizer=torch-adafactor (b2d=-0.8 eps2=1e-3 d=1.0 wd=0)",
    )
    print("  ckpt:", cfg.checkpoint, " (BFL double_blocks/single_blocks keys)")
    print("  cache:", cache_dir)

    var ctx = DeviceContext()
    var st = SafeTensors.open(cfg.checkpoint)

    # frozen residents: stack base (img_in/txt_in/final_layer) + approximator.
    var base = load_chroma_stack_base_bfl(st, NUM_DOUBLE, NUM_SINGLE, ctx)
    print("[load] BFL stack base resident (img_in/txt_in/final_layer.linear)")
    var approx = load_chroma_dit_cache_bfl(cfg.checkpoint, ctx)
    print("[load] BFL approximator resident (distilled_guidance_layer)")

    # The pinned-host bf16 store = the live model (~17.2GB host RAM).
    var store = build_chroma_host_bf16(st, NUM_DOUBLE, NUM_SINGLE, ctx)
    # Adafactor factored states, device-resident, flat (doubles*8 then singles*2).
    var af_states = build_chroma_ft_adafactor_states(store, ctx)

    # ── RESUME (base store built above; now overlay + sidecar) ───────────────
    var start_k = 1
    if resume_overlay != String(""):
        chroma_host_bf16_overlay_resume(store, resume_overlay)
        var exp_rows = List[Int]()
        var exp_cols = List[Int]()
        chroma_ft_state_shapes(store, exp_rows, exp_cols)
        var sc = full_ft_sidecar_load(
            full_ft_sidecar_path_for_overlay(resume_overlay),
            exp_rows, exp_cols, ctx,
        )
        if sc.seed_base != SEED_BASE:
            raise Error(
                String("chroma full-FT resume: sidecar seed_base ")
                + String(sc.seed_base) + String(" != trainer SEED_BASE ")
                + String(SEED_BASE)
                + String(" — the sigma/noise streams would not continue")
            )
        if sc.t_step >= run_steps:
            raise Error(
                String("chroma full-FT resume: sidecar t_step ")
                + String(sc.t_step) + String(" >= requested total steps ")
                + String(run_steps) + String(" — nothing to continue")
            )
        start_k = sc.t_step + 1
        af_states = sc^.take_states()
        print("[chroma-ft] RESUME from", resume_overlay,
              "| continuing at global step", start_k, "/", run_steps)

    # 3-axis RoPE tables (positions fixed for 512px; built once).
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, H, Dh](HT, WT, ctx, STDtype.BF16)
    var cos = rope[0].to_host(ctx)
    var sin = rope[1].to_host(ctx)
    print("[load] chroma 3-axis rope tables built (S*H x Dh/2)")

    var files = _list_cache(cache_dir)
    print("[cache] samples:", len(files))

    # ── inline sampling (FT arm; FT_INLINE_SAMPLING_PLAN_2026-07-08, #4) ─────
    # Fires after global step k when k % sample_every == 0 (never at k=0).
    # v1 does NOT save-before-sample (the FT arm saves at run end only —
    # documented delta vs ot_should_save_before_sample). Sampling reads the
    # LIVE pinned-host store (the optimizer writes back into it every block —
    # it IS the current model); conditioning = the model's resident-sampler
    # contract (cached caption T5 cond + zero uncond, no live TE).
    var sample_every = cfg.sample_every
    var sample_enabled = sample_every > 0
    if sample_enabled:
        print("[chroma-ft-sample] enabled every", sample_every,
              "steps (steps=", SAMPLE_STEPS, " cfg=", SAMPLE_CFG, ")")
        print("[chroma-ft-sample] v1: LIVE host-store weights, sample res",
              LAT_W * 8, "x", LAT_H * 8, ", cached-caption T5 cond + zero",
              "uncond,")
        print("[chroma-ft-sample]     NO save-before-sample (the FT arm saves",
              "at run end)")

    print("")
    print("step  loss  (full-FT; sigma policy = chroma's own logit-normal shift dispatch)")
    var train_start = perf_counter_ns()

    for k in range(start_k, run_steps + 1):
        var t0 = perf_counter_ns()
        var slot = (k - 1) % len(files)
        var st_c = SafeTensors.open(files[slot])
        var latent_cache = _load_chroma_cache_tensor(st_c, String("latent"), ctx)
        var lat_raw = _host_f32_for_step_math(latent_cache, ctx)         # [16*64*64]

        # t5_embed [1, seq, 4096] -> pad/truncate to [N_TXT, 4096] (zero pad rows).
        var t5_info = st_c.tensor_info(String("t5_embed"))
        var t5_seq = Int(t5_info.shape[1])
        var t5_cache = _load_chroma_cache_tensor(st_c, String("t5_embed"), ctx)
        var t5_flat = _host_f32_for_step_math(t5_cache, ctx)
        var txt_tokens = List[Float32]()
        for r in range(N_TXT):
            if r < t5_seq:
                for c in range(TXT_CH):
                    txt_tokens.append(t5_flat[r * TXT_CH + c])
            else:
                for _ in range(TXT_CH):
                    txt_tokens.append(Float32(0.0))

        # ── VAE shift/scale then pack_latents (chroma's own cond prep) ──
        for i in range(len(lat_raw)):
            lat_raw[i] = (lat_raw[i] - VAE_SHIFT) * VAE_SCALE
        var latent_packed = _pack_latents(lat_raw)

        # ── chroma's own sigma policy (the LoRA loop's non-smoke dispatch) ──
        var sigma = sample_timestep_logit_normal(SEED_BASE + UInt64(k), cfg.timestep_shift)
        var sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
        if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
            sigma_idx = NUM_TRAIN_TIMESTEPS - 1
        var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
        var t_model = Float32(sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

        # ── flow-match in PACKED latent space ──
        var noise = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + UInt64(k))
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        # ── frozen approximator -> per-step modulation table ──
        var pooled_tensor = _pooled_modulation_tensor(approx, t_model, ctx)
        var pooled = _host_f32_for_step_math(pooled_tensor, ctx)

        # ── streamed FT forward from the live host store ──
        var fwd = chroma_stack_ft_forward_streamed[H, Dh, N_IMG, N_TXT, S](
            noisy.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
            base, store, cos.copy(), sin.copy(),
            D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )

        # ── loss = MSE(pred, target) ; d_loss = (2/N)(pred - target) ──
        var nout = len(fwd.out)
        var d_loss = List[Float32]()
        var sse = 0.0
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            sse += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)
        var loss = Float32(sse / Float64(nout))

        # ── streamed FT backward + fused device Adafactor + SR + write-back ──
        var wrote = chroma_stack_ft_backward_streamed[H, Dh, N_IMG, N_TXT, S](
            d_loss.copy(), base, store, cos.copy(), sin.copy(), fwd,
            D, FMLP, OUT_CH, EPS,
            af_states,
            k, Float64(cfg.lr), Float64(-0.8), Float64(1e-3),
            Float64(1.0), Float64(0.0),
            SEED_BASE * UInt64(2654435761) + UInt64(k),
            ctx,
        )
        _ = wrote.grad_count

        var t1 = perf_counter_ns()
        var mi = ctx.get_memory_info()
        var used_gb = Float64(Int(mi[1]) - Int(mi[0])) / (1024.0 * 1024.0 * 1024.0)
        print(
            "[chroma-ft] step", k, "/", run_steps, "| loss", loss,
            "| sigma", sig,
            "| s/step", Float64(t1 - t0) / 1.0e9,
            "| avg", (Float64(t1 - train_start) / 1.0e9) / Float64(k - start_k + 1),
            "| vram_used_gb", used_gb,
        )

        # ── inline sampler: sample the LIVE host store, no save/reload ──────
        if sample_enabled and k % sample_every == 0:
            ctx.synchronize()
            cu_mempool_trim_current(0)   # release pool blocks before the render
            try:
                _chroma_ft_run_inline_samples(
                    base, approx, store, txt_tokens.copy(),
                    cos.copy(), sin.copy(), cfg, k, ctx,
                )
            except e:
                print("[chroma-ft-sample] sample FAILED (training continues):", e)
            ctx.synchronize()

    # Save the trained surface (host bytes -> safetensors overlay, no GPU).
    makedirs(String(LORA_DIR), exist_ok=True)
    var out_path = (
        String(LORA_DIR) + String("/chroma_full_ft_") + String(run_steps)
        + String(".safetensors")
    )
    chroma_host_bf16_save(store, out_path)
    # Resume sidecar NEXT TO the overlay: adafactor row/col states + t_step
    # (= completed global steps) + SEED_BASE (the stream continuity contract).
    full_ft_sidecar_save(
        af_states, run_steps, SEED_BASE,
        full_ft_sidecar_path_for_overlay(out_path), ctx,
    )
    print("[chroma-ft] DONE —", run_steps, "full-FT steps; weights:", out_path)
    print("[chroma-ft] v1 notes: surface = block matmuls (biases/norms/mods/")
    print("  embedders/final layer frozen); RESUME: pass the saved overlay as")
    print("  the arg after steps (adafactor sidecar derived from it); inline")
    print("  sampling: set sample_every>0 (samples the LIVE store at 512,")
    print("  cached-caption T5 cond + zero uncond, no save-before-sample).")


def main() raises:
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    var arg_base = 1
    if len(a) >= 2:
        var first = String(a[1])
        if first.endswith(String(".json")):
            cfg_path = first.copy()
            arg_base = 2

    var train_cfg = read_model_config(cfg_path)

    # ── FULL FINETUNE arm (CHROMA_FULL_FT): its own self-contained loop —
    # bypasses the entire LoRA machinery below (incl. the LoRA-recipe pins in
    # validate_chroma_train_config; the FT arm has its own fail-loud validate).
    # Gate-don't-fork (C13).
    comptime if CHROMA_FULL_FT:
        var ft_preflight = create_onetrainer_cache_preflight_plan(train_cfg)
        validate_onetrainer_cache_preflight_plan(ft_preflight)
        var ft_steps = DEFAULT_RUN_STEPS
        if len(a) > arg_base:
            ft_steps = _parse_nonnegative_int(String(a[arg_base]))
        # optional next arg = FT overlay to RESUME from (sidecar derived).
        var ft_resume = String("")
        if len(a) > arg_base + 1:
            ft_resume = String(a[arg_base + 1])
            if not ft_resume.endswith(String(".safetensors")):
                raise Error(
                    String("chroma full-FT resume: expected an overlay ")
                    + String(".safetensors path, got ") + ft_resume
                )
        if train_cfg.only_cache:
            raise Error("chroma full-FT v1: only_cache not wired (fail-loud)")
        _chroma_full_ft_run(
            train_cfg, ft_steps,
            chroma_cache_dir_from_train_config(train_cfg), ft_resume,
        )
        return

    validate_chroma_train_config(train_cfg)
    var cache_preflight = create_onetrainer_cache_preflight_plan(train_cfg)
    validate_onetrainer_cache_preflight_plan(cache_preflight)

    var run_steps = DEFAULT_RUN_STEPS
    if len(a) > arg_base:
        run_steps = _parse_nonnegative_int(String(a[arg_base]))
    elif train_cfg.only_cache:
        run_steps = 0

    var ckpt = chroma_checkpoint_from_train_config(train_cfg)
    var cache_dir = chroma_cache_dir_from_train_config(train_cfg)
    var sample_cadence = chroma_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = chroma_sampling_enabled(sample_cadence)

    print("=== Chroma (chroma1-hd) REAL LoRA training loop (block-swap offload) ===")
    print("  config:", cfg_path)
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " Fmlp=", FMLP, " out_ch=", OUT_CH)
    print("  depth: NUM_DOUBLE=", NUM_DOUBLE, " NUM_SINGLE=", NUM_SINGLE, " mod_index=", MOD_INDEX)
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", train_cfg.lora_rank, " alpha=", train_cfg.lora_alpha,
          " lr=", train_cfg.lr, " shift=", train_cfg.timestep_shift,
          " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  run_steps=", run_steps, " config_max_steps=", train_cfg.max_steps)
    print(
        "  cadence: save_every=", train_cfg.save_every,
        " sample_after=", sample_cadence.sample_after,
        " unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " skip_first=", sample_cadence.sample_skip_first,
        " sample_file=", sample_cadence.sample_definition_file_name,
    )
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  ckpt:", ckpt)
    print("  cache:", cache_dir)
    if train_cfg.enable_async_offloading:
        print("[offload] async offload requested by config; Chroma trainer currently uses synchronous TurboPlannedLoader")
    if train_cfg.only_cache:
        print("[Chroma] only_cache requested; no train steps will run in this trainer")
        return

    var ctx = DeviceContext()

    # ── stack-level base (frozen; x_embedder/context_embedder/proj_out) ──────
    print("[load] ChromaStackBase (x_embedder, context_embedder, proj_out)")
    var base_st = SafeTensors.open(ckpt)
    var base = load_chroma_stack_base(base_st, NUM_DOUBLE, NUM_SINGLE, ctx)
    print("[load] base resident")

    # ── frozen approximator (distilled_guidance_layer) ───────────────────────
    print("[load] approximator (distilled_guidance_layer)")
    var approx = ChromaDitCache.load(ckpt, ctx)
    print("[load] approximator resident")

    # ── block-swap offload loader ────────────────────────────────────────────
    var plan = build_chroma1_hd_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(ckpt, plan^, cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── 3-axis RoPE tables (positions fixed for 512px; built once, BF16) ─────
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, H, Dh](HT, WT, ctx, STDtype.BF16)
    var cos = rope[0].to_host(ctx)
    var sin = rope[1].to_host(ctx)
    print("[load] chroma 3-axis rope tables built (S*H x Dh/2)")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
    var n_adapters = total_adapters(lora)
    print("[lora] adapters:", n_adapters,
          " (", DBL_STREAM_SLOTS * 2, "x", NUM_DOUBLE, "double +",
          SGL_SLOTS, "x", NUM_SINGLE, "single)")

    var files = _list_cache(cache_dir)
    print("[cache] samples:", len(files))

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    # samples dir for sample-during-training PNGs (created once if enabled).
    var samples_dir = String(LORA_DIR) + String("/samples")
    if sample_enabled:
        makedirs(samples_dir, exist_ok=True)
        print("[cadence] sample-during-training WIRED -> ", samples_dir,
              " (steps=", SAMPLE_STEPS, " cfg=", SAMPLE_CFG, ")")
    if sample_enabled and should_sample_completed_step(sample_cadence, 0):
        print("[cadence] step 0 sample due (fires after the first completed step in this bounded loop)")
    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        var st = SafeTensors.open(files[slot])
        var latent_cache = _load_chroma_cache_tensor(st, String("latent"), ctx)
        var lat_raw = _host_f32_for_step_math(latent_cache, ctx)         # [16*64*64]

        # t5_embed [1, seq, 4096] -> pad/truncate to [N_TXT, 4096] (zero pad rows).
        var t5_info = st.tensor_info(String("t5_embed"))
        var t5_seq = Int(t5_info.shape[1])
        var t5_cache = _load_chroma_cache_tensor(st, String("t5_embed"), ctx)
        var t5_flat = _host_f32_for_step_math(t5_cache, ctx)
        var txt_tokens = List[Float32]()
        for r in range(N_TXT):
            if r < t5_seq:
                for c in range(TXT_CH):
                    txt_tokens.append(t5_flat[r * TXT_CH + c])
            else:
                for _ in range(TXT_CH):
                    txt_tokens.append(Float32(0.0))

        # ── VAE shift/scale then pack_latents ──
        for i in range(len(lat_raw)):
            lat_raw[i] = (lat_raw[i] - VAE_SHIFT) * VAE_SCALE
        var latent_packed = _pack_latents(lat_raw)

        # ── timestep ──
        var sigma_idx: Int
        if FIXED_SIGMA_SMOKE:
            sigma_idx = FIXED_SIGMA_IDX
        else:
            var sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)
            sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
            if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
                sigma_idx = NUM_TRAIN_TIMESTEPS - 1
        var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
        var t_model = Float32(sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

        # ── flow-match in PACKED latent space ──
        var noise = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        # ── frozen approximator -> modulation table ──
        var pooled_tensor = _pooled_modulation_tensor(approx, t_model, ctx)
        var pooled = _host_f32_for_step_math(pooled_tensor, ctx)

        # ── forward (offload, full depth) -> velocity [N_IMG, OUT_CH] ──
        var fwd = chroma_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            noisy.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
            base, loader, lora, cos.copy(), sin.copy(),
            D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )

        # ── loss = MSE(pred, target) ; d_loss = (2/N)(pred - target) ──
        var nout = len(fwd.out)
        var d_loss = List[Float32]()
        var sse = 0.0
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            sse += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)
        var loss = Float32(sse / Float64(nout))
        if k == 1:
            first_loss = loss
        last_loss = loss

        # ── backward (offload, full depth) ──
        var grads = chroma_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
            d_loss, noisy.copy(), txt_tokens.copy(), base, loader, lora,
            cos.copy(), sin.copy(), fwd,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, CLIP_GRAD_NORM)

        # ── AdamW ──
        var step_lr = ot_lr_for_optimizer_step(train_cfg, k)
        flux_lora_adamw_step(
            lora, grads, k, step_lr, ctx,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
            train_cfg.weight_decay,
        )

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            String("Chroma-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[Chroma-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

        var saved_this_step = False
        if chroma_should_save_checkpoint(train_cfg, k):
            var save_path = _step_lora_path(
                chroma_output_lora_path_from_train_config(train_cfg, run_steps), k
            )
            _ = save_chroma_lora(lora, save_path, ctx)
            var state_path = save_path + String(".state.safetensors")
            _ = save_chroma_lora_state(lora, state_path, ctx)
            saved_this_step = True
            print("[Chroma-lora] save_state step=", k, " path=", state_path)
        if sample_enabled and should_sample_completed_step(sample_cadence, k):
            if chroma_should_save_before_sample(sample_cadence, k, saved_this_step):
                var sample_path = _step_lora_path(
                    chroma_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                _ = save_chroma_lora(lora, sample_path, ctx)
                var sample_state = sample_path + String(".state.safetensors")
                _ = save_chroma_lora_state(lora, sample_state, ctx)
                print("[Chroma-lora] save_before_sample step=", k, " path=", sample_state)
            # Sample from the CURRENT frozen base + streamed blocks + LIVE LoRA.
            # v1 conditioning: this step's cached caption T5 embeds (txt_tokens)
            # as COND, zeros as UNCOND. See chroma_sample_resident.mojo header.
            print(
                "[cadence] sample due at completed_step=", k,
                " sample_file=", sample_cadence.sample_definition_file_name,
            )
            _chroma_run_sample(
                base, approx, loader, lora, txt_tokens.copy(),
                cos.copy(), sin.copy(), samples_dir, k, ctx,
            )

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(n_adapters):
        b_absum_final += _absum(lora.ad[i].b)
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        print("RESULT: REAL run OK — LoRA-B grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        var lora_out = chroma_output_lora_path_from_train_config(train_cfg, run_steps)
        _ = save_chroma_lora(lora, lora_out, ctx)
        var state_out = lora_out + String(".state.safetensors")
        _ = save_chroma_lora_state(lora, state_out, ctx)
        print("[Chroma-lora] save_state step=", run_steps, " path=", state_out)
    else:
        print("RESULT: FAIL trains=", trains)
