# train_flux_real.mojo — Flux.1-dev LoRA training loop.
#
# STATUS: not production-tested. This is Flux.1-dev only. Do not confuse it
# with Flux.2/Klein or dev2 paths. The shared progress display is wired for
# consistency, but Flux.1-dev trainer/sample/save/resume contract verification is
# a later task.
#
# TRANSLATION of EriDiffusion-v2 train_flux.rs onto the parity-verified Mojo
# Flux LoRA OFFLOAD stack (models/flux/flux_stack_lora.mojo). Real flux1-dev
# base weights (streamed block-by-block via TurboPlannedLoader), real prepared
# cache (latent + T5 + CLIP-pooled), full 19+38 block depth. No synthetic
# tensors. Mirrors train_zimage_real.mojo's loop structure (timing, grad clip,
# shared progress display) and train_flux.rs's recipe.
#
# Per step (translated from train_flux.rs main loop, lines 700-857):
#   1. load cached {latent [1,16,64,64] RAW, t5_embed [1,seq,4096], clip_pool [1,768]}
#   2. latent_scaled = (latent - SHIFT) * SCALE          (train_flux.rs:736)
#   3. pack_latents(latent_scaled): [1,16,h,w] -> [N_IMG, 64] channel-major
#      patchify, h_tok=h/2 w_tok=w/2                      (flux_sampler.rs:59-69)
#   4. sigma_idx = floor(logit_normal_sigma * 1000) clamp; sigma=(idx+1)/1000;
#      t_model = idx/1000                                 (train_flux.rs:767-813)
#   5. noisy = noise*sigma + latent_packed*(1-sigma)      (train_flux.rs:797-799)
#      target = noise - latent_packed   (rectified-flow)  (train_flux.rs:802)
#   6. flux_stack_lora_forward_offload(noisy_img_tokens, t5_txt_tokens,
#        timestep=t_model*1000, guidance=GUIDANCE*1000, vector=clip_pool) -> pred [N_IMG,64]
#   7. loss = MSE(pred, target); d_loss = (2/N)(pred - target)
#   8. flux_stack_lora_backward_offload -> LoRA grads; global-norm clip(1.0)
#   9. flux_lora_adamw_step; print shared progress display
#
# Recipe scalars (OneTrainer "#flux LoRA.json" preset + config defaults — verified
# against /home/alex/OneTrainer 2026-06-22):
#   lr=3e-4 (preset learning_rate), rank=16 (config default lora_rank),
#   alpha=1.0 (config default lora_alpha; preset does NOT override),
#   lr_warmup_steps=200 (config default; preset unset), lr_scheduler=CONSTANT,
#   timestep_shift=1.0 (config default; dynamic_timestep_shifting=false),
#   guidance=1.0 (config default transformer.guidance_scale; preset unset),
#   clip_grad_norm=1.0, betas=(0.9,0.999) eps=1e-8 weight_decay=1e-2 (ADAMW
#   default), SHIFT=0.1159, SCALE=0.3611, NUM_TRAIN_TIMESTEPS=1000.
# NOTE: OT preset resolution=768 (latent 96x96); this trainer is comptime-baked
# at 512px (latent 64x64). See the resolution-mismatch FLAG in the build request.
#
# MEMORY: the flux1-dev transformer is 11.9B params (47.6 GB F32 resident) — does
# NOT fit a 3090. The OFFLOAD path streams one block at a time
# (flux_stack_lora_forward_offload / _backward_offload, equivalence-gated vs the
# resident path at cos>=0.9999). The NON-streamed FluxStackBase (img_in/txt_in,
# 3 embed MLPs, PER-BLOCK modulation linears, final layer) is ~12.3 GB F32
# resident; with one streamed block (~0.84 GB) + activations + LoRA optimizer
# state it fits a 24 GB GPU. FULL 19+38 depth is the default.
#
# FIXED_SIGMA_SMOKE: when True, every step uses the SAME cache sample AND a fixed
# timestep+noise so a correct LoRA backward MUST drive loss DOWN monotonically
# (the canonical trainer-correctness gate, independent of per-step sampling
# variance — same probe as train_zimage_real / train_anima_real).
#
# Run (real smoke):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_flux_real.mojo -o /tmp/train_flux_real && \
#     /tmp/train_flux_real [steps]

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns
from std.os import listdir

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts

from serenitymojo.models.flux.weights import load_flux_stack_base
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxLoraGradSet, FluxStackLoraSet, build_flux_lora_set,
    build_flux_stack_lora_set, total_stack_adapters,
    flux_stack_lora_forward_offload, flux_stack_lora_backward_offload,
    flux_stack_lora_forward_offload_full, flux_stack_lora_backward_offload_full,
    flux_lora_adamw_step, flux_stack_lora_adamw_step,
    save_flux_lora, save_flux_lora_state,
    save_flux_lora_combined, save_flux_lora_state_combined, total_adapters,
)
from serenitymojo.models.flux.lora_block import DBL_STREAM_SLOTS, SGL_SLOTS
from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.offload.plan import build_flux1_dev_block_plan, OffloadConfig
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
    OT_GRAD_POLICY_ON_OR_CPU_OFFLOADED,
    ot_cache_dir_from_train_config,
    ot_output_lora_path_from_train_config,
    ot_sample_cadence_from_train_config,
    ot_sampling_enabled,
    ot_should_save_before_sample,
    ot_should_save_checkpoint,
    ot_step_lora_path,
    ot_lr_for_optimizer_step,
    validate_ot_gradient_checkpointing_policy,
    validate_ot_lora_adamw_loop_policy,
    validate_ot_train_math_policy,
)
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON, GRADIENT_CHECKPOINTING_CPU_OFFLOADED,
)
from serenitymojo.training.adapter_algo_policy import require_lora_or_locon_linear
from serenitymojo.training.onetrainer_cache_preflight import (
    create_onetrainer_cache_preflight_plan,
    validate_onetrainer_cache_preflight_plan,
)
from serenitymojo.training.flux_sample_resident import (
    flux_sample_offload, flux_decode_packed_to_png,
)
from std.os import makedirs

# ── FULL FINETUNE (-D FLUX_FULL_FT=1; FULL_FINETUNE_ROLLOUT_PLAN_2026-07-07,
# flux card — the chroma worked example's arm with the flux deltas) ───────────
from std.sys.defines import get_defined_int
from serenitymojo.models.flux.flux_full_ft import (
    flux_host_bf16_overlay_resume, flux_ft_state_shapes,
    FluxHostBf16, build_flux_host_bf16, build_flux_ft_adafactor_states,
    flux_stack_ft_forward_streamed, flux_stack_ft_backward_streamed,
    flux_host_bf16_save, flux_ckpt_has_guidance,
)
from serenitymojo.training.adafactor_device import AdafactorDeviceState
# FULL-FT resume sidecar (the fleet helper): adafactor row/col states + t_step
# + seed_base round-trip; sidecar path derived from the overlay path.
from serenitymojo.training.full_ft_sidecar import (
    full_ft_sidecar_save, full_ft_sidecar_load,
    full_ft_sidecar_path_for_overlay,
)
from serenitymojo.training.levers import levers_optimizer_active
# FULL-FT inline sampling (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #5):
# denoise from the LIVE pinned-host bf16 store via the FT streamed forward;
# schedule + decode are the LoRA sampler's parity-gated pieces
# (flux_decode_packed_to_png is already imported above for the LoRA arm).
from serenitymojo.models.flux.flux_stack import FluxStackBase
from serenitymojo.io.cap_cache import save_tensor_bin
from serenitymojo.sampling.flux1_dev import build_flux1_sigma_schedule
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current

# FULL FINETUNE arm (build with -D FLUX_FULL_FT=1): trains the block matmul
# surface (19 double x 8 mats + 38 single x 2 mats, ~8.6B params, ~17.2GB bf16)
# through the pinned-host bf16 both-ways store + fused device Adafactor + SR.
# Default 0 = every LoRA path below byte-unchanged (C13 gate-don't-fork).
#
# CHECKPOINT NOTE: the staged flux1-dev.safetensors is a SYMLINK to
# flux1-schnell.safetensors — SCHNELL weights under the dev name (23.8GB BF16,
# BFL keys, verified 2026-07-08). Schnell has NO guidance_in embedder; the FT
# arm auto-detects that (flux_ckpt_has_guidance) and loads the stack base with
# has_guidance=False, so NO guidance vector is fed. A real dev ckpt under the
# same path auto-detects True and feeds GUIDANCE*1000 — no code change.
comptime FLUX_FULL_FT = get_defined_int["FLUX_FULL_FT", 0]() != 0

# FT-arm defaults (config cache_dir/dataset_cache_dir overrides FT_CACHE_DIR).
# FT_CACHE_DIR is the DERIVED boxjana flux cache (latent+t5 verbatim from the
# chroma boxjana cache — shared flux VAE+T5 — plus REAL Mojo CLIP-L clip_pool
# per caption; pipeline/flux_cache_from_chroma.mojo). The LoRA arm's CACHE_DIR
# above points at a dead EriDiffusion path and is NOT used by the FT arm.
comptime FT_CACHE_DIR = "/home/alex/mojodiffusion/output/cache/boxjana_flux_512"
comptime FT_OUT_DIR = "/home/alex/mojodiffusion/output/flux_boxjana"

# FT inline sampling (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #5): denoise
# length by ckpt class — dev (guidance_in present) = the LoRA sampler's 20-step
# default; schnell (the on-box bytes under the dev name) = 4-step distilled,
# no guidance vector. SAME parity-gated flux1 sigma schedule helper either way
# (plan point 4: the model's existing sampler helpers, no new schedule code).
comptime FT_SAMPLE_STEPS_DEV = 20
comptime FT_SAMPLE_STEPS_SCHNELL = 4


# ── arch (flux1-dev; H/Dh/D fixed comptime, verified vs the checkpoint) ──────
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288          # mlp_hidden = D*4
comptime IN_CH = 64            # patch_dim = 16ch * 2*2
comptime TXT_CH = 4096         # T5 joint_attention_dim
comptime OUT_CH = 64
comptime T_DIM = 256           # timestep_dim
comptime VEC_DIM = 768         # CLIP-pooled
comptime NUM_DOUBLE = 19
comptime NUM_SINGLE = 38
comptime EPS = Float32(1e-06)
comptime MAX_PERIOD = Float32(10000.0)

# ── resolution (512px): latent [16,64,64] -> pack2 -> 32x32=1024 img tokens ──
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime PATCH = 2
comptime HT = LAT_H // PATCH   # 32
comptime WT = LAT_W // PATCH   # 32
comptime N_IMG = HT * WT       # 1024
comptime N_TXT = 512           # T5 padded length (BFL convention)
comptime S = N_TXT + N_IMG     # 1536

# ── recipe (OneTrainer "#flux LoRA.json" preset + config defaults) ───────────
comptime RANK = 16
comptime ALPHA = Float32(1.0)          # OT config default lora_alpha (preset unset)
comptime LR = Float32(3.0e-4)          # OT preset learning_rate 0.0003
comptime TIMESTEP_SHIFT = Float32(1.0)
comptime GUIDANCE = Float32(1.0)       # OT config default guidance_scale (preset unset)
comptime VAE_SHIFT = Float32(0.1159)
comptime VAE_SCALE = Float32(0.3611)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

# Overfit-correctness probe (see header). VERIFY monotone loss + LoRA-B growth.
comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_IDX = 500   # mid-schedule sigma when FIXED_SIGMA_SMOKE.

comptime CKPT = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"
comptime CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_flux_512_smoke"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/alina_flux"
comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/flux.json"
comptime DEFAULT_RUN_STEPS = 5

# ── sample-during-training (v1) ───────────────────────────────────────────────
# VAE for the sample decode (FLUX ae). The unpack uses the VAE in-channel count
# LAT_C (16); the packed patch dim is LAT_C*4 == IN_CH (64). HT/WT (32) are the
# patchified half-grid (IMG_H2/IMG_W2 in the inference CLI). Sample defaults match
# the gated inference CLI (steps 20, guidance == the trainer's GUIDANCE).
comptime VAE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"
comptime SAMPLE_STEPS = 20
comptime SAMPLE_SEED = UInt64(0xF10A_5A91)


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


def validate_flux_train_config(cfg: TrainConfig) raises:
    require_lora_or_locon_linear(cfg, String("Flux"))
    if cfg.checkpoint == String(""):
        raise Error("Flux trainer config must set checkpoint")
    if cfg.n_heads != H:
        raise Error(String("Flux config n_heads ") + String(cfg.n_heads) + String(" != H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("Flux config head_dim ") + String(cfg.head_dim) + String(" != Dh ") + String(Dh))
    if cfg.d_model != D:
        raise Error(String("Flux config d_model ") + String(cfg.d_model) + String(" != D ") + String(D))
    if cfg.in_channels != IN_CH:
        raise Error(String("Flux config in_channels ") + String(cfg.in_channels) + String(" != IN_CH ") + String(IN_CH))
    if cfg.joint_attention_dim != TXT_CH:
        raise Error(String("Flux config joint_attention_dim ") + String(cfg.joint_attention_dim) + String(" != TXT_CH ") + String(TXT_CH))
    if cfg.out_channels != OUT_CH:
        raise Error(String("Flux config out_channels ") + String(cfg.out_channels) + String(" != OUT_CH ") + String(OUT_CH))
    if cfg.num_double != NUM_DOUBLE or cfg.num_single != NUM_SINGLE:
        raise Error(
            String("Flux trainer requires double=") + String(NUM_DOUBLE)
            + String(" single=") + String(NUM_SINGLE)
            + String("; got double=") + String(cfg.num_double)
            + String(" single=") + String(cfg.num_single)
        )
    if cfg.mlp_hidden != FMLP:
        raise Error(String("Flux config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != FMLP ") + String(FMLP))
    if cfg.timestep_dim != T_DIM:
        raise Error(String("Flux config timestep_dim ") + String(cfg.timestep_dim) + String(" != T_DIM ") + String(T_DIM))
    if cfg.lora_rank != RANK:
        raise Error(
            String("Flux trainer is compiled for lora_rank=")
            + String(RANK)
            + String("; parsed ")
            + String(cfg.lora_rank)
        )
    if not _close_f32(cfg.lora_alpha, ALPHA):
        raise Error("Flux trainer lora_alpha does not match compiled constant")
    if not _close_f32(cfg.lr, LR, Float32(1.0e-9)):
        raise Error("Flux trainer learning_rate does not match compiled constant")
    if not _close_f32(cfg.timestep_shift, TIMESTEP_SHIFT):
        raise Error("Flux trainer timestep_shift does not match compiled constant")
    if not _close_f32(cfg.max_grad_norm, CLIP_GRAD_NORM):
        raise Error("Flux trainer max_grad_norm does not match compiled constant")
    validate_ot_lora_adamw_loop_policy(cfg, String("Flux trainer"))
    validate_ot_train_math_policy(cfg, String("Flux trainer"))
    validate_ot_gradient_checkpointing_policy(
        cfg, String("Flux trainer"), OT_GRAD_POLICY_ON_OR_CPU_OFFLOADED
    )


def flux_checkpoint_from_train_config(cfg: TrainConfig) -> String:
    if cfg.checkpoint != String(""):
        return cfg.checkpoint.copy()
    return String(CKPT)


def flux_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return ot_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def flux_output_lora_path_from_train_config(cfg: TrainConfig, completed_step: Int) -> String:
    return ot_output_lora_path_from_train_config(
        cfg, String(LORA_DIR), String("flux_lora"), completed_step
    )


def flux_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return ot_sample_cadence_from_train_config(cfg_path, cfg)


def flux_sampling_enabled(cadence: SampleCadence) -> Bool:
    return ot_sampling_enabled(cadence)


def flux_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return ot_should_save_checkpoint(cfg, completed_step)


def flux_should_save_before_sample(
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
    # stack-level LoRA grads share the SAME global clip norm (OT clips ALL trained
    # params together). Empty when stack-level LoRA disabled.
    for i in range(len(grads.st_d_a)):
        for j in range(len(grads.st_d_a[i])):
            ss += Float64(grads.st_d_a[i][j]) * Float64(grads.st_d_a[i][j])
        for j in range(len(grads.st_d_b[i])):
            ss += Float64(grads.st_d_b[i][j]) * Float64(grads.st_d_b[i][j])
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
    for i in range(len(grads.st_d_a)):
        for j in range(len(grads.st_d_a[i])):
            grads.st_d_a[i][j] = grads.st_d_a[i][j] * s
        for j in range(len(grads.st_d_b[i])):
            grads.st_d_b[i][j] = grads.st_d_b[i][j] * s
    return gn


# ── flux cache reader (prepare_flux.rs schema: latent / t5_embed / clip_pool) ─
def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("flux cache: no .safetensors in ") + dir)
    # simple insertion sort for reproducible order
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]
            fs[j - 1] = fs[j]
            fs[j] = tmp
            j -= 1
    return fs^


def _cache_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return t^


def _host_f32_for_step_math(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    """Stage cache tensors through their stored dtype before host step math."""
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


# ── pack_latents: [16,LAT_H,LAT_W] flat -> [N_IMG, 64] channel-major patchify ─
# Mirrors flux_sampler.rs pack_latents EXACTLY:
#   reshape [c, ht, p, wt, p] -> permute (ht, wt, c, p, p) -> [ht*wt, c*p*p].
# So token (ih,iw) carries [c, ph, pw] (c-major, then ph, then pw).
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


# ── FULL-FT run (self-contained; called from main behind FLUX_FULL_FT) ───────
# Mirrors _chroma_full_ft_run: cache fetch (latent + t5 + clip_pool) → FLUX's
# OWN sigma policy (logit-normal shift=cfg.timestep_shift, sigma_idx
# quantization — this file's LoRA-loop dispatch, lines tagged train_flux.rs:
# 767-813) → flow-match in PACKED latent space (VAE shift/scale + pack_latents,
# train_flux.rs:736) → FLUX conditioning (timestep*1000 + guidance*1000 when
# the ckpt has guidance_in + CLIP-L clip_pool vector — the flux_stack embed
# chain) → streamed FT forward from the pinned-host bf16 store → flux's own
# MSE loss → streamed FT backward with fused device Adafactor + SR +
# write-back → host-direct safetensors overlay save (ORIGINAL BFL key names).
# Bypasses the ENTIRE LoRA machinery (no turbo loader, no LoRA set, no AdamW).
def _validate_flux_full_ft_config(cfg: TrainConfig) raises:
    # arch checks (the LoRA validate minus the LoRA/recipe pins — full-FT has
    # no rank/alpha and takes lr from the config).
    if cfg.checkpoint == String(""):
        raise Error("Flux full-FT config must set checkpoint")
    if not cfg.checkpoint.endswith(String(".safetensors")):
        raise Error("Flux full-FT requires a single safetensors checkpoint")
    if cfg.n_heads != H or cfg.head_dim != Dh or cfg.d_model != D:
        raise Error("Flux full-FT config arch mismatch (n_heads/head_dim/d_model)")
    if cfg.in_channels != IN_CH or cfg.joint_attention_dim != TXT_CH or cfg.out_channels != OUT_CH:
        raise Error("Flux full-FT config arch mismatch (in/txt/out channels)")
    if cfg.num_double != NUM_DOUBLE or cfg.num_single != NUM_SINGLE:
        raise Error("Flux full-FT requires double=19 single=38")
    if cfg.mlp_hidden != FMLP:
        raise Error("Flux full-FT config mlp_hidden mismatch")
    if cfg.timestep_dim != T_DIM:
        raise Error("Flux full-FT config timestep_dim mismatch")
    # fail-loud guards (v1 scope: b1 only, no accum/EMA/dropout/levers-opt)
    if cfg.batch_size != 1:
        raise Error("flux full-FT v1: batch_size must be 1")
    if cfg.grad_accum_steps > 1:
        raise Error("flux full-FT v1: grad_accum_steps must be 1")
    if cfg.ema_enabled:
        raise Error("flux full-FT v1: EMA shadows not wired")
    if cfg.caption_dropout_prob > Float32(0.0):
        raise Error("flux full-FT v1: caption dropout not wired")
    if levers_optimizer_active(cfg):
        raise Error(
            "flux full-FT v1: the optimizer is FIXED device-Adafactor+SR "
            + "(the krea2 full-FT contract); unset the optimizer levers"
        )


# ── FULL-FT inline sampling (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #5) ───
# Rectified-flow Euler denoise whose transformer forward is the FT arm's OWN
# streamed forward (flux_stack_ft_forward_streamed) reading the LIVE
# pinned-host bf16 store — base+updates already merged = the current model,
# NO LoRA, NO reload — so sampling reuses the training memory footprint by
# construction. Schedule/step math is 1:1 with the parity-gated LoRA sampler
# (training/flux_sample_resident.mojo):
#   sigmas = build_flux1_sigma_schedule(n_steps, N_IMG)   # n_steps+1, 1->0
#   per step: pred = DiT(img, txt, t*1000, [g*1000], clip_pool)
#             img += dt*pred (dt<0), host F32 math.
# FLUX is GUIDANCE-DISTILLED: ONE forward per step, NO CFG / uncond pass (the
# load-bearing delta vs the chroma FT sampler — flux_sample_resident.mojo:28).
# Guidance-aware: the guidance vector is fed ONLY when the ckpt has
# guidance_in (the FT arm's has_guidance dispatch) — the on-box ckpt is
# SCHNELL bytes (no guidance_in, few-step distilled), and a real dev ckpt
# drop-in auto-feeds GUIDANCE*1000 + the 20-step default with no code change.
# No parity claim: smoke + artifact evidence only (the denoise forward is the
# already-gated training forward). Returns the denoised PACKED latent
# [N_IMG*IN_CH] host floats in trainer-scaled space (the FLUX VAE decode's
# internal z/scale+shift inverts the trainer's (latent-SHIFT)*SCALE).
def _flux_ft_sample_store_latent(
    base: FluxStackBase,
    store: FluxHostBf16,
    txt_tokens: List[Float32],    # [N_TXT*TXT_CH] cached caption T5 (padded)
    clip_pool: List[Float32],     # [VEC_DIM] cached caption CLIP-L pool
    init_noise: List[Float32],    # [N_IMG*IN_CH] t=1 packed noise (own stream)
    cos: List[Float32],
    sin: List[Float32],
    has_guidance: Bool,
    n_steps: Int,
    ctx: DeviceContext,
) raises -> List[Float32]:
    if n_steps < 1:
        raise Error("flux FT inline sampler: steps must be >= 1")
    var sigmas = build_flux1_sigma_schedule(n_steps, N_IMG)
    var img = init_noise.copy()   # [N_IMG*IN_CH], evolves in place each step
    var n = len(img)

    # guidance vec (dev only): pre-scaled *1000 (BFL time_factor) — the FT
    # arm's OWN training dispatch. Schnell: None (and the streamed forward
    # ignores it when base.has_guidance=False anyway).
    var guidance = Optional[List[Float32]](None)
    if has_guidance:
        var gl = List[Float32]()
        gl.append(GUIDANCE * Float32(1000.0))
        guidance = Optional[List[Float32]](gl^)

    print("[flux-ft-sample] steps=", n_steps,
          " (guidance-distilled single fwd, no CFG) has_guidance=", has_guidance)
    for step in range(n_steps):
        var t_curr = sigmas[step]
        var t_next = sigmas[step + 1]
        var dt = t_next - t_curr   # < 0 (down the schedule)

        # timestep pre-scaled *1000 (BFL time_factor; the trainer convention).
        var timestep = List[Float32]()
        timestep.append(t_curr * Float32(1000.0))

        # single forward: streamed FT fwd from the LIVE host store.
        var fwd = flux_stack_ft_forward_streamed[H, Dh, N_IMG, N_TXT, S](
            img, txt_tokens, timestep, guidance, clip_pool,
            base, store, cos, sin,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
        )
        var v = fwd.out.copy()
        if len(v) != n:
            raise Error("flux FT inline sampler: velocity length mismatch")
        # DROP the forward tape before the Euler update — its 57 saved block
        # inputs are dead weight here (no backward runs on samples).
        _ = fwd^
        # Euler: img += dt*pred (flux_sample_resident.mojo:152-155).
        for i in range(n):
            img[i] = img[i] + dt * v[i]
        ctx.synchronize()
        if step == 0 or step + 1 == n_steps or (step + 1) % 5 == 0:
            print("[flux-ft-sample] step", step + 1, "/", n_steps,
                  " sigma=", t_curr)
    return img^


# FT-arm cadence body (called from `_flux_full_ft_run` behind FLUX_FULL_FT).
# Conditioning = the model's existing resident-sampler contract
# (flux_sample_resident.mojo header): the CURRENT step's cached caption
# t5_embed (padded to [N_TXT,4096]) + clip_pool [768] — no live text encoder.
# Sample-noise seed = its OWN deterministic stream derived from
# (SEED_BASE, completed_step) — DISJOINT from all three training streams
# (SEED_BASE+k / SEED_BASE*7919+k / SEED_BASE*2654435761+k); _host_noise is a
# pure function of its seed, so sampling consumes NO training RNG and the
# wobble-class resume gate (BYTE-EQUAL sigmas) is untouched.
def _flux_ft_run_inline_samples(
    base: FluxStackBase,
    store: FluxHostBf16,
    txt_tokens: List[Float32],   # [N_TXT*TXT_CH] this step's cached caption T5
    clip_pool: List[Float32],    # [VEC_DIM]      this step's cached CLIP pool
    cos: List[Float32],
    sin: List[Float32],
    cfg: TrainConfig,
    completed_step: Int,
    has_guidance: Bool,
    ctx: DeviceContext,
) raises:
    var samples_dir = String(FT_OUT_DIR) + String("/samples")
    makedirs(samples_dir, exist_ok=True)

    # schnell (the on-box bytes) = 4-step distilled; dev = the LoRA sampler's
    # 20-step default. Same parity-gated flux1 sigma schedule either way.
    var n_steps = FT_SAMPLE_STEPS_DEV if has_guidance else FT_SAMPLE_STEPS_SCHNELL

    # OWN sample-noise stream (the shipped krea2/klein/zimage/chroma
    # derivation): (SEED_BASE XOR "SAMPLE") folded with the global step
    # (+ prompt idx = 0 — flux samples ONE image per event, the cached-caption
    # conditioning).
    var sample_seed = (
        (SEED_BASE ^ UInt64(0x53414D504C45))          # "SAMPLE"
        + UInt64(completed_step) * UInt64(1000003)
        + UInt64(0)
    )
    print("[flux-ft-sample] step", completed_step, " steps=", n_steps,
          " has_guidance=", has_guidance, " seed=", sample_seed)
    var init_noise = _host_noise(N_IMG * IN_CH, sample_seed)

    var latent = _flux_ft_sample_store_latent(
        base, store, txt_tokens, clip_pool, init_noise,
        cos, sin, has_guidance, n_steps, ctx,
    )

    var out_png = (
        samples_dir + String("/ft_sample_step") + String(completed_step)
        + String(".png")
    )
    # Persist the LATENT first so a process-separated decode can ALWAYS
    # produce the PNG, then attempt the in-process 512 decode; its failure is
    # non-fatal. The flux pack is byte-compatible with chroma's ([N_IMG,64]
    # flux-VAE trainer-scaled packing, same unpack permute + same
    # load_flux1_ldm_decoder) — chroma_decode_latent is the offline CLI.
    var lat_bin = out_png + String(".lat.bin")
    var lat_t = Tensor.from_host(
        latent.copy(), [N_IMG, IN_CH], STDtype.F32, ctx
    )
    save_tensor_bin(lat_t, lat_bin, ctx)
    ctx.synchronize()
    cu_mempool_trim_current(0)   # release denoise transients pre-decode
    # FLUX VAE (ae.safetensors) — the LoRA sampler's decode contract; cfg.vae
    # overrides when set (must be the BFL flux VAE layout).
    var vae_path = String(VAE_PATH)
    if cfg.vae.byte_length() > 0:
        vae_path = cfg.vae.copy()
    try:
        flux_decode_packed_to_png[N_IMG, HT, WT, LAT_H, LAT_W, LAT_C](
            latent, vae_path, out_png, ctx
        )
        print("[flux-ft-sample] wrote", out_png,
              " (", LAT_W * 8, "x", LAT_H * 8, ")")
    except e:
        print("[flux-ft-sample] in-process decode failed (latent saved):",
              lat_bin, " err:", String(e))
        print("[flux-ft-sample] decode offline (fresh GPU pool): ",
              "chroma_decode_latent ", lat_bin, " ", vae_path, " ", out_png)


# resume_overlay ("" = fresh run): a prior FT run's saved overlay; the adafactor
# sidecar is derived from it (full_ft_sidecar_path_for_overlay). Resume = base
# ckpt store build THEN overlay bytes THEN sidecar states/t_step/seed; the loop
# continues at global step t_step+1 with the SAME SEED_BASE, so the sigma/noise/
# SR streams (SEED_BASE+k, SEED_BASE*7919+k, SEED_BASE*2654435761+k) continue
# exactly. Fail-loud on any seed/shape/count mismatch.
def _flux_full_ft_run(
    cfg: TrainConfig, run_steps: Int, cache_dir: String, resume_overlay: String
) raises:
    _validate_flux_full_ft_config(cfg)
    if run_steps < 1:
        raise Error("flux full-FT v1: run_steps must be >= 1")
    print("==== flux FULL FINETUNE (v1: 19-double + 38-single matmul surface, device adafactor) ====")
    print(
        "lr=", cfg.lr, " steps=", run_steps,
        " SR=on  optimizer=torch-adafactor (b2d=-0.8 eps2=1e-3 d=1.0 wd=0)",
    )
    print("  ckpt:", cfg.checkpoint)
    print("  NOTE: on-box flux1-dev.safetensors -> flux1-schnell.safetensors symlink")
    print("        (SCHNELL weights under the dev name; 23.8GB all-BF16 BFL keys)")
    print("  cache:", cache_dir)

    var ctx = DeviceContext()
    var st = SafeTensors.open(cfg.checkpoint)

    # schnell detection: no guidance_in.* keys in the header -> no guidance vec.
    var has_guidance = flux_ckpt_has_guidance(st)
    if has_guidance:
        print("[load] guidance_in PRESENT (dev-class ckpt): guidance vec =", GUIDANCE, "* 1000")
    else:
        print("[load] guidance_in ABSENT (SCHNELL ckpt): has_guidance=False, no guidance vec fed")

    # frozen residents: the full FluxStackBase (img_in/txt_in, time/guidance/
    # vector embed MLPs, per-block mod.lin linears, final layer) — the flux
    # delta vs chroma's tiny base+approximator (~6.6GB device for the mod.lins).
    var base = load_flux_stack_base(st, NUM_DOUBLE, NUM_SINGLE, has_guidance, ctx)
    print("[load] FluxStackBase resident (embedders + per-block mod.lin + final layer)")

    # The pinned-host bf16 store = the live model (~17.2GB host RAM).
    var store = build_flux_host_bf16(st, NUM_DOUBLE, NUM_SINGLE, ctx)
    # Adafactor factored states, device-resident, flat (doubles*8 then singles*2).
    var af_states = build_flux_ft_adafactor_states(store, ctx)

    # ── RESUME (base store built above; now overlay + sidecar) ───────────────
    var start_k = 1
    if resume_overlay != String(""):
        flux_host_bf16_overlay_resume(store, resume_overlay)
        var exp_rows = List[Int]()
        var exp_cols = List[Int]()
        flux_ft_state_shapes(store, exp_rows, exp_cols)
        var sc = full_ft_sidecar_load(
            full_ft_sidecar_path_for_overlay(resume_overlay),
            exp_rows, exp_cols, ctx,
        )
        if sc.seed_base != SEED_BASE:
            raise Error(
                String("flux full-FT resume: sidecar seed_base ")
                + String(sc.seed_base) + String(" != trainer SEED_BASE ")
                + String(SEED_BASE)
                + String(" — the sigma/noise streams would not continue")
            )
        if sc.t_step >= run_steps:
            raise Error(
                String("flux full-FT resume: sidecar t_step ")
                + String(sc.t_step) + String(" >= requested total steps ")
                + String(run_steps) + String(" — nothing to continue")
            )
        start_k = sc.t_step + 1
        af_states = sc^.take_states()
        print("[flux-ft] RESUME from", resume_overlay,
              "| continuing at global step", start_k, "/", run_steps)

    # 3-axis RoPE tables (positions fixed for 512px; built once).
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, H, Dh](HT, WT, ctx, STDtype.BF16)
    var cos = rope[0].to_host(ctx)
    var sin = rope[1].to_host(ctx)
    print("[load] flux 3-axis rope tables built (S*H x Dh/2)")

    var files = _list_cache(cache_dir)
    print("[cache] samples:", len(files))

    # ── inline sampling (FT arm; FT_INLINE_SAMPLING_PLAN_2026-07-08, #5) ─────
    # Fires after global step k when k % sample_every == 0 (never at k=0).
    # v1 does NOT save-before-sample (the FT arm saves at run end only —
    # documented delta vs ot_should_save_before_sample). Sampling reads the
    # LIVE pinned-host store (the optimizer writes back into it every block —
    # it IS the current model); conditioning = the model's resident-sampler
    # contract (cached caption t5_embed + clip_pool, no live TE).
    var sample_every = cfg.sample_every
    var sample_enabled = sample_every > 0
    if sample_enabled:
        var banner_steps = (
            FT_SAMPLE_STEPS_DEV if has_guidance else FT_SAMPLE_STEPS_SCHNELL
        )
        print("[flux-ft-sample] enabled every", sample_every,
              "steps (steps=", banner_steps,
              " guidance-distilled single fwd, no CFG)")
        print("[flux-ft-sample] v1: LIVE host-store weights, sample res",
              LAT_W * 8, "x", LAT_H * 8, ", cached-caption t5+clip cond,")
        print("[flux-ft-sample]     NO save-before-sample (the FT arm saves",
              "at run end)")

    print("")
    print("step  loss  (full-FT; sigma policy = flux's own logit-normal shift dispatch)")
    var train_start = perf_counter_ns()

    for k in range(start_k, run_steps + 1):
        var t0 = perf_counter_ns()
        var slot = (k - 1) % len(files)
        var st_c = SafeTensors.open(files[slot])
        var lat_cache = _cache_tensor(st_c, String("latent"), ctx)
        var lat_raw = _host_f32_for_step_math(lat_cache, ctx)              # [16*64*64]
        var clip_pool_cache = _cache_tensor(st_c, String("clip_pool"), ctx)   # [1,768]
        var clip_pool = _host_f32_for_step_math(clip_pool_cache, ctx)
        if len(clip_pool) != VEC_DIM:
            raise Error("flux full-FT: cache clip_pool is not [1,768] (real CLIP-L pool required)")

        # t5_embed [1, seq, 4096] -> pad/truncate to [N_TXT, 4096] (zero pad rows).
        var t5_info = st_c.tensor_info(String("t5_embed"))
        var t5_seq = Int(t5_info.shape[1])
        var t5_cache = _cache_tensor(st_c, String("t5_embed"), ctx)
        var t5_flat = _host_f32_for_step_math(t5_cache, ctx)
        var txt_tokens = List[Float32]()
        for r in range(N_TXT):
            if r < t5_seq:
                for c in range(TXT_CH):
                    txt_tokens.append(t5_flat[r * TXT_CH + c])
            else:
                for _ in range(TXT_CH):
                    txt_tokens.append(Float32(0.0))

        # ── VAE shift/scale (train_flux.rs:736) then pack_latents ──
        for i in range(len(lat_raw)):
            lat_raw[i] = (lat_raw[i] - VAE_SHIFT) * VAE_SCALE
        var latent_packed = _pack_latents(lat_raw)

        # ── flux's own sigma policy (this file's LoRA-loop non-smoke dispatch) ──
        var sigma = sample_timestep_logit_normal(SEED_BASE + UInt64(k), cfg.timestep_shift)
        var sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
        if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
            sigma_idx = NUM_TRAIN_TIMESTEPS - 1
        var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
        var t_model = Float32(sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)
        # caller pre-scales t by 1000 (BFL time_factor; flux1_dit.mojo convention).
        var timestep = List[Float32]()
        timestep.append(t_model * Float32(1000.0))
        # guidance (dev only): pre-scaled *1000; None for schnell.
        var guidance = Optional[List[Float32]](None)
        if has_guidance:
            var gl = List[Float32]()
            gl.append(GUIDANCE * Float32(1000.0))
            guidance = Optional[List[Float32]](gl^)

        # ── flow-match in PACKED latent space ──
        var noise = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + UInt64(k))
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        # ── streamed FT forward from the live host store ──
        var fwd = flux_stack_ft_forward_streamed[H, Dh, N_IMG, N_TXT, S](
            noisy, txt_tokens, timestep, guidance, clip_pool,
            base, store, cos, sin,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
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
        var wrote = flux_stack_ft_backward_streamed[H, Dh, N_IMG, N_TXT, S](
            d_loss, base, store, cos, sin, fwd,
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
            "[flux-ft] step", k, "/", run_steps, "| loss", loss,
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
                _flux_ft_run_inline_samples(
                    base, store, txt_tokens, clip_pool,
                    cos, sin, cfg, k, has_guidance, ctx,
                )
            except e:
                print("[flux-ft-sample] sample FAILED (training continues):", e)
            ctx.synchronize()

    # Save the trained surface (host bytes -> safetensors overlay, no GPU).
    makedirs(String(FT_OUT_DIR), exist_ok=True)
    var out_path = (
        String(FT_OUT_DIR) + String("/flux_full_ft_") + String(run_steps)
        + String(".safetensors")
    )
    flux_host_bf16_save(store, out_path)
    # Resume sidecar NEXT TO the overlay: adafactor row/col states + t_step
    # (= completed global steps) + SEED_BASE (the stream continuity contract).
    full_ft_sidecar_save(
        af_states, run_steps, SEED_BASE,
        full_ft_sidecar_path_for_overlay(out_path), ctx,
    )
    print("[flux-ft] DONE —", run_steps, "full-FT steps; weights:", out_path)
    print("[flux-ft] v1 notes: surface = block matmuls (biases/norms/mod.lins/")
    print("  embedders/final layer frozen — OT trains those too, documented delta);")
    print("  RESUME: pass the saved overlay as the arg after steps (adafactor")
    print("  sidecar derived from it); inline sampling: set sample_every>0")
    print("  (samples the LIVE store at 512, cached-caption t5+clip cond,")
    print("  guidance-distilled single fwd, NO save-before-sample); the base")
    print("  ckpt is SCHNELL bytes under the dev name.")


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

    # ── FULL FINETUNE arm (FLUX_FULL_FT): its own self-contained loop —
    # bypasses the entire LoRA machinery below (incl. the LoRA-recipe pins in
    # validate_flux_train_config; the FT arm has its own fail-loud validate).
    # Gate-don't-fork (C13).
    comptime if FLUX_FULL_FT:
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
                    String("flux full-FT resume: expected an overlay ")
                    + String(".safetensors path, got ") + ft_resume
                )
        if train_cfg.only_cache:
            raise Error("flux full-FT v1: only_cache not wired (fail-loud)")
        _flux_full_ft_run(
            train_cfg, ft_steps,
            ot_cache_dir_from_train_config(train_cfg, String(FT_CACHE_DIR)),
            ft_resume,
        )
        return

    validate_flux_train_config(train_cfg)
    var cache_preflight = create_onetrainer_cache_preflight_plan(train_cfg)
    validate_onetrainer_cache_preflight_plan(cache_preflight)

    var run_steps = DEFAULT_RUN_STEPS
    if len(a) > arg_base:
        run_steps = _parse_nonnegative_int(String(a[arg_base]))
    elif train_cfg.only_cache:
        run_steps = 0

    var ckpt = flux_checkpoint_from_train_config(train_cfg)
    var cache_dir = flux_cache_dir_from_train_config(train_cfg)
    var sample_cadence = flux_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = flux_sampling_enabled(sample_cadence)
    # sample-during-training output dir (<lora_dir>/samples). Created up front so a
    # step-0 / early sample has somewhere to write. Sampling reuses the SAME cached
    # conditioning (txt_tokens + clip_pool) the current step already loaded — see
    # flux_sample_resident.mojo header (v1 conditioning).
    var samples_dir = String(LORA_DIR) + String("/samples")
    if sample_enabled:
        makedirs(samples_dir, exist_ok=True)

    print("=== Flux (flux1-dev) REAL LoRA training loop (block-swap offload) ===")
    print("  config:", cfg_path)
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " Fmlp=", FMLP, " out_ch=", OUT_CH)
    print("  depth: NUM_DOUBLE=", NUM_DOUBLE, " NUM_SINGLE=", NUM_SINGLE, " (FULL flux1-dev)")
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", train_cfg.lora_rank, " alpha=", train_cfg.lora_alpha,
          " lr=", train_cfg.lr, " shift=", train_cfg.timestep_shift,
          " guidance=", GUIDANCE, " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
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
        print("[offload] async offload requested by config; Flux trainer currently uses synchronous TurboPlannedLoader")
    if train_cfg.only_cache:
        print("[Flux] only_cache requested; no train steps will run in this trainer")
        return

    var ctx = DeviceContext()

    # ── stack-level base (frozen; resident ~12.3 GB F32) ─────────────────────
    print("[load] FluxStackBase (img/txt_in, embedders, per-block mod.lin, final layer)")
    var base_st = SafeTensors.open(ckpt)
    var base = load_flux_stack_base(base_st, NUM_DOUBLE, NUM_SINGLE, True, ctx)
    print("[load] base resident")

    # ── block-swap offload loader (streams attn/mlp blocks one at a time) ────
    var plan = build_flux1_dev_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(ckpt, plan^, cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── 3-axis RoPE tables (positions fixed for 512px; built once) ───────────
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, H, Dh](HT, WT, ctx, STDtype.BF16)
    var cos = rope[0].to_host(ctx)
    var sin = rope[1].to_host(ctx)
    print("[load] flux 3-axis rope tables built (S*H x Dh/2)")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    # OneTrainer "#flux LoRA.json" default (empty layer_filter) LoRAs EVERY
    # transformer Linear: the block-projection adapters (build_flux_lora_set) AND
    # the stack-level adapters (build_flux_stack_lora_set: per-block modulation
    # linears + the embedder / input-projection / final linears). Both B=0 at init.
    var lora = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
    var n_adapters = total_adapters(lora)
    var stack_lora = build_flux_stack_lora_set(
        NUM_DOUBLE, NUM_SINGLE, D, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, True, RANK, ALPHA
    )
    var n_stack = total_stack_adapters(stack_lora)
    print("[lora] block adapters:", n_adapters,
          " (", DBL_STREAM_SLOTS * 2, "x", NUM_DOUBLE, "double +",
          SGL_SLOTS, "x", NUM_SINGLE, "single)")
    print("[lora] stack adapters:", n_stack,
          " (per-block mod.lin + embedders + input-proj + final = full OT default)")
    print("[lora] TOTAL trained LoRA modules:", n_adapters + n_stack)

    # ── cache ────────────────────────────────────────────────────────────────
    var files = _list_cache(cache_dir)
    print("[cache] samples:", len(files))

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    # guidance is pre-scaled *1000 (BFL time_factor; same as timestep).
    var guidance_list = List[Float32]()
    guidance_list.append(GUIDANCE * Float32(1000.0))
    var guidance = Optional[List[Float32]](guidance_list^)

    if sample_enabled and should_sample_completed_step(sample_cadence, 0):
        # step-0 sample (untrained LoRA == identity) is skipped: the in-loop
        # sampler conditions on the CURRENT step's cached caption embeds, which
        # are only loaded once the loop starts. First real sample fires at the
        # first completed step that hits the cadence (see the in-loop callsite).
        print("[cadence] step-0 sample skipped (untrained LoRA); first sample at next cadence step")
    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        # ── load sample ──
        var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        var st = SafeTensors.open(files[slot])
        var lat_cache = _cache_tensor(st, String("latent"), ctx)        # [16*64*64]
        var clip_pool_cache = _cache_tensor(st, String("clip_pool"), ctx)   # [768]
        var lat_raw = _host_f32_for_step_math(lat_cache, ctx)
        var clip_pool = _host_f32_for_step_math(clip_pool_cache, ctx)

        # t5_embed [1, seq, 4096] -> pad/truncate to [N_TXT, 4096] (zero pad rows).
        var t5_info = st.tensor_info(String("t5_embed"))
        var t5_seq = Int(t5_info.shape[1])
        var t5_cache = _cache_tensor(st, String("t5_embed"), ctx)       # [seq*4096]
        var t5_flat = _host_f32_for_step_math(t5_cache, ctx)
        var txt_tokens = List[Float32]()
        for r in range(N_TXT):
            if r < t5_seq:
                for c in range(TXT_CH):
                    txt_tokens.append(t5_flat[r * TXT_CH + c])
            else:
                for _ in range(TXT_CH):
                    txt_tokens.append(Float32(0.0))

        # ── VAE shift/scale (train_flux.rs:736) then pack_latents ──
        for i in range(len(lat_raw)):
            lat_raw[i] = (lat_raw[i] - VAE_SHIFT) * VAE_SCALE
        var latent_packed = _pack_latents(lat_raw)                 # [N_IMG*64]

        # ── timestep (train_flux.rs:767-813) ──
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
        # caller pre-scales t by 1000 (BFL time_factor; flux1_dit.mojo convention).
        var timestep = List[Float32]()
        timestep.append(t_model * Float32(1000.0))

        # ── flow-match in PACKED latent space ──
        # noisy = noise*sigma + latent*(1-sigma) ; target = noise - latent.
        var noise = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        # ── forward (offload, full depth) -> velocity [N_IMG, OUT_CH] ──
        # _full path applies BOTH block-projection LoRA (`lora`) and stack-level
        # LoRA (`stack_lora`) — the complete OneTrainer default surface.
        var fwd = flux_stack_lora_forward_offload_full[H, Dh, N_IMG, N_TXT, S](
            noisy.copy(), txt_tokens.copy(), timestep.copy(), guidance, clip_pool.copy(),
            base, loader, lora, stack_lora, cos.copy(), sin.copy(),
            D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
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
        # `clip_pool` (CLIP-pooled) is the text_embedder lin1 input, needed for
        # that adapter's d_a in the stack-level backward.
        var grads = flux_stack_lora_backward_offload_full[H, Dh, N_IMG, N_TXT, S](
            d_loss, noisy.copy(), txt_tokens.copy(), base, loader, lora,
            stack_lora, clip_pool.copy(), cos.copy(), sin.copy(), fwd,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
        )

        # ── grad norm + configured clip (block + stack grads, one global norm) ──
        var gn_before = _clip(grads, train_cfg.max_grad_norm)

        # ── AdamW (block adapters, then stack adapters) ──
        var step_lr = ot_lr_for_optimizer_step(train_cfg, k)
        flux_lora_adamw_step(
            lora, grads, k, step_lr, ctx,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
            train_cfg.weight_decay,
        )
        flux_stack_lora_adamw_step(
            stack_lora, grads, k, step_lr, ctx,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
            train_cfg.weight_decay,
        )

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        var b_absum = Float32(0.0)
        var b_nonzero = 0
        for i in range(n_adapters):
            var bs2 = _absum(lora.ad[i].b)
            b_absum += bs2
            if bs2 > 0.0:
                b_nonzero += 1
        print_trainer_progress(
            String("Flux-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[Flux-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

        var saved_this_step = False
        if flux_should_save_checkpoint(train_cfg, k):
            var save_path = _step_lora_path(
                flux_output_lora_path_from_train_config(train_cfg, run_steps), k
            )
            _ = save_flux_lora_combined(lora, stack_lora, save_path, ctx)
            var state_path = save_path + String(".state.safetensors")
            _ = save_flux_lora_state_combined(lora, stack_lora, state_path, ctx)
            saved_this_step = True
            print("[Flux-lora] save_state step=", k, " path=", state_path)
        if sample_enabled and should_sample_completed_step(sample_cadence, k):
            if flux_should_save_before_sample(sample_cadence, k, saved_this_step):
                var sample_path = _step_lora_path(
                    flux_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                _ = save_flux_lora_combined(lora, stack_lora, sample_path, ctx)
                var sample_state = sample_path + String(".state.safetensors")
                _ = save_flux_lora_state_combined(lora, stack_lora, sample_state, ctx)
                print("[Flux-lora] save_before_sample step=", k, " path=", sample_state)
            # ── sample-during-training (v1; guidance-distilled, single-fwd Euler) ─
            # Denoise from the CURRENT frozen base + streamed blocks + live LoRA,
            # conditioned on THIS step's cached caption embeds (txt_tokens +
            # clip_pool — the v1 conditioning, see flux_sample_resident.mojo).
            # WARNING: each sample re-streams all 57 blocks SAMPLE_STEPS times via
            # the same `loader`; rare cadence only. Fail-loud — any raise aborts.
            print(
                "[cadence] sample due at completed_step=", k,
                " sample_file=", sample_cadence.sample_definition_file_name,
                " — denoising", SAMPLE_STEPS, "steps (re-streams blocks)",
            )
            var sample_packed = flux_sample_offload[
                H, Dh, N_IMG, N_TXT, S, IN_CH, OUT_CH
            ](
                base, loader, lora,
                txt_tokens.copy(), clip_pool.copy(), cos.copy(), sin.copy(),
                GUIDANCE, SAMPLE_STEPS, SAMPLE_SEED + UInt64(k),
                D, FMLP, TXT_CH, T_DIM, VEC_DIM, EPS, ctx,
            )
            var sample_png = (
                samples_dir + String("/step_") + String(k) + String(".png")
            )
            flux_decode_packed_to_png[
                N_IMG, HT, WT, LAT_H, LAT_W, LAT_C
            ](sample_packed, String(VAE_PATH), sample_png, ctx)
            print("[Flux-lora] sample step=", k, " -> ", sample_png)

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(n_adapters):
        b_absum_final += _absum(lora.ad[i].b)
    # stack-level LoRA-B growth (per OT default, these must also train).
    var stack_b_final = Float32(0.0)
    for slot in range(len(stack_lora.level)):
        if stack_lora.level[slot]:
            stack_b_final += _absum(stack_lora.level[slot].value().b)
    for i in range(len(stack_lora.dbl_img_mod)):
        if stack_lora.dbl_img_mod[i]:
            stack_b_final += _absum(stack_lora.dbl_img_mod[i].value().b)
        if stack_lora.dbl_txt_mod[i]:
            stack_b_final += _absum(stack_lora.dbl_txt_mod[i].value().b)
    for i in range(len(stack_lora.sgl_mod)):
        if stack_lora.sgl_mod[i]:
            stack_b_final += _absum(stack_lora.sgl_mod[i].value().b)
    print("[lora] stack LoRA-B |.|_1 final =", stack_b_final, " (expect > 0 — trained)")
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0) and (stack_b_final > 0.0)
    if trains and (last_loss == last_loss):
        print("RESULT: REAL run OK — block LoRA-B grew 0 ->", b_absum_final,
              "; stack LoRA-B ->", stack_b_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        var lora_out = flux_output_lora_path_from_train_config(train_cfg, run_steps)
        _ = save_flux_lora_combined(lora, stack_lora, lora_out, ctx)
        var state_out = lora_out + String(".state.safetensors")
        _ = save_flux_lora_state_combined(lora, stack_lora, state_out, ctx)
        print("[Flux-lora] save_state step=", run_steps, " path=", state_out)
    else:
        print("RESULT: FAIL trains=", trains)
