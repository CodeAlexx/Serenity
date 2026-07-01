# training/train_l2p_real.mojo — Z-Image L2P (pixel-space) LoRA REAL training loop.
#
# FAITHFUL to ai-toolkit (authoritative reference) + EDv2 train_l2p.rs (the
# parity-verified Rust port). L2P = Z-Image-Turbo DiT body (reused VERBATIM) +
# 16×16 pixel-space patchify x_embedder + FROZEN MicroDiffusionModel U-Net head
# (the `local_decoder`). LoRA trains ONLY the 30 main DiT `layers` blocks.
#
# Reference (read FULL):
#   ai-toolkit  extensions_built_in/diffusion_models/z_image/z_image_l2p_model.py
#               (MicroDiffusionModel + L2P forward + FakeVAE pixel path)
#   ai-toolkit  toolkit/samplers/custom_flowmatch_sampler.py (add_noise, linear ts)
#   ai-toolkit  jobs/process/BaseSDTrainProcess.py (uniform timestep, noise, loss)
#   EDv2        crates/eridiffusion-cli/src/bin/train_l2p.rs (cross-ref recipe)
#
# RECIPE (ai-toolkit / EDv2, all four prior divergences FIXED here):
#   1. CACHE: reads {pixel [3,512,512] F32, cap_feats [1,seq,2560] F32} via
#      L2PCache (NOT the Klein {latent,text_embedding,text_mask} contract).
#      cap_feats seq VARIES per sample and is ALREADY trimmed to valid tokens —
#      there is NO text_mask; valid_cap := cap_feats.shape[1].
#   2. HEAD: runs the REAL FROZEN local_decoder (MicroDiffusionModel U-Net)
#      forward+backward (models/l2p/local_decoder_train.mojo). The DiT's last
#      image-token hidden [N_IMG, D] IS the feature map (NO final layer-norm /
#      modulate / linear — ai-toolkit has none). pred = local_decoder(noisy, feat).
#   3. TIMESTEP: UNIFORM UNSHIFTED — t_int = randint(0, NUM_TRAIN_TIMESTEPS)+1,
#      sigma = t_int / NUM_TRAIN_TIMESTEPS (ai-toolkit timestep_type='linear').
#      shift=3.0 is the INFERENCE sigma schedule only; it does NOT apply here.
#   4. LoRA: 30 main blocks, 7 Z-Image slots (to_q/to_k/to_v/to_out.0/w1/w3/w2).
#      PEFT save keys via save_zimage_lora_main_only.
#      NOTE (#4 partial): ai-toolkit/EDv2 also LoRA the per-block
#      adaLN_modulation.0 (8th target). The Mojo Z-Image LoRA infra is hardwired
#      to ZIMAGE_SLOTS=7 with no adaLN slot; adding it is a cross-cutting change
#      to the shared Z-Image LoRA stack (struct + fwd + bwd + AdamW + save) that
#      also touches the production zimage trainer. NOT done here — see the
#      BUILD REQUEST / DELIVERABLE notes. This trainer matches the 7-slot set.
#
# FLOW-MATCH (rectified):
#   noisy = (1 - sigma) * pixel + sigma * noise
#   target = noise - pixel                      (v-target in PIXEL space)
#   pred  = local_decoder(noisy, feat)          (returns -v_raw via DiT/decoder)
#   loss  = mean((pred - target)^2)             (F32)
#   We negate the DiT/decoder output (pred = -decoder_out) to match Python's
#   `model_fn_z_image` which returns -DiT(...); target stays noise - pixel.
#
# DTYPE:
#   * DiT base weights: bf16 (large) + f32 (norms) — mixed (as loaded).
#   * LoRA A/B masters/grads: F32.
#   * local_decoder convs: F32 (the conv/pool/silu backward kernels are F32-only).
#   * Pixels, noise, feat, loss: F32 host / device.
#
# COMPILE-ONLY GATE (orchestrator owns the compile):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo build -I . serenitymojo/training/train_l2p_real.mojo -o /tmp/train_l2p_real

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.ffi import sys_system
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.zimage_stack import ZImageStackForward
from serenitymojo.models.zimage.lora_block import ZIMAGE_SLOTS
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraSet, ZImageLoraGrads, build_zimage_lora_set,
    zimage_lora_set_to_device,
    zimage_stack_lora_forward_main_device,
    zimage_stack_lora_backward_main_device_nofinal,
    zimage_lora_adamw_step_main_only, save_zimage_lora_main_only,
    save_zimage_lora_main_only_state, load_zimage_lora_main_only_state,
    ZIMAGE_DIRECT_24_GIB,
    empty_zimage_direct_dora_set, empty_zimage_direct_oft_set,
    zimage_direct_dense_carrier_bytes,
    zimage_direct_dora_preflight, zimage_direct_oft_preflight,
    build_zimage_direct_dora_set_from_main_blocks,
    build_zimage_direct_oft_set_for_main_blocks,
    zimage_stack_direct_dora_forward_main_device,
    zimage_stack_direct_oft_forward_main_device,
    zimage_stack_direct_dora_backward_main_device_nofinal,
    zimage_stack_direct_oft_backward_main_device_nofinal,
    zimage_direct_dora_grad_norm, zimage_direct_dora_clip_grads,
    zimage_direct_dora_adamw_step, zimage_direct_dora_zero_leg_l1,
    zimage_direct_dora_trainable_bytes, save_zimage_direct_dora,
    zimage_direct_oft_grad_norm, zimage_direct_oft_clip_grads,
    zimage_direct_oft_adamw_step, zimage_direct_oft_vec_l1,
    zimage_direct_oft_trainable_bytes, save_zimage_direct_oft,
)
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectOFTSet,
)
from serenitymojo.models.zimage.zimage_lokr_stack import (
    ZImageLoKrSet, empty_zimage_lokr_set, build_zimage_lokr_set,
    zimage_lokr_carrier_lists, zimage_lokr_carrier_total_bytes,
    zimage_lokr_chain_all, zimage_lokr_adamw_step, zimage_lokr_grad_norm,
    zimage_lokr_clip_grads, zimage_lokr_zero_leg_l1, save_zimage_lokr,
)
from serenitymojo.models.zimage.zimage_loha_stack import (
    ZImageLoHaSet, empty_zimage_loha_set, build_zimage_loha_set,
    zimage_loha_carrier_lists, zimage_loha_carrier_total_bytes,
    zimage_loha_chain_all, zimage_loha_adamw_step, zimage_loha_grad_norm,
    zimage_loha_clip_grads, zimage_loha_zero_leg_l1, save_zimage_loha,
)
from serenitymojo.models.l2p.weights import (
    L2PRealAux, load_l2p_real_aux, load_l2p_block_weights_prefixed,
    build_l2p_adaln, build_l2p_block_modvecs, build_l2p_cap_seq,
    build_l2p_x_seq, build_l2p_rope, build_l2p_positions,
)
from serenitymojo.models.l2p.local_decoder_train import (
    L2PDecoderF32, l2p_decoder_f32_from_gate,
    l2p_decoder_forward, l2p_decoder_backward,
)
from serenitymojo.models.dit.zimage_l2p_local_decoder import ZImageL2PLocalDecoderGate
from serenitymojo.training.klein_dataset import L2PCache
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import (
    TrainConfig,
    TRAIN_ADAPTER_ALGO_LORA,
    TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOHA,
    TRAIN_ADAPTER_ALGO_LOKR,
    TRAIN_ADAPTER_ALGO_DORA,
    TRAIN_ADAPTER_ALGO_OFT,
    TRAIN_ADAPTER_ALGO_BOFT,
    TRAIN_ADAPTER_ALGO_LOCON,
)
from serenitymojo.training.adapter_algo_policy import adapter_algo_name
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES


# ── arch (Z-Image L2P; IDENTICAL body to Z-Image base) ───────────────────────
comptime H = 30
comptime Dh = 128
comptime D = H * Dh          # 3840
comptime F = 10240           # SwiGLU per-gate hidden
comptime CAP_DIM = 2560      # Qwen3 hidden
comptime ADALN_DIM = 256     # t_embedder output dim
comptime T_SCALE = Float32(1000.0)
comptime ROPE_THETA = Float32(256.0)
comptime AXIS0 = 32
comptime AXIS1 = 48
comptime AXIS2 = 48
comptime EPS = Float32(1e-5)
comptime FINAL_EPS = Float32(1e-6)

# ── pixel-space L2P specifics ─────────────────────────────────────────────────
comptime PIX_C = 3           # RGB channels (in_channels=3 per l2p.json)
comptime PATCH = 16          # patchify16
comptime PATCH_VEC = PIX_C * PATCH * PATCH  # 768

# ── resolution: 512x512 training bucket -> 32x32 = 1024 image tokens (no pad) ─
comptime PIX_H = 512
comptime PIX_W = 512
comptime HT = PIX_H // PATCH   # 32  (feat grid H; also p4 grid after 4 pools)
comptime WT = PIX_W // PATCH   # 32
comptime N_IMG = HT * WT       # 1024 (1024 % 32 == 0, no padding needed)

# ── caption sequence: bucketed to CAP_LEN; valid rows from cap_feats.shape[1] ─
comptime CAP_LEN = 224

# ── unified sequence ──────────────────────────────────────────────────────────
comptime N_TXT = CAP_LEN
comptime S = N_IMG + N_TXT    # 1248

# ── depth (full L2P = 2 NR + 2 CR + 30 main; refiners excluded from LoRA) ────
comptime NUM_NR = 2
comptime NUM_CR = 2
comptime MAIN_DEPTH = 30

# ── recipe (ai-toolkit / EDv2 / l2p.json) ────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(3.0e-4)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

# ── paths ─────────────────────────────────────────────────────────────────────
comptime CHECKPOINT_PATH = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
comptime CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/alina_l2p"

# Adapter slice: NR+CR blocks are allocated; only MAIN blocks are trained.
comptime TRAIN_ADAPTER_START = (NUM_NR + NUM_CR) * ZIMAGE_SLOTS
comptime N_ADAPTERS_TOTAL = (NUM_NR + NUM_CR + MAIN_DEPTH) * ZIMAGE_SLOTS
comptime L2P_OFT_BLOCK_SIZE = 4


# ── host math helpers ─────────────────────────────────────────────────────────

def _host_noise_l2p(n: Int, seed: UInt64) -> List[Float32]:
    """Box-Muller PCG Gaussian noise N(0,1) — same LCG as zimage trainer."""
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int(state >> 11)) * (1.0 / 9007199254740992.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int(state >> 11)) * (1.0 / 9007199254740992.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


def _uniform_t_int(seed: UInt64, num_steps: Int) -> Int:
    """Uniform integer in [1, num_steps] (ai-toolkit/EDv2: randint(0,num)+1)."""
    var state = seed * 6364136223846793005 + 1442695040888963407
    var u = UInt64((state >> 11)) % UInt64(num_steps)
    return Int(u) + 1


def _absum_l2p[dt: DType](v: List[Scalar[dt]]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = Float32(v[i])
        s += x if x >= 0.0 else -x
    return s


def _parse_nonnegative_int_l2p(s: String) raises -> Int:
    if s.byte_length() == 0:
        raise Error("train_l2p_real: expected non-negative integer")
    var out = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        if bs[i] < 0x30 or bs[i] > 0x39:
            raise Error(String("train_l2p_real: expected non-negative integer, got ") + s)
        out = out * 10 + Int(bs[i] - 0x30)
    return out


def _close_l2p(a: Float32, b: Float32, tol: Float32 = Float32(1.0e-7)) -> Bool:
    var d = a - b
    if d < Float32(0.0):
        d = -d
    return d <= tol


def validate_l2p_train_config(cfg: TrainConfig) raises:
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[L2P-locon] network_algorithm=locon: using the linear LoRA-compatible down/up path")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR:
        print("[L2P-lokr] network_algorithm=lokr: using main-layer carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA:
        print("[L2P-loha] network_algorithm=loha: using main-layer carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA or cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT:
        print(
            String("[L2P-") + adapter_algo_name(cfg.adapter_algo)
            + String("] network_algorithm=") + adapter_algo_name(cfg.adapter_algo)
            + String(": using direct main-layer W_eff substitution; BOFT remains excluded")
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error("L2P trainer: BOFT is intentionally excluded; use lora, locon, loha, lokr, dora, or oft where wired")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        raise Error("L2P trainer: full finetune is not wired; supported here: lora, locon, loha, lokr, dora, oft")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error(
            String("L2P trainer: adapter algorithm ")
            + adapter_algo_name(cfg.adapter_algo)
            + String(" is not wired; supported here: lora, locon, loha, lokr, dora, oft")
        )
    if cfg.name != String("l2p") and cfg.name != String("zimage_l2p"):
        raise Error(String("L2P trainer config requires model_type=l2p, got ") + cfg.name)
    if cfg.checkpoint == String(""):
        raise Error("L2P trainer config must set checkpoint")
    if cfg.dataset_cache_dir == String("") and cfg.cache_dir == String(""):
        raise Error("L2P trainer config must set cache_dir")
    if cfg.n_heads != H:
        raise Error(String("L2P config num_heads ") + String(cfg.n_heads) + String(" != H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("L2P config head_dim ") + String(cfg.head_dim) + String(" != Dh ") + String(Dh))
    if cfg.d_model != D:
        raise Error(String("L2P config inner_dim ") + String(cfg.d_model) + String(" != D ") + String(D))
    if cfg.in_channels != PIX_C:
        raise Error(String("L2P config in_channels ") + String(cfg.in_channels) + String(" != PIX_C ") + String(PIX_C))
    if cfg.joint_attention_dim != CAP_DIM:
        raise Error(String("L2P config joint_attention_dim ") + String(cfg.joint_attention_dim) + String(" != CAP_DIM ") + String(CAP_DIM))
    if cfg.out_channels != PIX_C:
        raise Error(String("L2P config out_channels ") + String(cfg.out_channels) + String(" != PIX_C ") + String(PIX_C))
    if cfg.num_double != 0 or cfg.num_single != MAIN_DEPTH:
        raise Error(
            String("L2P trainer requires num_double=0 num_single=")
            + String(MAIN_DEPTH)
            + String("; got double=") + String(cfg.num_double)
            + String(" single=") + String(cfg.num_single)
        )
    if cfg.mlp_hidden != F:
        raise Error(String("L2P config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != F ") + String(F))
    if cfg.lora_rank != RANK:
        raise Error(String("L2P trainer is compiled for lora_rank=") + String(RANK))
    if not _close_l2p(cfg.lora_alpha, ALPHA):
        raise Error("L2P trainer lora_alpha does not match compiled constant")
    if not _close_l2p(cfg.lr, LR, Float32(1.0e-9)):
        raise Error("L2P trainer learning_rate does not match compiled constant")
    if not _close_l2p(cfg.max_grad_norm, CLIP_GRAD_NORM):
        raise Error("L2P trainer max_grad_norm does not match compiled constant")


def _global_norm_l2p(grads: ZImageLoraGrads, start: Int, end: Int) -> Float64:
    var ss = 0.0
    for i in range(start, end):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip_l2p(
    mut grads: ZImageLoraGrads, max_norm: Float32, start: Int, end: Int
) -> Float64:
    var gn = _global_norm_l2p(grads, start, end)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(start, end):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


# ── feat map seam: [N_IMG, D] host (token-major, t = ih*WT+iw) <-> NCHW [1,D,HT,WT]
def _tokens_to_feat_nchw(
    x_final_host: List[Float32], ctx: DeviceContext
) raises -> Tensor:
    """Image rows [0,N_IMG) of x_final [S,D] -> feat map NCHW [1,D,HT,WT].
    Token order is row-major (ih,iw): t = ih*WT + iw, matching build_l2p_x_seq."""
    var feat = List[Float32]()
    for _ in range(D * N_IMG):
        feat.append(Float32(0.0))
    for ih in range(HT):
        for iw in range(WT):
            var t = ih * WT + iw
            for d in range(D):
                # NCHW flat: ((0*D + d)*HT + ih)*WT + iw
                feat[(d * HT + ih) * WT + iw] = x_final_host[t * D + d]
    return Tensor.from_host(feat^, [1, D, HT, WT], STDtype.F32, ctx)


def _feat_nchw_to_tokens(d_feat: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    """d_feat NCHW [1,D,HT,WT] -> d_x_full [S,D] (image rows filled, cap rows 0)."""
    var dh = d_feat.to_host(ctx)
    var out = List[Float32]()
    for _ in range(S * D):
        out.append(Float32(0.0))
    for ih in range(HT):
        for iw in range(WT):
            var t = ih * WT + iw
            for d in range(D):
                out[t * D + d] = dh[(d * HT + ih) * WT + iw]
    return out^


# ── per-step result ───────────────────────────────────────────────────────────
@fieldwise_init
struct L2PStepResult(Copyable, Movable):
    var loss: Float32
    var grad: Float32
    var secs: Float32
    var lora_b_sum: Float32
    var nonfinite: Int


def _train_one_step_l2p(
    k: Int,
    run_steps: Int,
    slot: Int,
    step_seed: UInt64,
    cache: L2PCache,
    aux: L2PRealAux,
    dec: L2PDecoderF32,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    mut lora: ZImageLoraSet,
    lokr_active: Bool,
    loha_active: Bool,
    dora_active: Bool,
    oft_active: Bool,
    mut lokr_masters: ZImageLoKrSet,
    mut loha_masters: ZImageLoHaSet,
    mut direct_dora: FlatDirectDoRASet,
    mut direct_oft: FlatDirectOFTSet,
    direct_targets: Int,
    step_lr: Float32,
    max_grad_norm: Float32,
    beta1: Float32,
    beta2: Float32,
    optimizer_eps: Float32,
    weight_decay: Float32,
    train_start_ns: UInt,
    ctx: DeviceContext,
) raises -> L2PStepResult:
    var t0 = perf_counter_ns()

    # ── load cached sample: pixel [3,H,W] F32, cap_feats [1,seq,2560] F32 ──────
    var s = cache.load(slot, ctx)
    var psh = s.pixel.shape()
    if len(psh) != 3 or psh[0] != PIX_C or psh[1] != PIX_H or psh[2] != PIX_W:
        raise Error("train_l2p_real: pixel shape mismatch — expected [3,512,512]")
    var csh = s.cap_feats.shape()
    if len(csh) != 3 or csh[0] != 1 or csh[2] != CAP_DIM:
        raise Error("train_l2p_real: cap_feats shape mismatch — expected [1,seq,2560]")
    var valid_cap = csh[1]      # cap_feats already trimmed to valid tokens; NO mask
    if valid_cap <= 0 or valid_cap > CAP_LEN:
        raise Error("train_l2p_real: caption length out of range")

    var pix_h = cast_tensor(s.pixel, STDtype.F32, ctx).to_host(ctx)  # [3,512,512] flat

    # ── timestep: UNIFORM UNSHIFTED (ai-toolkit timestep_type='linear') ───────
    var t_int = _uniform_t_int(SEED_BASE + step_seed, NUM_TRAIN_TIMESTEPS)
    var sigma = Float32(t_int) / Float32(NUM_TRAIN_TIMESTEPS)
    # DiT timestep input: v_in = (1 - sigma). build_l2p_adaln does t_val*T_SCALE
    # with NO internal inversion, and the verified inference contract is
    # zimage_l2p_model_timestep(sigma) = (1-sigma)*1000 (zimage_l2p_contract.mojo:105;
    # ai-toolkit (1000-timestep)/1000; EDv2 dit.rs t=(1-v)*time_scale).
    var t_value = Float32(1.0) - sigma

    # ── pixel noise + noisy pixels (rectified flow) ──────────────────────────
    var noise_pix = _host_noise_l2p(PIX_C * PIX_H * PIX_W, SEED_BASE * UInt64(7919) + step_seed)
    var noisy_pix_h = List[Float32]()
    for i in range(len(pix_h)):
        noisy_pix_h.append(pix_h[i] * (Float32(1.0) - sigma) + noise_pix[i] * sigma)
    var noisy_pixel_t = Tensor.from_host(noisy_pix_h^, [1, PIX_C, PIX_H, PIX_W], STDtype.F32, ctx)

    # ── adaln + modvecs ───────────────────────────────────────────────────────
    var adaln = build_l2p_adaln(aux, t_value, T_SCALE, ctx)
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod.append(build_l2p_block_modvecs(
            aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln, D, ctx
        ))
    var main_mod = List[ZImageModVecs]()
    for i in range(MAIN_DEPTH):
        main_mod.append(build_l2p_block_modvecs(
            aux.main_mod_w[i][], aux.main_mod_b[i][], adaln, D, ctx
        ))

    # ── x_seq: patchify16(noisy_pixels) -> Linear -> [N_IMG, D] ──────────────
    var x_t_host = build_l2p_x_seq(aux, noisy_pixel_t, PIX_H, PIX_W, ctx)

    # ── cap_seq from cap_feats (valid_cap rows, pad rest with cap_pad_token) ──
    var cap_feats = cast_tensor(s.cap_feats, STDtype.F32, ctx)   # [1,seq,2560]
    var cap_full = cap_feats.to_host(ctx)
    var cap_vals = List[Float32]()
    for r in range(CAP_LEN):
        var src_r = r if r < valid_cap else valid_cap - 1
        for c in range(CAP_DIM):
            cap_vals.append(cap_full[src_r * CAP_DIM + c])
    var cap2 = Tensor.from_host(cap_vals^, [CAP_LEN, CAP_DIM], STDtype.F32, ctx)
    var cap_seq = build_l2p_cap_seq(aux, cap2, EPS, ctx)
    var cap_pad_h = aux.cap_pad_token[].to_host(ctx)
    for r in range(valid_cap, CAP_LEN):
        for c in range(D):
            cap_seq[r * D + c] = cap_pad_h[c]

    # ── rope ──────────────────────────────────────────────────────────────────
    var pos_step = build_l2p_positions(N_IMG, HT, WT, CAP_LEN, valid_cap)
    var x_pos = pos_step[0].copy()
    var cap_pos = pos_step[1].copy()
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var xr = build_l2p_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos = xr[0].copy(); var x_sin = xr[1].copy()
    var ur = build_l2p_rope(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos = ur[0].copy(); var uni_sin = ur[1].copy()
    var crr = build_l2p_rope(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos = crr[0].copy(); var cap_sin = crr[1].copy()

    var t_prep = perf_counter_ns()

    # ── DiT stack forward (last-block hidden = the feature map; NO final layer)
    # We reuse the proven stack forward. ai-toolkit L2P has no final layer-norm/
    # modulate/linear; we pass an IDENTITY-shaped final layer purely so the
    # existing forward runs, but we IGNORE fwd.out and use saved.x_final (the
    # last main-block output) as the feature source. f_scale=zeros, out_ch=D,
    # final_lin_w=identity[D,D], final_lin_b=zeros — these only affect fwd.out,
    # which we discard. The backward we call (nofinal) ignores them entirely.
    var f_scale_zeros = List[Float32]()
    for _ in range(D):
        f_scale_zeros.append(Float32(0.0))
    var ident_host = List[Float32]()
    for _ in range(D * D):
        ident_host.append(Float32(0.0))
    for d in range(D):
        ident_host[d * D + d] = Float32(1.0)
    var ident_w = Tensor.from_host(ident_host^, [D, D], STDtype.F32, ctx)
    var zero_b_host = List[Float32]()
    for _ in range(D):
        zero_b_host.append(Float32(0.0))
    var zero_b = Tensor.from_host(zero_b_host^, [D], STDtype.F32, ctx)

    var lora_dev = zimage_lora_set_to_device(lora, ctx)
    var t_lora = perf_counter_ns()

    var fwd: ZImageStackForward
    if dora_active:
        fwd = zimage_stack_direct_dora_forward_main_device[H, Dh, N_IMG, N_TXT, S](
            x_t_host.copy(), cap_seq.copy(),
            nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, direct_dora,
            direct_targets, f_scale_zeros.copy(),
            ident_w, zero_b,
            x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
            D, F, D, EPS, FINAL_EPS, ctx,
        )
    elif oft_active:
        fwd = zimage_stack_direct_oft_forward_main_device[H, Dh, N_IMG, N_TXT, S](
            x_t_host.copy(), cap_seq.copy(),
            nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, direct_oft,
            direct_targets, f_scale_zeros.copy(),
            ident_w, zero_b,
            x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
            D, F, D, EPS, FINAL_EPS, ctx,
        )
    else:
        fwd = zimage_stack_lora_forward_main_device[H, Dh, N_IMG, N_TXT, S](
            x_t_host.copy(), cap_seq.copy(),
            nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora_dev,
            f_scale_zeros.copy(),
            ident_w, zero_b,
            x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
            D, F, D, EPS, FINAL_EPS, ctx,
        )
    var t_fwd = perf_counter_ns()

    # ── feature map [1, D, HT, WT] from the last-block image-token hidden ─────
    var x_final_host = fwd.x_final[].to_host(ctx)   # [S, D]
    var feat_nchw = _tokens_to_feat_nchw(x_final_host, ctx)

    # ── REAL local_decoder forward (FROZEN): pred [1,3,512,512] ───────────────
    var dec_fwd = l2p_decoder_forward[PIX_H, PIX_W, HT, WT](
        dec, noisy_pixel_t, feat_nchw, ctx
    )
    var pred_h = dec_fwd.pred_nchw.to_host(ctx)     # [3,512,512] flat
    var t_dec = perf_counter_ns()

    # ── loss: target = noise - pixel ; pred = -decoder_out ; mean MSE (F32) ──
    var npix = PIX_C * PIX_H * PIX_W
    var d_pred_h = List[Float32]()
    for _ in range(npix):
        d_pred_h.append(Float32(0.0))
    var sse = 0.0
    var inv_n = Float32(2.0) / Float32(npix)
    for i in range(npix):
        var pred = -pred_h[i]
        var target = noise_pix[i] - pix_h[i]
        var diff = pred - target
        sse += Float64(diff) * Float64(diff)
        # dL/d(decoder_out) = dL/dpred * dpred/d(decoder_out) = (2/N)*diff * (-1)
        d_pred_h[i] = -inv_n * diff
    var loss = Float32(sse / Float64(npix))
    var d_pred_t = Tensor.from_host(d_pred_h^, [1, PIX_C, PIX_H, PIX_W], STDtype.F32, ctx)
    var t_loss = perf_counter_ns()

    # ── local_decoder backward (FROZEN): d_pred -> d_feat [1,D,HT,WT] ─────────
    var d_feat = l2p_decoder_backward[PIX_H, PIX_W, HT, WT](
        dec, dec_fwd.acts, d_pred_t, ctx
    )
    var d_x_full = _feat_nchw_to_tokens(d_feat, ctx)   # [S,D], image rows filled
    var t_dbwd = perf_counter_ns()

    if dora_active:
        var dg = zimage_stack_direct_dora_backward_main_device_nofinal[
            H, Dh, N_IMG, N_TXT, S
        ](
            d_x_full, main_blocks, main_mod, direct_dora, direct_targets,
            uni_cos[], uni_sin[], fwd, D, F, EPS, ctx,
        )
        var t_bwd = perf_counter_ns()
        var gn_before = zimage_direct_dora_grad_norm(dg.grads)
        if gn_before > Float64(max_grad_norm):
            zimage_direct_dora_clip_grads(dg.grads, max_grad_norm / Float32(gn_before))
        zimage_direct_dora_adamw_step(
            direct_dora, dg.grads, k, step_lr, beta1, beta2, optimizer_eps,
            weight_decay,
        )
        var t_opt = perf_counter_ns()
        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        var b_absum = Float32(zimage_direct_dora_zero_leg_l1(direct_dora))
        print_trainer_progress(
            String("L2P-dora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start_ns) / 1.0e9,
        )
        if dg.nonfinite_grads != 0:
            print("[L2P-dora] warning nonfinite_direct_grads=", dg.nonfinite_grads)
        print("[TIMING step=", k,
              "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
              " direct=", Float32(Float64(t_lora - t_prep) / 1.0e9),
              " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
              " dec=", Float32(Float64(t_dec - t_fwd) / 1.0e9),
              " loss=", Float32(Float64(t_loss - t_dec) / 1.0e9),
              " dbwd=", Float32(Float64(t_dbwd - t_loss) / 1.0e9),
              " bwd=", Float32(Float64(t_bwd - t_dbwd) / 1.0e9),
              " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
        return L2PStepResult(loss, Float32(gn_before), Float32(secs), b_absum, dg.nonfinite_grads)

    if oft_active:
        var og = zimage_stack_direct_oft_backward_main_device_nofinal[
            H, Dh, N_IMG, N_TXT, S
        ](
            d_x_full, main_blocks, main_mod, direct_oft, direct_targets,
            uni_cos[], uni_sin[], fwd, D, F, EPS, ctx,
        )
        var t_bwd = perf_counter_ns()
        var gn_before = zimage_direct_oft_grad_norm(og.grads)
        if gn_before > Float64(max_grad_norm):
            zimage_direct_oft_clip_grads(og.grads, max_grad_norm / Float32(gn_before))
        zimage_direct_oft_adamw_step(
            direct_oft, og.grads, k, step_lr, beta1, beta2, optimizer_eps,
            weight_decay,
        )
        var t_opt = perf_counter_ns()
        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        var b_absum = Float32(zimage_direct_oft_vec_l1(direct_oft))
        print_trainer_progress(
            String("L2P-oft"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start_ns) / 1.0e9,
        )
        if og.nonfinite_grads != 0:
            print("[L2P-oft] warning nonfinite_direct_grads=", og.nonfinite_grads)
        print("[TIMING step=", k,
              "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
              " direct=", Float32(Float64(t_lora - t_prep) / 1.0e9),
              " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
              " dec=", Float32(Float64(t_dec - t_fwd) / 1.0e9),
              " loss=", Float32(Float64(t_loss - t_dec) / 1.0e9),
              " dbwd=", Float32(Float64(t_dbwd - t_loss) / 1.0e9),
              " bwd=", Float32(Float64(t_bwd - t_dbwd) / 1.0e9),
              " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
        return L2PStepResult(loss, Float32(gn_before), Float32(secs), b_absum, og.nonfinite_grads)

    # ── DiT stack backward (no final layer): d_x_full -> LoRA grads ───────────
    var grads = zimage_stack_lora_backward_main_device_nofinal[H, Dh, N_IMG, N_TXT, S](
        d_x_full, main_blocks, main_mod, lora_dev,
        uni_cos[], uni_sin[], fwd,
        D, F, EPS, ctx,
    )
    var t_bwd = perf_counter_ns()

    # ── clip + optimize (main adapters only) ──────────────────────────────────
    var gn_before: Float64
    var progress_label = String("L2P-lora")
    if lokr_active:
        progress_label = String("L2P-lokr")
        var mg = zimage_lokr_chain_all(lokr_masters, grads.d_a, grads.d_b)
        var mnorm = zimage_lokr_grad_norm(mg)
        gn_before = mnorm
        if mnorm > Float64(max_grad_norm):
            zimage_lokr_clip_grads(mg, max_grad_norm / Float32(mnorm))
        zimage_lokr_adamw_step(
            lokr_masters, mg, k, step_lr, beta1, beta2, optimizer_eps,
            weight_decay,
        )
        var carriers = zimage_lokr_carrier_lists(lokr_masters, D, F)
        lora.ad = carriers^
        print("[L2P-lokr] step=", k, " master_grad_norm=", Float32(mnorm),
              " zero_leg_l1=", zimage_lokr_zero_leg_l1(lokr_masters))
    elif loha_active:
        progress_label = String("L2P-loha")
        var mg = zimage_loha_chain_all(loha_masters, grads.d_a, grads.d_b)
        var mnorm = zimage_loha_grad_norm(mg)
        gn_before = mnorm
        if mnorm > Float64(max_grad_norm):
            zimage_loha_clip_grads(mg, max_grad_norm / Float32(mnorm))
        zimage_loha_adamw_step(
            loha_masters, mg, k, step_lr, beta1, beta2, optimizer_eps,
            weight_decay,
        )
        var carriers = zimage_loha_carrier_lists(loha_masters, D, F)
        lora.ad = carriers^
        print("[L2P-loha] step=", k, " master_grad_norm=", Float32(mnorm),
              " zero_leg_l1=", zimage_loha_zero_leg_l1(loha_masters))
    else:
        gn_before = _clip_l2p(grads, max_grad_norm, TRAIN_ADAPTER_START, N_ADAPTERS_TOTAL)
        zimage_lora_adamw_step_main_only(
            lora, grads, k, step_lr, ctx, beta1, beta2, optimizer_eps,
            weight_decay,
        )
    var t_opt = perf_counter_ns()

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    var b_absum = Float32(0.0)
    if lokr_active:
        b_absum = Float32(zimage_lokr_zero_leg_l1(lokr_masters))
    elif loha_active:
        b_absum = Float32(zimage_loha_zero_leg_l1(loha_masters))
    else:
        for i in range(TRAIN_ADAPTER_START, N_ADAPTERS_TOTAL):
            b_absum += _absum_l2p(lora.ad[i].b)

    print_trainer_progress(
        progress_label, k, run_steps, 1,
        loss, Float64(gn_before), secs, 0.0,
        Float64(t1 - train_start_ns) / 1.0e9,
    )
    if grads.nonfinite_lora_grads != 0:
        print("[L2P-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)
    print("[TIMING step=", k,
          "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
          " lora=", Float32(Float64(t_lora - t_prep) / 1.0e9),
          " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
          " dec=", Float32(Float64(t_dec - t_fwd) / 1.0e9),
          " loss=", Float32(Float64(t_loss - t_dec) / 1.0e9),
          " dbwd=", Float32(Float64(t_dbwd - t_loss) / 1.0e9),
          " bwd=", Float32(Float64(t_bwd - t_dbwd) / 1.0e9),
          " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
    return L2PStepResult(loss, Float32(gn_before), Float32(secs), b_absum, grads.nonfinite_lora_grads)


# ── main ──────────────────────────────────────────────────────────────────────
def main() raises:
    var ctx = DeviceContext()
    var a = argv()
    var arg_base = 1
    var has_config = False
    var train_cfg = TrainConfig.default()
    if len(a) >= 2:
        var first = String(a[1])
        if first.endswith(String(".json")):
            train_cfg = read_model_config(first)
            has_config = True
            arg_base = 2
            validate_l2p_train_config(train_cfg)

    var run_steps = 5
    if has_config and train_cfg.max_steps > 0:
        run_steps = train_cfg.max_steps
    if len(a) > arg_base:
        run_steps = _parse_nonnegative_int_l2p(String(a[arg_base]))
    var start_step = 0
    if len(a) > arg_base + 1:
        start_step = _parse_nonnegative_int_l2p(String(a[arg_base + 1]))
    var resume_state = String("")
    if len(a) > arg_base + 2:
        resume_state = String(a[arg_base + 2])
    if run_steps < 1:
        raise Error("train_l2p_real: run_steps must be >= 1")
    if start_step > run_steps:
        raise Error("train_l2p_real: start_step cannot exceed run_steps")

    var ckpt_path = String(CHECKPOINT_PATH)
    var cache_dir = String(CACHE_DIR)
    if has_config:
        ckpt_path = train_cfg.checkpoint.copy()
        if train_cfg.dataset_cache_dir != String(""):
            cache_dir = train_cfg.dataset_cache_dir.copy()
        elif train_cfg.cache_dir != String(""):
            cache_dir = train_cfg.cache_dir.copy()
    var adapter_rank = RANK
    var adapter_alpha = ALPHA
    var step_lr = LR
    var max_grad_norm = CLIP_GRAD_NORM
    var beta1 = Float32(0.9)
    var beta2 = Float32(0.999)
    var optimizer_eps = Float32(1.0e-8)
    var weight_decay = Float32(0.01)
    if has_config:
        adapter_rank = train_cfg.lora_rank
        adapter_alpha = train_cfg.lora_alpha
        step_lr = train_cfg.lr
        max_grad_norm = train_cfg.max_grad_norm
        beta1 = train_cfg.beta1
        beta2 = train_cfg.beta2
        optimizer_eps = train_cfg.eps
        weight_decay = train_cfg.weight_decay
    var lokr_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var dora_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var carrier_active = lokr_active or loha_active
    var direct_active = dora_active or oft_active
    var direct_targets = 1 if train_cfg.lokr_targets == 1 else 2

    print("=== Z-Image L2P REAL LoRA training loop (ai-toolkit faithful) ===")
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " F=", F)
    print("  pixel input: C=", PIX_C, " H=", PIX_H, " W=", PIX_W,
          " patch=", PATCH, " feat grid=", HT, "x", WT)
    print("  depth: NR=", NUM_NR, " CR=", NUM_CR, " MAIN=", MAIN_DEPTH)
    print("  bucket: 512x512 -> N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", adapter_rank, " alpha=", adapter_alpha, " lr=", step_lr,
          " timestep=UNIFORM UNSHIFTED")
    print("  checkpoint:", ckpt_path)
    print("  cache:", cache_dir)
    print("  head: REAL FROZEN local_decoder (MicroDiffusionModel U-Net) fwd+bwd")

    # ── cache first: fail before loading the ~19 GB checkpoint ───────────────
    var cache = L2PCache(cache_dir.copy())
    print("[cache] samples:", cache.count())
    var k0 = cache.peek_key(0, ctx)
    print("[cache] first entry: C=", k0.c, " H=", k0.h, " W=", k0.w, " cap_seq=", k0.seq)
    if k0.c != PIX_C or k0.h != PIX_H or k0.w != PIX_W:
        raise Error("train_l2p_real: cache pixel shape mismatch — expected [3,512,512]")

    # ── load checkpoint ───────────────────────────────────────────────────────
    print("[load] opening single-file checkpoint")
    var st = SafeTensors.open(ckpt_path.copy())
    print("[load] tensors in checkpoint:", st.count())
    print("[load] aux (embedders + adaLN per block)")
    var aux = load_l2p_real_aux(st, NUM_NR, MAIN_DEPTH, ctx)
    print("[load] blocks: NR + CR + MAIN")
    var nr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_NR):
        nr_blocks.append(load_l2p_block_weights_prefixed(
            st, String("noise_refiner.") + String(i), ctx
        ))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(load_l2p_block_weights_prefixed(
            st, String("context_refiner.") + String(i), ctx
        ))
    var main_blocks = List[ZImageBlockWeights]()
    for i in range(MAIN_DEPTH):
        main_blocks.append(load_l2p_block_weights_prefixed(
            st, String("layers.") + String(i), ctx
        ))
    print("[load] resident:", len(nr_blocks), "nr +", len(cr_blocks), "cr +",
          len(main_blocks), "main blocks")

    # ── FROZEN local_decoder (load BF16 gate, cast convs to F32 once) ─────────
    print("[load] local_decoder (MicroDiffusionModel U-Net, FROZEN)")
    var dec_gate = ZImageL2PLocalDecoderGate.load(ckpt_path.copy(), ctx)
    var dec = l2p_decoder_f32_from_gate(dec_gate, ctx)

    # ── LoRA / LyCORIS carrier set ────────────────────────────────────────────
    var lora = build_zimage_lora_set(NUM_NR, NUM_CR, MAIN_DEPTH, D, F, adapter_rank, adapter_alpha)
    var lokr_masters = empty_zimage_lokr_set()
    var loha_masters = empty_zimage_loha_set()
    var direct_dora = empty_zimage_direct_dora_set()
    var direct_oft = empty_zimage_direct_oft_set()
    if (carrier_active or direct_active) and resume_state != String("") and resume_state != String("-"):
        raise Error("L2P LyCORIS direct/carrier path: resume state is not wired")
    if (not carrier_active) and (not direct_active) and resume_state != String("") and resume_state != String("-"):
        print("[L2P-lora] loading resume state:", resume_state)
        lora = load_zimage_lora_main_only_state(
            NUM_NR, NUM_CR, MAIN_DEPTH, adapter_rank, adapter_alpha, D, F,
            resume_state, ctx,
        )
    if carrier_active:
        if lokr_active:
            lokr_masters = build_zimage_lokr_set(
                NUM_NR, NUM_CR, MAIN_DEPTH, D, F,
                adapter_rank, adapter_alpha,
                train_cfg.lokr_factor, train_cfg.lokr_decompose_both,
                train_cfg.lokr_full_matrix, direct_targets,
                train_cfg.seed * UInt64(53) + UInt64(11),
            )
            for i in range(TRAIN_ADAPTER_START):
                lokr_masters.active[i] = False
            var carrier_bytes = zimage_lokr_carrier_total_bytes(lokr_masters, D, F)
            print("[L2P-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
            if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
                raise Error(
                    String("L2P LoKr carrier set needs ")
                    + String(carrier_bytes) + String(" bytes on device (> budget ")
                    + String(LOKR_CARRIER_MAX_DEVICE_BYTES) + String(")")
                )
            var carriers = zimage_lokr_carrier_lists(lokr_masters, D, F)
            lora = ZImageLoraSet(carriers^, NUM_NR, NUM_CR, MAIN_DEPTH, adapter_rank)
            print("[L2P-lokr] carrier set materialized:", len(lora.ad), "adapters")
        elif loha_active:
            loha_masters = build_zimage_loha_set(
                NUM_NR, NUM_CR, MAIN_DEPTH, D, F,
                adapter_rank, adapter_alpha, direct_targets,
                train_cfg.seed * UInt64(53) + UInt64(11),
            )
            for i in range(TRAIN_ADAPTER_START):
                loha_masters.active[i] = False
            var carrier_bytes = zimage_loha_carrier_total_bytes(loha_masters, D, F)
            print("[L2P-loha] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
            if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
                raise Error(
                    String("L2P LoHa carrier set needs ")
                    + String(carrier_bytes) + String(" bytes on device (> budget ")
                    + String(LOKR_CARRIER_MAX_DEVICE_BYTES) + String(")")
                )
            var carriers = zimage_loha_carrier_lists(loha_masters, D, F)
            lora = ZImageLoraSet(carriers^, NUM_NR, NUM_CR, MAIN_DEPTH, adapter_rank)
            print("[L2P-loha] carrier set materialized:", len(lora.ad), "adapters")
    elif dora_active:
        var dense_bytes = zimage_direct_dense_carrier_bytes(MAIN_DEPTH, D, F, direct_targets)
        var direct_bytes = zimage_direct_dora_preflight(
            MAIN_DEPTH, D, F, adapter_rank, direct_targets,
            ZIMAGE_DIRECT_24_GIB,
        )
        print("[L2P-dora] dense carrier bytes:", dense_bytes,
              " direct trainable bytes:", direct_bytes,
              " budget:", ZIMAGE_DIRECT_24_GIB)
        direct_dora = build_zimage_direct_dora_set_from_main_blocks(
            main_blocks, D, F, adapter_rank, adapter_alpha, direct_targets,
            train_cfg.seed * UInt64(53) + UInt64(29), False, ctx,
        )
        print("[L2P-dora] direct trainable bytes materialized:",
              zimage_direct_dora_trainable_bytes(direct_dora))
    elif oft_active:
        var dense_bytes = zimage_direct_dense_carrier_bytes(MAIN_DEPTH, D, F, direct_targets)
        var direct_bytes = zimage_direct_oft_preflight(
            MAIN_DEPTH, D, F, L2P_OFT_BLOCK_SIZE, direct_targets,
            ZIMAGE_DIRECT_24_GIB,
        )
        print("[L2P-oft] dense carrier bytes:", dense_bytes,
              " direct trainable bytes:", direct_bytes,
              " budget:", ZIMAGE_DIRECT_24_GIB)
        direct_oft = build_zimage_direct_oft_set_for_main_blocks(
            MAIN_DEPTH, D, F, L2P_OFT_BLOCK_SIZE, direct_targets,
        )
        print("[L2P-oft] direct trainable bytes materialized:",
              zimage_direct_oft_trainable_bytes(direct_oft))
    print("[lora] adapters:", MAIN_DEPTH * ZIMAGE_SLOTS, "trainable main;",
          N_ADAPTERS_TOTAL, "allocated total")
    var b_absum_init = Float32(0.0)
    if lokr_active:
        b_absum_init = Float32(zimage_lokr_zero_leg_l1(lokr_masters))
        print("[L2P-lokr] zero-leg L1 at init =", b_absum_init, " (expect 0.0)")
    elif loha_active:
        b_absum_init = Float32(zimage_loha_zero_leg_l1(loha_masters))
        print("[L2P-loha] zero-leg L1 at init =", b_absum_init, " (expect 0.0)")
    elif dora_active:
        b_absum_init = Float32(zimage_direct_dora_zero_leg_l1(direct_dora))
        print("[L2P-dora] zero-leg L1 at init =", b_absum_init, " (expect 0.0)")
    elif oft_active:
        b_absum_init = Float32(zimage_direct_oft_vec_l1(direct_oft))
        print("[L2P-oft] OFT vec L1 at init =", b_absum_init, " (expect 0.0)")
    else:
        for i in range(TRAIN_ADAPTER_START, N_ADAPTERS_TOTAL):
            b_absum_init += _absum_l2p(lora.ad[i].b)
        print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var train_start = perf_counter_ns()

    for k in range(start_step + 1, run_steps + 1):
        var slot = (k - 1) % cache.count()
        var step_seed = UInt64(k)
        var r = _train_one_step_l2p(
            k, run_steps, slot, step_seed, cache, aux, dec,
            nr_blocks, cr_blocks, main_blocks, lora,
            lokr_active, loha_active, dora_active, oft_active,
            lokr_masters, loha_masters, direct_dora, direct_oft,
            direct_targets,
            step_lr, max_grad_norm, beta1, beta2, optimizer_eps, weight_decay,
            train_start, ctx,
        )
        if k == start_step + 1:
            first_loss = r.loss
        last_loss = r.loss

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    if lokr_active:
        b_absum_final = Float32(zimage_lokr_zero_leg_l1(lokr_masters))
    elif loha_active:
        b_absum_final = Float32(zimage_loha_zero_leg_l1(loha_masters))
    elif dora_active:
        b_absum_final = Float32(zimage_direct_dora_zero_leg_l1(direct_dora))
    elif oft_active:
        b_absum_final = Float32(zimage_direct_oft_vec_l1(direct_oft))
    else:
        for i in range(TRAIN_ADAPTER_START, N_ADAPTERS_TOTAL):
            b_absum_final += _absum_l2p(lora.ad[i].b)
    var trains = b_absum_final > 0.0
    if trains and (last_loss == last_loss):
        var train_label = String("LyCORIS zero-leg") if (carrier_active or dora_active) else (String("OFT vec") if oft_active else String("LoRA-B"))
        print("RESULT: REAL L2P TRAIN OK — ", train_label, " grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        _ = sys_system(String("mkdir -p ") + String(LORA_DIR))
        var lora_out = String(LORA_DIR) + String("/l2p_lora_step") + String(run_steps) + String(".safetensors")
        if lokr_active:
            var nmods = save_zimage_lokr(lokr_masters, lora_out, ctx)
            print("[L2P-lokr] saved:", lora_out, " modules=", nmods)
        elif loha_active:
            var nmods = save_zimage_loha(loha_masters, lora_out, ctx)
            print("[L2P-loha] saved:", lora_out, " modules=", nmods)
        elif dora_active:
            var nmods = save_zimage_direct_dora(direct_dora, lora_out, ctx)
            print("[L2P-dora] saved:", lora_out, " modules=", nmods)
        elif oft_active:
            var nmods = save_zimage_direct_oft(direct_oft, lora_out, ctx)
            print("[L2P-oft] saved:", lora_out, " modules=", nmods)
        else:
            _ = save_zimage_lora_main_only(lora, lora_out, ctx)
            var state_out = lora_out + String(".state.safetensors")
            _ = save_zimage_lora_main_only_state(lora, state_out, ctx)
            print("[L2P-lora] saved:", lora_out)
            print("[L2P-lora] state:", state_out)
    else:
        print("RESULT: FAIL trains=", trains)
