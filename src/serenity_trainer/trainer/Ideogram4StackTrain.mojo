# Ideogram4StackTrain.mojo — optimizer bridge for Ideogram4 block-stack LoRA.
#
# The model module computes d_A/d_B for every repeated transformer block. This
# trainer module owns AdamW state and applies those gradients to the LoRA set.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import zeros_device

from serenity_trainer.model.Ideogram4LoRABlock import (
    I4_SLOTS_PER_BLOCK,
    Ideogram4LoraSet,
    Ideogram4StackLoraGrads,
    ideogram4_stack_lora_backward,
    ideogram4_stack_lora_backward_graph,
    ideogram4_stack_lora_forward,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.util.optimizer.adamw_extensions import adamw_step

comptime TArc = ArcPointer[Tensor]

# ── autograd_v2 GRAPH SWAP (P7 rollout, klein KLEIN_V2_GRAPH precedent) ───────
# When IDEOGRAM4_V2_GRAPH_PATH is on, the per-block recompute + hand-chain
# backward pair in the stack backward is driven by the autograd_v2 graph engine
# (ideogram4_stack_lora_backward_graph — same conductor loop, same slot fan-in,
# per-block coarse mini-graphs whose backward calls the hand-chain oracle;
# SAME-PROCESS bit gate: autograd_v2/tests/ideogram4_block_parity.mojo).
# ideogram4 is COARSE stage-1 = engine only, NO slab/capture (like Klein P6).
# False = the previous hand-chain path (C13 gate-don't-delete: the hand-chain in
# block.mojo stays compiled + reachable; flag default-OFF is byte-identical to
# today).
comptime IDEOGRAM4_V2_ENGINE = True
# DEFAULT-ON (2026-06-25): all trainers must dispatch backward through autograd_v2
# (Alex mandate). Gated by the per-block bit gate autograd_v2/tests/ideogram4_block_parity
# (engine grad == hand-chain grad BIT-EQUAL, lead-verified). Hand-chain stays reachable
# in the else branch (C13). End-to-end trainer N-step anchor pending the eri2 cache.
comptime IDEOGRAM4_V2_GRAPH = True
comptime IDEOGRAM4_V2_GRAPH_PATH = IDEOGRAM4_V2_ENGINE and IDEOGRAM4_V2_GRAPH


struct Ideogram4LoraAdamState(Movable):
    var m_a: List[TArc]
    var v_a: List[TArc]
    var m_b: List[TArc]
    var v_b: List[TArc]

    def __init__(
        out self,
        var m_a: List[TArc],
        var v_a: List[TArc],
        var m_b: List[TArc],
        var v_b: List[TArc],
    ):
        self.m_a = m_a^
        self.v_a = v_a^
        self.m_b = m_b^
        self.v_b = v_b^


@fieldwise_init
struct Ideogram4StackTrainResult(Copyable, Movable):
    var adapters: Int
    var slots_per_block: Int
    var grad_b_l1: Float32
    var adapter_b_l1: Float32


def make_ideogram4_lora_adam_state(
    loras: Ideogram4LoraSet, ctx: DeviceContext
) raises -> Ideogram4LoraAdamState:
    var m_a = List[TArc]()
    var v_a = List[TArc]()
    var m_b = List[TArc]()
    var v_b = List[TArc]()
    for i in range(len(loras.ad)):
        m_a.append(TArc(zeros_device(loras.ad[i][].a.shape(), STDtype.BF16, ctx)))
        v_a.append(TArc(zeros_device(loras.ad[i][].a.shape(), STDtype.BF16, ctx)))
        m_b.append(TArc(zeros_device(loras.ad[i][].b.shape(), STDtype.BF16, ctx)))
        v_b.append(TArc(zeros_device(loras.ad[i][].b.shape(), STDtype.BF16, ctx)))
    return Ideogram4LoraAdamState(m_a^, v_a^, m_b^, v_b^)


def apply_ideogram4_lora_grads(
    loras: Ideogram4LoraSet,
    mut state: Ideogram4LoraAdamState,
    grads: Ideogram4StackLoraGrads,
    optimizer_step: Int,
    cfg: TrainConfig,
    ctx: DeviceContext,
) raises -> Ideogram4StackTrainResult:
    if len(loras.ad) != len(grads.d_a) or len(loras.ad) != len(grads.d_b):
        raise Error("apply_ideogram4_lora_grads: adapter/grad count mismatch")
    if len(loras.ad) != len(state.m_a) or len(loras.ad) != len(state.m_b):
        raise Error("apply_ideogram4_lora_grads: optimizer state count mismatch")

    # Real LoRA-B gradient L1 (the 10-step gate reads result.grad_b_l1). Sum
    # |d_b| across every adapter BEFORE the optimizer step, so it reflects the
    # gradients that drove this step — matches the live trainer's grad_norm and
    # the levers path. Replaces the old grad_b_l1=0.0 / adapter_b_l1=0.0 stubs
    # that made the gate fail and the trainer look dead while it was learning.
    var grad_b_l1 = Float32(0.0)
    for i in range(len(loras.ad)):
        grad_b_l1 += _l1_sum(grads.d_b[i][], ctx)

    for i in range(len(loras.ad)):
        adamw_step(
            loras.ad[i][].a,
            state.m_a[i][],
            state.v_a[i][],
            grads.d_a[i][],
            optimizer_step,
            cfg.learning_rate,
            cfg.beta1,
            cfg.beta2,
            cfg.eps,
            cfg.weight_decay,
            cfg.stochastic_rounding,
            cfg.seed + UInt32(optimizer_step + i),
            ctx,
        )
        adamw_step(
            loras.ad[i][].b,
            state.m_b[i][],
            state.v_b[i][],
            grads.d_b[i][],
            optimizer_step,
            cfg.learning_rate,
            cfg.beta1,
            cfg.beta2,
            cfg.eps,
            cfg.weight_decay,
            cfg.stochastic_rounding,
            cfg.seed + UInt32(optimizer_step + 100000 + i),
            ctx,
        )

    # Post-step LoRA-B param L1 (the live trainer reads result.adapter_b_l1 for
    # its last_b summary; mirrors the levers path's returned b_l1).
    var adapter_b_l1 = Float32(0.0)
    for i in range(len(loras.ad)):
        adapter_b_l1 += _l1_sum(loras.ad[i][].b, ctx)

    return Ideogram4StackTrainResult(
        len(loras.ad), I4_SLOTS_PER_BLOCK, grad_b_l1, adapter_b_l1
    )


def train_ideogram4_stack_lora_step[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    x_in: Tensor,
    adaln_input: Tensor,
    cosf: Tensor,
    sinf: Tensor,
    transformer_weights: ShardedSafeTensors,
    loras: Ideogram4LoraSet,
    mut opt_state: Ideogram4LoraAdamState,
    d_stack_out: Tensor,
    optimizer_step: Int,
    cfg: TrainConfig,
    ctx: DeviceContext,
) raises -> Ideogram4StackTrainResult:
    var fwd = ideogram4_stack_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
        x_in, adaln_input, cosf, sinf, transformer_weights, loras, ctx
    )
    comptime if IDEOGRAM4_V2_GRAPH_PATH:
        # P7: per-block graph-engine backward (same conductor loop, same slot
        # fan-in, same arg list — drop-in for the hand-chain call below; bit
        # gate = ideogram4_block_parity).
        var grads = ideogram4_stack_lora_backward_graph[S, Hidden, Heads, Dh, FF, Adaln](
            d_stack_out, adaln_input, cosf, sinf, transformer_weights, loras, fwd^, ctx
        )
        return apply_ideogram4_lora_grads(
            loras, opt_state, grads^, optimizer_step, cfg, ctx
        )
    else:
        var grads = ideogram4_stack_lora_backward[S, Hidden, Heads, Dh, FF, Adaln](
            d_stack_out, adaln_input, cosf, sinf, transformer_weights, loras, fwd^, ctx
        )
        return apply_ideogram4_lora_grads(
            loras, opt_state, grads^, optimizer_step, cfg, ctx
        )


def _l1_sum(x: Tensor, ctx: DeviceContext) raises -> Float32:
    var host = x.to_host(ctx)
    var s = Float32(0.0)
    for i in range(len(host)):
        var v = host[i]
        if v < Float32(0.0):
            s -= v
        else:
            s += v
    return s


# ══════════════════════════════════════════════════════════════════════════════
# T1.C OPTIMIZER LEVERS BRIDGE (TIER1_PARITY_CAMPAIGN_2026-06-11.md).
#
# serenitymojo's levers optimizers (adafactor / schedule-free AdamW,
# training/levers.mojo levers_optimizer_step_host) step HOST
# serenitymojo.training.train_step.LoraAdapter mirrors (List[BFloat16] a/b),
# while ideogram4's adapters are DEVICE Tensors (module/LoRAModule.LoraAdapter
# inside Ideogram4LoraSet) stepped in place by the fused adamw_step above. The
# structs do not match, so this bridge implements the documented
# adapter-mirror host path:
#   lazy init (step 1): download every device a/b into a host mirror set
#   per step:           grads D2H -> levers_optimizer_step_host on the mirrors
#                       -> RNE-bf16 mirrors re-uploaded into the LoraSet's
#                          device tensors (Tensor.from_host_bf16, the same
#                          host->device construction the LoRA loader uses)
# The mirrors stay authoritative for params while the lever is active (the
# device AdamW never runs), so no per-step download is needed after init.
# Correctness over speed: ~2 H2D/D2H sweeps of the 204-adapter set per step.
#
# DEFAULT-OFF CONTRACT (C13): the trainer seam is
#   `if levers_optimizer_active(lcfg): ideogram4_levers_optimizer_step(...)
#    else: <the existing literal apply_ideogram4_lora_grads call>`
# — optimizer=ADAMW (the config default) never enters this section.
#
# RESUME: levers optimizer state has no save/resume sidecar (levers.mojo T1.C
# header) — levers_optimizer_step_host fails loud when its first call arrives
# at k != 1, which covers ideogram4 resume runs.
# ══════════════════════════════════════════════════════════════════════════════

from serenitymojo.training.train_step import LoraAdapter as SmLoraAdapter
from serenitymojo.training.levers import (
    LeversOptimizerState,
    levers_optimizer_active,
    levers_optimizer_step_host,
)
from serenitymojo.training.train_config import TrainConfig as LeversConfig


struct Ideogram4LeversBridge(Movable):
    """Host mirror set + levers optimizer state for one train run. Also the
    EMA tracking target (training/lora_ema.mojo wants the same host-mirror
    shape): with the default AdamW the driver refreshes the mirrors from
    device after each step; with a levers optimizer they are already live."""

    var mirrors: List[SmLoraAdapter]
    var mirrors_inited: Bool
    var opt_st: LeversOptimizerState

    def __init__(out self):
        self.mirrors = List[SmLoraAdapter]()
        self.mirrors_inited = False
        self.opt_st = LeversOptimizerState()


def ideogram4_levers_mirrors_init(
    mut bridge: Ideogram4LeversBridge,
    loras: Ideogram4LoraSet,
    ctx: DeviceContext,
) raises:
    """Download the device LoRA set into fresh host mirrors (bf16-exact:
    device bf16 -> F32 upcast -> the SmLoraAdapter ctor's RNE bf16 re-round is
    the identity on bf16-representable values). Adam moment lists are left
    EMPTY — the levers optimizers and lora_ema only touch a/b/shape fields."""
    if bridge.mirrors_inited:
        return
    for i in range(len(loras.ad)):
        var a_sh = loras.ad[i][].a.shape()   # [rank, in]
        var b_sh = loras.ad[i][].b.shape()   # [out, rank]
        var rank = loras.ad[i][].rank
        if len(a_sh) != 2 or len(b_sh) != 2 or a_sh[0] != rank or b_sh[1] != rank:
            raise Error(
                String("ideogram4_levers_mirrors_init: adapter ") + String(i)
                + String(" has unexpected a/b shape")
            )
        bridge.mirrors.append(
            SmLoraAdapter(
                loras.ad[i][].a.to_host(ctx),
                loras.ad[i][].b.to_host(ctx),
                rank, a_sh[1], b_sh[0],
                loras.ad[i][].scale(),
                List[Float32](), List[Float32](),
                List[Float32](), List[Float32](),
            )
        )
    bridge.mirrors_inited = True


def ideogram4_levers_refresh_mirrors(
    mut bridge: Ideogram4LeversBridge,
    loras: Ideogram4LoraSet,
    ctx: DeviceContext,
) raises:
    """Re-pull the live device a/b into the host mirrors (verbatim bf16).
    Needed each step ONLY on the default fused-AdamW path when EMA is on —
    the levers optimizer path keeps the mirrors authoritative itself."""
    if not bridge.mirrors_inited or len(bridge.mirrors) != len(loras.ad):
        raise Error("ideogram4_levers_refresh_mirrors: mirrors not initialized")
    for i in range(len(loras.ad)):
        bridge.mirrors[i].a = loras.ad[i][].a.to_host_bf16(ctx)
        bridge.mirrors[i].b = loras.ad[i][].b.to_host_bf16(ctx)


def ideogram4_levers_optimizer_step(
    lcfg: LeversConfig,
    loras: Ideogram4LoraSet,
    mut bridge: Ideogram4LeversBridge,
    grads: Ideogram4StackLoraGrads,
    k: Int,
    step_lr: Float32,
    ctx: DeviceContext,
) raises -> Float32:
    """The ONE ideogram4 call for the T1.C optimizer lever: host
    adafactor/schedule-free step over ALL adapters' mirrors, then re-upload
    into the LoraSet's live device tensors. step_lr = the trainer-scheduled lr
    (ideogram4 has no LR scheduler -> the constant cfg.learning_rate; the
    schedule-free path ignores it and uses lcfg.lr per the levers contract).
    Returns the post-step LoRA-B |.|_1 (host, for the progress summary).
    Call ONLY when levers_optimizer_active(lcfg) (C13)."""
    ideogram4_levers_mirrors_init(bridge, loras, ctx)
    if len(grads.d_a) != len(bridge.mirrors) or len(grads.d_b) != len(bridge.mirrors):
        raise Error("ideogram4_levers_optimizer_step: grad/mirror count mismatch")

    var d_a_h = List[List[Float32]]()
    var d_b_h = List[List[Float32]]()
    for i in range(len(bridge.mirrors)):
        d_a_h.append(grads.d_a[i][].to_host(ctx))
        d_b_h.append(grads.d_b[i][].to_host(ctx))

    levers_optimizer_step_host(
        lcfg, bridge.mirrors, d_a_h, d_b_h, k, step_lr,
        0, len(bridge.mirrors), bridge.opt_st,
    )

    var b_l1 = Float32(0.0)
    for i in range(len(bridge.mirrors)):
        loras.ad[i][].a = Tensor.from_host_bf16(
            bridge.mirrors[i].a, loras.ad[i][].a.shape(), ctx
        )
        loras.ad[i][].b = Tensor.from_host_bf16(
            bridge.mirrors[i].b, loras.ad[i][].b.shape(), ctx
        )
        for j in range(len(bridge.mirrors[i].b)):
            var v = bridge.mirrors[i].b[j].cast[DType.float32]()
            if v < Float32(0.0):
                b_l1 -= v
            else:
                b_l1 += v
    return b_l1
