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
    ideogram4_stack_lora_forward,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.util.optimizer.adamw_extensions import adamw_step

comptime TArc = ArcPointer[Tensor]


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

    var grad_b_l1 = Float32(0.0)
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

    return Ideogram4StackTrainResult(
        len(loras.ad), I4_SLOTS_PER_BLOCK, grad_b_l1, Float32(0.0)
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
