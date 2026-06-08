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
from serenitymojo.io.dtype import STDtype
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
from serenity_trainer.trainer.Ideogram4LoRATrainStep import (
    Ideogram4LoRATrainResult,
    ideogram4_lora_train_step_resident,
)
from serenity_trainer.trainer.Ideogram4StackTrain import (
    Ideogram4LoraAdamState,
    make_ideogram4_lora_adam_state,
)
from serenity_trainer.trainer.TrainState import TrainProgress
from serenity_trainer.util.config.TrainConfig import TrainConfig


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

    var last_loss = Float32(0.0)
    var last_b = Float32(0.0)
    var train_start = perf_counter()
    var smooth_loss = Float32(0.0)
    var smooth_inited = False
    for local_step in range(run_cfg.steps):
        var sample_index = progress.global_step % cache.len()
        var seed = run_cfg.noise_seed + UInt64(opt_step + local_step)
        var sample = cache.sample[NT, GH, GW](
            sample_index, run_cfg.default_t_flow, seed, ctx
        )

        var result: Ideogram4LoRATrainResult = ideogram4_lora_train_step_resident[NT, GH, GW](
            weights,
            sample.noisy[],
            sample.clean[],
            sample.noise[],
            sample.t_flow,
            sample.llm_features[],
            loras,
            opt,
            opt_step + 1,
            cfg,
            ctx,
        )
        last_loss = result.loss
        last_b = result.adapter_b_l1
        opt_step += 1
        progress.next_step(cfg.batch_size)
        if not smooth_inited:
            smooth_loss = result.loss
            smooth_inited = True
        else:
            smooth_loss = smooth_loss * Float32(0.99) + result.loss * Float32(0.01)

        if run_cfg.progress_file_path.byte_length() > 0:
            var elapsed = perf_counter() - train_start
            var speed = elapsed / Float64(local_step + 1)
            _append_ideogram4_live_progress(
                run_cfg.progress_file_path,
                progress,
                run_cfg.steps,
                cfg,
                result.loss,
                smooth_loss,
                Float32(speed),
                elapsed,
            )

        if run_cfg.save_every_steps > 0 and opt_step % run_cfg.save_every_steps == 0:
            _save_lora(loras, _step_lora_path(run_cfg.output_dir, opt_step), ctx)

        if (
            run_cfg.checkpoint_every_steps > 0
            and opt_step % run_cfg.checkpoint_every_steps == 0
        ):
            var ckpt_dir = _step_state_dir(run_cfg.output_dir, opt_step)
            save_ideogram4_lora_train_state(ckpt_dir, opt, progress, opt_step, ctx)

    var train_elapsed = perf_counter() - train_start
    var seconds_per_step = train_elapsed / Float64(run_cfg.steps)

    _save_lora(loras, final_lora_path, ctx)
    save_ideogram4_lora_train_state(final_state_dir, opt, progress, opt_step, ctx)

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


def _append_ideogram4_live_progress(
    path: String,
    progress: TrainProgress,
    max_steps: Int,
    cfg: TrainConfig,
    loss: Float32,
    smooth_loss: Float32,
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
        + String(" | grad_norm 0.0000 | ")
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
