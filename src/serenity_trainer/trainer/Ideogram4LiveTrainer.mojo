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
#
# The UI launches this as a background process. Progress is written as
# Serenity-shaped callback lines so TrainerRuntimeBridge can tail the file.

from std.gpu.host import DeviceContext
from std.os import makedirs
from std.sys import argv

from serenity_trainer.trainer.Ideogram4LoRATrainer import (
    Ideogram4LoRATrainRunConfig,
    train_ideogram4_lora_from_cache,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig


comptime DEFAULT_TRANSFORMER = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime DEFAULT_CACHE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_predict.safetensors"
comptime DEFAULT_OUTPUT = "/home/alex/mojodiffusion/output"
comptime DEFAULT_PROGRESS = "target/serenity_trainer_progress.log"

comptime NT = 651
comptime GH = 16
comptime GW = 16


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

    if steps < 1:
        steps = 1
    if save_every_steps < 0:
        save_every_steps = 0

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
