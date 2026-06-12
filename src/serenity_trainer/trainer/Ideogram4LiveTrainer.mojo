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
