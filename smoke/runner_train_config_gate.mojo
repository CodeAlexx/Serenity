# runner_train_config_gate.mojo — gate the UI->runner train-config seam.
#
# Proves that trainer_ui_runner_train_config_json (the JSON the UI writes for
# the config-driven live runners) round-trips through serenitymojo's REAL
# read_model_config with the model identity, architecture dims, and recipe the
# UI selected. This is the contract the chroma/ernie/anima/sdxl runners
# re-validate at startup.

from serenity_trainer.ui.TrainerConfigModel import (
    TrainerUIConfig,
    trainer_ui_apply_model_preset,
    trainer_ui_ideogram4_levers_path_or_skip,
    trainer_ui_ideogram4_levers_set,
    trainer_ui_runner_train_config_json,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import (
    LOSS_FN_MSE, LOSS_FN_HUBER, LOSS_FN_SMOOTH_L1,
    TRAIN_OPTIMIZER_ADAMW, TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW, TRAIN_OPTIMIZER_ADAMW_8BIT,
)


def _check(name: String, cond: Bool, detail: String) raises:
    if cond:
        print("  PASS ", name, " ", detail)
    else:
        raise Error(String("FAIL ") + name + String(" ") + detail)


def _gate_target(
    model_type_index: Int32,
    expect_target: String,
    expect_model_type: String,
    expect_d_model: Int,
    expect_single: Int,
    expect_heads: Int,
) raises:
    var ui = TrainerUIConfig()
    ui.model_type_index = model_type_index
    trainer_ui_apply_model_preset(ui, True)
    _check(
        expect_target.copy(),
        ui.backend_target == expect_target,
        String("backend_target=") + ui.backend_target.copy(),
    )
    ui.lora_rank = 8.0
    ui.lora_alpha = 4.0
    ui.learning_rate = 0.00025
    ui.max_train_steps = 123.0
    ui.save_every = 50.0

    var json = trainer_ui_runner_train_config_json(ui)
    var path = String("/tmp/serenity_ui_") + expect_target.copy() + String("_cfg_gate.json")
    var f = open(path.copy(), "w")
    f.write(json)
    f.close()

    var cfg = read_model_config(path.copy())
    _check(expect_target.copy(), cfg.name == expect_model_type, String("model_type=") + cfg.name.copy())
    if expect_d_model > 0:
        _check(
            expect_target.copy(),
            cfg.d_model == expect_d_model,
            String("inner_dim=") + String(cfg.d_model),
        )
        _check(
            expect_target.copy(),
            cfg.num_single == expect_single,
            String("num_single=") + String(cfg.num_single),
        )
        _check(
            expect_target.copy(),
            cfg.n_heads == expect_heads,
            String("num_heads=") + String(cfg.n_heads),
        )
    _check(expect_target.copy(), cfg.lora_rank == 8, String("lora_rank=") + String(cfg.lora_rank))
    var alpha_ok = cfg.lora_alpha > 3.99 and cfg.lora_alpha < 4.01
    _check(expect_target.copy(), alpha_ok, String("lora_alpha=") + String(cfg.lora_alpha))
    var lr_ok = cfg.lr > 0.000249 and cfg.lr < 0.000251
    _check(expect_target.copy(), lr_ok, String("lr=") + String(cfg.lr))
    _check(expect_target.copy(), cfg.max_steps == 123, String("max_steps=") + String(cfg.max_steps))
    _check(expect_target.copy(), cfg.save_every == 50, String("save_every=") + String(cfg.save_every))
    _check(
        expect_target.copy(),
        cfg.checkpoint == ui.base_model_name,
        String("checkpoint=") + cfg.checkpoint.copy(),
    )
    _check(
        expect_target.copy(),
        cfg.dataset_cache_dir == ui.cache_dir,
        String("cache_dir=") + cfg.dataset_cache_dir.copy(),
    )


def _close32(v: Float32, exp: Float32) -> Bool:
    var d = v - exp
    if d < Float32(0.0):
        d = -d
    return d < Float32(1.0e-5)


def _gate_zimage_loss_levers() raises:
    # T1.A: the UI loss-lever keys (loss_fn/huber_delta/smooth_l1_beta/
    # min_snr_gamma_flow) must round-trip through serenitymojo's REAL
    # read_model_config for the zimage target — default OFF, and flipped
    # values land in TrainConfig (training/levers.mojo consumes them).
    var ui = TrainerUIConfig()
    ui.model_type_index = 7
    trainer_ui_apply_model_preset(ui, True)

    # default emission == lever off (mse / 1.0 / 1.0 / 0.0)
    var json0 = trainer_ui_runner_train_config_json(ui)
    var p0 = String("/tmp/serenity_ui_zimage_levers_default_gate.json")
    var f0 = open(p0.copy(), "w")
    f0.write(json0)
    f0.close()
    var c0 = read_model_config(p0.copy())
    _check(String("zimage-levers-default"), c0.loss_fn == LOSS_FN_MSE,
           String("loss_fn=") + String(c0.loss_fn))
    _check(String("zimage-levers-default"), _close32(c0.huber_delta, Float32(1.0)),
           String("huber_delta=") + String(c0.huber_delta))
    _check(String("zimage-levers-default"), _close32(c0.smooth_l1_beta, Float32(1.0)),
           String("smooth_l1_beta=") + String(c0.smooth_l1_beta))
    _check(String("zimage-levers-default"), _close32(c0.min_snr_gamma_flow, Float32(0.0)),
           String("min_snr_gamma_flow=") + String(c0.min_snr_gamma_flow))

    # flipped from the UI cfg → parsed TrainConfig carries the lever
    ui.loss_fn = String("huber")
    ui.huber_delta = 0.25
    ui.smooth_l1_beta = 2.0
    ui.min_snr_gamma_flow = 5.0
    var json1 = trainer_ui_runner_train_config_json(ui)
    var p1 = String("/tmp/serenity_ui_zimage_levers_flipped_gate.json")
    var f1 = open(p1.copy(), "w")
    f1.write(json1)
    f1.close()
    var c1 = read_model_config(p1.copy())
    _check(String("zimage-levers-flipped"), c1.loss_fn == LOSS_FN_HUBER,
           String("loss_fn=") + String(c1.loss_fn))
    _check(String("zimage-levers-flipped"), _close32(c1.huber_delta, Float32(0.25)),
           String("huber_delta=") + String(c1.huber_delta))
    _check(String("zimage-levers-flipped"), _close32(c1.smooth_l1_beta, Float32(2.0)),
           String("smooth_l1_beta=") + String(c1.smooth_l1_beta))
    _check(String("zimage-levers-flipped"), _close32(c1.min_snr_gamma_flow, Float32(5.0)),
           String("min_snr_gamma_flow=") + String(c1.min_snr_gamma_flow))

    # smooth_l1 selector tag parses too
    ui.loss_fn = String("smooth_l1")
    var json2 = trainer_ui_runner_train_config_json(ui)
    var p2 = String("/tmp/serenity_ui_zimage_levers_sl1_gate.json")
    var f2 = open(p2.copy(), "w")
    f2.write(json2)
    f2.close()
    var c2 = read_model_config(p2.copy())
    _check(String("zimage-levers-sl1"), c2.loss_fn == LOSS_FN_SMOOTH_L1,
           String("loss_fn=") + String(c2.loss_fn))


def _gate_caption_dropout_prob() raises:
    # T1.D: the recipe JSON must emit caption_dropout_prob and round-trip it
    # through read_model_config — default OFF (0.0, the C13 companion flip of
    # the UI's old 0.05 default) and a flipped value.
    var ui = TrainerUIConfig()
    ui.model_type_index = 7
    trainer_ui_apply_model_preset(ui, True)

    var json0 = trainer_ui_runner_train_config_json(ui)
    var p0 = String("/tmp/serenity_ui_caption_dropout_default_gate.json")
    var f0 = open(p0.copy(), "w")
    f0.write(json0)
    f0.close()
    var c0 = read_model_config(p0.copy())
    _check(String("caption-dropout-default"),
           _close32(c0.caption_dropout_prob, Float32(0.0)),
           String("caption_dropout_prob=") + String(c0.caption_dropout_prob))

    ui.caption_dropout = 0.1
    var json1 = trainer_ui_runner_train_config_json(ui)
    var p1 = String("/tmp/serenity_ui_caption_dropout_flipped_gate.json")
    var f1 = open(p1.copy(), "w")
    f1.write(json1)
    f1.close()
    var c1 = read_model_config(p1.copy())
    _check(String("caption-dropout-flipped"),
           _close32(c1.caption_dropout_prob, Float32(0.1)),
           String("caption_dropout_prob=") + String(c1.caption_dropout_prob))


def _gate_ema() raises:
    # T1.B: the UI EMA keys (ema/ema_decay/ema_update_step_interval) must
    # round-trip through read_model_config — default OFF (ema_enabled False),
    # and the UI's "EMA" dropdown choice (TrainerConfigModel ema_options is
    # OFF/EMA) must parse to enabled + carry decay/interval into TrainConfig
    # (train_zimage_real.mojo lora_ema wiring consumes them).
    var ui = TrainerUIConfig()
    ui.model_type_index = 7
    trainer_ui_apply_model_preset(ui, True)

    var json0 = trainer_ui_runner_train_config_json(ui)
    var p0 = String("/tmp/serenity_ui_ema_default_gate.json")
    var f0 = open(p0.copy(), "w")
    f0.write(json0)
    f0.close()
    var c0 = read_model_config(p0.copy())
    _check(String("ema-default"), not c0.ema_enabled,
           String("ema_enabled=") + String(c0.ema_enabled))
    _check(String("ema-default"), _close32(c0.ema_decay, Float32(0.999)),
           String("ema_decay=") + String(c0.ema_decay))
    _check(String("ema-default"), c0.ema_update_step_interval == 5,
           String("ema_update_step_interval=") + String(c0.ema_update_step_interval))

    ui.ema_mode = String("EMA")
    ui.ema_decay = 0.99
    ui.ema_update_step_interval = 2.0
    var json1 = trainer_ui_runner_train_config_json(ui)
    var p1 = String("/tmp/serenity_ui_ema_flipped_gate.json")
    var f1 = open(p1.copy(), "w")
    f1.write(json1)
    f1.close()
    var c1 = read_model_config(p1.copy())
    _check(String("ema-flipped"), c1.ema_enabled,
           String("ema_enabled=") + String(c1.ema_enabled))
    _check(String("ema-flipped"), _close32(c1.ema_decay, Float32(0.99)),
           String("ema_decay=") + String(c1.ema_decay))
    _check(String("ema-flipped"), c1.ema_update_step_interval == 2,
           String("ema_update_step_interval=") + String(c1.ema_update_step_interval))


def _gate_optimizer_runner() raises:
    # T1.C: the zimage emission carries the optimizer enum (from the existing
    # UI optimizer dropdown via optimizer_runner_value) + optimizer_warmup_
    # steps (:= learning_rate_warmup_steps) and round-trips through
    # serenitymojo's REAL read_model_config — default ADAMW/0 (lever OFF),
    # flipped ADAFACTOR and SCHEDULE_FREE_ADAMW land in TrainConfig, and an
    # unsupported dropdown value (CAME) FAILS LOUD at config load.
    var ui = TrainerUIConfig()
    ui.model_type_index = 7
    trainer_ui_apply_model_preset(ui, True)

    # default emission == lever off (ADAMW, warmup 0)
    var json0 = trainer_ui_runner_train_config_json(ui)
    var p0 = String("/tmp/serenity_ui_optimizer_default_gate.json")
    var f0 = open(p0.copy(), "w")
    f0.write(json0)
    f0.close()
    var c0 = read_model_config(p0.copy())
    _check(String("optimizer-default"), c0.optimizer == TRAIN_OPTIMIZER_ADAMW,
           String("optimizer=") + String(c0.optimizer))
    _check(String("optimizer-default"), c0.optimizer_warmup_steps == 0,
           String("optimizer_warmup_steps=") + String(c0.optimizer_warmup_steps))

    # flipped: ADAFACTOR (dropdown index 3)
    ui.optimizer_index = 3
    var json1 = trainer_ui_runner_train_config_json(ui)
    var p1 = String("/tmp/serenity_ui_optimizer_adafactor_gate.json")
    var f1 = open(p1.copy(), "w")
    f1.write(json1)
    f1.close()
    var c1 = read_model_config(p1.copy())
    _check(String("optimizer-adafactor"),
           c1.optimizer == TRAIN_OPTIMIZER_ADAFACTOR,
           String("optimizer=") + String(c1.optimizer))

    # flipped: SCHEDULE_FREE_ADAMW (dropdown index 5, appended T1.C) with
    # warmup from learning_rate_warmup_steps
    ui.optimizer_index = 5
    ui.learning_rate_warmup_steps = 25.0
    var json2 = trainer_ui_runner_train_config_json(ui)
    var p2 = String("/tmp/serenity_ui_optimizer_sf_gate.json")
    var f2 = open(p2.copy(), "w")
    f2.write(json2)
    f2.close()
    var c2 = read_model_config(p2.copy())
    _check(String("optimizer-schedulefree"),
           c2.optimizer == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW,
           String("optimizer=") + String(c2.optimizer))
    _check(String("optimizer-schedulefree"), c2.optimizer_warmup_steps == 25,
           String("optimizer_warmup_steps=") + String(c2.optimizer_warmup_steps))

    # flipped: ADAMW8BIT (dropdown index 0) — T2.A: optimizer_runner_value
    # maps the UI label to the runner enum ADAMW_8BIT (bnb block-wise 8-bit
    # AdamW, serenitymojo training/adamw8bit.mojo via the levers dispatch)
    ui.optimizer_index = 0
    var json8 = trainer_ui_runner_train_config_json(ui)
    var p8 = String("/tmp/serenity_ui_optimizer_adamw8bit_gate.json")
    var f8 = open(p8.copy(), "w")
    f8.write(json8)
    f8.close()
    var c8 = read_model_config(p8.copy())
    _check(String("optimizer-adamw8bit"),
           c8.optimizer == TRAIN_OPTIMIZER_ADAMW_8BIT,
           String("optimizer=") + String(c8.optimizer))

    # unsupported dropdown value (CAME, index 2) fails loud at config load
    ui.optimizer_index = 2
    var json3 = trainer_ui_runner_train_config_json(ui)
    var p3 = String("/tmp/serenity_ui_optimizer_came_gate.json")
    var f3 = open(p3.copy(), "w")
    f3.write(json3)
    f3.close()
    var raised = False
    try:
        var _c3 = read_model_config(p3.copy())
    except _e:
        raised = True
    _check(String("optimizer-came-fails-loud"), raised, String("raised=") + String(raised))


def _gate_hidream() raises:
    # HiDream-O1 P4: the "hidream" emission (train_hidream_o1_real trailing
    # argv [config.json]) must round-trip read_model_config — default-off
    # levers + quantized_resident "OFF" (C13), and flipped levers +
    # "fp8_e4m3" land in TrainConfig. Unsupported quantized_resident tags
    # FAIL LOUD at config load.
    var ui = TrainerUIConfig()
    ui.model_type_index = 11  # HIDREAM_O1 (appended last)
    trainer_ui_apply_model_preset(ui, True)
    _check(String("hidream-preset"), ui.backend_target == String("hidream"),
           String("backend_target=") + ui.backend_target.copy())
    _check(String("hidream-preset"), Int(ui.lora_rank) == 32,
           String("lora_rank=") + String(ui.lora_rank))

    var json0 = trainer_ui_runner_train_config_json(ui)
    var p0 = String("/tmp/serenity_ui_hidream_default_gate.json")
    var f0 = open(p0.copy(), "w")
    f0.write(json0)
    f0.close()
    var c0 = read_model_config(p0.copy())
    _check(String("hidream-default"), c0.name == String("hidream_o1"),
           String("model_type=") + c0.name.copy())
    _check(String("hidream-default"), c0.checkpoint == ui.base_model_name,
           String("checkpoint=") + c0.checkpoint.copy())
    _check(String("hidream-default"), c0.dataset_cache_dir == ui.cache_dir,
           String("cache_dir=") + c0.dataset_cache_dir.copy())
    _check(String("hidream-default"), c0.lora_rank == 32,
           String("lora_rank=") + String(c0.lora_rank))
    _check(String("hidream-default"), c0.loss_fn == LOSS_FN_MSE,
           String("loss_fn=") + String(c0.loss_fn))
    _check(String("hidream-default"),
           c0.quantized_resident == String("OFF"),
           String("quantized_resident=") + c0.quantized_resident.copy())
    _check(String("hidream-default"), c0.optimizer == TRAIN_OPTIMIZER_ADAMW,
           String("optimizer=") + String(c0.optimizer))
    _check(String("hidream-default"), not c0.ema_enabled,
           String("ema_enabled=") + String(c0.ema_enabled))

    # flipped levers + fp8_e4m3 land in TrainConfig
    ui.loss_fn = String("huber")
    ui.huber_delta = 0.25
    ui.min_snr_gamma_flow = 5.0
    ui.ema_mode = String("EMA")
    ui.optimizer_index = 3  # ADAFACTOR
    ui.quantized_resident = String("fp8_e4m3")
    var json1 = trainer_ui_runner_train_config_json(ui)
    var p1 = String("/tmp/serenity_ui_hidream_flipped_gate.json")
    var f1 = open(p1.copy(), "w")
    f1.write(json1)
    f1.close()
    var c1 = read_model_config(p1.copy())
    _check(String("hidream-flipped"), c1.loss_fn == LOSS_FN_HUBER,
           String("loss_fn=") + String(c1.loss_fn))
    _check(String("hidream-flipped"), _close32(c1.huber_delta, Float32(0.25)),
           String("huber_delta=") + String(c1.huber_delta))
    _check(String("hidream-flipped"),
           _close32(c1.min_snr_gamma_flow, Float32(5.0)),
           String("min_snr_gamma_flow=") + String(c1.min_snr_gamma_flow))
    _check(String("hidream-flipped"), c1.ema_enabled,
           String("ema_enabled=") + String(c1.ema_enabled))
    _check(String("hidream-flipped"),
           c1.optimizer == TRAIN_OPTIMIZER_ADAFACTOR,
           String("optimizer=") + String(c1.optimizer))
    _check(String("hidream-flipped"),
           c1.quantized_resident == String("fp8_e4m3"),
           String("quantized_resident=") + c1.quantized_resident.copy())

    # unsupported quantized_resident tag fails loud at config load
    ui.quantized_resident = String("int4_makebelieve")
    var json2 = trainer_ui_runner_train_config_json(ui)
    var p2 = String("/tmp/serenity_ui_hidream_badquant_gate.json")
    var f2 = open(p2.copy(), "w")
    f2.write(json2)
    f2.close()
    var raised = False
    try:
        var _c2 = read_model_config(p2.copy())
    except _e:
        raised = True
    _check(String("hidream-badquant-fails-loud"), raised,
           String("raised=") + String(raised))


def _gate_ideogram4_levers() raises:
    # Ideogram4 lever delivery (Ideogram4LiveTrainer argv 10/11 contract):
    # default-off config -> levers_set False and argv 11 == "-" (skip
    # sentinel, C13 byte-identical default runs); any lever flipped ->
    # the levers JSON path is delivered and the emission round-trips
    # read_model_config with the lever values.
    var ui = TrainerUIConfig()
    ui.model_type_index = 0  # IDEOGRAM_4
    trainer_ui_apply_model_preset(ui, True)
    _check(String("ideogram4-preset"),
           ui.backend_target == String("ideogram4"),
           String("backend_target=") + ui.backend_target.copy())

    # the exact path the bridge passes (_runner_train_config_path).
    var levers_path = String("target/serenity_ideogram4_train_config.json")

    # default-off: no levers JSON, argv 11 = "-", argv 10 carrier = 0.0
    _check(String("ideogram4-default"),
           not trainer_ui_ideogram4_levers_set(ui),
           String("levers_set=") + String(trainer_ui_ideogram4_levers_set(ui)))
    var skip = trainer_ui_ideogram4_levers_path_or_skip(ui, levers_path.copy())
    _check(String("ideogram4-default"), skip == String("-"),
           String("argv11=") + skip.copy())
    _check(String("ideogram4-default"), String(ui.caption_dropout) == String("0.0"),
           String("argv10=") + String(ui.caption_dropout))
    # the default emission still parses to all-default levers (sanity)
    var json0 = trainer_ui_runner_train_config_json(ui)
    var p0 = String("/tmp/serenity_ui_ideogram4_levers_default_gate.json")
    var f0 = open(p0.copy(), "w")
    f0.write(json0)
    f0.close()
    var c0 = read_model_config(p0.copy())
    _check(String("ideogram4-default"), c0.name == String("ideogram4"),
           String("model_type=") + c0.name.copy())
    _check(String("ideogram4-default"), c0.loss_fn == LOSS_FN_MSE,
           String("loss_fn=") + String(c0.loss_fn))
    _check(String("ideogram4-default"), not c0.ema_enabled,
           String("ema_enabled=") + String(c0.ema_enabled))
    _check(String("ideogram4-default"), c0.optimizer == TRAIN_OPTIMIZER_ADAMW,
           String("optimizer=") + String(c0.optimizer))

    # huber lever -> JSON delivered + parses
    ui.loss_fn = String("huber")
    ui.huber_delta = 0.25
    _check(String("ideogram4-huber"), trainer_ui_ideogram4_levers_set(ui),
           String("levers_set=") + String(trainer_ui_ideogram4_levers_set(ui)))
    var path1 = trainer_ui_ideogram4_levers_path_or_skip(ui, levers_path.copy())
    _check(String("ideogram4-huber"), path1 == levers_path,
           String("argv11=") + path1.copy())
    var json1 = trainer_ui_runner_train_config_json(ui)
    var p1 = String("/tmp/serenity_ui_ideogram4_levers_huber_gate.json")
    var f1 = open(p1.copy(), "w")
    f1.write(json1)
    f1.close()
    var c1 = read_model_config(p1.copy())
    _check(String("ideogram4-huber"), c1.loss_fn == LOSS_FN_HUBER,
           String("loss_fn=") + String(c1.loss_fn))
    _check(String("ideogram4-huber"), _close32(c1.huber_delta, Float32(0.25)),
           String("huber_delta=") + String(c1.huber_delta))

    # each remaining lever flips levers_set on its own
    var ui_ema = TrainerUIConfig()
    ui_ema.model_type_index = 0
    trainer_ui_apply_model_preset(ui_ema, True)
    ui_ema.ema_mode = String("EMA")
    _check(String("ideogram4-ema-lever"),
           trainer_ui_ideogram4_levers_set(ui_ema),
           String("levers_set=") + String(trainer_ui_ideogram4_levers_set(ui_ema)))
    var ui_opt = TrainerUIConfig()
    ui_opt.model_type_index = 0
    trainer_ui_apply_model_preset(ui_opt, True)
    ui_opt.optimizer_index = 3  # ADAFACTOR
    _check(String("ideogram4-opt-lever"),
           trainer_ui_ideogram4_levers_set(ui_opt),
           String("levers_set=") + String(trainer_ui_ideogram4_levers_set(ui_opt)))
    var ui_snr = TrainerUIConfig()
    ui_snr.model_type_index = 0
    trainer_ui_apply_model_preset(ui_snr, True)
    ui_snr.min_snr_gamma_flow = 5.0
    _check(String("ideogram4-snr-lever"),
           trainer_ui_ideogram4_levers_set(ui_snr),
           String("levers_set=") + String(trainer_ui_ideogram4_levers_set(ui_snr)))
    # caption dropout does NOT force the levers JSON (argv 10 carries it)
    var ui_drop = TrainerUIConfig()
    ui_drop.model_type_index = 0
    trainer_ui_apply_model_preset(ui_drop, True)
    ui_drop.caption_dropout = 0.25
    _check(String("ideogram4-dropout-argv10"),
           not trainer_ui_ideogram4_levers_set(ui_drop),
           String("levers_set=") + String(trainer_ui_ideogram4_levers_set(ui_drop)))
    _check(String("ideogram4-dropout-argv10"),
           String(ui_drop.caption_dropout) == String("0.25"),
           String("argv10=") + String(ui_drop.caption_dropout))


def main() raises:
    print("== runner train config gate ==")
    # model_type option indices: 4=CHROMA_1, 5=ERNIE_IMAGE, 6=ANIMA, 2=SDXL,
    # 7=Z_IMAGE, 8=Z_IMAGE_L2P
    _gate_target(4, String("chroma"), String("chroma"), 3072, 38, 24)
    _gate_target(5, String("ernie"), String("ernie_image"), 4096, 36, 32)
    _gate_target(6, String("anima"), String("anima"), 2048, 28, 16)
    _gate_target(2, String("sdxl"), String("sdxl"), 0, 0, 0)
    _gate_target(7, String("zimage"), String("zimage"), 3840, 30, 30)
    _gate_target(8, String("l2p"), String("l2p"), 3840, 30, 30)
    _gate_zimage_loss_levers()
    _gate_caption_dropout_prob()
    _gate_ema()
    _gate_optimizer_runner()
    _gate_hidream()
    _gate_ideogram4_levers()
    print("ALL GATES PASS — UI runner train-config seam OK")
