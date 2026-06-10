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
    trainer_ui_runner_train_config_json,
)
from serenitymojo.io.train_config_reader import read_model_config


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
    print("ALL GATES PASS — UI runner train-config seam OK")
