# ideogram4_prepare_cache.mojo — stage B of the Ideogram-4 cache prepare:
# the staged images + rendered prompts (scripts/ideogram4_stage_images.py)
# go through the GATED Mojo encoders (Ideogram4VaeEncoder, Qwen3Tokenizer +
# ideogram4_encode_text 13-tap) into the indexed safetensors training cache
# the Ideogram4CacheReader streams (clean.<i> [1,128,GH,GW] F32 +
# llm.<i> [1,NT,53248] BF16).
#
# WHY THIS EXISTS (measured 2026-06-11): the UI's "cache" default was the
# ideogram4_fx_predict PARITY FIXTURE (one fixed sample) — a 3000-step run
# completed with loss ~1.3e-4 / grad_norm 0.0000 = trained on garbage.
# This tool builds the REAL multi-sample cache.
#
# Text padding: token ids padded (or truncated) to NT with 151643 (the
# eos/pad id the tokenizer-probe oracle sequence ends with) BEFORE encoding,
# so llm features are the encoder's real output at fixed NT.
# HYPOTHESIS to cross-check vs DiffSynth-Studio: their pad strategy.
#
# Run (after stage A; GPU free; ~70 samples => one Qwen3-VL + VAE pass each):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -I /home/alex/serenity-trainer/src \
#     -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib \
#     -Xlinker -lsqlite3 \
#     /home/alex/serenity-trainer/smoke/ideogram4_prepare_cache.mojo \
#     -o /tmp/ideogram4_prepare
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     /tmp/ideogram4_prepare <stage_dir> <out_cache.safetensors> <n>

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenity_trainer.model.Ideogram4VAE import Ideogram4VaeEncoder
from serenity_trainer.model.Ideogram4TextEncoder import (
    ideogram4_load_text_encoder_default,
    ideogram4_encode_text,
)

comptime TArc = ArcPointer[Tensor]
comptime TOK = "/home/alex/.serenity/models/ideogram-4-fp8/tokenizer/tokenizer.json"
comptime NT = 256          # text bucket: ids padded with PAD_ID, then encoded
comptime PAD_ID = 151643   # Qwen3 eos/pad (the tokenizer-probe oracle tail)
comptime LH = 64           # 512px -> VAE latent 64x64 -> packed [1,128,32,32]
comptime LW = 64


def _read_text(path: String) raises -> String:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    return s^


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("usage: ideogram4_prepare <stage_dir> <out.safetensors> <n>")
    var stage_dir = String(args[1])
    var out_path = String(args[2])
    var n = Int(String(args[3]))

    var ctx = DeviceContext()
    print("[prepare] loading VAE encoder (512px bucket)")
    var venc = Ideogram4VaeEncoder[LH, LW].load_default(ctx)
    print("[prepare] loading Qwen3-VL text encoder")
    var tenc = ideogram4_load_text_encoder_default(ctx)
    var tok = Qwen3Tokenizer(String(TOK))

    var imgs = ShardedSafeTensors.open(stage_dir + "/images.safetensors")

    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(n):
        # ── image -> packed normalized latent [1,128,32,32] F32 ──
        var img = cast_tensor(
            Tensor.from_view(imgs.tensor_view(String("image.") + String(i)), ctx),
            STDtype.BF16, ctx,
        )
        var lat = venc.encode(img, ctx)
        var clean = cast_tensor(lat, STDtype.F32, ctx)

        # ── caption -> ids (pad/truncate NT) -> 13-tap features BF16 ──
        var prompt = _read_text(stage_dir + "/prompt." + String(i) + ".txt")
        var ids = tok.encode(prompt)
        if len(ids) > NT:
            var cut = List[Int]()
            for j in range(NT):
                cut.append(ids[j])
            ids = cut^
        while len(ids) < NT:
            ids.append(PAD_ID)
        var feats = ideogram4_encode_text(tenc, ids, ctx)
        var llm = cast_tensor(feats, STDtype.BF16, ctx)

        names.append(String("clean.") + String(i))
        tensors.append(TArc(clean^))
        names.append(String("llm.") + String(i))
        tensors.append(TArc(llm^))
        print("[prepare] sample", i, "done (ids", len(ids), ")")

    save_safetensors(names, tensors, out_path, ctx)
    print("[prepare] WROTE", out_path, "samples=", n)
