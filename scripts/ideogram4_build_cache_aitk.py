#!/usr/bin/env python3
# build_ideogram4_cache.py — derive a REAL small ideogram4 training cache
# (the dead default /home/alex/trainings/ideogram4_giger_cache/cache.safetensors)
# for the IDEOGRAM4_FULL_FT smoke, since no cache exists on-box and the Mojo
# stage-B prepare (Serenity smoke/ideogram4_prepare_cache.mojo) is asset-blocked:
# its encoders expect /home/alex/.serenity/models/ideogram-4-fp8/{vae,text_encoder,
# tokenizer} which are absent (only transformer/ exists), and the Mojo
# load_ideogram_qwen3vl loads the WHOLE 8B encoder dequanted-bf16 device-resident
# (~16GB > the 5080's 16GB).
#
# This builder produces the EXACT schema stage B writes (Ideogram4CacheReader
# "indexed" layout): clean.<i> [1,128,32,32] F32, llm.<i> [1,256,53248] BF16,
# text_len.<i> [1] F32 — using the AI-TOOLKIT PRODUCTION encode paths (the
# MJ-1041 oracle), same code the parity oracles import:
#   image  : stage-A transform (scripts/ideogram4_stage_images.py:84-92 —
#            center-crop, LANCZOS 512, [-1,1] CHW) → ai-toolkit AutoEncoder
#            encode_images path (ideogram4.py:556-578 via the vae-oracle recipe:
#            moments=encoder(img bf16); mean=moments[:,:32]; patchify(2);
#            (patched-shift)/scale bf16, get_latent_norm) → F32 store.
#   caption: stage-A render_prompt (digest_caption_string + the gate-verified
#            chat-template string) → tokenizer ids (natural length, cap 256)
#            → ai-toolkit get_qwen3_vl_features (pipeline.py:108-158 verbatim:
#            per-layer walk, 13-tap stack/permute/reshape) → rows zero-padded
#            to NT=256 (== the Mojo prepare's pad-then-zero: causal attention
#            makes trailing-pad encode identical at real rows; pad rows are
#            zeroed in both).
#   text encoder: the on-box BF16 Qwen3-VL text tower (Ideogram-4-bf16-Diffusers
#            HF snapshot text_encoder, keys language_model.* — ai-toolkit
#            production loads the public bf16 tower; fp8-dequant vs bf16 was
#            gated negligible). Runs on CPU bf16 (16GB weights don't fit the
#            16GB 5080); VAE runs on GPU bf16.
import json, sys, glob
from pathlib import Path
import importlib.util

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file, load_file

sys.path.insert(0, "/home/alex/ai-toolkit")
from toolkit.ideogram_caption import digest_caption_string  # noqa: E402

SRC = "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/ideogram4/src"


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_vae = _load("aitk_i4_vae", f"{SRC}/vae.py")
_ln = _load("aitk_i4_latent_norm", f"{SRC}/latent_norm.py")

DATASET = Path("/home/alex/datasets/gigerver3")
OUT_DIR = Path("/home/alex/trainings/ideogram4_giger_cache")
OUT = OUT_DIR / "cache.safetensors"
N = 8
SIZE = 512
NT = 256
FP8_SNAP = glob.glob(
    "/home/alex/.cache/huggingface/hub/models--ideogram-ai--ideogram-4-fp8/snapshots/*"
)[0]
BF16_SNAP = glob.glob(
    "/home/alex/.cache/huggingface/hub/models--CalamitousFelicitousness--"
    "Ideogram-4-bf16-Diffusers/snapshots/*"
)[0]
VAE_PATH = f"{FP8_SNAP}/vae/diffusion_pytorch_model.safetensors"
TOK_DIR = f"{FP8_SNAP}/tokenizer"
TE_DIR = f"{BF16_SNAP}/text_encoder"

QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)


def render_prompt(caption: str) -> str:
    # scripts/ideogram4_stage_images.py:49-64 (gate-verified == apply_chat_template)
    p = digest_caption_string(caption)
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"


def load_image(path: Path) -> torch.Tensor:
    # scripts/ideogram4_stage_images.py:84-92
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr.transpose(2, 0, 1))[None]  # [1,3,512,512] f32


def patchify_latents(z: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    # ai-toolkit src/pipeline.py:80-88 (inlined verbatim, same as the vae oracle)
    b, ae_ch, h8, w8 = z.shape
    ph = pw = patch_size
    gh, gw = h8 // ph, w8 // pw
    z = z.view(b, ae_ch, gh, ph, gw, pw)
    z = z.permute(0, 3, 5, 1, 2, 4).reshape(b, ph * pw * ae_ch, gh, gw)
    return z


@torch.no_grad()
def get_qwen3_vl_features(language_model, token_ids, attention_mask, pos_2d):
    # ai-toolkit src/pipeline.py:108-158 VERBATIM (transformers 5.5.x kwargs).
    from transformers.masking_utils import create_causal_mask

    inputs_embeds = language_model.embed_tokens(token_ids)
    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]
    causal_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)
    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        if layer_idx in tap_set:
            captured[layer_idx] = hidden_states
    selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    batch_size, seq_len = token_ids.shape
    stacked = torch.stack(selected, dim=0)
    stacked = torch.permute(stacked, (1, 2, 3, 0))
    stacked = stacked.reshape(batch_size, seq_len, -1)
    text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
    stacked = stacked * text_mask
    return stacked


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    jpgs = sorted(
        DATASET.glob("*.jpg"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 30,
    )
    pairs = [(p, p.with_suffix(".txt")) for p in jpgs if p.with_suffix(".txt").exists()][:N]
    assert len(pairs) == N, f"need {N} img+txt pairs, found {len(pairs)}"
    print(f"[cache] dataset {DATASET} -> {len(pairs)} samples")

    # ── VAE (GPU bf16, ai-toolkit AutoEncoder — the vae-oracle load) ──
    ae = _vae.AutoEncoder(_vae.AutoEncoderParams())
    ae.load_state_dict(_vae.convert_diffusers_state_dict(load_file(VAE_PATH)))
    ae.to(device=dev, dtype=torch.bfloat16).eval().requires_grad_(False)
    shift_f32, scale_f32 = _ln.get_latent_norm()
    shift = shift_f32.view(1, -1, 1, 1).to(dev, torch.bfloat16)
    scale = scale_f32.view(1, -1, 1, 1).to(dev, torch.bfloat16)
    ae_ch = ae.params.z_channels
    print(f"[cache] ai-toolkit AutoEncoder loaded (z_channels={ae_ch})")

    cleans = []
    for i, (jp, _) in enumerate(pairs):
        img = load_image(jp).to(dev, torch.bfloat16)
        with torch.no_grad():
            moments = ae.encoder(img)
            mean = moments[:, :ae_ch]                    # [1,32,64,64]
            patched = patchify_latents(mean, 2)          # [1,128,32,32]
            lat = (patched - shift) / scale              # bf16 (prod norm)
        cleans.append(lat.float().cpu())
        print(f"[cache] vae {i}: latent std={lat.float().std():.4f} "
              f"mean={lat.float().mean():+.4f}")
    del ae
    torch.cuda.empty_cache()

    # ── tokenizer + captions ──
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    ids_list, lens = [], []
    for i, (_, cp) in enumerate(pairs):
        rendered = render_prompt(cp.read_text().strip())
        ids = tok(rendered, add_special_tokens=False).input_ids
        nat = min(len(ids), NT)
        ids_list.append(ids[:nat])
        lens.append(nat)
        print(f"[cache] tok {i}: natural_len={nat}")

    # ── text encoder (CPU bf16 — the on-box bf16 Qwen3-VL text tower) ──
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from safetensors import safe_open

    cfg = AutoConfig.from_pretrained(TE_DIR)
    lm = Qwen3VLTextModel._from_config(cfg.get_text_config(), dtype=torch.bfloat16)
    idx = json.load(open(f"{TE_DIR}/model.safetensors.index.json"))
    sd = {}
    for shard in sorted(set(idx["weight_map"].values())):
        with safe_open(f"{TE_DIR}/{shard}", framework="pt") as f:
            for k in f.keys():
                sd[k.replace("language_model.", "", 1)] = f.get_tensor(k)
    missing, unexpected = lm.load_state_dict(sd, strict=False, assign=True)
    assert not missing, f"missing keys: {missing[:5]}"
    print(f"[cache] Qwen3VL text tower loaded on CPU bf16 "
          f"(unexpected={len(unexpected)}: {unexpected[:3]})")
    lm.eval().requires_grad_(False)
    # PROOF of real weights: byte-compare one loaded tensor vs disk.
    with safe_open(f"{TE_DIR}/{idx['weight_map']['language_model.layers.0.self_attn.q_proj.weight']}",
                   framework="pt") as f:
        disk_q = f.get_tensor("language_model.layers.0.self_attn.q_proj.weight")
    got_q = lm.layers[0].self_attn.q_proj.weight
    assert torch.equal(got_q, disk_q), "loaded q_proj != disk (random init?)"
    print("[cache] REAL weights verified (layer0 q_proj byte-matches disk)")

    llms = []
    for i, ids in enumerate(ids_list):
        token_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(token_ids)
        pos_2d = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)
        feats = get_qwen3_vl_features(lm, token_ids, attention_mask, pos_2d)  # [1,L,53248] bf16
        L = feats.shape[1]
        full = torch.zeros(1, NT, feats.shape[2], dtype=torch.bfloat16)
        full[:, :L] = feats  # pad rows stay ZERO (== Mojo prepare's masked pad rows)
        llms.append(full)
        print(f"[cache] llm {i}: L={L} |feat|_mean={feats.float().abs().mean():.4f}")

    # ── NON-DEGENERATE conditioning check (fail loud) ──
    flat = torch.stack([l[0, : min(lens)].float().reshape(-1) for l in llms])
    cos = torch.nn.functional.cosine_similarity(flat[None, :], flat[:, None], dim=-1)
    off = cos[~torch.eye(N, dtype=torch.bool)]
    print(f"[cache] cross-sample llm cos: min={off.min():.4f} max={off.max():.4f} "
          f"mean={off.mean():.4f}")
    assert off.max() < 0.999, "DEGENERATE conditioning: llm features near-identical"
    lat_flat = torch.stack([c.reshape(-1) for c in cleans])
    lcos = torch.nn.functional.cosine_similarity(lat_flat[None, :], lat_flat[:, None], dim=-1)
    loff = lcos[~torch.eye(N, dtype=torch.bool)]
    print(f"[cache] cross-sample latent cos: min={loff.min():.4f} max={loff.max():.4f}")
    assert loff.max() < 0.999, "DEGENERATE latents"

    # ── write the stage-B schema (Ideogram4CacheReader indexed layout) ──
    tensors = {}
    for i in range(N):
        tensors[f"clean.{i}"] = cleans[i]                                   # F32
        tensors[f"llm.{i}"] = llms[i]                                       # BF16
        tensors[f"text_len.{i}"] = torch.tensor([float(lens[i])], dtype=torch.float32)
    save_file(tensors, str(OUT))
    print(f"[cache] WROTE {OUT} samples={N}")


if __name__ == "__main__":
    main()
