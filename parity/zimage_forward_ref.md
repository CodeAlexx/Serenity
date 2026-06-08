# Z-Image Transformer Forward Reference

Numeric forward reference generated from Serenity's **real** diffusers
`ZImageTransformer2DModel`. Use it to verify a Mojo port of the transformer
forward.

- **Reference source:** Serenity + diffusers ONLY (never EriDiffusion/Rust).
- **Generator:** `parity/gen_zimage_forward_ref.py`
- **Run:** `/home/alex/Serenity/venv/bin/python parity/gen_zimage_forward_ref.py`
  (cwd `/home/alex/Serenity` so `modules` imports resolve)
- **Model:** `/home/alex/.serenity/models/zimage_base/transformer`
  (dim=3840, n_heads=30, head_dim=128, n_layers=30, n_refiner_layers=2,
  cap_feat_dim=2560, axes_dims=[32,48,48], patch_size=2, rope_theta=256,
  in_channels=16, t_scale=1000)
- **Compute dtype:** bf16, CUDA, eval, `torch.no_grad()`.

## How Serenity calls the transformer

From `modules/modelSetup/BaseZImageSetup.py:125-135`:

```python
latent_input = scaled_noisy_latent_image.unsqueeze(2)   # [B,16,1,H,W]
latent_input_list = list(latent_input.unbind(dim=0))    # list of [16,1,H,W]
output_list = model.transformer(
    latent_input_list,
    (1000 - timestep) / 1000,
    text_encoder_output,
    return_dict=True,
).sample
predicted_flow = -torch.stack(output_list, dim=0).squeeze(dim=2)
```

## text_encoder_output structure (the 3rd positional arg `cap_feats`)

`text_encoder_output` is Serenity's `embeddings_list` built in
`modules/model/ZImageModel.py:encode_text` (lines 170-173):

```python
bool_attention_mask = tokens_mask.bool()
embeddings_list = [sample[bool_attention_mask[i]] for i, sample in enumerate(text_encoder_output)]
```

i.e. the Qwen3 `hidden_states[-2]`, masked per sample.

- **Python type:** `list[torch.Tensor]`, one entry per batch sample.
- **Each entry shape:** `[caption_len, feat_dim]` = `[64, 2560]` (bf16).

diffusers `forward(self, x, t, cap_feats, ...)`:
- `x`: `list[Tensor]`, each `[C, F, H, W]` = `[16, 1, 16, 16]`.
- `t`: flow time `(1000 - timestep)/1000`.
- `cap_feats`: `list[Tensor]`, each `[64, 2560]`.

`omni_mode = isinstance(x[0], list)`. Here `x[0]` / `cap_feats[0]` are Tensors,
so **basic (non-omni) mode**: single image + single caption per sample.

## Fixed inputs (reproducible)

- numpy seed `1234`, `np.random.default_rng(SEED).standard_normal(...)`.
- Latent fed to transformer: `[1, 16, 1, 16, 16]` bf16 (no extra scaling — this
  isolates the transformer forward).
- `t_model = (1000 - 250)/1000 = 0.75`.
- cap_feats: `[64, 2560]` bf16.

## Velocity convention

`zi_fwd_velocity.bin` is the **RAW** transformer `.sample`
(`torch.stack(output_list,0).squeeze(2)` → `[1,16,16,16]`).
The Mojo wrapper returns this same RAW velocity.

`predicted_flow = -velocity` (negation applied OUTSIDE the transformer; it is
**NOT** baked into the dumped bin).

## Dumped files (all float32, little-endian, row-major / C-order)

| File | Shape | dtype | Bytes | Contents |
|------|-------|-------|-------|----------|
| `zi_fwd_latent.bin`   | `[1,16,16,16]` | float32 LE | 16384  | Input latent (F=1 squeezed). The `[1,16,1,16,16]` transformer input is this reshaped. Mojo casts to bf16 on upload. |
| `zi_fwd_cap.bin`      | `[64,2560]`    | float32 LE | 655360 | Caption hidden states fed as `cap_feats[0]`. |
| `zi_fwd_velocity.bin` | `[1,16,16,16]` | float32 LE | 16384  | RAW transformer `.sample` output. `predicted = -velocity` (separate). |
| `zi_fwd_meta.json`    | -              | JSON       | -      | Shapes, seed, t_model, exact text_encoder_output structure, velocity stats. |

### Byte layout (row-major)

- **`zi_fwd_latent.bin`** `[1,16,16,16]` index `(b,c,h,w)`:
  offset = `((c*16 + h)*16 + w) * 4` bytes (b=0).
- **`zi_fwd_cap.bin`** `[64,2560]` index `(s,f)`:
  offset = `(s*2560 + f) * 4` bytes.
- **`zi_fwd_velocity.bin`** `[1,16,16,16]` index `(b,c,h,w)`:
  offset = `((c*16 + h)*16 + w) * 4` bytes (b=0).

Each value is a 4-byte IEEE-754 little-endian float32.

## Velocity stats (this run, seed 1234, t=0.75)

```
shape:     (1, 16, 16, 16)
mean:      -0.02331649
std:        1.65836108
min:       -6.06250000
max:        6.12500000
nonfinite:  0
```

(bf16 compute → values land on bf16-representable points, e.g. min/max are
exact bf16 ticks.)
