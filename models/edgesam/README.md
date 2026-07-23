# EdgeSAM — promptable segmentation, 1024×1024

Segment Anything for the edge: click a point or drag a box, get a mask. EdgeSAM
keeps SAM's promptable split but distils the ViT-H image encoder down to a
lightweight **RepViT** backbone, so it runs on-device.
Upstream: [chongzhou96/EdgeSAM](https://github.com/chongzhou96/EdgeSAM).

Three `.hbm` — one encoder, two decoders:

| build | BCDL consumes | on-board |
|---|---|---|
| `edge_sam_encoder_nashm.hbm` | image → `image_embeddings` [1,256,64,64], run **once per image** | not measured |
| `edge_sam_decoder_sp1_nashm.hbm` | 1-point prompt → mask, run **per prompt** | not measured |
| `edge_sam_decoder_bp2_nashm.hbm` | box (2 corners) prompt → mask, run **per prompt** | not measured |

**Licence tier D — EdgeSAM upstream terms are not yet reviewed.** This directory
is a recipe only; check the upstream licence and the weights' terms before
compiling for redistribution or shipping a `.hbm`.

## The traps

**1. The encoder/decoder split is not optional — and it is the whole point.**
The encoder is the expensive part (the RepViT backbone) and it depends only on
the image, so it runs **once per frame**. The decoder is tiny and runs **once per
prompt** — every click, every box. A single fused graph would re-run the whole
encoder on every interaction. Splitting lets BCDL encode once, cache the
`[1,256,64,64]` embedding, and pay only the cheap decoder per prompt. This is the
same split SAM ships; here it is load-bearing for interactive latency.

**2. Fixed prompts, static graph — each arity is its own compiled decoder.**
SAM's reference ONNX decoder takes a *dynamic* number of prompt points plus
`mask_input` / `has_mask_input` / `orig_im_size`. A BPU graph is **static**: the
number of prompt points is baked into the input shapes at export time. So there
is no single "decoder" — there is one decoder **per prompt arity**:

| build | points | `point_coords` | `point_labels` | meaning |
|---|---|---|---|---|
| `sp1` | 1 | `[1,1,2]` | `[1,1]` | one foreground point (label 1) |
| `bp2` | 2 | `[1,2,2]` | `[1,2]` | a box as two corners (labels 2, 3) |

SAM's point-label convention: `1` = foreground, `0` = background, `2` = box
top-left, `3` = box bottom-right, `-1` = padding. Add a third prompt shape (e.g.
point + box, or multi-point) and you compile a third decoder. The dynamic
dense-prompt path (`mask_input`) and the `orig_im_size` resize are dropped from
the graph; the low-res mask upsample to the original frame is CPU
post-processing in BCDL.

**3. The decoder must be calibrated on REAL encoder embeddings, not random
tensors.** This is the one that bites. The decoder has three inputs, and they are
not independent: `image_embeddings` is the manifold the prompt attends against.
Calibrate it on `np.random` tensors and you gather activation statistics on a
distribution the deployed decoder never sees — and, as everywhere in this repo,
it **compiles clean, loads, runs at full speed, and segments to garbage**. So
`calib.py --part decoder` loads the *exported encoder ONNX*, runs it on the same
calibration images, and feeds its real outputs as the embedding calibration set;
the prompts (`point_coords`, `point_labels`) are sampled inside each image's valid
(non-padded) region so the coordinate statistics are realistic too. All three
inputs go through `common/calib_pack.py` as `;`-separated positional lists — the
multi-input pattern from `las2`, extended from two inputs to three.

**4. `raw_image` + `no_preprocess` means normalisation is baked into the graph.**
SAM normalises with `pixel_mean` / `pixel_std`. `export.py` bakes that into the
encoder graph (a Sub/Div at the front), which is why the input is named
`raw_image` and `config_encoder.yaml` declares `no_preprocess` — the board hands
the graph raw pixels and the graph normalises internally. The resize-longest-side-
to-1024 + pad-to-square letterbox stays on the CPU (it is per-image and
aspect-dependent, so it cannot be static). If your upstream export instead leaves
normalisation on the CPU, export with `--no-bake-norm` and run `calib.py` with
`--cpu-normalize` so the calibration domain matches what the board feeds. Confirm
which one your export does before trusting the encoder output — this is the single
detail to verify against the exporter in hand.

## Why RepViT helps here

Unlike a ViT image encoder, EdgeSAM's RepViT backbone is a **CNN**, so it carries
none of the LayerNorm-precision concern a ViT stack forces on the quantiser. The
encoder config sets no `optimization` directive and lets the compiler pick the
bit width per layer. If a mask-IoU check shows the embedding degrading, measure
`set_all_nodes_int16` on the board before hand-writing a mixed-precision config —
mixed precision is usually a trap here too (see `CONVERSION.md`).

## Running it

All three builds go through `compile.sh`, which mounts this directory at `/ws`
and hands the container one GPU (`--gpu` is an nvidia-smi index; the card must be
Ampere or newer).

```bash
# [1/3] upstream checkpoint -> encoder + two decoder ONNX
python export.py --repo /path/to/EdgeSAM \
                 --ckpt /path/to/EdgeSAM/weights/edge_sam.pth --out-dir .

# [2/3] calibration
#   encoder: SAM-letterboxed 1024x1024 images (raw pixels; norm baked in graph)
python calib.py --config config_encoder.yaml --part encoder \
                --images /path/to/images --limit 64
#   decoders: embeddings from the REAL encoder + sampled prompts
python calib.py --config config_decoder_sp1.yaml --part decoder \
                --images /path/to/images --encoder-onnx edge_sam_encoder_bpu.onnx
python calib.py --config config_decoder_bp2.yaml --part decoder \
                --images /path/to/images --encoder-onnx edge_sam_encoder_bpu.onnx

# [3/3] compile each build
./compile.sh --config config_encoder.yaml     --gpu 2
./compile.sh --config config_decoder_sp1.yaml --gpu 2
./compile.sh --config config_decoder_bp2.yaml --gpu 2
```

Domain match beats sample count: encode the kind of images you will actually
segment. 20–100 is plenty.

## Status

Recipe **reconstructed**, not re-run. The salvaged pieces are the three build
configs (encoder + two fixed-prompt decoders), the featuremap/`no_preprocess`
input contract, and the prompt arities. `export.py` drives the upstream EdgeSAM
export and cuts the split; the exact upstream entry point (`sam_model_registry`)
is best-effort against the checkout — adjust it to the repo in hand. `calib.py`
is the faithful, load-bearing part: it is the multi-input decoder calibration that
feeds real encoder embeddings plus sampled prompts.

**No board numbers survived** — every measured field in `expected.json` is `null`
on purpose. The three-layer close-out (A/B/C) and the mask-IoU task metric have
not been run here; measure on an S100P before quoting anything. And **check the
upstream licence first** (tier D — not yet reviewed).
