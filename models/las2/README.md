# Lite Any Stereo V2 (LAS2) — stereo disparity, 480×640

Zero-shot stereo matching, fully on the BPU, int16, single `.hbm`.
Upstream: [LiteAnyStereo](https://github.com/TomTomTommi/LiteAnyStereo).

| build | BCDL consumes | on-board |
|---|---|---|
| `las2_m_crop_nashm.hbm` | stereo disparity (centre-crop letterboxing) | ~14 ms / 71 FPS |
| `las2_m_int16_nashm.hbm` | stereo disparity (resize letterboxing) | ~14 ms / 71 FPS |

Licence tier **D** — recipe only, upstream terms not yet reviewed.

## Which build is wrong for you

The two shipped builds are **not** better/worse. They differ only in how the
calibration pairs were letterboxed, and **using the wrong one costs accuracy
silently** — the model loads, runs at full speed, and returns a plausible
disparity map:

- `_crop` — calibrated on **centre-cropped** pairs (`gen_calib.py --mode crop`)
- `_int16` — calibrated on **resized** pairs (`gen_calib.py --mode resize`)

Match it to how your runtime letterboxes the input pair. BCDL's stereo test
feeds a centre-cropped pair and therefore prefers `_crop`.

## The three traps on this model

**1. `int8` is not an option, and mixed precision is a trap.**
Disparity is `soft-argmax`: `Σ softmax(cost_volume) · disparity_index`. Small
quantisation errors in the softmax distribution do not cancel — they accumulate
into a **systematic disparity offset**. Measured on identical calibration data:

| quantisation | EPE | on-board FPS | |
|---|---|---|---|
| int8 (all) | 1.23 px | 174 | ~1 px systematic offset + speckle |
| mixed (cost volume + head int16) | 0.25 px | **12** ⚠️ | requant at scattered int8↔int16 boundaries dominates — *slowest of the three* |
| **int16 (all)** | **0.12 px** | 71 | uniform dataflow, no conversions ✅ |

The mixed-precision result is the counter-intuitive one and worth internalising
beyond this model: **unless int16 layers coalesce into large contiguous blocks,
the boundary requant cost exceeds what you save.** Reaching for mixed precision
because "only the sensitive layers need int16" made this model 6× slower than
just quantising everything to int16.

**2. `torch.onnx` cannot export `aten::unfold`** (fails with *input size not
accessible*). LAS2 uses it twice — building the correlation volume, and the
context upsample. `shifts.py` rewrites both into pad+slice over standard
primitives, numerically equivalent to **1e-6 px EPE**. This is the general
technique for putting an unusual model on the BPU: rewrite the unsupported
operator as equivalent standard tensor primitives rather than giving up on the
model.

**3. The model has `@torch.compile` on internal functions.** Without triton
installed the export dies, so `export.py` forces `TORCHDYNAMO_DISABLE=1`. Pure
eager — it does not change the exported graph.

## Why this model at all

Picked over FoundationStereo, which compiles fully to BPU but runs a 3D cost
volume at **~40 s/frame** — not a BPU-suitable architecture. LAS2 is
BPU-friendly by construction: FasterNet 2D CNN backbone (not ViT), cost
aggregation **100% 2D** (disparity folded into channels, no 3D conv),
feed-forward with no GRU iteration, and BatchNorm rather than LayerNorm (which
avoids the int16 LayerNorm question entirely).

Operator profile at M/480×640: 2259 nodes, all 2D conv, max rank 4 (no 5D), one
nearest-neighbour Resize, and no unfold / 3D-conv / LayerNorm / GridSample.

## Running it

```bash
# 1. upstream checkpoint -> BPU-friendly ONNX (~6s)
python export.py --repo /path/to/LiteAnyStereo --size m --hw 480 640 \
                 --out las2_m_480x640.onnx

# 2. calibration. The config selects the build; --mode must agree with it.
python calib.py --config config.yaml --mode crop \
                --sbs-dir '/path/to/captures/Explorer_HD2K*.png'

# 3. compile (config.yaml = crop, config_resize.yaml = resize)
./compile.sh --config config.yaml --gpu 2
```

`--sbs-dir` takes a **glob**, not just a directory: capture folders usually hold
more than the stereo set, and the extras are both out-of-domain (RULE 2) and
often too small to crop.

`--gpu` is an **nvidia-smi index**, handed to docker as `--gpus device=N`. It is
deliberately not `CUDA_VISIBLE_DEVICES`, which numbers devices differently — see
[CONVERSION.md](../../CONVERSION.md#traps). The card must be Ampere or newer.

Both configs target `nash-m` (S100/S100P). `jobs` lives in the config, not on
the command line.

## Status

Re-run end to end on OpenExplorer 3.7.0 and verified at each step:

- **export** reproduces the original model — 2259 nodes, identical op histogram
  and I/O signature, all 278 initializers matching to 7e-7. Not byte-identical
  (ONNX metadata differs), which is expected.
- **calibration** writes 5 same-domain pairs per RULE 2, through
  `common/calib_pack.py`.
- **layer A** (quantisation fidelity) measured **cosine 0.999954** on `disp`,
  against the 0.99 gate in `expected.json`.

Layers B and C need the board and have not been re-measured in this repo; the
figures in `expected.json` come from the original adaptation work.
