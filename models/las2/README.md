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
python export.py   --ckpt <upstream .pth> --out las2_m_480x640.onnx
python calib.py    --mode crop   --out cal_crop/        # or --mode resize
./compile.sh --workdir . --onnx las2_m_480x640.onnx --calib cal_crop \
             --prefix las2_m_crop_nashm --march nash-m --gpu 0
```

`--march nash-m` targets S100P. The calibration GPU needs Ampere or newer — the
PTQ CUDA kernels fail with `cudaErrorInvalidDevice` on older cards.

## Status

The scripts here are the originals from the LAS2 adaptation work and are
complete. `calib.py` is still named `gen_calib.py` and does not yet route
through `common/calib_pack.py`; the accuracy figures above were measured with
these scripts as they stand.
