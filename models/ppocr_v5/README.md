# PP-OCRv5 — text detection + recognition + angle classification

Upstream: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
Upstream licence: **Apache-2.0**.

| build | shape | output |
|---|---|---|
| `ppocrv5_server_det_960x960.hbm` | 1×3×960×960 | 1×1×960×960 DB probability map |
| `ppocrv5_server_rec_48x320.hbm` | 1×3×48×320 | 1×T×~18385 CTC logits |
| `ppocrv5_lcnet_cls_80x160.hbm` | 1×3×80×160 | 1×2 — textline orientation (0° / 180°) |
| `ppocr_keys_v5_<N>.txt` | — | dictionary, paired with the rec build |

## Why v5 exists alongside v6

The newer [PP-OCRv6 recipe](../ppocr_v6/README.md) supersedes v5 for detection
and recognition, so **v5 det/rec are kept only as a documented fallback**. The
one piece that is *not* superseded is the **angle classifier**: PP-OCRv6 ships no
cls model, so the v5 `PP-LCNet_x1_0_textline_ori` classifier stays in service. It
decides whether a detected line is upright or rotated 180° before recognition —
a step v6 has no replacement for. That is the reason this directory is here.

All three sub-models compile as `featuremap` / `no_preprocess`, matching v6 so
the runtime's OCR preprocessing path is uniform across versions.

## The trap that makes v5 harder than v6

v6 publishes ONNX directly. **v5 ships only Paddle inference models**, so every
sub-model must go through `paddle2onnx` first — and that is exactly where the
original v5 recipe rotted. Two independent bugs were baked into the surviving
export scripts, and both compile cleanly into a wrong `.hbm`:

**1. The detector script pointed at a SEAL detector.** The salvaged
`export_onnx_det.sh` ran paddle2onnx on `PP-OCRv4_server_seal_det` — a *seal*
detector, and even the wrong major version — instead of the general text
detector `PP-OCRv5_server_det`. paddle2onnx converts it without a complaint: you
get a valid ONNX of the wrong network, which compiles, loads, and then detects
nothing on ordinary document text. Copying that script verbatim silently
converts the wrong model.

**2. The shape-fix was commented out for det and rec.** The
`paddle2onnx.optimize --input_shape_dict` call that pins the graph to a static
shape (det → `1×3×960×960`, rec → `1×3×48×320`) was **commented out** in the det
and rec scripts — a naive re-run therefore produces a *dynamic-shape* ONNX that
the BPU compiler will not accept. Only the cls script was clean: it named the
right model (`PP-LCNet_x1_0_textline_ori`) with the shape fix on (80×160).

`export.py` fixes both: it hard-codes the correct Paddle model directory per
task (and refuses a `*_seal_det` directory outright), and it always runs the
shape fix. This is the load-bearing part of the whole recipe.

## What else goes wrong (shared with v6)

**3. The dictionary must match the model's exact class count.** BCDL's CTC
decoder indexes `dict[argmax]`, so a dictionary of the wrong length does not
error — it decodes to confident, wrong text, offset by the mismatch. The v5
server charset is expected to be **18385 classes** (the characters + a `blank`
placeholder at index 0 + a trailing space). `export.py` extracts the charset
from the same `inference.yml` that ships beside the weights, appends blank and
space, and **asserts the count equals the model's real output width** — so a
mismatched pair fails at export instead of at read time. The file is written in
BCDL's format: line 0 is a literal `blank`, then the characters in upstream
order, then a trailing space, making the line count equal the class count.

**4. Recognition (and cls) preprocessing is aspect-preserving pad, NOT a
stretch.** This is the one that bites hardest and is identical to v6's. PaddleOCR
scales a line crop to the target height keeping its aspect ratio, then
**right-pads with zeros** to the target width. A naive `cv2.resize(img, (W, H))`
instead *stretches* a short line to fill the width, distorting every glyph and —
just as bad — handing the calibrator a distribution the runtime never produces.
On v6 this single mistake made int8 look far worse than it is (it even dropped
the int8 output cosine by ~0.007) before any bit-width change. `calib.py` does
the correct pad for both rec (48×320) and cls (80×160); detection uses an
anamorphic resize to 960×960, which is correct for a page.

Channel order is **RGB** (BGR read, R into channel 0) to match BCDL/ccdl.
Normalisation is PaddleOCR's own and lives in `calib.py`, not the config,
because these compile as `no_preprocess`:

    det : (x/255 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]   (ImageNet)
    rec : (x/255 - 0.5) / 0.5                                   ([-1, 1])
    cls : (x/255 - 0.5) / 0.5                                   ([-1, 1])

## Running it

```bash
SRC=<dir holding the PP-OCRv5 *_infer Paddle models>

python export.py --src $SRC --task det     # -> ppocrv5_server_det_960x960.onnx
python export.py --src $SRC --task rec     # -> ppocrv5_server_rec_48x320.onnx + dict
python export.py --src $SRC --task cls     # -> ppocrv5_lcnet_cls_80x160.onnx

python calib.py --config config_det.yaml --task det --images <page images>
python calib.py --config config_rec.yaml --task rec --images <line crops>
python calib.py --config config_cls.yaml --task cls --images <line crops>

./compile.sh --config config_det.yaml --gpu <ampere-or-newer>
./compile.sh --config config_rec.yaml --gpu <ampere-or-newer>
./compile.sh --config config_cls.yaml --gpu <ampere-or-newer>
```

`export.py` needs `paddle2onnx` and the Paddle inference models on the host;
`onnxslim` simplifies and re-asserts the static shape, and the ONNX IR is capped
at 9 (HBDK rejects IR > 9). `--gpu` is an **nvidia-smi index** handed to docker
as `--gpus device=N`, not `CUDA_VISIBLE_DEVICES` — see
[CONVERSION.md](../../CONVERSION.md#traps). The card must be Ampere or newer.
`jobs` lives in the config, not on the command line.

Calibrate detection on **pages** that look like your deployment pages, and
recognition / cls on **real line crops** — ideally cut by the detector, which is
exactly the distribution those two models see at deployment.

## Status

Reconstructed from the salvaged hb_compile configs and the three (rotted) export
scripts. **Not re-run through the toolchain here** — no measured cosine, `.hbm`
size or latency for v5 survived the salvage, so `expected.json` leaves those
null rather than inventing them.

- **export** is the load-bearing reconstruction: correct model ids for all three
  sub-models (the salvaged det script named a seal detector) and the shape fix
  forced on (it was commented out for det and rec).
- **dictionary** count (~18385) is the expected v5 server value but was
  reconstructed from prior notes, not a surviving artifact; `export.py` asserts
  it against the ONNX output width, so a wrong value fails loudly at export.
- **calibration** routes through `common/calib_pack.py` with the correct
  aspect-preserving pad for rec/cls.

For the *measured* behaviour of the shared DB-detection and CTC-recognition
stacks, the [PP-OCRv6 recipe](../ppocr_v6/README.md) is the reference — it is the
same architecture family, compiled and board-verified. Expect v5 det/rec to
behave similarly (notably: v6's recogniser was best as **all-int16**, which beat
the compiler's default mixed precision on accuracy, size *and* latency; measure
`set_all_nodes_int16` on v5 rec before accepting the default mix). Detection and
recognition here are the fallback; **the cls model is the reason to keep v5**.
