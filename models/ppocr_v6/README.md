# PP-OCRv6 — text detection + recognition

Upstream: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) v3.7.0.
Licence tier **A** (Apache-2.0).

| build | shape | output |
|---|---|---|
| `ppocrv6_medium_det_960x960.hbm` | 1×3×960×960 | 1×1×960×960 DB probability map |
| `ppocrv6_medium_rec_48x320.hbm` | 1×3×48×320 | 1×40×18710 CTC logits |
| `ppocr_keys_v6_18710.txt` | — | dictionary, paired with the rec model |

There is **no v6 angle classifier**. Upstream ships only det and rec for v6, so
keep using the v5 PP-LCNet textline classifier if you need one.

## Why this is easier than v5

v5 needed `paddle2onnx`, and that step is where its recipe rotted: the surviving
export script pointed at a **seal detector** rather than the v5 detector, and
the shape-fixing call was commented out. **v6 publishes ONNX directly**, so
`export.py` only fixes shapes — there is no Paddle dependency and no model
selection to get wrong.

## What goes wrong here instead

**1. The dictionary must match the exact model that produced it.**
`medium` and `small` share an 18710-class charset. **`tiny` is a different
charset with 6906 classes.** Decoding a tiny model with the medium dictionary
does not error — it produces confident, wrong text. `export.py` extracts the
dictionary from the same `inference.yml` that ships beside the weights and
asserts the entry count equals the model's real output width, so a mismatched
pair fails at export instead of at read time.

The file format is BCDL's: line 0 is a literal `blank` placeholder, then the
characters in upstream order, then a trailing space — so the line count equals
the class count. PaddleOCR itself stores only the characters and adds blank and
space at runtime; the +2 is applied during export.

**2. 48×320 silently truncates long lines, and this is separate from
quantisation.** The recogniser sees a fixed 320-wide input, i.e. a 6.7:1 aspect.
Feed it a crop longer than that and the squeeze merges characters — the decode
does not fail, it just drops some. This is a **float** effect, with no
quantisation involved. Measured on a 48×951 crop (19.8:1), decoding the float
ONNX:

| model | rec width | chars emitted (truth = 30) |
|---|---|---|
| v6 float | 320 | 26 — drops 4 |
| v5 float | 320 | 17 — drops 13 |
| v6 float | 960 | 30 — exact |

Both float models drop characters at 320; the fix is width, not precision. **This
is not a v6 regression — v5 has the same limit, worse.**

**Should the rec ONNX just be exported wider?** It depends on your text. With the
correct aspect-preserving preprocessing (trap #3), 320 already handles any line
up to 6.7:1 without distortion — a short line is scaled to its natural width and
right-padded, not stretched. Only genuinely long lines (a full sentence, a long
digit string like the 19.8:1 sample above) overflow 320 and get truncated. For
those, export wider — `export.py --task rec --hw 48 960` produces a valid
`1×3×48×960 → 1×120×18710` graph — but the `.hbm` grows with input **area**
(48×960 is 3× the instructions of 48×320) and every short line then wastes most
of that width on padding. Usually the better answer is to **split long lines
upstream** (the detector already yields per-line boxes) and keep the 320 model.
Reserve a wide build for a workload that is genuinely long-line-dominated.

The practical consequence for **evaluating quantisation**: never measure int8 vs
int16 on a crop wider than 6.7:1, or the aspect loss swamps the quantisation
signal. The board section below uses aspect-fitting crops for exactly this
reason.

**3. Recognition preprocessing is aspect-preserving + right-pad, NOT a stretch.**
This is the one that bites hardest, and it is easy to get wrong. PaddleOCR's
`resize_norm_img` (and BCDL's `packNchwPad`) scale the crop to height 48 keeping
its aspect ratio, then **right-pad with zeros** to width 320. A naive
`cv2.resize(img, (320, 48))` instead **stretches** a short line to fill the full
width, which distorts every glyph. Measured impact, decoding the **float** ONNX
(no quantisation at all) on 20 short crops: stretch vs pad changed the decode on
**16 of 20**, and pad was almost always right —

| stretch (wrong) | pad (correct) |
|---|---|
| `CT\|NHO` | `COUTINHO` |
| `C` | `Englis` |
| `''` | `P2` |
| `0CHARD` | `RCHARD` |

The distortion dwarfs the int8/int16 difference. `calib.py` does the correct
pad, so the calibration distribution matches deployment; if you evaluate with a
stretch you will blame the quantiser for what the preprocessing broke.

Channel order is **RGB** (BGR read, R into channel 0) to match BCDL/ccdl.
Upstream PaddleOCR's own inference happens to feed BGR to this same graph; the
deployment target here is BCDL, and calibration follows the deployment.

**4. Normalisation is not in `config.yaml`.** Both models compile as
`featuremap` / `no_preprocess`, matching v5 so the runtime's OCR path needs no
change. The compiler applies nothing, so the whole normalisation-and-resize
contract lives between `calib.py` and the runtime's CPU preprocessing rather than
in the yaml. This is the one model here where `calib_pack.py` cannot enforce that
invariant for you — trap #3 is exactly what goes wrong when it drifts.

## Running it

```bash
SRC=<dir holding the PP-OCRv6_* huggingface repos>

python export.py --src $SRC --task det --size medium     # -> 960x960 onnx
python export.py --src $SRC --task rec --size medium     # -> 48x320 onnx + dict

python calib.py --config config_det.yaml --task det --images <page images>
python calib.py --config config_rec.yaml --task rec --images <line crops>

./compile.sh --config config_det.yaml --gpu <ampere-or-newer>
./compile.sh --config config_rec.yaml --gpu <ampere-or-newer>
```

Calibrate recognition on **real line crops** — ideally cut by the detector —
not on whole pages scaled to 48 high. The activation distribution of a squashed
page does not resemble a text line. The concrete way to get those crops is to
run the detection ONNX over the same page images and cut its boxes: that is
exactly the distribution the recogniser sees at deployment, and it costs one
`onnxruntime` pass. The reference build here was calibrated that way — 24
detection pages (from PaddleOCR's own dataset sample images) fed the detector,
whose boxes produced 64 recognition line crops.

## Status

Verified as far as it can be without the board:

- **export** produces fully static graphs from the dynamic upstream ONNX;
  det `1×3×960×960 → 1×1×960×960`, rec `1×3×48×320 → 1×40×18710`.
- **dictionary** extracted and asserted against the model width for medium
  (18710) and tiny (6906).
- **decode verified end to end in ONNX Runtime**: at 48×960 the greedy CTC
  decode, using BCDL's exact convention (blank at index 0, collapse repeats,
  `dict[argmax]`), reproduces PaddleOCR's own output character for character.
  This confirms the dictionary, the class indexing and the preprocessing
  constants before any BPU work.

### Compiled

Both models compiled to `.hbm` on OE 3.7.0 (HBRT 4.7.5), all int8 PTQ:

| | hbm | output cosine (vs float) | quantisation | calibration |
|---|---|---|---|---|
| det | 22.4 MB | 0.9817 | all int8 | 24 dataset sample pages |
| rec (int8) | 22.7 MB | 0.9819 | mixed (compiler-chosen: 162 int8 + 27 int16) | 64 line crops cut from those pages by the det ONNX |
| rec (int16) | 24.0 MB | 0.9974 | all int16 (`config_rec_int16.yaml`) | same 64 crops |

The recogniser's default mix was **chosen by the compiler**, not requested: with
no `optimization` directive, hb_compile promoted 27 nodes to int16 on its own —
almost certainly the LayerNorm-adjacent layers a transformer stack is sensitive
at. Worth knowing before reaching for a manual mixed-precision config: the
default already does the obvious promotions.

The rec cosine here is **0.9819, up from 0.9747** in an earlier build — the only
thing that changed was fixing the calibration preprocessing (trap #3): the same
stretch-vs-pad error that distorts inference also distorted the calibration set,
so getting it right improved the quantisation itself, before any bit-width change.

One fix the compile forced that reading could not: the **detector exports at
ONNX IR10**, which HBDK rejects (max IR9). `export.py` now caps it; the
recogniser at IR6 was unaffected. See
[CONVERSION.md](../../CONVERSION.md#traps).

Both output cosines sit **below the 0.99 layer-A gate**. That is int8 on
transformer-ish OCR heads and is expected — DB detection thresholds its
probability map and CTC recognition is argmax-per-step, so both absorb some
fidelity loss. Whether it costs real recall or character accuracy is a **board**
question; compare against an int16 build only if it does, and do not fail these
on cosine alone (see [CONVERSION.md](../../CONVERSION.md#verification)).

The `Erf` (GELU) question **is settled for detection**: all 28 GELU nodes show
`ON=BPU` in `node_info.csv`, and `ON=BPU` is trustworthy (unlike `--`). No CPU
fallback. Latency itself is still unmeasured until the board.

### Board close-out

Both models were run on an S100P (HBRT 4.7.5):

- **det, layer B**: **bit-identical** to the host `.bc` (cosine 1.0, error 0).
- **det, layer C**: cosine **0.9817** vs float ONNX. Since layer B is exact,
  this is pure int8 quantisation loss, not a port defect. Whether 0.98 costs box
  recall needs a boxed comparison on real pages, not a cosine.
- **rec, layer B**: cosine 1.0 but **not** bit-identical (max abs error 2e-4).
  The board and host nonetheless decode to **identical text**
  (`120250215/020427A026`). This model is mixed int8/int16/int32 with float CPU
  ops, so the last-place float difference is expected; the right layer-B check
  here is that the decode matches, and it does.
- **rec, layer B**: cosine 1.0 but not bit-identical (2e-4). Same reason as
  above; board and host decode identically.

The int8 output cosine being below 0.99 is **not** a port problem — layer B
proves the board matches the host. It is the quantisation itself.

### int8 vs int16 recognition (corrected preprocessing)

Measured on 20 aspect-fitting crops (≤6.7:1, so the aspect limit is out of the
picture), with the **correct pad preprocessing** on both calibration and
evaluation, each board build compared to the float ONNX decode:

| rec build | exact match | char error rate | layer B | hbm |
|---|---|---|---|---|
| int8 (default) | 9 / 20 | **12.6 %** | cosine 1.0, 2e-4 drift | 22.7 MB |
| int16 (`config_rec_int16.yaml`) | 16 / 20 | **4.9 %** | bit-identical | 24.0 MB |

Two readings of the same data:

- **int16 more than halves the character error** (4.9 % vs 12.6 %) and is the
  build to use when recognition accuracy matters. Its layer B is bit-identical
  again — a pure int16 graph has no float CPU ops to drift through, unlike the
  mixed default.
- **int8 is better than it first looked.** Most of its errors are single-glyph:
  `bywolu→bywolun`, `PA→IPA`, simplified-vs-traditional `大國→大国`, and in one
  case int8 is *more* complete than the float reference (`TheKig'→TheKing'`).
  Exact-match 9/20 reads worse than the 12.6 % CER because one wrong glyph fails
  a whole line. For easy, well-cropped Latin text int8 is fine.

An earlier version of this table (stretch preprocessing) reported int8 at 8/20
and made int8 look far worse — that was the preprocessing distortion, not the
quantiser. Fixing the preprocessing lifted both builds and shrank the gap; the
int16 advantage is real but smaller than a stretched measurement implied.

**Still not done:** no latency measurement (needs `hrt_model_exec perf` on a
quiet board), and no end-to-end accuracy against a labelled page set.
