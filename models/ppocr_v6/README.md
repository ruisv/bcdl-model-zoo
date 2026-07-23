# PP-OCRv6 — text detection + recognition

Upstream: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) v3.7.0.
Licence tier **A** (Apache-2.0).

| build | shape | output |
|---|---|---|
| `ppocrv6_medium_det_960x960.hbm` | 1×3×960×960 | 1×1×960×960 DB probability map |
| `ppocrv6_medium_rec_48x320.hbm` | 1×3×48×320 | 1×40×18710 CTC logits — short/mixed lines |
| `ppocrv6_medium_rec_int16_48x960.hbm` | 1×3×48×960 | 1×120×18710 — **long lines, int16 (recommended)** |
| `ppocr_keys_v6_18710.txt` | — | dictionary, paired with any rec build |

There are four rec builds (320/960 × int8/int16) because two independent axes
matter: **width** (320 truncates lines over 6.7:1, 960 takes ~20:1) and
**precision** (int16 roughly halves the character error). The two recommended
points are **320 int8** for short/mixed content and **960 int16** for long-line
workloads; see [the rec build guide](#which-rec-build) below.

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

**Should the rec ONNX just be exported wider?** For long-line workloads, yes — a
960 build is provided and works well (see [Which rec build](#which-rec-build)).
`export.py --task rec --hw 48 960` produces a valid `1×3×48×960 → 1×120×18710`
graph, and on the board it decodes long lines the 320 model truncates: a 79-char
Chinese sentence exactly, `test.jpg`'s 30-char string with a single insertion.
The cost is real — the `.hbm` grows with input **area** (48×960 is 3× the
instructions of 48×320) and short lines waste most of that width on padding — so
for mixed content the better answer is still to **split long lines upstream** (the
detector already yields per-line boxes) and keep the 320 model. Reserve 960 for a
workload that is genuinely long-line-dominated.

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
| rec 320 (int8) | 22.7 MB | 0.9819 | mixed (compiler-chosen: 162 int8 + 27 int16) | 64 line crops cut from those pages by the det ONNX |
| rec 320 (int16) | 24.0 MB | 0.9974 | all int16 (`config_rec_int16.yaml`) | same 64 crops |
| rec 960 (int8) | 30.7 MB | — | mixed | 64 crops, padded to 960 |
| rec 960 (int16) | 27.7 MB | 0.9981 | all int16 (`config_rec_int16_960.yaml`) | same |

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

### Which rec build

All four measured on the board against the matching float ONNX decode, with the
**correct pad preprocessing** on both calibration and evaluation. The 320 and 960
rows use different test sets — 320 on short aspect-fitting crops, 960 on long
lines — so read *int8-vs-int16 within a width*, not across widths:

| build | config | char error rate | latency | hbm |
|---|---|---|---|---|
| 320 int8 | `config_rec.yaml` | 12.6 % | 4.69 ms | 22.7 MB |
| 320 int16 | `config_rec_int16.yaml` | **4.9 %** | **2.17 ms** | 24.0 MB |
| 960 int8 | `config_rec_960.yaml` | 17.5 % | 12.78 ms | 30.7 MB |
| 960 int16 | `config_rec_int16_960.yaml` | **7.5 %** | **5.30 ms** | 27.7 MB |

(CER measured against float — 320 rows on short aspect-fitting crops, 960 rows on
long lines, so compare int8-vs-int16 within a width, not across. Latency is a
single BPU thread on a quiet S100P; 4-thread throughput adds <11 %, so these are
BPU-bound.)

**int16 wins on every axis — accuracy, size, AND speed — so there is no reason to
ship the int8 rec build:**

- It roughly **halves the character error** (4.9 vs 12.6, 7.5 vs 17.5).
- It is **2.2–2.4× faster** (2.17 vs 4.69 ms at 320; 5.30 vs 12.78 at 960).
- At 960 it is even **smaller** (27.7 vs 30.7 MB).

The counter-intuitive part is the speed: quantising *more* aggressively is
faster here. The reason is that the "int8" build is **not pure int8** — with no
directive the compiler picks mixed precision (164 int8 + 25 int16 + 2 int32), and
every int8↔int16 boundary costs a requant. All-int16 is one contiguous
fixed-point stream with no requant at all. It is the same trap LAS2 documents:
scattered mixed precision loses to uniform int16. Its layer B is also
bit-identical (no float CPU ops to drift through, unlike the mixed build's 2e-4).

**Ship 320 int16 for short/mixed content and 960 int16 for long lines.** The int8
builds are kept only to demonstrate the cost. Detection stays int8 — it is a
plain CNN with no requant problem, 8.49 ms / 118 FPS at 960×960.

On the **960 long lines int16 is near-perfect** where it matters: a 79-char
Chinese sentence decoded exactly, 74–76-char lines with one error each, and
`test.jpg` with a single insertion. The 7.5 % is carried by the short, hard crops
mixed into that set, not by the long lines 960 exists for.

> An earlier 320 table (stretch preprocessing) reported int8 8/20 vs int16 17/20,
> exaggerating both the loss and the gap. Fixing the preprocessing lifted both and
> shrank the difference; these are the corrected numbers.

960 int16 costs **2.4× the latency** of 320 int16 (5.30 vs 2.17 ms) for ~3× the
area — cheap enough at 189 FPS that a long-line workload should just use it.

**Still not done:** no end-to-end accuracy against a labelled page set (the CER
figures are vs the float model, not vs ground truth).
