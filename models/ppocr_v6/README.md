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

**2. 48×320 silently truncates long lines.** The recogniser sees a fixed
320-wide input, i.e. a 6.7:1 aspect. Feed it a crop longer than that and the
squeeze merges characters — the decode does not fail, it just drops some.
Measured on a 48×951 crop (19.8:1):

| rec width | timesteps | decoded |
|---|---|---|
| 320 | 40 | `1030520250215/20240427A026` |
| 960 | 120 | `1018308520250215/20240427A0226` ✅ |

The 960 result is character-exact against PaddleOCR's own output. **This is not
a v6 regression — v5 at 48×320 has the same limit.** If your lines are long,
compile a wider recogniser, but note the `.hbm` grows with input area: 48×960 is
3× the area of 48×320 for the same per-character work. Splitting long lines
upstream of the recogniser is usually the better trade.

**3. Normalisation is not in `config.yaml`.** Both models compile as
`featuremap` / `no_preprocess`, matching v5 so the runtime's OCR path needs no
change. That means the compiler applies nothing, and the normalisation contract
lives between `calib.py` and the runtime's CPU preprocessing rather than in the
yaml. This is the one model here where `calib_pack.py` cannot enforce that
invariant for you — if you change the runtime preprocessing, change `calib.py`
to match, or the calibration distribution silently stops matching deployment.

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
| rec | 22.6 MB | 0.9747 | mixed (compiler-chosen: 162 int8 + 27 int16) | 64 line crops cut from those pages by the det ONNX |

The recogniser's mix was **chosen by the compiler**, not requested: with no
`optimization` directive, hb_compile promoted 27 nodes to int16 on its own —
almost certainly the LayerNorm-adjacent layers a transformer stack is sensitive
at. Worth knowing before reaching for a manual mixed-precision config: the
default already does the obvious promotions.

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
- **rec, layer C**: cosine **0.9493**, per-step top-1 agreement 33/40, and the
  int8 decode **drops characters** the float ONNX keeps
  (`1030520250215/…` → `120250215/…`). This is the concrete answer to "is int8
  recognition good enough": on a hard crop, visibly not. An int16 recogniser is
  the next experiment if accuracy matters.

The two int8 output cosines being below 0.99 is therefore **not** a port
problem — layer B proves the board matches the host. It is the quantisation
itself, and it is a real cost for recognition on hard inputs.

**Still not done:** no latency measurement (needs `hrt_model_exec perf` on a
quiet board), and no end-to-end accuracy against a labelled page set.
