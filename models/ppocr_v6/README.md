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
page does not resemble a text line.

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

**Not yet done:** neither model has been compiled to `.hbm` or run on the
board, so there are no quantisation, latency or accuracy figures. `expected.json`
records gates, not measurements. Note also that both graphs contain `Erf`
(GELU); whether it lands on BPU or falls back to CPU is unmeasured, and for the
detector that matters — check `<prefix>_node_info.csv` after the first compile.
