# bcdl-model-zoo

ONNX → `.hbm` conversion recipes for the D-Robotics RDK **S100 / S100P / S600**
BPU, for the models that [BCDL](https://github.com/ruisv/bcdl) can decode.

BCDL is the on-board runtime: it loads a compiled `.hbm` and runs inference,
pre/post-processing, codecs and pipelines. **It does not convert models.** This
repo is the other half — it runs on an x86 host with a GPU and the D-Robotics
OpenExplorer toolchain, and produces the `.hbm` files BCDL consumes.

## What this repo does and does not ship

It ships **recipes**: the ONNX export (including graph surgery), the calibration
data generator, the `hb_compile` config, and the acceptance thresholds — enough
to reproduce a model from its upstream checkpoint.

It ships **almost no binaries**. A compiled `.hbm` is a derivative of upstream
weights, and those weights carry their own licences — several of them copyleft
or non-commercial. Only models whose upstream licence clearly permits it are
distributed as compiled binaries; for everything else you run the recipe.

> **Read this before assuming you can reproduce the BCDL benchmark table.**
> The models BCDL's detection / classification / pose / segmentation / OBB
> benchmarks run on are the YOLO family, which is **AGPL-3.0**. We do not host
> binaries for those. You can rebuild them from the recipes here, but you cannot
> download them from us.

| tier | licence | models | what you get |
|---|---|---|---|
| **A** | Apache-2.0 / MIT / BSD | PP-OCRv5 det·cls·rec, ViTPose-S, XFeat, SPAN, Real-ESRGAN Compact, YOLOP, PIDNet-S, Depth-Anything-V2 **Small** | recipe **+ compiled `.hbm`**, with upstream LICENCE and provenance |
| **B** | AGPL-3.0 | YOLO26 / YOLOv8 / YOLOE (Ultralytics) | **recipe only** — hosting AGPL derivative binaries pulls AGPL obligations onto the distributor |
| **C** | non-commercial weights | SCRFD / ArcFace (insightface), Depth-Anything-V2 Base+ | **recipe only**; for a commercial build see the alternatives noted per model |
| **D** | upstream unverified | LAS2, EdgeSAM, OSNet | **recipe only**, pending a per-model licence review |

## Layout

```
common/
  calib_pack.py     # the ONLY way calibration .npy files get written
  verify_cosine.py  # three-layer cosine check (A/B/C, see below)
models/<name>/
  README.md         # what goes wrong on this model, and which build is correct
  export.py         # upstream checkpoint -> ONNX, including graph rewrites
  calib.py          # regenerate calibration data from a public dataset
  config.yaml       # hb_compile configuration
  expected.json     # acceptance thresholds + measured on-board reference
```

## Two rules that exist because they were learned the hard way

**1. Calibration data must be pre-normalised, and only `calib_pack.py` writes
it.** When `cal_data_type` is float32, the compiler's `norm_type` does **not**
apply to the calibration data. Get this wrong and the input thresholds come out
wrong — and the model still compiles without a single warning, then decodes to
garbage. This is not a footnote; it is why PIDNet needed three attempts.

**2. Every model's README documents the *wrong* build, not just the right one.**
The failure mode across nearly all of these models is that the bad build
**compiles cleanly**. A document that only describes the correct path offers no
protection. So each README leads with what the sibling build does to you.

The same reason drives the naming rule inherited from BCDL: **a file's name
matches the model name inside it.** Suffixes like `_cut`, `_v3`, `_aligned`,
`_qat` are *which build this is*, not decoration.

## Verification

A recipe is accepted when `expected.json` is met and measured:

- **A — quantisation fidelity**: `.bc` vs ONNX, cosine > 0.99
- **B — execution consistency**: on-board `.hbm` vs host `.bc`, bit-identical (cosine = 1.0)
- **C — end-to-end**: on-board `.hbm` vs ONNX on real images, cosine > 0.99

**Cosine alone is not sufficient and must not be the only gate where a task
metric exists.** OSNet is the worked example: int8 PTQ produces well-formed unit
vectors that look fine to a cosine check, while Market-1501 Rank-1 collapses
from 85% to 51%. Any model with a measurable downstream metric declares that
metric in `expected.json` too.

## Status

Recipes are being reconstructed from the original conversion work; not every
model in BCDL's catalogue is here yet. **A recipe being present does not mean it
has been re-run** — each model's README states exactly how far it got, and
`expected.json` distinguishes gates from measurements.

| model | recipe | verified to |
|---|---|---|
| [`las2`](models/las2/) — stereo disparity | complete | export reproduces the original graph; **layer A cosine 0.999954**; compiled on OE 3.7.0 (38.8 MB hbm). Rebuild is not the shipped binary — calibration set differs; board layers B/C not measured here |
| [`ppocr_v6`](models/ppocr_v6/) — OCR det + rec | complete | decode verified character-exact against PaddleOCR in ONNX Runtime; **both compiled to `.hbm`** on OE 3.7.0 (det 0.98 / rec 0.97 int8 output cosine). Not board-run |

Recipes still to reconstruct, roughly in cost order: `superres`, `xfeat`,
`yoloe` (scripts survive intact); `span`, `vitpose` (one piece missing each);
`face`, `edgesam` (recoverable by diffing the derived ONNX against its parent);
`pidnet`, `yolop`, `osnet` (the three whose write-ups are worth the most);
`yolo26` (hardest — its compile config was never saved and must be reconstructed
from the compile log).

## Licence

The recipes and scripts in this repo are MIT. **The models are not ours** — each
one carries its upstream licence, recorded in its model directory. Check it
before you redistribute or ship commercially.
