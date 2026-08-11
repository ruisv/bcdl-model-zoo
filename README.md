# bcdl-model-zoo

ONNX → `.hbm` conversion recipes for the D-Robotics RDK **S100 / S100P / S600**
BPU, for the models that [BCDL](https://github.com/ruisv/bcdl) can decode.

BCDL is the on-board runtime: it loads a compiled `.hbm` and runs inference,
pre/post-processing, codecs and pipelines. **It does not convert models.** This
repo is the other half — it runs on an x86 host with a GPU and the D-Robotics
OpenExplorer toolchain, and produces the `.hbm` files BCDL consumes.

## What this repo ships

**Recipes**: the ONNX export (including graph surgery), the calibration data
generator, the `hb_compile` config, and the acceptance thresholds — enough to
reproduce any model from its upstream checkpoint. The compiled `.hbm` files
themselves are large and not committed (`.gitignore` excludes `*.hbm`); the
recipe rebuilds them.

**The model weights and the `.hbm` compiled from them follow their own upstream
licences.** A compiled `.hbm` is a derivative of upstream weights, so check the
model's licence before redistributing or shipping commercially — some are
copyleft (AGPL) and some are non-commercial.

**The licence on the repository is usually not the licence on the weights**, and
that gap is where this goes wrong. A checkpoint inherits restrictions the code
licence never mentions — from the **weights it was fine-tuned from** (ViTPose
starts at MAE, which is CC-BY-NC) and from the **dataset it was trained on**
(Cityscapes bars commercial use of models trained on it, explicitly). The table
below is grouped by what you may actually do with a compiled `.hbm`, not by the
badge on the repo. Rows marked *unresolved* are ones no upstream source answers:
silence is not permission.

| what you may do with the `.hbm` | models |
|---|---|
| **Redistribute, including commercially** | PP-OCRv6/v5 det·cls·rec (Apache-2.0 on code and every model card) · XFeat (weights committed in the Apache-2.0 repo) · Real-ESRGAN Compact (BSD-3, weights released by the copyright holder) · LingBot-Depth (Apache-2.0 on code and weights) · Depth-Anything-V2 **Small** |
| **Redistribute, but non-commercial** | ViTPose-S — Apache-2.0 code, but the checkpoint is fine-tuned from **MAE (CC-BY-NC-4.0)** and trained on COCO-WholeBody · PIDNet-S — MIT code, but **Cityscapes** permits distributing trained models and bars commercial use of them in the same breath · YOLOP — MIT code and in-repo weights, but **BDD100K** limits commercial use to member organizations |
| **Redistributable, with one thing worth checking** | LAS2 — MIT on code and weights, but the release was distilled from FoundationStereo, whose NVIDIA licence is research-only. That is upstream's compliance question rather than a term passed down · SPAN — Apache-2.0 by project statement, but the checkpoints ship from a separate file host with no terms attached to the artefact |
| **Recipe only — copyleft** | YOLO26 / YOLOv8 / YOLOE (Ultralytics) — **AGPL-3.0**; commercial use needs Ultralytics' Enterprise licence |
| **Recipe only — non-commercial weights** | SCRFD / ArcFace (insightface) · Depth-Anything-V2 **Base and up**; for a commercial build see the alternatives noted per model |
| **Unresolved — do not redistribute yet** | EdgeSAM (terms not reviewed) · OSNet — MIT code, but every ReID-useful checkpoint is trained on MSMT17 / Market-1501 under academic-only agreements and no upstream source says whether that reaches the weights |

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

Every converted model in BCDL's catalogue now has a recipe. **A recipe being
present does not mean it has been re-run** — three are board-verified end to end
(`complete`), the rest reconstruct the method faithfully from the original
conversion work (`recipe`). Each model's README states exactly how far it got, and
`expected.json` distinguishes gates from measurements — where no board numbers
survived, the measured fields are `null` on purpose, never invented.

| model | recipe | verified to |
|---|---|---|
| [`las2`](models/las2/) — stereo disparity | complete | **full three-layer close-out on the board**: A 0.999954, B bit-identical, C 0.9998. Rebuild differs from the shipped binary (calibration set) |
| [`ppocr_v6`](models/ppocr_v6/) — OCR det + rec | complete | compiled and **board-verified**; det layer B bit-identical. Recognition in **four builds** (320/960 × int8/int16); int16 halves the character error at both widths and 960 int16 decodes long lines near-perfectly (79-char sentence exact). Preprocessing corrected to aspect-preserving pad |
| [`osnet`](models/osnet/) — person ReID | complete | **the first task-metric recipe**: int8 PTQ collapses (Rank-1 51% vs 85%) so the shipped build is **QAT self-distillation** — no labels, ~20 min, Rank-1 recovered to 84.6%. Board-validated in the original M10 work (cosine 0.9799, 0.82 ms/crop). Includes a reproducible int8-collapse path |
| [`yolop`](models/yolop/) — panoptic driving | recipe | the `_cut` surgery: the uncut graph bakes the anchor decode via `ScatterND`, which compiles clean and **never writes the objectness/class columns** (silent, total detector failure). `export.py` cuts before the decode and emits the 3 raw heads; BCDL decodes. Not re-run through the toolchain here |
| [`pidnet`](models/pidnet/) — real-time seg | recipe | the `_v3` pre-normalisation trap: un-normalised calibration data compiles clean and segments to noise. `calib.py` routes through `calib_pack` so calibration matches the runtime distribution. Export reconstructed (opset-19, no-norm); calibration is the faithful, load-bearing part |
| [`superres`](models/superres/) — Real-ESRGAN ×4 | recipe | scripts intact. Documents the tile-**area** size trap (~37 MB @128 vs ~148 MB @256 for identical per-pixel quality). No board numbers survived — measured fields left null |
| [`span`](models/span/) — SPAN ×4 | recipe | scripts intact except calibration (reconstructed by analogy). The fidelity half of the keep-both super-res pair (~1/6 of Compact's size). Measured fields null |
| [`xfeat`](models/xfeat/) — sparse features | recipe | scripts intact, two numerically-checked graph rewrites (InstanceNorm → CPU preprocessing; `_unfold2d` → SpaceToDepth). Single-channel featuremap input. Measured fields null |
| [`yoloe`](models/yoloe/) — open-vocab det + seg | recipe | scripts intact. Head surgery freezes the open vocabulary (CLIP text embeddings folded into `cv3`) and emits raw heads; DFL/anchor/mask decode on the CPU. **AGPL-3.0 — recipe only, no redistributable binary.** Config reconstructed |
| [`vitpose`](models/vitpose/) — whole-body pose | recipe | 133-keypoint ViTPose-S. Export reconstructed (only prep + reference survived). Documents the ViT-on-BPU LayerNorm-precision concern (let the compiler auto-promote to int16). Measured fields null |
| [`yolo26`](models/yolo26/) — detection (main) | recipe | BCDL's main detector. Same DFL split-head cut as YOLOE det, minus the open-vocab fuse. Highest-risk reconstruction: the compile config was never saved, so `config.yaml` is reconstructed and flagged to verify. **AGPL-3.0 — recipe only, no binary** |
| [`ppocr_v5`](models/ppocr_v5/) — OCR det + rec + cls | recipe | kept for the **angle classifier v6 lacks** (v5 det/rec are a fallback). The export rot is the write-up: the surviving PaddleOCR script pointed at a `PP-OCRv4_server_seal_det` (wrong model *and* version) with the shape-fix commented out. Dictionary count asserted against the ONNX output width |
| [`face`](models/face/) — SCRFD det + ArcFace rec | recipe | both ONNX come from the insightface `buffalo_l` pack (locate + tiny edit, not a torch export). The `_aligned` calibration is the shipped build; int8/int16 recognition builds. **insightface weights non-commercial — recipe only** |
| [`edgesam`](models/edgesam/) — promptable seg | recipe | encoder + two fixed-prompt decoders (sp1 / bp2). The multi-input decoder calibration feeds real encoder embeddings + sampled prompts. Upstream terms unreviewed — recipe only |
| [`lingbot_depth`](models/lingbot_depth/) — RGB-D depth refinement | complete | compiled and **board-verified** (C 0.989, absrel 2.92%, mask IoU 0.959). The upstream graph is not compilable as written — it drops depth tokens by validity, so the sequence length depends on the data; the rewrite is **bit-identical** to upstream with that masking off. int8 collapses the output's dynamic range (233% absrel) and the per-layer report **predicted it**. **DDR-bound at 1457 ms**, with a measured token-budget/accuracy curve for trading it down |

Every model here is one BCDL has a decoder for and has run on the board. Not in
this repo, by design: the **download-only** entries in BCDL's catalogue (Depth-
Anything-V2, SigLIP, DeepLabV3+, YOLOv8, the YOLO26n zoo builds) — their "recipe"
is a URL, so they live in BCDL's `fetch_models.sh`, not here — and the
exploratory conversions with no BCDL decoder (dinov3, rf-detr, rt-detr, rtmpose,
mono3d), which would ship a recipe with no end-to-end path to validate it.

## Getting the compiled binaries

Recipes rebuild any model from its upstream checkpoint, and for most models that
is the only route offered here. Only the first row of the licence table above —
the models whose **weights**, not merely whose code, carry a licence that permits
redistribution — is eligible to be hosted as a compiled binary. Everything else
is **recipe-only** by design: build it yourself from weights you are licensed to
use. Each model's `expected.json` records its licence finding, including which
part of the chain (code, pretrained weights, or training data) the restriction
comes from.

## Licence

The recipes and scripts in this repo are MIT. **The models are not ours** — each
one carries its upstream licence, recorded in its model directory. Check it
before you redistribute or ship commercially.
