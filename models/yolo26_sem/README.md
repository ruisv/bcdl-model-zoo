# YOLO26n-sem — semantic segmentation, 640×640, Cityscapes 19-class

Upstream: [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO26n-sem, the
`semantic` task added alongside detect/seg/pose/obb/cls. Unlike YOLO26's other
heads (see [`../yolo26/`](../yolo26/)) this one needs **no forward-cut** at
export time — see `export.py`.

| build | BCDL consumes | note |
|---|---|---|
| `yolo26n_sem_nashm_640x640_nv12.hbm` (int16) | one dense-logit tensor `[1,19,640,640]`; BCDL argmaxes | AGPL — **recipe only, no binary** |

**Licence: AGPL-3.0.** Same terms as `../yolo26/`: recipe only, build your own
`.hbm` from your own licensed weights.

## The recipe: no cut needed, just get the quantisation right

Every other YOLO26 head in this repo needs its forward replaced before export,
because the stock head bakes NMS, anchor math or a DFL integral into the
graph. `SemanticSegment` does not: at inference it is a classifier conv fused
with a bilinear upsample back to input resolution — no boxes, no instances,
nothing to cut. `model.export(format="onnx")` already produces exactly what
BCDL wants. Verified op profile: Conv/Sigmoid/Mul backbone, one
Softmax/MatMul/Transpose attention block (the same neck stage YOLO26's
detector already runs on this BPU/march), three `Resize` nodes (two "nearest"
FPN upsamples identical to every other yolo26 head, one "linear" for the final
8× upsample — the same op `../pidnet/` already validates on this stack). Max
tensor rank 4, no Unfold/GridSample. **No operator rewrite.**

BCDL's existing `bcdl::Segmenter`/`decodeSeg` (used for DeepLabV3+ and
PIDNet-S) already consumes this exact shape — `[1,C,H,W]` logits,
channels-first, CPU argmax — so **no new C++ decode path was needed either**.
Wired into `tests/board_models.py` as `semseg_yolo26`, next to `semseg`
(DeepLabV3+) and `semseg_rt` (PIDNet-S): same three Cityscapes-19 heads, same
decoder, three different backbones.

## int8 vs int16 — read this before recompiling

**The all-int8 build passes every per-node and per-tensor cosine gate
(node_info.csv: every layer >0.95; output layer 0.9898 calibrated / 0.9867
quantised on a real held-out image) and is still wrong.** Measured on a real
street photo (not in the calibration set): the int8 build agrees with the
float model on only **61% of pixel-level argmax decisions**. This is not
random noise — whole classes swap wholesale (one class's pixel count went
2260→148849; another collapsed 131089→7577 in the same 640×640 frame).

This is the exact failure mode the project's quantisation rules name:
*"对回归/softmax/概率类数值敏感的输出，int8 位宽不足会系统性掉点"* — a dense
19-way softmax classifier is decision-boundary sensitive in a way a single
aggregate cosine number cannot see, because nearby-logit classes flip argmax
while the overall tensor stays "close" by cosine. Every other YOLO26 head in
this repo ends in a handful of numbers (box coordinates, class scores that get
thresholded) where this kind of noise washes out; this one's output *is* a
per-pixel decision, so it doesn't.

`optimization: set_all_nodes_int16` (see `config.yaml`) fixes most of it, and
switching the PTQ calibration set to the model's actual training domain fixes
almost all of the rest:

| build | calibration | cosine (bc vs float onnx, real image) | pixel argmax agreement | FPS / latency | DDR |
|---|---|---|---|---|---|
| int8 (`config_int8.yaml`, NOT shipped) | 22 nuScenes CAM_FRONT frames | 0.9867 | **61%** | 809 / 1.24ms | 37 MB |
| int16 | 22 nuScenes CAM_FRONT frames | 0.9948 | **85%** | 137 / 7.28ms | 183 MB |
| int16 (`config.yaml`, **shipped**) | 24 real Cityscapes val frames | 0.9956 | **96%** | 137 / 7.28ms | 183 MB |

int16 costs ~6× the latency and ~5× the DDR footprint of int8 — expected for
a full-int16 graph (see the project's mixed-precision rule: sparse int16
islands cost more in requant traffic than they save, so a small model like
this one goes all-int16 rather than mixed). 137 FPS / 7.3ms is still
comfortably real-time for a 640×640 segmentation task. Calibration domain
costs nothing at all — same compile settings, same latency/DDR, only the 24
input images changed.

**Two separate fixes, two separate causes.** YOLO26n-sem is pretrained on
Cityscapes (German urban street scenes, 19 classes). The recipe first
calibrated on 22 nuScenes CAM_FRONT frames (real street scenes, but a
different camera geometry/FOV and a different city/continent's visual
statistics) because no Cityscapes images were available on the convert host —
int16 alone got pixel agreement from 61%→85% under that calibration set. The
project's quantisation rules are explicit that domain match matters more than
sample count, and this model is the confirming case: downloading 24 real
Cityscapes validation frames (an ungated Parquet mirror on Hugging Face,
`Chris1/cityscapes` — the official site requires an account) and
recalibrating with the *same* int16 config took the same real held-out test
image from 85%→**96%** agreement, with identical latency/memory. The int8
failure mode (whole classes swapping) is a bit-width problem; the residual
85%→96% gap was purely a "calibrated on the wrong visual distribution"
problem — same lesson as PIDNet's `_v3` fix, different mechanism.

A qualitative check on a real Cityscapes photo (via BCDL's actual
`Segmenter`/`seg_colorize`, not just the raw tensor compare above) shows
coherent, plausible segmentation — road, sidewalk, vehicles, buildings/sky,
and a motorcyclist all cleanly separated — for both the nuScenes- and
Cityscapes-calibrated builds; the remaining ~4% pixel-level disagreement with
the float model concentrates at class boundaries rather than being random.

## Running it

```bash
# 1. checkpoint -> ONNX (no cut needed)
python export.py --weights yolo26n-sem.pt --output yolo26n_sem_bpu.onnx --imgsz 640

# 2. calibrate on REAL Cityscapes-domain frames -- see the int8-vs-int16-vs-
#    domain table above for why this is not a "nice to have". The official
#    site (cityscapes-dataset.com) needs a free account; an ungated Parquet
#    mirror also works (e.g. huggingface.co/datasets/Chris1/cityscapes,
#    columns `image`/`semantic_segmentation`, no login needed) -- just decode
#    the `image` column's bytes to files and point --images at that directory.
python calib.py --config config.yaml --images /path/to/cityscapes_frames --limit 30

# 3. compile (int16 -- see config.yaml; config_int8.yaml is kept only as the
#    documented cautionary example, not something to ship)
./compile.sh --config config.yaml --gpu 1
```

## Verification

Three-layer cosine (see [[accuracy-verification-method]]):
`verify_prep.py` (outside the OE docker, needs cv2) letterboxes a real image
and writes the NV12 + normalised-RGB feeds; `verify_gen_ref.py` (inside the OE
docker, needs `horizon_tc_ui`) runs the float ONNX and the quantised `.bc` on
that same input and prints layer A; `verify_cmp.py` (on the S100P board) feeds
the same NV12 planes through the compiled `.hbm` and prints layers B and C.

Measured (shipped int16 build, calibrated on real Cityscapes frames, real
held-out nuScenes CAM_FRONT frame -- a held-out image from a domain the model
was NOT calibrated on, deliberately, since that is the harder check):

| layer | meaning | measured |
|---|---|---|
| A — quantisation fidelity | `.bc` vs float onnx | cosine 0.9956 |
| B — execution consistency | board `.hbm` vs `.bc` | cosine 1.0, **bit-identical** |
| C — end-to-end | board `.hbm` vs float onnx | cosine 0.9956 |

B being exact says the board runtime agrees with the host simulation, so the
whole int8→int16→calibration-domain story above lives entirely in
quantisation (A/C), not plumbing.

## Status

**Freshly converted and board-validated in this recipe** (not reconstructed
from a lost build, unlike `../yolo26/`). Compiled, three-layer verified, and
run through BCDL's actual `Segmenter` on a real S100P board
(`tests/test_board_models.py::test_semantic_seg_cityscapes_yolo26`) producing
a coherent segmentation on a real Cityscapes photo. Both open items from the
first pass — int8 vs int16, and calibration domain — are closed: shipped
build is int16, calibrated on 24 real Cityscapes validation frames, 96%
pixel-level argmax agreement with the float model on a held-out image.
