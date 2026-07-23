# ViTPose-S whole-body — 133-keypoint pose, 256×192

Top-down whole-body 2D pose: one person crop in, 133 keypoint heatmaps out
(17 body + 6 foot + 68 face + 42 hand, COCO-WholeBody). A ViT-Small backbone plus
a lightweight heatmap decoder. Upstream export path:
[easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) (Apache-2.0). The
pretrained wholebody weights follow their own terms — check before redistribution.

| build | BCDL consumes | on-board |
|---|---|---|
| `vitpose_s_wholebody_nashm_256x192.hbm` | 133×64×48 heatmaps for a person crop | not recorded (see Status) |

Input is `featuremap` + `no_preprocess`: the crop is padded to 3:4, resized to
192×256, `/255` and ImageNet z-scored **outside** the graph (in `calib.py` and in
the on-board preprocessor), and the ONNX takes the already-normalised tensor.

## The traps on this model

**1. It is a *top-down* model — feeding it a whole frame is the silent wrong
input.** ViTPose does not find people; it assumes it has already been handed a
single person cropped and letterboxed to 3:4. Run a detector first, widen each box
(`PAD_BBOX=10` px/side), pad to aspect, resize to 192×256. Get the crop geometry
wrong and the heatmaps are still well-formed and plausible — the peaks just land in
the wrong place. The calibration set has to be *person crops through that same
preprocessing*, not scene images; calibrating on whole frames trains the
quantisation on a distribution the model never sees at runtime.

**2. This is the repo's first ViT, and LayerNorm is the quantisation-sensitive
operator on the Nash BPU.** int8 LayerNorm is where a transformer loses accuracy
here. The right move is *not* to reach for a knob: with no `optimization`
directive the compiler already promotes the LayerNorm-adjacent layers to int16 on
its own (the exact behaviour this repo saw on the PP-OCRv6 recogniser, where the
auto-chosen int16 nodes were the LayerNorm-adjacent ones). So `config.yaml`
deliberately forces **no** global bit width. Read `Output Data Type` in
`<prefix>_node_info.csv` to see the split it picked, and only hand-write a mixed
config to *override* it — never to discover it. And do not reflexively apply
`set_all_nodes_int16`: on LAS2 a global int16 was right, on the PP-OCRv6
recogniser the default mix was faster; measure before deciding.

**3. Heatmap decode is not in the graph, and must not be compared through.** The
`.hbm` emits raw 64×48 heatmaps. Argmax / soft-argmax to sub-pixel coordinates and
the crop→image un-mapping happen on the **CPU downstream**. Keep them out of the
board-vs-float comparison: compare the heatmap *tensors*, not decoded keypoints.
Otherwise a preprocessing difference and a quantisation difference become
indistinguishable in the result (this is exactly why the salvaged reference runner
saves inputs + heatmaps and stops there).

**4. `export.py` is reconstructed — treat the output shape as unconfirmed.** The
64×48 heatmap size is the standard ViTPose resolution at 256×192 (input/4), but it
was not pinned down by anything that survived. `export.py` asserts the 133-channel
count (which *is* confirmed) and prints a NOTE if the spatial size differs; if it
does, fix `expected.json`'s `output_shape` before compiling.

## Running it

All three steps run on the convert host; the compile step runs inside the
OpenExplorer GPU container via `compile.sh` (`--gpu` is an nvidia-smi index, card
must be Ampere or newer).

```bash
# [1/3] checkpoint -> static, slimmed, IR9 ONNX at 256x192 (RECONSTRUCTED export)
python export.py --repo /path/to/easy_ViTPose \
                 --cfg  /path/to/easy_ViTPose/configs/ViTPose_small_wholebody_256x192.py \
                 --ckpt /path/to/vitpose-s-wholebody.pth \
                 --out  vitpose_s_wholebody_static.onnx

# [2/3] person-crop calibration set (domain-matched; through common/calib_pack.py)
python calib.py --config config.yaml --images /path/to/person_images \
                --detector /path/to/yolo_person.pt --limit 64

# [3/3] compile -> out/vitpose_s_wholebody_nashm_256x192.hbm
./compile.sh --config config.yaml --gpu 2
```

## Why this model

ViTPose is the strong, simple recipe for 2D pose: a plain ViT backbone with a
minimal decoder, no task-specific necks. The Small variant keeps it cheap enough
to run per detected person, and the whole-body head (133 keypoints) covers face and
hands as well as the body — the thing a body-only COCO-17 pose model cannot do.

## Status

**Recipe partially reconstructed; not re-run in this repo.**

- **`export.py` is RECONSTRUCTED.** The original checkpoint→ONNX export did not
  survive — only a downstream static-shape prep and a float reference runner did.
  `export.py` rebuilds the missing first step from the salvaged I/O contract plus
  the standard easy_ViTPose export path. Faithful to the method, not copied from
  the lost original.
- **`calib.py` follows the salvaged preprocessing** (`vitpose_prep.py`), re-routed
  through `common/calib_pack.py` so there is one calibration writer.
- **`config.yaml` is from salvage**, with `cal_data_dir` normalised to the `/ws`
  mount used by `compile.sh`.
- **No measured numbers survived.** Latency, FPS, cosine and any task metric are
  `null` in `expected.json` with a note, not fabricated. The shipped
  `vitpose_s_wholebody_nashm_256x192.hbm` exists on the board, but its measured
  figures were not recorded here. A rebuild should run the three cosine layers and
  a COCO-WholeBody AP check, comparing heatmap tensors rather than decoded points.
