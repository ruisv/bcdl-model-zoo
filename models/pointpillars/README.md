# PointPillars — lidar 3-D detection, and a 46× layout fix

A KITTI point cloud in, oriented 3-D boxes out (car / pedestrian / cyclist).
Upstream: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) PointPillars,
checkpoint `pointpillar_7728.pth`.

| build | BCDL consumes | on-board |
|---|---|---|
| `pointpillars_vfe_conv_int8_nashm.hbm` | pillar feature encoding | **1.62 ms** |
| `pointpillars_bbhead_int8_nashm.hbm` | BEV backbone + anchor head | **7.20 ms** |
| | **BPU total** | **8.82 ms** (published stages: 403.7 ms) |

Upstream code licence: Apache-2.0. **The weights are KITTI-trained and KITTI is
CC BY-NC-SA, so no binary is redistributed here** — this is a recipe, and BCDL
ships the runtime. See `expected.json` for the chain.

> D-Robotics publishes its own prebuilt PointPillars stages, and BCDL fetches
> those by default. This recipe exists because they are **46× slower on the BPU**
> for reasons that have nothing to do with the network.

## The layout trap, which is the whole point of this recipe

The published VFE spends **290 ms** on a 4→64 MLP of about **95 MFLOP**. That is
roughly **0.3 GFLOPS** on an 80+ TOPS part, and the ratio is the diagnosis: it is
not compute-bound, it is layout-bound.

Its input is `(11546, 32, 4)` — 11546 pillars × 32 points × 4 features. To a
convolution engine that is a huge spatial axis against a **4-wide channel axis**,
which is the shape it is worst at. Nothing about the arithmetic is expensive; the
tensor is simply the wrong shape to feed it.

Rewrite the same arithmetic as an image and it disappears:

| upstream | here |
|---|---|
| per-point linear 4→64 (after in-graph expansion to 10) | **1×1 convolution**, 10 → 64 channels |
| BatchNorm | folded into the convolution |
| max over the 32 points | **`(1, 32)` max-pool** |
| input `(pillars, points, 4)` | input `(1, 10, pillars, 32)` |

**1.62 ms. Same numbers, 179×.** The cost of the rewrite is that the 10-channel
per-point expansion — raw point, offset from the pillar's centroid, offset from
the pillar's centre — has to happen on the host. That is 1.1 ms of C++
(`bcdl::pillarPointFeatures()`); `calib.py` here has the numpy equivalent.

The lesson generalises past this model: **when a small graph is slow, check the
achieved FLOPS before touching the network.** Two orders of magnitude below the
part's rating means the tensor shape is wrong, not the model.

## The head cannot be compiled as published

Upstream's `AnchorHeadSingle` bakes the anchor decode into the graph with
`ScatterND`. That compiles **without a single warning** into a model whose output
columns are never written — the same silent failure YOLOP has in this repo, from
the same operator.

`export.py` cuts the head at its three convolutions (`cls` 18, `box` 42, `dir`
12 at 248×216) and leaves the decode on the host, where BCDL does it. Cut, the
head compiles at **cls cosine 1.000000 / box 0.999946**.

Once the head is three 1×1 convolutions it may as well ride along with the
backbone, so `export.py` emits them as one graph: one Engine, one round trip,
**113.7 ms → 7.20 ms**. (More than fusion alone explains — the published backbone
is a nash-e int16 build and this is nash-m int8.)

**The anchor grid was recovered, not written from the paper.** BCDL's
`generateAnchors3d` reproduces the published graph's own anchor constants to
**7.6e-06**, and that agreement is what pinned three conventions that produce
well-formed boxes in the wrong places if you guess them: a **closed** linspace
over the range (not cell centres), **class-major then rotation** ordering, and
anchor **z as the centre** rather than the base.

## int8 beats int16 here, which is not the usual direction

| build | cosine vs the float ONNX | latency |
|---|---|---|
| **int8 (shipped)** | **0.9988** | 1.62 ms |
| `set_all_nodes_int16` | 0.9567 | 1.36 ms |

The compiler's own per-layer report agrees (conv 0.9926 int16 vs 0.9999 int8).
Neither config carries an `optimization` directive for that reason. **Do not
"upgrade" this to int16 without re-measuring.**

## You cannot swap one stage

The published `.hbm` is **not this checkpoint**. Best-matching-channel median
correlation between the two VFEs is 0.904, but only 3 of 64 channels exceed 0.99
and the identity-order correlation is **0.000** — the channel ordering differs.
So all three published stages have to be replaced together, which is why the
backbone and head are re-exported here rather than reused.

Since the two chains are different weights, **a cosine between them means
nothing**, and the acceptance gate is the task's own unit: on the same sweep this
chain finds 14 boxes against the published chain's 16, **12 of them the same
class within 1 m**, and the six highest-scoring agree to under 0.1 m in position
and 0.06 rad in heading.

## One trap that is not in the graph: the output is stride-padded

The VFE's output is `[1, 64, 11546, 1]` and each channel is padded by **6 float
slots**. Read the device buffer flat and channel *c* is displaced by 6*c* — the
last channel by 378 pillars. That scores **0.70** against the float reference and
looks exactly like a quantisation failure. It is not: de-strided, the board
matches the compiler's own reported cosine bit for bit.

Worth stating because of how the diagnosis went: clipping accounted for 0.002 of
the gap, simulating the compiler's declared thresholds in numpy predicted 0.999,
and a linear probe showed the weights and bias were correct — three results all
saying "the artifact is fine" before anyone checked the reader. **When your
number and the compiler's disagree by two orders of magnitude, suspect your
reader before their reporter.**

BCDL's `Engine.infer()` gathers padded outputs since the fix; anything reading
`outputData()` directly must honour `outputStride()`.

## Running it

```bash
# [1/3] export both graphs from the checkpoint
python export.py --ckpt pointpillar_7728.pth

# [2/3] calibration, from real KITTI sweeps
python calib.py --config config_vfe.yaml    --velodyne /path/to/kitti/velodyne --limit 8
python calib.py --config config_bbhead.yaml --velodyne /path/to/kitti/velodyne \
                --vfe-onnx vfe_conv.onnx --limit 8

# [3/3] compile
./compile.sh --config config_vfe.yaml    --gpu 1
./compile.sh --config config_bbhead.yaml --gpu 1
```

The backbone's calibration set only exists downstream of the VFE, so `calib.py`
**runs the exported float `vfe_conv.onnx`** to build it. Calibrating it on
anything else — random data, or the published VFE's output — describes a
distribution the deployed stack never produces.

**Sample count is bounded by size here, not by taste**: one VFE sample is 15 MB
and one canvas sample is 53 MB, so 64 frames is 4.3 GB. Eight is the
recommendation; the shipped build used **one**, which is thin. Accuracy held
(0.9988), but if a rebuild degrades, raise this first.

## What is not here

KITTI mAP. The gate is agreement with the published chain plus every box sitting
on real returns (a standing check in BCDL as `tests/test_lidar_board_py.py`), not
a benchmark score.

And the remaining cost is no longer the model: of BCDL's 43 ms end to end, 8.8 ms
is BPU and about **17 ms is host-to-device copying**, 12 of that the 55 MB BEV
canvas. The next lever is scattering straight into the engine's input buffer, not
a smaller graph.
