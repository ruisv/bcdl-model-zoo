# NeuFlow v2 — dense optical flow, 512×384

Two frames in, a per-pixel displacement field out. Upstream:
[NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2), checkpoint
`neuflow_mixed.pth`.

| build | BCDL consumes | on-board |
|---|---|---|
| `neuflow_v2_int16_nashm_512x384.hbm` | dense flow field | **19.45 ms / 51 FPS** (26.4 ms end to end) |

> [!IMPORTANT]
> **Recipe only — no binary.** The code is Apache-2.0 *and* the weights are
> committed in the same repository, which normally settles it. The training data
> does not: FlyingThings3D is research-only, and its terms say **nothing** about
> models trained on it. Silence is not permission. See [Licence](#licence).

## The lesson: int8 compiles clean and is unusable

This is the model that pins down when a cosine gate is worthless.

| build | static-pair noise floor | EPE on known 2–16 px shifts | BPU |
|---|---|---|---|
| float ONNX | 0.025 px | 0.036–0.076 px | — |
| **int8** | **1.936 px** | 1.61–2.15 px | 14.4 ms |
| **int16 (shipped)** | **0.070 px** | 0.055–0.081 px | 19.4 ms |

The int8 error is a **flat noise floor**: a 2 px displacement and a 16 px
displacement come back with the same ~1.9 px error. So fast pixels look fine —
12% relative error — and slow pixels are entirely noise. On a real frame pair
whose true displacement is 0.05 px, the int8 output's cosine against float is
**0.02**.

**The compiler reported 0.9954 output cosine for that build.** A field with a
60 px dynamic range absorbs 2 px of noise without moving a cosine much. int16
costs 34% latency and gets essentially float accuracy back.

**A head that regresses a physical quantity has to be scored in that quantity's
units.** Cosine cannot see this failure, and neither can `node_info.csv`.

## Ground truth without a dataset

Flow has a property most tasks lack: **exact ground truth can be manufactured.**
Translate an image by a known integer number of pixels and the correct field is
that constant, everywhere. Drop a 48 px border so the wrap-around does not
pollute the measurement, and you have a real error number with no dataset and no
labels.

The complementary test: feed the **same frame twice**. The true answer is zero,
so whatever comes back is the noise floor, measured directly.

Both numbers are in the table above, and both are why the int8 build was caught.

## Exporting it: three things to copy from `infer.py`

Miss any of these and you still get a graph — a wrong one.

1. **Fuse Conv+BN before tracing.** The released checkpoints are *not* fused;
   upstream fuses at load time. An unfused export is not the network anybody
   measured.
2. **Call `init_bhwd()` before tracing.** It precomputes the correlation
   lookup's `linspace`/`meshgrid` into constants for one fixed resolution.
   Without it they trace as live `Range`/`Shape` ops and the graph carries
   dynamic shapes.
3. **Export in fp32.** The demo runs half precision; tracing that path exports a
   half graph the toolchain will not take.

The iteration counts are export-time constants (1 at 1/16 scale, 8 at 1/8), so
the refinement loops unroll completely — **the graph has no control flow at
all**. After `onnxslim`: 890 nodes, opset 17, nothing dynamic left.

The only red-flag operator is **`GridSample` ×9** (the correlation lookup), and
it is natively supported on this BPU. The 4 `LayerNorm`s belong to the 1/16-scale
cross-attention and are where bit width matters — which the all-int16 build
settles.

## Calibration: use 12 Hz frames, not 2 Hz keyframes

The shipped build calibrated on **nuScenes sweeps** (12 Hz), 30 logs × 2 pairs =
60 pairs. Keyframes are 2 Hz — half a second apart, by which point the car has
moved several metres, which is **not the displacement range this model is asked
about**. Calibrating there sizes the activation thresholds for a distribution
the runtime never produces.

The tensors are **BGR, 0–255, not normalised**: the `/255` is inside the graph,
which is why the config says `no_preprocess`. Two inputs means two directories
matched positionally — a count or ordering mismatch calibrates the model on
pairs that are not pairs, silently.

## Running it

```bash
python export.py --ckpt neuflow_mixed.pth --repo /path/to/NeuFlow_v2
onnxslim neuflow_512x384.onnx neuflow_512x384_slim.onnx
python calib.py --config config.yaml --sweeps /path/to/frames --limit 60
./compile.sh --gpu 1
```

## What is not here

The `sintel` and `things` checkpoints were not tried — the shipped build uses
`mixed`. No resolution other than 512×384 was compiled; hbm size and latency
scale with input **area**, so a 768×432 build is a separate compile.

Two measurement notes worth carrying to the next model: do **not** time a loop
that reads from a compressed `.npz` (the first attempt measured 87.5 ms end to
end, 66 ms of which was re-decompressing images every iteration — the real
number is 26.4 ms), and 6.3 ms of that end-to-end time is preprocessing two
frames with a hand-written bilinear resize, which OpenCV or NEON would cut.

## Licence

**Apache-2.0** on the code, and the pretrained weights are committed inside that
Apache-2.0 repository — which is normally the cleanest case there is.

**The data chain is what makes this recipe-only.** `neuflow_mixed.pth` is
trained on FlyingThings3D and Sintel. FlyingThings3D's terms of use read, in
full:

> This dataset is provided for research purposes only and without any warranty.
> Any commercial use is prohibited.

Those terms address **the dataset**. They say nothing about models trained on
it — they neither extend the restriction to derivatives nor grant permission for
them. This repo's rule is that **silence is not permission**, so this sits where
OSNet sits: recipe public, binary not. (Compare PIDNet, which *is* redistributed
non-commercially, because Cityscapes explicitly permits distributing models
trained on it while barring commercial use of them.)

Sintel is a second open question: some sources report CC BY 3.0, others
CC BY-NC-SA 3.0.

If you need a flow model you can redistribute, the architecture is Apache-2.0 —
retrain it on data you can account for. That is exactly the part this recipe
makes reproducible.
