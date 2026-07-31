# LingBot-Depth — RGB-D depth refinement, 480×640

Takes a depth map you already have — a stereo or ToF reading, noisy and full of
holes — plus the aligned RGB frame, and returns hole-filled metric depth with a
per-pixel trust mask. It **refines rather than estimates**, so it composes with
LAS2 or a depth camera instead of competing with them.
Upstream: [lingbot-depth](https://github.com/Robbyant/lingbot-depth) (MDM v2,
DINOv2 ViT-L/14 RGB-D encoder + conv neck and two heads, 321M parameters).

| build | BCDL consumes | on-board |
|---|---|---|
| `lingbot_depth_v05_int16_nashm.hbm` | RGB-D depth refinement | **1457 ms / 0.69 FPS** |

Upstream licence: Apache-2.0.

> **This is a seconds-per-frame model.** It is an order of magnitude slower than
> anything else in this repo. Use it to refine a keyframe on demand — a grasp
> pose, a mapping snapshot — not as a stage in a video loop. The reason is
> structural and is explained under trap 3.

## The four traps on this model

### 1. The upstream graph cannot be compiled as written — the sequence length depends on the data

`vision_transformer.depth_masking` drops every depth token whose 14×14 patch
holds no valid reading, so the transformer's sequence length is a function of
the input. No static graph can express that.

`export.py` rebuilds token preparation without the masking, which makes the
sequence exactly `1 + N_img + N_depth`. That is what `enable_depth_mask=False`
does in the upstream forward, and the export verifies itself against upstream's
own path: **max|Δ| = 0.0, bit-identical**, when the RGB resize antialiases.

The masking itself is not free to drop. Measured across the eight bundled
reference scenes, keeping every token costs:

| | masking on (upstream) | masking off (deployed) |
|---|---|---|
| mean absolute relative error | — | **0.06%** |
| worst-scene RMSE | — | 4.8 cm |
| mask IoU | — | 0.9999 |

Four of the eight scenes come out **bit-identical**, because no patch in them is
entirely invalid and the masking therefore dropped nothing.

**The caveat that matters:** those scenes are 87–100% valid depth. Sparse input —
a lidar sweep, a few hundred feature-point depths — was *not* measured, and is
exactly the case where dropped tokens carry information. Note also that upstream
feeds invalid pixels as log-depth `0`, which collides with a genuine 1 m reading;
with masking on, the token was dropped instead of being ambiguous.

If sparse input turns out to matter, the fix is already scoped and needs no
retraining: `dinov2_rgbd/layers/attention.py` already routes through
`F.scaled_dot_product_attention(q, k, v, attn_bias)`. A **per-key** bias of shape
`[1,1,1,S]` holding about `-30` on the dead tokens is mathematically identical to
removing them, because an attention output depends on which *keys* exist, not on
which queries do — and it costs `S` extra values rather than the `S²` an ordinary
attention mask would. Use `-30`, not `-inf` or `-1e4`: the score matrix is
quantised, and a huge negative constant would wreck its range.

### 2. int8 PTQ is a total loss, and the per-layer report says so in advance

A 24-layer ViT-L is the case the all-int16 rule exists for.

| quantisation | absrel | cosine | output range | mask IoU | on-board |
|---|---|---|---|---|---|
| int8 (all) | **233%** | 0.526 | 2.9–13.8 m ⚠️ | 0.558 | 255 ms |
| **int16 (all)** | **2.92%** | **0.989** | 0.96–46.05 m ✅ | 0.959 | 1457 ms |

The float model's range is 0.97–45.79 m. int8 does not merely add noise — the
**dynamic range collapses**, and it still compiles without a single warning.

Unusually, `*_quant_info.json` predicted this before any board time was spent:
LayerNorm and softmax `Mul` nodes came back at **0.006** cosine, and 1056 of 1418
nodes sat below 0.90. Contrast that with LAS2, where two thirds of layers sit
below 0.99 and the model output is still 0.99995 — per-layer cosine is normally a
localisation tool rather than a gate, but a whole *class* of node collapsing to
~0 is a different signal from a broad mild sag. Read it that way: look at the
distribution by operator type, not at the median.

### 3. It is DDR-bound, not compute-bound, and that is why int16 costs 5.7×

int16 is usually a ~2× tax. Here it is 255 ms → 1457 ms. The perf report says
where it goes: the attention matmuls **compute for 4.3 ms inside a 21.9 ms
slot**, and the model moves **23 GB of DDR per frame** at int16 against 4.2 GB at
int8.

The cause is the attention score matrix. Image tokens and depth tokens are
**concatenated**, so the sequence is `1 + 2N`; at N = 1200 that is 2401, and
`16 heads × 2401 × 2401` at int16 is 184 MB — materialised to DDR, twice per
block, 24 blocks deep.

Two consequences:

- **hbdk's own latency estimate is not trustworthy for this graph.** It predicted
  680 ms against 1457 ms measured (2.1× optimistic), while on the int8 build it
  was within 9%. The estimate models compute better than it models DDR.
- **The token budget is the only real latency lever**, and it is quadratic.

### 4. The token budget is a free-ish speed knob, because quantisation error already dominates

The network has no fixed resolution: a token count plus the aspect ratio sets the
encoder grid. Upstream's range is `[1200, 3600]`. Measured in fp32 against a
3600-token reference on the bundled scenes:

| tokens | grid | encoder input | sequence | attention cost | absrel | median | δ<1.05 |
|---|---|---|---|---|---|---|---|
| **1200** (shipped) | 30×40 | 420×560 | 2401 | 0.112× | 1.20% | 0.18% | 95.2% |
| 768 | 24×32 | 336×448 | 1537 | 0.046× | 1.75% | 0.29% | 91.7% |
| 432 | 18×24 | 252×336 | 865 | 0.015× | 2.56% | 0.46% | 88.7% |

int16 quantisation alone costs 2.92%. **Every row in that table is cheaper than
the quantisation you are already paying for**, so if 0.69 FPS is too slow, drop
the token count before reaching for anything cleverer. Pass `--tokens 768` to
`export.py` and re-run the recipe; nothing else changes.

1200 was shipped first because it is the bottom of the trained range, and because
its decoder pyramid lands exactly on 480×640 (30×40 → ×16), which makes the
graph's final resize the identity.

## Host-side preprocessing is part of the model

Everything that is not a static tensor op was pushed out of the graph, so the
`.hbm` takes two clean F32 featuremaps and the host owns the rest:

```
host  ->  RGB   : area-average resize to 420×560, BGR->RGB, /255, ImageNet mean/std
          depth : nearest resize to 420×560, zero invalid readings, natural log
BPU   ->  patch embed -> 24 blocks -> conv neck -> depth head + mask head
host  ->  threshold the mask logits, optionally unproject to a point cloud
```

Two details are load-bearing:

- **The RGB downscale must be antialiased.** The reference implementation uses
  torch's antialiased bilinear. An area average (what BCDL's C++ does, so the
  task layer needs no OpenCV) costs **6.1e-4** mean absolute relative error;
  plain bilinear costs **9.1e-4**. Both are small, but plain bilinear is the
  worse half of a free choice.
- **Depth enters as log depth with 0 meaning "no reading"** and the graph output
  is already metric — this checkpoint sets `remap_depth_out: linear`, so there is
  no exponential to apply. Keeping the log remap on the host also puts the
  quantiser on a well-behaved range.

`calib.py` is the single definition of that preprocessing; BCDL's
`src/tasks/depth_refine.cc` mirrors it, and BCDL's test suite asserts the two
agree to within cv2's uint8 rounding.

## Build it

```bash
# 1/3  export a static-shape ONNX (verifies itself against upstream)
python export.py --repo /path/to/lingbot-depth --ckpt model_v05.pt \
                 --tokens 1200 --out lingbot_depth_v05_480x640.onnx
onnxslim lingbot_depth_v05_480x640.onnx lingbot_depth_v05_480x640_slim.onnx

# 2/3  calibration data + the fp32 golden, in deployment preprocessing
python calib.py --config config.yaml --scenes /path/to/lingbot-depth/examples \
                --flip --golden --repo /path/to/lingbot-depth --ckpt model_v05.pt

# 3/3  compile (about 2 h of codegen, longer on a contended host)
./compile.sh --gpu 0
```

Then verify on the board with `common/verify_cosine.py` and the `val_*.npy`
files `calib.py --golden` leaves next to the calibration set.

One model-specific wrinkle: **calibration falls back from batch 8 to batch 1**
on this graph, because the traced reshapes carry a literal batch of 1. hb_compile
reports it and recovers on its own. Harmless at 15 samples (~3 min); if you ever
calibrate on hundreds, make those reshapes batch-agnostic first.

## Calibration domain

The shipped build was calibrated on the **eight reference scenes upstream
bundles**, plus horizontal mirrors — 15 samples of real indoor sensor depth at
640×480, one scene held out as the golden. That is domain-matched to the model's
own examples, not to any particular camera.

If you are deploying against a specific depth source — a stereo rig through LAS2,
a particular ToF module — **recalibrate on that source**. It is a ten-minute job
and it is the highest-value change available: this repo's own measurements
elsewhere (LAS2, PIDNet) show domain match beating sample count every time.
