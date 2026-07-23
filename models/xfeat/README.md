# XFeat — sparse local features, 640×480

A lightweight local-feature extractor for image matching: one shared trunk, three
output maps — a dense **64-d descriptor** field, a **keypoint-logit** map and a
**reliability heatmap**. Keypoint NMS, top-k selection and sparse descriptor
sampling are CPU post-processing that BCDL owns; the `.hbm` produces only the
three maps. Upstream: [accelerated_features](https://github.com/verlab/accelerated_features)
(Apache-2.0).

| build | BCDL consumes | on-board |
|---|---|---|
| `xfeat_nashm_640x480.hbm` | sparse local features (dense descriptor + keypoint + reliability heads) | not recorded — see Status |

Upstream licence: **Apache-2.0** (both the code and the shipped `xfeat.pt` weights
follow the accelerated_features repo terms — check before redistribution).

## The traps on this model

**1. The board input is not an image — it is a pre-standardised featuremap, and
that is deliberate.** XFeat starts with an `InstanceNorm2d`. That norm is a
*per-image, data-dependent* statistic (subtract this image's mean, divide by this
image's std), which is exactly the shape of thing PTQ quantises badly. So the
recipe **lifts the norm out of the graph** into CPU preprocessing: `export.py`
exports a model that takes an already grayscale + instance-normalised
`[1,1,480,640]` tensor, and `config.yaml` declares a single-channel `featuremap`
input with `norm_type: no_preprocess`. The board hands the BPU a standardised
tensor; the standardisation happens on the CPU first. Lifting it out also parks
the input in a fixed, standardised domain that quantises cleanly.

**2. Grayscale is a CHANNEL MEAN, not a luma weighting.** XFeat's front end does
`x.mean(dim=1)` — a plain average of R,G,B — not the usual `0.299R+0.587G+0.114B`.
`calib.py` reproduces exactly that, then applies the same `(x-mean)/sqrt(var+1e-5)`
(biased variance, InstanceNorm2d's default eps). Reach for `cv2.cvtColor(...GRAY)`
out of habit and the calibration domain drifts off what the runtime feeds — and,
as everywhere in this repo, the model still compiles clean, loads, and runs at
full speed. Preprocess for calibration exactly the way the CPU preprocesses at
deploy.

**3. `_unfold2d` doesn't lower — rewrite it, then prove it.** The keypoint head's
window gather (`_unfold2d`, window size 8) has no clean `torch.onnx` lowering. It
is replaced with `F.pixel_unshuffle(x, 8)`, which exports to a single
`SpaceToDepth`. For the single-channel input the two are the **same permutation**
(window offset `(dy,dx)` → channel `dy*8+dx`), and `export.py` asserts
`torch.equal(...)` before it exports anything rather than trusting the claim. This
is the same discipline LAS2 uses for its unfold rewrite: replace the operator the
exporter can't handle with an equivalent standard-primitive one, then verify
numerically. `export.py` also re-checks the whole rewritten module against the
original end-to-end (max |diff| < 1e-4 on all three heads).

**4. A plausible descriptor cosine is not proof the matches survive.** This is a
sparse-feature extractor, and — like OSNet's ReID embedding — a quantised
descriptor field can hold a decent-looking cosine while the actual keypoint set
and mutual-NN matches degrade. The honest end-to-end check is the **task metric**:
run the reference CPU decode (`xfeat_ref.py`: heatmap → NMS kernel 5 → sparse
interpolated scoring → top-k 4096 @ threshold 0.05 → L2-normalised descriptors →
mutual-NN match at min-cossim 0.82) on both the float ONNX and the board `.hbm`,
and compare kept keypoints and match count on a real image pair. Cosine on `feats`
alone is a localisation tool, not the gate.

## The ops profile

The interesting shape of this model is the three-headed output over a shared,
mostly-2D-conv trunk at 1/8 stride (60×80 maps from the 480×640 input):

- **`feats`** — `1×64×60×80`, the dense descriptor field, built by fusing three
  trunk scales (`x3 + interp(x4) + interp(x5)` → `block_fusion`). Two bilinear
  `Resize` nodes live here.
- **`heatmap`** — `1×1×60×80`, reliability, from `heatmap_head(feats)`.
- **`keypoints`** — `1×65×60×80`, from `keypoint_head(pixel_unshuffle(x,8))`: the
  8×8 `SpaceToDepth` of the input gives 64 sub-pixel cells and the head adds the
  dustbin channel.

No LayerNorm, no attention, no 3D conv, no GridSample — a plain CNN once the
InstanceNorm is lifted out, which is why it is a comfortable BPU target. (The full
op histogram is printed by `export.py` at export time; it was not captured in the
salvaged material, so it is not reproduced here as a fixed number.)

## Running it

```bash
# [1/3] upstream checkpoint -> BPU-friendly ONNX (lifts InstanceNorm, rewrites
#       unfold; both checked numerically). Then slim it.
python export.py --repo /path/to/accelerated_features \
                 --weights /path/to/accelerated_features/weights/xfeat.pt \
                 --hw 480 640 --out xfeat_640x480.onnx
onnxslim xfeat_640x480.onnx xfeat_640x480_slim.onnx

# [2/3] calibration. Texture-mixed set; preprocess must match the CPU deploy path
#       (channel-mean grayscale + instance norm), which calib.py does.
python calib.py --config config.yaml \
                --kitti /path/to/kitti/image_2 \
                --eth3d /path/to/eth3d \
                --coco  /path/to/coco/images --limit 100

# [3/3] compile
./compile.sh --gpu 2
```

`--gpu` is an **nvidia-smi index**, handed to docker as `--gpus device=N` (not
`CUDA_VISIBLE_DEVICES`, which numbers devices differently — see
[CONVERSION.md](../../CONVERSION.md#traps)). The card must be Ampere or newer.
`jobs` lives in the config, not on the command line. Target is `nash-m`
(S100/S100P).

## Status

Recipe **reconstructed from the original conversion scripts**; not re-run in this
repo.

- **export** — faithful to the salvaged `xfeat_export.py`: the two graph rewrites
  (InstanceNorm → CPU, `_unfold2d` → `pixel_unshuffle`/`SpaceToDepth`) and their
  in-script numerical checks are preserved verbatim. Parametrised with argparse
  and given an IR≤9 cap for the toolchain; those are the only additions.
- **calib** — faithful to `xfeat_calib.py` (channel-mean grayscale + instance
  norm, KITTI/ETH3D/COCO texture mix), rerouted through `common/calib_pack.py` so
  it writes through the one calibration path and drops a manifest. The specific
  dataset paths/counts in the salvage were machine-local and are now CLI args.
- **config** — the salvaged `config.yaml`, with `cal_data_dir` moved to the repo's
  `/ws/…` convention. No `optimization` directive was set upstream, so the
  compiler picks bit width per layer (predominantly int8); this was **not**
  re-verified, and no int8-vs-int16 comparison was recorded.
- **compile** — the standard container wrapper (LAS2's, one build).

**Not recorded anywhere in the salvage:** on-board latency / FPS, the `.hbm` size,
any A/B/C cosine, and the task-metric acceptance numbers. `expected.json` leaves
all of these `null` rather than inventing them. What is genuine is the *method* —
the two verified rewrites and the deploy-matched preprocessing. To close this out,
build it, run the reference decode from `xfeat_ref.py`, and record the match count
and per-output cosines on an S100P.
