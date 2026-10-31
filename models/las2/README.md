# Lite Any Stereo V2 (LAS2) — stereo disparity, 480×640

Zero-shot stereo matching, fully on the BPU, int16, single `.hbm`.
Upstream: [LiteAnyStereo](https://github.com/TomTomTommi/LiteAnyStereo).

| build | config | BCDL consumes | on-board |
|---|---|---|---|
| `las2_m_crop2_nashm.hbm` | `config.yaml` | stereo disparity (centre-crop letterboxing) | 13.9 ms / 72 FPS |
| `las2_m_resize2_nashm.hbm` | `config_resize.yaml` | stereo disparity (resize letterboxing) | 13.9 ms / 72 FPS |
| `las2_l_crop2_nashm.hbm` | `config_l.yaml` | the same, L size — more accurate up close | 22.9 ms / 44 FPS |
| `las2_l_resize2_nashm.hbm` | `config_l_resize.yaml` | the same, L size, resize | 22.9 ms / 44 FPS |

The `2` is the calibration range — see trap 4. The first two builds,
`las2_m_crop_nashm` and `las2_m_int16_nashm`, clip disparity and are retired.

Upstream licence: not yet reviewed — check LiteAnyStereo's terms before
redistribution.

## Which build is wrong for you

Within a size, `_crop2` and `_resize2` are **not** better/worse. They differ
only in how the calibration pairs were letterboxed, and **using the wrong one
costs accuracy silently** — the model loads, runs at full speed, and returns a
plausible disparity map:

- `_crop2` — calibrated on **centre-cropped** pairs (`calib.py --mode crop`)
- `_resize2` — calibrated on **resized** pairs (`calib.py --mode resize`)

Match it to how your runtime letterboxes the input pair. BCDL's stereo test
feeds a centre-cropped pair and therefore prefers `_crop2`.

### M or L

L has twice M's aggregation depth at every scale; everything else (2D-only
cost aggregation, no iteration, all int16) is the same. On the board, against
ground truth on held-out ETH3D and Middlebury 2021 scenes (33 for crop, 51 for
resize — different sets, so compare within a mode):

| build | EPE | bad 1 px | bad 2 px | BPU latency | `.hbm` |
|---|---|---|---|---|---|
| M crop2 | 0.998 px | 14.2% | 6.0% | 13.9 ms / 72 FPS | 40 MB |
| L crop2 | 0.928 px | 11.0% | 5.0% | 22.9 ms / 44 FPS | 71 MB |
| M resize2 | 0.597 px | 8.5% | 3.4% | 13.9 ms / 72 FPS | 41 MB |
| L resize2 | 0.568 px | 6.7% | 2.9% | 22.9 ms / 44 FPS | 71 MB |

The gain is all close up: on Middlebury, crop bad-1px goes 19.2% → 14.5%
(resize 14.4% → 10.7%); on ETH3D, where disparities are small, the two are
level. Error against each build's own float model is about 0.3 px for all
four, so L's lead is the network, not the quantisation.

## The four traps on this model

**1. `int8` is not an option, and mixed precision is a trap.**
Disparity is `soft-argmax`: `Σ softmax(cost_volume) · disparity_index`. Small
quantisation errors in the softmax distribution do not cancel — they accumulate
into a **systematic disparity offset**. Measured on identical calibration data:

| quantisation | EPE | on-board FPS | |
|---|---|---|---|
| int8 (all) | 1.23 px | 174 | ~1 px systematic offset + speckle |
| mixed (cost volume + head int16) | 0.25 px | **12** ⚠️ | requant at scattered int8↔int16 boundaries dominates — *slowest of the three* |
| **int16 (all)** | **0.12 px** | 71 | uniform dataflow, no conversions ✅ |

These EPEs are against the float model on a ZED pair that was **inside** the
calibration set, so they compare the three options fairly but flatter all of
them. On held-out scenes the int16 build's error against the float model is
~0.2–0.3 px — see trap 4.

The mixed-precision result is the counter-intuitive one and worth internalising
beyond this model: **unless int16 layers coalesce into large contiguous blocks,
the boundary requant cost exceeds what you save.** Reaching for mixed precision
because "only the sensitive layers need int16" made this model 6× slower than
just quantising everything to int16.

**2. `torch.onnx` cannot export `aten::unfold`** (fails with *input size not
accessible*). LAS2 uses it twice — building the correlation volume, and the
context upsample. `shifts.py` rewrites both into pad+slice over standard
primitives, numerically equivalent to **1e-6 px EPE**. This is the general
technique for putting an unusual model on the BPU: rewrite the unsupported
operator as equivalent standard tensor primitives rather than giving up on the
model.

**3. The model has `@torch.compile` on internal functions.** Without triton
installed the export dies, so `export.py` forces `TORCHDYNAMO_DISABLE=1`. Pure
eager — it does not change the exported graph.

**4. The calibration set caps the disparity.** The output's int16 threshold
is the largest disparity the calibration pairs contain, and everything above it
is clipped — no error, no warning, a plausible map whose near field reads as
farther away. The first builds were calibrated on ZED scenes with no near
field: the crop build clipped at **131 px** and the resize build at **47.7 px**,
out of the model's 0–192. Measured on held-out scenes with ground truth (ETH3D
and Middlebury 2021, none in the calibration set), the worst crop scene was
14.7 px off the float model and the worst resize scene 33 px; 20 of 51 resize
scenes were past 0.5 px.

The fix stays in-domain: `calib.py --extend-range` adds a copy of each pair
with the right view shifted left by *k* px, which raises every disparity by
exactly *k* and changes nothing else, with *k* chosen from the float model so
each copy tops out near 185 px. `--pool` adds a few real large-disparity pairs
on top (the shipped builds used six from MiddEval3, which is disjoint from the
evaluation sets). After the fix the worst crop scene is 0.30–0.43 px; scenes
that were inside the old range pay about 0.07 px for the wider one.

Range in the data is not enough on its own, because the toolchain reads only
one calibration sample in eight (see `common/calib_pack.py`). The first resize
rebuild had full-range data and still clipped at 47.7 px: of its 16 pairs, only
pairs 0 and 8 — both originals, no near field — set the thresholds. The crop
rebuild only escaped because pair 32 happened to be a large-disparity one.
`calib_pack.py` now writes every pair eight times, and the configs use plain
`max` calibration (`max_percentile: 1.0`), which int16 has the resolution for.

Why nothing caught it: BCDL's own stereo pair spans 44–129 px — two pixels
under the crop cap. Its test now shifts that pair up to ~175 px and checks the
output follows (`test_full_disparity_range`).

## Why this model at all

Picked over FoundationStereo, which compiles fully to BPU but runs a 3D cost
volume at **~40 s/frame** — not a BPU-suitable architecture. LAS2 is
BPU-friendly by construction: FasterNet 2D CNN backbone (not ViT), cost
aggregation **100% 2D** (disparity folded into channels, no 3D conv),
feed-forward with no GRU iteration, and BatchNorm rather than LayerNorm (which
avoids the int16 LayerNorm question entirely).

Operator profile at M/480×640: 2259 nodes, all 2D conv, max rank 4 (no 5D), one
nearest-neighbour Resize, and no unfold / 3D-conv / LayerNorm / GridSample.

## Running it

```bash
# 1. upstream checkpoint -> BPU-friendly ONNX (~6s)
python export.py --repo /path/to/LiteAnyStereo --size m --hw 480 640 \
                 --out las2_m_480x640.onnx

# 2. calibration. The config selects the build; --mode must agree with it.
#    --extend-range needs the float model (--repo) to measure each pair.
python calib.py --config config.yaml --mode crop \
                --sbs-dir '/path/to/captures/Explorer_HD2K*.png' \
                --extend-range --repo /path/to/LiteAnyStereo \
                --pool /path/to/MiddEval3/trainingH /path/to/MiddEval3/testH

# 3. compile (config.yaml = crop, config_resize.yaml = resize)
./compile.sh --config config.yaml --gpu 2

# resize: resizing shrinks the pool's disparities too, so lower the bar —
# the shipped resize builds used --pool-min 60.
python calib.py --config config_resize.yaml --mode resize ... --pool-min 60

# L size: export with --size l, and use config_l.yaml / config_l_resize.yaml.
# calib.py also takes --size l, so the shifts are sized by the model you ship.
```

`--sbs-dir` takes a **glob**, not just a directory: capture folders usually hold
more than the stereo set, and the extras are both out-of-domain (RULE 2) and
often too small to crop.

`--gpu` is an **nvidia-smi index**, handed to docker as `--gpus device=N`. It is
deliberately not `CUDA_VISIBLE_DEVICES`, which numbers devices differently — see
[CONVERSION.md](../../CONVERSION.md#traps). The card must be Ampere or newer.

Both configs target `nash-m` (S100/S100P). `jobs` lives in the config, not on
the command line.

## Status

Re-run end to end on OpenExplorer 3.7.0 (HBRT 4.7.5), verified at each step:

- **export** reproduces the original graph — 2259 nodes, identical op histogram
  and I/O signature, all 278 initializers matching to 7e-7. Not byte-identical
  (ONNX metadata differs), which is expected.
- **calibration** written through `common/calib_pack.py`.
- **layer A** (quantisation fidelity) measured **cosine 0.999954** on `disp`,
  against the 0.99 gate.
- **compile** produced a 38.8 MB `.hbm` in ~39 min on one RTX-class GPU.

### The `2` builds

Rebuilt with `--extend-range`, the MiddEval3 pool, 8× repetition and plain
`max` calibration. The thresholds that matter, read from each build's
`*_quant_info.json`:

| build | calibration samples | upsample `Mul` | output |
|---|---|---|---|
| M crop2 | 36 pairs × 8 | 188.0 | 182.9 |
| L crop2 | 36 pairs × 8 | 188.0 | 187.1 |
| M resize2 | 16 pairs × 8 | 188.0 | 181.2 |
| L resize2 | 16 pairs × 8 | 187.9 | 180.5 |

The retired builds had 131 (crop) and 47.7 (resize) on the output. Against
the retired resize build on the same 51 held-out scenes, EPE went 2.17 → 0.60
px and Middlebury bad-3px 17.3% → 3.6%. The crop rebuild with 8× repetition
matched the one without to 0.01 px — it had escaped the one-in-eight problem
by luck (trap 4).

These still start from the uncommitted capture set below, so they are not
reproducible from public data alone; their manifests record every source
file, shift and pool pair, so at least the inputs are now provable.

### The first rebuild was not the shipped one, and could not be

40,658,816 bytes against the shipped model's 40,700,800 — a 0.1% difference, and
they are not interchangeable byte-for-byte. The cause is calibration data: the
shipped crop build used **15 stereo pairs from a capture set that was never
committed anywhere**, and the only pairs available from the upstream repository
are the **5** in its `assets/`. Different calibration set, different activation
thresholds, different instruction stream.

This is the failure mode the manifest mechanism exists to prevent going forward:
`cal_crop/*.manifest.json` records each source file and its hash, so a future
rebuild can at least *prove* whether it used the same inputs. It cannot help
retroactively here, because the original set was never recorded.

Practically: this recipe produces a **working, verified** LAS2 model, not a
reproduction of the exact binary in BCDL's `models/`. If you need the shipped
model's numbers specifically, use the shipped model.

### Board close-out

Both board layers were measured on an S100P (HBRT 4.7.5), on this rebuild:

- **layer B** (`.hbm` vs host `.bc`): **bit-identical** — cosine 1.0, max abs
  error 0. The board runtime and the host simulation agree exactly, so the
  plumbing (input stride, layout) is correct.
- **layer C** (`.hbm` vs ONNX, real stereo pair): **cosine 0.999814**. The
  largest errors are in occlusion and border pixels; the disparity field is
  otherwise sub-pixel.

This is the repo's first full three-layer close-out. The EPE, latency and FPS
figures elsewhere in `expected.json` are from the original adaptation work on
the **shipped** build, not this rebuild.
