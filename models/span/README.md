# SPAN x4 (ch48) — single-image super-resolution, 128×128 tile

A ×4 upscaler run tile-by-tile on the BPU. In, a 128×128 RGB tile in `[0,1]`;
out, its 512×512 super-resolved version. The runtime tiles a full frame, runs
each tile through this `.hbm`, and blends the overlaps.
Upstream: [SPAN](https://github.com/hongyuanyu/SPAN) (`spanx4_ch48`), Apache-2.0.

| build | BCDL consumes | on-board |
|---|---|---|
| `spanx4_ch48_nashm_128.hbm` | ×4 super-resolution, 128 tile (blend downstream) | not re-measured — see Status |

The compiled model is **~37 MB**. That is almost entirely instruction stream, not
weights, which is the first trap below.

## The traps on this model

**1. The output convention is undefined upstream, and getting it wrong is
silent.** SPAN normalises its input with `(x - mean) * img_range` inside
`forward` but **never undoes it**, and the repo ships no inference script or
config that says what the raw output means. So there are two live candidates —
the output is already `[0,1]`, or it needs `raw/255 + mean` — and they produce
*different pixels*. Pick wrong and the model loads, runs at full speed, and
returns a plausibly-shaped image with shifted colour and levels. `export.py`
does not assume: it scores **both** conventions by PSNR against a real
ground-truth HR image and folds the winner into the graph so the exported model
always returns `[0,1]`. This is why `export.py` needs a `--ref-hr` / `--ref-lr`
pair.

**2. Conv3XC recomputes its own reparameterisation inside `forward`.** Each
Conv3XC block fuses a `1×1 → 3×3 → 1×1` branch plus a `1×1` skip into a single
`3×3` at eval — an exact fusion, but one that runs *inside* `forward()`. Trace
that and the export captures the weight arithmetic as graph operators instead of
a clean conv. `export.py` runs the fusion once, replaces `forward` with the plain
`eval_conv`, and asserts the replacement is numerically identical (`max|diff| <
1e-5`) before exporting.

**3. The `.hbm` scales with tile AREA, not weights — so compile the small tile.**
The same SPAN network at a 256×256 tile is a **~148 MB** model against **~37 MB**
at 128×128, for **identical per-pixel throughput**. The binary is mostly
instruction stream; doubling each side quadruples the area and roughly
quadruples the size while buying nothing per pixel. Since the runtime tiles the
frame anyway, the 128 tile is the right build — a larger tile only enlarges the
model. (This 128 vs 256 = 37 vs 148 MB relationship is the repo's canonical
example of area-scaling.)

**4. `featuremap` + `no_preprocess` means the input domain is your job.** The
config declares the input as an arbitrary tensor with no normalisation, so the
toolchain does no colour conversion and no scaling. The model's actual domain is
**RGB, CHW, `[0,1]`** (pixels ÷ 255). Calibration tiles must be written in
exactly that domain — hand it raw `0–255` and the activation ranges are gathered
on a distribution the deployed model never sees. `calib.py` produces `[0,1]`
tiles and routes them through `common/calib_pack.py`, the one entry point that
writes a calibration set.

## Why this model — and why it is kept alongside Real-ESRGAN Compact

SPAN and Real-ESRGAN Compact are a **keep-both pair, not better/worse**:

- **SPAN is the fidelity choice.** It wins on *clean* input, reconstructs true
  detail rather than hallucinating it, and is **~1/6 the size** of Real-ESRGAN
  Compact. Use it when the source is a genuine low-resolution image.
- **Compact is the perceptual choice.** It is tuned for *degraded* input —
  blur, compression artefacts, noise — where a fidelity model has nothing clean
  to reconstruct and Compact's perceptual training produces the better-looking
  result.

Match the model to the input, not to a quality ranking. Both share the same
128-tile, `featuremap`/`no_preprocess`, `[0,1]` interface, so the runtime swaps
one `.hbm` for the other with no plumbing change.

## Running it

```bash
# [1/3] upstream checkpoint -> BPU-friendly ONNX.
#   --ref-hr / --ref-lr score the output convention against a real image;
#   --ref-lr is the 4x INTER_AREA downscale of --ref-hr.
python export.py --repo /path/to/SPAN --ckpt /path/to/spanx4_ch48.pth \
                 --ref-hr /path/to/hr.png --ref-lr /path/to/lr.png \
                 --tile 128 --out spanx4_ch48_128.onnx

# [2/3] calibration: 128x128 RGB [0,1] tiles, native + 4x-down mix, via calib_pack
python calib.py --config config.yaml \
                --images '/path/to/images/*.png' '/path/to/more/*.jpg' --limit 60

# [3/3] compile (mounts this dir at /ws; --gpu is an nvidia-smi index, Ampere+)
./compile.sh --config config.yaml --gpu 2
```

`--gpu` is an **nvidia-smi index**, handed to docker as `--gpus device=N`. It is
deliberately not `CUDA_VISIBLE_DEVICES`, which numbers devices differently — see
[CONVERSION.md](../../CONVERSION.md#traps). The card must be Ampere or newer.
`jobs` lives in the config, not on the command line.

## Status

**Reconstructed, not re-verified.** `export.py`, `config.yaml` and `compile.sh`
are rebuilt from the salvaged span convert scripts (`span_export.py`,
`run_span_compile.sh`, `span_ref.py`).

**`calib.py` is reconstructed by analogy — there was no salvaged calibration
script for span.** It is modelled on the sibling super-resolution calib and on
`span_ref.py`'s preprocessing: 128×128 RGB tiles scaled by 1/255, alternating
native crops and 4×-downscaled crops so both input domains are represented, all
written through `common/calib_pack.py`.

The salvaged material carried **no board measurements** — no latency, FPS,
cosine or PSNR. `expected.json` leaves every such field `null` rather than
inventing one; the only number stated is the ~37 MB tile-area size, which is the
repo-documented area relation, not a fresh measurement. A rebuild still needs a
full three-layer close-out and a PSNR-vs-bicubic score on the board before this
recipe is "verified".
