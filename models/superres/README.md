# Real-ESRGAN Compact x4 (superres) — tile super-resolution, 128×128

A x4 super-resolution tile network: 128×128 RGB in, 512×512 out. Real-ESRGAN
Compact (`SRVGGNetCompact`, the `realesr-general-x4v3` checkpoint) — a plain
Conv/PReLU stack into a PixelShuffle, plus a nearest-neighbour skip so the net
only learns the residual over an enlargement. Fully convolutional, all standard
ops, no attention: it quantises to int8 without drama. The on-board runtime
tiles a frame, upscales each tile and blends the overlaps.

Upstream: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause).

| build | BCDL consumes | tile | size |
|---|---|---|---|
| `realesr_general_x4v3_nashm_128.hbm` | x4 tile upscale (runtime tiles + blends) | 128×128 | ~37 MB |
| `realesr_general_x4v3_nashm_256.hbm` *(not shipped)* | same, larger tile | 256×256 | ~148 MB |

## The traps on this model

**1. The tile size is the whole decision, and the larger tile is the wrong
one.** A compiled `.hbm` is mostly **instruction stream, not weights**, and it
scales with input **AREA**. The *same* network at a 256×256 tile is a ~148 MB
model against ~37 MB at 128×128 — 4× the pixels, ~4× the size — for **identical
per-pixel throughput and identical quality**. The 256 build is not sharper; it
is just bigger. Since the runtime already tiles the frame, the 128 tile is
correct and the 256 tile only makes sense for a caller that cannot tile. Reach
for the bigger tile expecting better output and you pay 4× the footprint for
nothing. This is why `config.yaml` is the 128 build and `config_256.yaml` exists
only to make the cost visible.

**2. Calibrate on both input domains or the model saturates on the one it never
saw.** This net is fed two genuinely different distributions depending on what
the caller is doing:

- **native** crops — a real image cropped without rescaling: sharp edges, sensor
  detail (the *sharpen a large image* case);
- **downscaled** crops — a crop taken after a 4× area-downscale: soft and
  band-limited (the *enlarge a small image* case).

`calib.py` makes half the set each. Calibrate on only one and the other domain's
activation ranges are unrepresented, so the quantised model clips exactly when
it meets the input it was never shown. As always here, **domain match beats
sample count** — 20–100 tiles from content that looks like the deployment is
worth more than a large out-of-domain pile.

**3. The tile is a raw tensor, not an image.** The input is `featuremap` /
`no_preprocess`: RGB, CHW, in **[0,1]**. The compiler applies no normalisation,
so `calib.py` divides by 255 itself before handing the tiles to
`calib_pack.pack()`. Feed 0–255 tiles here and the calibration statistics are
gathered on a range the deployed [0,1] path never sees — the model compiles
clean and washes out. The write still goes through `calib_pack` so there is only
one way to produce a calibration set.

## Compact is the perceptual pick, not "the worse SPAN"

Real-ESRGAN Compact and SPAN are a **keep-both pair, not better/worse.** Compact
is the *perceptual* choice — it was trained with a GAN objective and wins on
degraded input: blurred, compressed, noisy, real-world low-quality frames, where
it hallucinates plausible detail. SPAN is the *fidelity* choice — it optimises
PSNR/SSIM and wins on clean input where you want the reconstruction to stay
faithful to the source rather than invent texture. Pick Compact when the input
is ugly and you want it to *look* good; pick SPAN when the input is clean and you
want it to *be* accurate. Neither replaces the other.

## Running it

```bash
# [1/3] upstream checkpoint -> ONNX at the shipped 128 tile
python export.py --weights /path/to/realesr-general-x4v3.pth --tile 128 \
                 --out realesr_general_x4v3_128.onnx

# [2/3] calibration: native + 4x-downscaled tiles, [0,1] RGB, written via calib_pack
python calib.py --config config.yaml --images '/path/to/images/*.png' --limit 60

# optional fp32 golden for layer C (one isolated tile, no blending)
python calib.py --config config.yaml --images '/path/to/images/*.png' \
                --golden --onnx realesr_general_x4v3_128.onnx \
                --golden-image /path/to/an_hr_image.png

# [3/3] compile (config.yaml = 128 tile; config_256.yaml = the 256 tile)
./compile.sh --config config.yaml --gpu 2
```

`--gpu` is an **nvidia-smi index**, handed to docker as `--gpus device=N` — not
`CUDA_VISIBLE_DEVICES`, which numbers devices differently. The card must be
Ampere or newer. `jobs` lives in the config, not on the command line. Both
configs target `nash-m` (S100/S100P).

The honest way to score it: take a real image as ground-truth HR, 4×-downscale
it to make the LR input, upscale, and compare — a useful upscaler has to beat
bicubic. The single-tile golden lets the board be checked against the float model
directly, before any tile blending is in the picture.

## Status

Recipe reconstructed from the original conversion scripts (`sr_export.py`,
`sr_calib.py` / `sr_calib128.py`, the per-tile configs and `sr_ref.py`). The
export, the [0,1] calibration with its native/downscaled domain mix, and the
int8 compile are faithful to the salvage.

**Not re-verified in this repo.** No latency, PSNR or cosine measurement
survived in the salvaged scripts, so `expected.json` carries those as `null`
rather than invented figures — the `~37 MB` / `~148 MB` sizes are the documented
tile-area expectation, not a measured byte count for a rebuild. The method
(including the PSNR-vs-bicubic task metric and the single-tile layer-C golden) is
reproducible from the scripts here; run the three-layer check on an S100P and
fill in the numbers before quoting any.
