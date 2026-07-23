# PIDNet-S — real-time semantic segmentation, 2048×1024

Cityscapes 19-class semantic segmentation at video rate. Upstream:
[PIDNet](https://github.com/XuJiacong/PIDNet).

| build | BCDL consumes | note |
|---|---|---|
| `pidnet_s_nashm_1024x2048_nv12_v3.hbm` | 19-class seg logits; BCDL takes the argmax | `_v3` = calibrated on **pre-normalised** data |

Upstream licence: check PIDNet's terms before redistribution.

## The trap: the wrong calibration data compiles clean and segments to noise

`_v3` is a suffix that records *which calibration was used*, and the earlier
builds without it are the cautionary tale. The failure is the same silent kind
that runs through this whole repo, and PIDNet is its clearest instance:

**When `cal_data_type` is float32, the compiler's `norm_type` / `mean_value` /
`scale_value` apply to the runtime input path only — NOT to the calibration
data.** So if you hand the compiler raw 0-255 pixels while the config declares an
ImageNet mean and scale, the activation statistics are gathered on a distribution
the deployed model never sees. The input thresholds come out wrong, and the model
**compiles without a single warning, loads, runs at full frame rate, and segments
to noise.** There is no loud signal to catch it.

`_v3` is the build calibrated on **pre-normalised** data. The defence is
structural, not a note in a doc: `calib.py` writes calibration data only through
`common/calib_pack.py`, which applies exactly the normalisation `config.yaml`
declares. There is no second path that could skip it.

Two normalisation steps have to line up for this to be correct, and the recipe
keeps them in one place each:

1. **The ONNX takes RAW pixels** (`pidnet_op19_nonorm.onnx`) — no normalisation
   baked into the graph. An earlier build baked a partial job (`data_scale` 1/255
   with *no* mean subtraction), which is the same clean-compile failure with a
   different missing step.
2. **The config does all the normalisation** (`data_mean_and_scale`, ImageNet
   mean/std), for both the runtime input and — through `calib_pack` — the
   calibration data. One declaration, applied identically to both paths.

## The other detail: opset 19

The export is opset 19 so PIDNet's bilinear `Resize` lowers cleanly on the
toolchain. Lower opsets export without error but can drop or approximate the
upsample — worth knowing because, again, it would not fail loudly.

## Running it

```bash
# 1. PIDNet-S checkpoint -> no-norm opset-19 ONNX (raw-pixel input)
python export.py --repo /path/to/PIDNet \
                 --weights PIDNet_S_Cityscapes_val.pt --out pidnet_op19_nonorm.onnx

# 2. calibrate on street frames (Cityscapes val or any urban set), PRE-NORMALISED
python calib.py --config config.yaml --images /path/to/cityscapes_frames --limit 64

# 3. compile
./compile.sh --config config.yaml --gpu 2
```

`--gpu` is an nvidia-smi index handed to docker; the card must be Ampere or
newer. `jobs` lives in the config.

## Status

**Reconstructed.** The original export script was not recovered; `export.py`
reproduces the method from the surviving config (`pidnet_op19_nonorm.onnx`, opset
19, raw-pixel input, `data_mean_and_scale`) and the upstream PIDNet-S inference
forward — verify the seg-output resolution against your own PIDNet checkout. The
load-bearing part of this recipe, the pre-normalised calibration through
`calib_pack`, is faithful and is the whole reason `_v3` exists. Not re-run through
the toolchain in this repo; the shipped build is board-validated in BCDL's
real-time segmentation path.
