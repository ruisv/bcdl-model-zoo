# Converting a model: the general method

What applies to every model here. Per-model specifics — which graph to cut, what
bit width, which build is the right one — live in `models/<name>/README.md`.

Everything below was verified against OpenExplorer **3.7.0** targeting a board
running system software **4.0.5**. Version-sensitive claims say so.

- [The pipeline](#the-pipeline)
- [config.yaml](#configyaml)
- [Multi-input models](#multi-input-models)
- [Calibration](#calibration)
- [Choosing a bit width](#choosing-a-bit-width)
- [Verification](#verification)
- [Traps](#traps)

## The pipeline

Three steps, and they run in two different places:

```
export.py    upstream checkpoint -> ONNX        (host, torch env)
calib.py     public data         -> .npy set    (host)
compile.sh   ONNX + calibration  -> .hbm        (host, inside the OE container)
```

Only the `.hbm` crosses to the board. The board never converts anything, and the
host never runs inference on a `.hbm`.

## config.yaml

Four groups, and every model here uses the same skeleton:

```yaml
model_parameters:        # what to compile and where the output goes
  onnx_model: "model.onnx"
  march: "nash-m"                        # S100/S100P. S600 has its own.
  working_dir: "./out"
  output_model_file_prefix: "model_nashm_640x640"
input_parameters:        # what the model expects at runtime
  input_name: "images"
  input_type_rt: "nv12"                  # what the board feeds it
  input_type_train: "rgb"                # what the ONNX was trained on
  input_layout_train: "NCHW"
  norm_type: "data_mean_and_scale"
  mean_value: "123.675 116.28 103.53"
  scale_value: "0.01712475 0.017507 0.01742919"
calibration_parameters:
  cal_data_dir: "/ws/cal"
  calibration_type: "default"
compiler_parameters:
  optimize_level: "O2"
  compile_mode: "latency"
  jobs: 32                               # NOT a CLI flag; hb_compile has no --jobs
```

`march` decides which board the `.hbm` runs on. **A `nash-e` model does run on an
S100P** — that is march compatibility, and it is separate from whether the head
layout matches your decoder. A model can load and run and still decode to zero
detections because the head is laid out differently.

The compiled `.hbm` is mostly **instruction stream, not weights**, and it scales
with input **area**. The same super-resolution network at a 256×256 tile is a
148 MB model against 37 MB at 128×128, for identical per-pixel throughput. If
the runtime already tiles, compile the small tile.

## Multi-input models

Every per-input field is a **`;`-separated list**, and they must line up
positionally with `input_name`:

```yaml
input_parameters:
  input_name: "left;right"
  input_type_rt: "featuremap;featuremap"
  input_type_train: "featuremap;featuremap"
  norm_type: "no_preprocess;no_preprocess"
calibration_parameters:
  cal_data_dir: "/ws/cal/left;/ws/cal/right"
```

`cal_data_type` is per-input too, if you use it at all — a scalar on a two-input
model is rejected with *"Num of cal_data_type given: 1 is not equal to input num
2"*. `common/calib_pack.py` checks the lengths itself, because the compiler's
error does not always name the field that is short.

`featuremap` means "an arbitrary tensor": no colour-order conversion and no
normalisation, and you supply the arrays yourself.

## Calibration

**Always through `common/calib_pack.py`.** It reads mean/scale from the same
`config.yaml` hb_compile reads, so the two cannot drift. See the module
docstring for why that matters more than it sounds — the short version is that
un-normalised calibration data compiles perfectly and decodes to garbage.

- **Format: `.npy`.** `cal_data_type` is deprecated in 3.7.0 and the toolchain
  asks for npy directly. `.bin` still works and older configs here use it.
- **Nothing but samples in `cal_data_dir`.** Every file in that directory is
  read as a sample; a stray metadata file fails with *"cannot reshape array of
  size 196 into shape (1,3,480,640)"*. `calib_pack.py` writes its manifest
  beside the directory rather than inside it.
- **Domain match beats sample count.** 20-100 samples is plenty. Data from the
  actual deployment domain wins: on LAS2, 5 same-domain stereo pairs gave EPE
  0.12px and adding 54 out-of-domain pairs made it *worse* (0.26px).
- **Preprocess exactly as you deploy.** Same letterboxing, same resolution. On
  a stereo model, calibrating with a resize and deploying with a crop changes
  the disparity range and the output collapses.

`norm_type` is also formally deprecated (normalisation is inferred from whether
mean/scale are present), but it is still honoured and every config here sets it
explicitly. It documents intent, and `calib_pack.py` keys off it.

## Choosing a bit width

Start at int8. Move up only with evidence, and **measure before assuming mixed
precision is the compromise** — it frequently is not:

| | LAS2 int8 | LAS2 mixed | LAS2 all-int16 |
|---|---|---|---|
| accuracy | 1.23 px EPE | 0.25 px | **0.12 px** |
| speed | 174 FPS | **12 FPS** | 71 FPS |

Mixed precision was both slower than everything and less accurate than plain
int16. Scattered int8/int16 boundaries each need a requant, and that cost
dominates. **Unless the int16 layers coalesce into large contiguous blocks,
quantise the whole graph.**

Some networks cannot be rescued by bit width at all. OSNet's int8 PTQ compiles
cleanly and emits well-formed unit vectors whose Market-1501 Rank-1 is 51%
against the float model's 85%; it needed QAT self-distillation to reach 84.6%.
Use `<prefix>_node_info.csv` (per-layer BPU/CPU placement and per-layer cosine)
to find out whether the loss is one isolated layer or spread across the graph —
if it is spread, mixed precision has nothing to target.

## Verification

Three layers, via `common/verify_cosine.py`. Run all three: each isolates a
different failure.

| layer | compares | gate |
|---|---|---|
| A quantisation fidelity | host `.bc` vs ONNX | cosine > 0.99 |
| B execution consistency | board `.hbm` vs host `.bc` | **bit-identical** |
| C end-to-end | board `.hbm` vs ONNX, real images | cosine > 0.99 |

B is exact by construction — it is the same compiled graph run twice. Drift
there is a plumbing fault, never a quantisation one, and the first thing to
check is **input stride**: a row-padded buffer fed as if packed produces
plausible output and once drove a face-recognition cosine to 0.015 while every
compile-time check stayed green.

hb_compile prints a per-output cosine at the end of calibration, which is
layer A for free.

**Where a task metric exists, cosine is not sufficient** — see OSNet above.
Declare the task metric in `expected.json` and measure it.

## Traps

**GPU selection: nvidia-smi index ≠ CUDA index.** CUDA defaults to
`CUDA_DEVICE_ORDER=FASTEST_FIRST` and reorders by compute capability, while
nvidia-smi orders by PCI bus. `--gpus all -e CUDA_VISIBLE_DEVICES=2` therefore
does *not* reliably select the card `nvidia-smi` calls 2. Hand the container one
device with `--gpus "device=N"`. Getting this wrong surfaces late, as
`cudaErrorInvalidDevice: invalid device ordinal` partway into calibration.

**The calibration GPU must be Ampere or newer.** The PTQ kernels fail on older
cards with the same `cudaErrorInvalidDevice`, which is indistinguishable from
the mistake above — check both.

**Container invocation.** Entrypoint is `/bin/bash`, so commands go through
`-lc "..."`. Use `--user "$(id -u):$(id -g)"` or the outputs land owned by root,
and point `HOME` and `MPLCONFIGDIR` at a writable path inside the mount or the
toolchain fails on startup.

**A wrong build compiles cleanly.** This is the theme. Nearly every failure
documented in this repo produced a model that compiled without warnings, loaded,
and ran at full speed. Treat a successful compile as no evidence at all — the
gate is `expected.json`, not the absence of errors.

**The file name is which build it is.** `_cut`, `_v3`, `_aligned`, `_qat`,
`_crop` are not decoration. Keep the file name matching the model name inside
it; renaming a `.hbm` to something convenient has already caused one round of
"which calibration was this?" archaeology.
