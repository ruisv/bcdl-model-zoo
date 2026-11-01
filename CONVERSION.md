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
- **Batch-8 calibration can drop samples — check, don't assume.** With every
  input at batch 1, hb_compile calibrates in batches of 8. If the optimised
  graph cannot run at batch 8 the log says *"Reset batch_size=1 and execute
  forward again"* and every sample counts; most models take that path. The trap
  is a graph that runs at batch 8 but mixes the batch after constant folding:
  on LAS2 the optimiser froze a Resize's output size at batch 1, so the nodes
  after it saw only the first sample of each batch and the disparity output
  capped at 47.7 px on a set spanning 0–188 px. The check: run the
  `*_optimized_float_model.onnx` on two different samples as one batch (with
  the toolchain's own executor, since it has custom ops) and compare against
  running them one at a time. Only if that fails, write each sample 8 times in
  a row (`pack(..., repeat=CALIB_REPEAT)`). Not by default: duplicates do not
  move max or percentile thresholds, but they move kl ones.
- **Domain match beats sample count.** 20-100 samples is plenty, and data from
  the actual deployment domain is what you want. The LAS2 measurement that used
  to be quoted here (5 same-domain pairs 0.12 px, adding 54 out-of-domain pairs
  0.26 px) was taken while that graph's output nodes saw one sample in eight,
  so treat it as anecdote, not evidence.
- **Preprocess exactly as you deploy — and confirm what "deploy" actually does.**
  Same letterboxing, same resolution, same channel order. On a stereo model,
  calibrating with a resize and deploying with a crop changes the disparity range
  and the output collapses. On PP-OCRv6 recognition, a plain `resize(320,48)`
  *stretches* short lines while the runtime does aspect-preserving resize +
  right-pad; that single mistake distorted the calibration set AND every
  evaluation, made int8 look far worse than it was, and even lifted the int8
  output cosine from 0.9747 to 0.9819 once fixed. **Read the deployment code's
  preprocessing, don't assume it.** When a task metric exists, the fastest way to
  catch a preprocessing mismatch is to run the *float* model through your
  pipeline first: if float already degrades, the bug is upstream of quantisation.

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

The PP-OCRv6 recogniser is the second worked example, and it makes the point
sharper because the mixed build is what you get by *default*. With no directive
the compiler chose 164 int8 / 25 int16 / 2 int32, and board latency came out
**4.69 ms against all-int16's 2.17 ms** — the more aggressively quantised build
was 2.2× faster. Same cause: requant at every int8↔int16 boundary. The lesson
for a transformer-ish graph: measure `set_all_nodes_int16` on the board before
accepting the default mix. Here all-int16 won on latency, accuracy, and size at
once.

Before hand-writing a mixed-precision config, see what the compiler does on its
own. With no `optimization` directive, hb_compile already promotes the obviously
sensitive layers: the PP-OCRv6 recogniser came out 162 int8 / 27 int16 / 2 int32
without being asked, and the int16 nodes were the LayerNorm-adjacent ones a
transformer stack needs. Read `Output Data Type` in `node_info.csv` to see the
split it chose. A manual config is for overriding that, not for discovering it.

Some networks cannot be rescued by bit width at all. OSNet's int8 PTQ compiles
cleanly and emits well-formed unit vectors whose Market-1501 Rank-1 is 51%
against the float model's 85%; it needed QAT self-distillation to reach 84.6%.
Use `<prefix>_node_info.csv` (per-layer placement and per-layer cosine) to find
out whether the loss is one isolated layer or spread across the graph — if it is
spread, mixed precision has nothing to target.

**Per-layer cosine is a localisation tool, not a quality gate.** Do not read a
low number there as a failure. Measured on the accepted all-int16 LAS2 build:

| | value |
|---|---|
| median per-layer quantised cosine | 0.9863 |
| worst per-layer | 0.9436 |
| layers below 0.99 | 430 of 651 |
| **model output cosine** | **0.999954** |

Two thirds of the layers sit under the 0.99 that gates the *model*, and the
model is nonetheless near-perfect: intermediate error does not simply
accumulate, and a per-layer figure is not comparable to an end-to-end one. Use
the column to compare candidate builds against each other and to find outliers
worth attention — never to accept or reject a build. That decision belongs to
`expected.json`.

**The one shape that does predict failure: a whole operator class collapsing.**
LingBot-Depth's int8 build is the counter-example that sharpens the rule. Its
per-layer report is not a broad mild sag like LAS2's — it has *every* LayerNorm
and softmax `Mul` node at ~0.006 cosine, with 1056 of 1418 nodes below 0.90, and
the build really is unusable (233% absolute relative error, dynamic range
collapsed). The difference is categorical, not a matter of degree, so **group the
report by operator type before judging it**. A median that sags is normal; an
operator class pinned at zero means that operator has no usable range at this bit
width, and no amount of end-to-end luck will rescue it. On a 24-layer ViT-L this
saved a board trip — the verdict was legible before anything was compiled to
`.hbm`.

Two more columns worth reading correctly:

- **`Output Data Type`** is the real evidence of what your quantisation
  directive did — the LAS2 int16 build shows `si16` on 642 nodes and `si8` on
  16. Check this rather than trusting the yaml.
- **`ON`** is not a clean BPU/CPU split. It shows `--` for nodes that do not
  execute standalone, including fused ones and the graph's own output node, so a
  `--` is **not** proof of CPU fallback. If you need to know where an operator
  really ran, measure it; do not infer it from this column.

## Verification

Three layers, via `common/verify_cosine.py`. Run all three: each isolates a
different failure.

| layer | compares | gate |
|---|---|---|
| A quantisation fidelity | host `.bc` vs ONNX | cosine > 0.99 |
| B execution consistency | board `.hbm` vs host `.bc` | **bit-identical** |
| C end-to-end | board `.hbm` vs ONNX, real images | cosine > 0.99 |

B compares the same compiled graph run in two places, so a *gross* drift there
is a plumbing fault, never a quantisation one — and the first thing to check is
**input stride**: a row-padded buffer fed as if packed produces plausible output
and once drove a face-recognition cosine to 0.015 while every compile-time check
stayed green.

**But "bit-identical" is the right bar only for pure fixed-point graphs.**
Measured on the board: LAS2 (all-int16) and the PP-OCRv6 detector (all-int8)
are bit-identical between `.hbm` and `.bc`, max abs error exactly 0. The
PP-OCRv6 recogniser is not — it is mixed int8/int16/int32 with float CPU
operators, and the board and host float implementations differ in the last
places: cosine 1.0, max abs error 2e-4, and an **identical CTC decode**. So when
the graph has float CPU ops, read layer B as *task equivalence* (same decode,
same boxes) rather than bitwise equality. A 2e-4 logit drift that leaves the
argmax unchanged is a pass; a stride bug is not, and the two are not hard to
tell apart — the stride bug collapses the cosine, this does not.

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

**ONNX IR version must be ≤ 9.** HBDK 4.x rejects anything newer — *"The ir
version of the model is 10, which is greater than the maximum supported ir
version of 9."* Recent exporters emit IR10 by default (the PP-OCRv6 detector
does; the recogniser, at IR6, does not). onnxslim preserves the source IR, so
cap it afterwards: load the model, set `ir_version = 9`, save. IR9 covers up to
opset 20, so this is a header change, not a downgrade of any operator. The
per-model `export.py` handles it, but a hand-run conversion has to remember it.

**A wrong build compiles cleanly — and a noisy compile can still be right.** This
is the theme, and it cuts both ways. Nearly every failure in this repo produced a
model that compiled without warnings, loaded, and ran at full speed; treat a
successful compile as no evidence at all. But the converse also holds: the
PP-OCRv6 960 recogniser logged two onnxruntime `Reshape` errors during
calibration (a 5-D attention Squeeze that onnxslim did not resolve for the wider,
120-timestep shape), and it compiled anyway to a model that decodes long lines
correctly on the board. Errors in the log are a prompt to **go verify on the
board**, not an automatic reject. The gate is `expected.json` and a board run,
never the presence or absence of compile-time messages.

**The file name is which build it is.** `_cut`, `_v3`, `_aligned`, `_qat`,
`_crop` are not decoration. Keep the file name matching the model name inside
it; renaming a `.hbm` to something convenient has already caused one round of
"which calibration was this?" archaeology.

**hbdk's latency estimate models compute, not DDR.** The number in
`<prefix>.json` (`performance.latency (ms)`) was within 9% of the board on
LingBot-Depth's int8 build — and 2.1× optimistic on the int16 build of the same
graph (680 ms estimated, 1457 ms measured). The int16 build is DDR-bound: its
attention matmuls compute for 4.3 ms inside a 21.9 ms slot, and the graph moves
23 GB per frame against int8's 4.2 GB. Before trusting the estimate, check
`DDR megabytes per run` and compare `computing cost (no DDR)` against
`active period of time` per layer. When those diverge, the estimate is a lower
bound and the board is the only answer.

**Size a big transformer with truncated probes before paying for a full
compile.** Export the same graph with N and M transformer blocks (everything else
identical, so every op type and tensor shape survives), run `hb_compile
--fast-perf` on both — no calibration data needed — and the per-block cost and
fixed overhead fall out of two points. On LingBot-Depth, 1-block 15.65 ms and
3-block 33.42 ms gave 8.885 ms/block plus 6.77 ms fixed, predicting 220 ms for
the real 24-block model against 255 ms measured. Two five-minute compiles instead
of a two-hour one, and it answers "is this model even in the right order of
magnitude" before you commit to calibrating it.

**When the graph is DDR-bound on attention, resolution is the lever.** Attention
cost is quadratic in sequence length, so a model with a tunable token count has a
much better speed knob than bit width. LingBot-Depth's own measurements: dropping
1200 → 768 tokens makes attention 2.4× cheaper for +0.55pp absolute relative
error — while the int16 quantisation it is already paying for costs 2.92%. Check
whether the accuracy you would trade away is smaller than the accuracy you have
already spent; often it is.
