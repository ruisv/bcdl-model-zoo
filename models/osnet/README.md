# OSNet-AIN — person ReID embedding, 256×128

A 512-d appearance embedding per person crop, for multi-object tracking.
Upstream: [torchreid](https://github.com/KaiyangZhou/deep-person-reid) (MIT);
weights are OSNet-AIN (`osnet_ain_x1_0`) trained on MSMT17.

| build | BCDL consumes | on-board |
|---|---|---|
| `osnet_ain_qat_nashm_256x128.hbm` | 512-d ReID embedding (L2-normalised downstream) | 0.82 ms / 1220 FPS |

`osnet.py` / `osnet_ain.py` are vendored verbatim from torchreid (MIT) so the
export, QAT and deploy steps import identical model definitions without a
torchreid install inside the toolchain container. The **weights'** own terms are
not reviewed — check before redistribution.

## Why there is no "wrong build" table here — and why that is the danger

Every other model in this repo has a build you can pick wrong and lose accuracy
*visibly enough that a cosine catches it*. OSNet has the opposite problem, and it
is worse: **the wrong build is invisible.** A quantised ReID tower that has
genuinely fallen apart still emits well-formed 512-d unit vectors. Its cosine
against the float reference does **not** clearly say "broken", the model loads,
runs at full speed, and tracking quietly degrades.

So this model is the reason the repo insists on a **task metric** and not just a
cosine gate. The number that tells the truth is Market-1501 Rank-1 over a
labelled gallery (`market.py`): int8 PTQ scores **51%** where the float model
scores **85%**, and only the retrieval metric shows it.

## The four traps on this model

**1. int8 PTQ does not work on this network, at all.** Not "a bit worse" —
Rank-1 51.4% against the float 84.8%, on a build that compiles without a single
warning. Every knob was tried and none rescued it: three calibration methods,
calibration set grown 64→400 crops (400 was *slightly worse*). You can reproduce
the collapse with `calib.py` + `config.yaml` + `market.py`. This is why the whole
QAT apparatus below exists.

**2. InstanceNorm is a quantisation *asset*, not a liability — this is
backwards from the usual intuition.** The three OSNet variants differ only in how
much InstanceNorm they carry, and that turned out to *be* the axis that decides
int8 survival:

| variant | InstanceNorm | int8 PTQ Rank-1 |
|---|---|---|
| `osnet_ain_x1_0` | IN in the residual blocks | **51.4%** |
| `osnet_ibn_x1_0` | IN paired with BN, early stages | 5.2% |
| `osnet_x1_0` | BatchNorm only | 6.6% |

Reaching for the plain-BN model "because BatchNorm quantises more predictably"
makes it 8× worse. And you cannot dodge the problem by swapping backbone either:
ResNet50's *float* cross-domain Rank-1 (46.3) is already below OSNet-AIN's (70.1),
so a "more quantisation-friendly" backbone loses more than quantisation ever did.

**3. The fix is QAT self-distillation, and it is hours not days.** The defect is
precisely stated: the quantised embedding has drifted off the float one (cosine
~0.47 where it should be ~0.99). That is a **function-matching** problem, not an
identity-learning one, so:

- the FP32 model is the only supervision needed — **no identity labels, no
  triplet mining, no ReID training set**;
- any pile of person crops works (`crops.py` cuts ~11k from COCO + KITTI);
- the loss is `1 - cos(student, teacher)` — **cosine, not L2**, because the
  embedding is L2-normalised before use so its magnitude is not in the contract;
- **Market-1501 is never trained on** — it stays the held-out benchmark.

6–8 epochs, ~20 minutes on one GPU, cosine-vs-FP32 climbs 0.40 → 0.978, Rank-1
recovers to **84.6%**. And it is debuggable offline: the QAT graph reproduces the
board's failure in torch (cosine 0.5049 torch vs 0.475 board), so you iterate in
seconds against the number the board will actually show.

**4. Export the QAT model the vendor's way; `convert_fx` is a dead end.** The
real→fake quant conversion belongs at the **HBIR** level, not the PyTorch level:

```python
set_fake_quantize(model, FakeQuantState.VALIDATION)
qat_hbir       = export(model, example)              # export the QAT model as-is
quantized_hbir = hbdk4.compiler.convert(qat_hbir, march)   # this resolves it
hbdk4.compiler.compile(quantized_hbir, path, march)
```

The directly-exported QAT graph still carries `qnt.const_fake_quant` nodes that
hbdk marks `illegal` — **that is expected**, and `convert` is what clears them.
Trying to convert in PyTorch with `convert_fx` instead walks into the plugin's
quantised-mean gap: quantised InstanceNorm *and* `AdaptiveAvgPool2d(1)` both
reduce over `dim=(2,3)`, and the quantised `mean` has no multi-dim implementation
(typeguard rejects the tuple; silence it and it does `scale * x.shape[dim]` on
the tuple and dies). `deploy.py` follows the vendor flow and sidesteps all of it.
The general lesson: **when the vendor ships a reference exporter, copy it — do
not patch your way down the stack from the first error message.**

(A fifth, smaller one lives in `export.py`: the head emits a 4-D `[N,512,1,1]`
embedding rather than torchreid's 2-D vector, because the 2-D form bakes the
batch size into a Reshape constant that makes batch-8 calibration silently fall
back to one-sample-at-a-time. Same 512 floats to the board either way.)

## Why this model

OSNet-AIN is the strongest *cross-domain* small ReID backbone available: a
tracker never trains on the scene it runs in, and AIN's instance normalisation is
exactly what buys the domain robustness (and, per trap 2, the quantisation
robustness). At 6.6 MB and 0.82 ms/crop it is cheap enough to embed every box in
every frame.

## Running it

All four steps run inside the OpenExplorer GPU container via `compile.sh`, which
mounts this directory at `/ws` and hands in one GPU (`--gpu` is an nvidia-smi
index; the card must be Ampere or newer).

```bash
# [1/4] FP32 checkpoint -> ONNX (teacher + float baseline + int8 input)
python export.py --weights osnet_ain_x1_0_msmt17.pth \
                 --out osnet_ain_x1_0_256x128.onnx

# establish the float Rank-1 baseline and write the board-comparison bundle
python market.py --root /path/to/Market-1501-v15.09.15 \
                 --onnx osnet_ain_x1_0_256x128.onnx --n-ids 100 \
                 --bundle market_ref.npz

# [2/4] cut the unlabelled distillation crop set (~11k COCO + KITTI crops)
python crops.py --coco /path/to/coco --kitti /path/to/kitti --out crops/

# [3/4] QAT self-distillation -> qat checkpoint
./compile.sh --gpu 2 python qat.py \
    --weights osnet_ain_x1_0_msmt17.pth --crops crops/ --out qat_osnet_ain.pth

# [4/4] QAT -> .hbm (the shipped build)
./compile.sh --gpu 2 python deploy.py \
    --weights osnet_ain_x1_0_msmt17.pth --qat qat_osnet_ain.pth --crops crops/ \
    --out out/osnet_ain_qat_nashm_256x128.hbm
```

To watch int8 collapse for yourself (trap 1):

```bash
python calib.py --config config.yaml --coco /path/to/coco --kitti /path/to/kitti
./compile.sh --gpu 2 hb_compile -c config.yaml     # compiles clean, and is wrong
# then score out_int8/osnet_ain_int8_nashm_256x128 against market_ref.npz -> ~51%
```

## Status

Recipe reconstructed from the original M10 adaptation scripts. The shipped-build
numbers in `expected.json` (Rank-1 84.62 vs 84.84 float, board-vs-FP32 cosine
0.9799, 0.82 ms/crop) are from that board-validated adaptation on S100P (HBRT
4.7.5). A from-scratch rebuild through this recipe reproduces the *method* — the
crop set and QAT seed — not necessarily the exact binary; the acceptance is the
task metric, not byte-identity.
