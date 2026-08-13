# GLASS — one-class industrial anomaly detection, 288×288

Trained on one product's **defect-free** images only; returns a per-pixel anomaly
map for that product. Upstream: [GLASS](https://github.com/cqylunlun/GLASS)
(ECCV 2024), WideResNet-50 backbone, MVTec-AD `bottle`.

| build | BCDL consumes | on-board |
|---|---|---|
| `glass_bottle_int8_nashm_288x288.hbm` | per-pixel anomaly map | **72.6 ms / 14 FPS** (81.0 ms end to end) |

Hugging Face: [`ruisv/bcdl-glass-bottle`](https://huggingface.co/ruisv/bcdl-glass-bottle)
— **non-commercial**, see [Licence](#licence).

> [!IMPORTANT]
> **This model detects defects in one kind of bottle.** It is an *example*, not a
> general anomaly detector. What transfers to your product is the recipe, and
> that is cheaper than it sounds — see below.

## Why "retrain it for your part" is a reasonable ask

The backbone is a **frozen ImageNet WideResNet-50**. Per product, only two small
heads are trained — `pre_projection` and `discriminator`, about **16 MB**. You
need defect-free images of your part and no labelled defects at all, because the
method never sees one during training.

That structure is also why a library cannot ship a useful general model here, and
why this recipe is worth more than the binary attached to it.

## int8, at an output cosine of 0.9409

That number is far below any threshold anyone would normally accept. It costs
nothing:

| build | image AUROC | pixel AUROC | output cosine |
|---|---|---|---|
| float ONNX | 0.9905 | 0.9907 | — |
| **int8 (shipped)** | **0.9905** | **0.9905** | **0.9409** |
| int16 | 0.9897 | 0.9905 | 0.9918 |

**AUROC is rank-based.** Quantisation noise nudges every score slightly and does
not reorder them, so a metric built on ordering does not notice. int16 buys back
the cosine and is *marginally worse* on the metric that matters.

Read this next to [`neuflow`](../neuflow/), where the compiler reported **0.9954**
for a build that was unusable. Together they are the whole argument:

> **Cosine is not a proxy for task quality in either direction. Score the task in
> its own units.**

## What the AUROC hides, and what you have to decide

Good and defective scores **overlap**: the best normal image scores 0.953, the
worst defect 0.882. So on this product:

- zero misses costs about **50% false alarms**
- zero false alarms misses about **5%** of defects

The misses are all `contamination` — small foreign objects. **The threshold is a
line-side decision**, not a property of the model, and the figure it comes from
belongs to your production line, not to this repo.

The image score is the map's **maximum, not its mean**. How bad a part is depends
on its worst patch; a mean lets a large clean region hide a small defect, which is
precisely the case this method exists to catch.

## Exporting it

The one operator worth worrying about is `torch.nn.Unfold` (3×3 neighbourhood
aggregation), which the BPU does not support — and it **needs no rewrite**,
because the torch exporter lowers it into standard operators. After `onnxslim`
the graph is 136 nodes with **no red-flag operator at all**: Conv, Relu, Pool,
Reshape, Transpose, Gemm, MatMul, Sigmoid.

Calibration is **48 normal images of the same product**. Unusually, the
calibration domain being "normal only" is not a compromise — the model has never
seen a defect and is not meant to, so normal images *are* the honest
distribution.

```bash
python export.py --ckpt /path/to/bottle_ckpt.pth --repo /path/to/GLASS
onnxslim glass_bottle.onnx glass_bottle_slim.onnx
python calib.py --config config.yaml --good /path/to/mvtec/bottle/train/good --limit 48
./compile.sh --gpu 1
```

## To go faster

**Change the backbone.** 72.6 ms is a WideResNet-50 at 288², and that is the
whole cost — preprocessing is 1.6 ms and decoding 6.8 ms (6.3 of which is the
Gaussian blur; `blur_kernel=0` skips it at the price of diverging from the
reference post-processing). Lean GLASS, with a MobileNetV2 backbone, is the
ready-made next step.

## Licence

**MIT** on the code and on the released weights. **The shipped checkpoint is
trained on MVTec AD, which is CC BY-NC-SA 4.0** — attribution, no commercial
use, and derivatives under the same terms.

Unlike a bare "research only" notice, a Creative Commons licence has an explicit
framework for adapted material, so a model trained on it may be redistributed
under the same licence. Note the **share-alike**, which ViTPose's CC-BY-NC does
not carry: a redistributed `.hbm` travels under CC BY-NC-SA 4.0 too.

**This does not reach a real deployment, and that is the point.** A deployed
GLASS is retrained on *your* defect-free parts, so it derives from your data, not
from MVTec's. What is encumbered is this example checkpoint — not the capability,
and not the recipe.

> A correction worth recording: an earlier note in this project claimed the MVTec
> AD dataset was already staged on the conversion host. It was not — that
> directory held **HALCON**, MVTec's machine-vision software, which is a
> different thing from the company's dataset. The dataset was downloaded
> separately (5.27 GB).
