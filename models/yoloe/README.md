# YOLOE-11s — open-vocab detection + segmentation, frozen to COCO-80

Upstream: [Ultralytics YOLOE](https://github.com/ultralytics/ultralytics)
(YOLOE-11s-seg). Open-vocabulary: the classifier compares image features to CLIP
text embeddings, so the class set is normally a *runtime* prompt. Upstream
licence: **AGPL-3.0** — copyleft.

> **Recipe only — no `.hbm` in this repo.** A compiled `.hbm` is a derivative of
> the AGPL Ultralytics weights, so it carries AGPL obligations and is **not
> committed or published here**. Rebuild it from this recipe. Commercial use
> needs an Ultralytics Enterprise licence. See [Licence](#licence).

| build | input | outputs | on-board |
|---|---|---|---|
| `yoloe_11s_coco80_det_bpu_nashm_640x640_nv12.hbm` | 1×3×640×640 nv12 | 6 = 3×[cls, box] NHWC | not measured |
| `yoloe_11s_coco80_seg_bpu_nashm_640x640_nv12.hbm` | 1×3×640×640 nv12 | 10 = 3×[cls, box, coef] + proto NHWC | not measured |

Both freeze the open vocabulary to the **80 COCO classes** at export time.

## What "open-vocab" survives, and what doesn't

YOLOE's selling point is runtime text prompts. **A static `.hbm` cannot do that**
— the contrastive classifier is a matmul against an *external* text-embedding
tensor, and there is nowhere to feed that tensor into a compiled graph. So the
recipe pins one vocabulary at export: `get_text_pe(COCO80)` computes the 80 text
embeddings, `set_classes` installs them, and `head.fuse()` folds them into the
`cv3` convolution. After fusing, `cv3[i](x)` emits 80-channel class **logits**
directly; the contrastive head collapses to Identity.

The consequence: the shipped models are **fixed 80-class detectors/segmenters**.
To change the class set, re-run `export_*.py` with a different list — you get a
different `.hbm`. There is no runtime prompting on the board.

## Traps on this model

**1. The head is rewritten — raw convs on the BPU, decode on the CPU.** This is
the whole recipe, and it is what makes YOLOE (or any Ultralytics YOLO) go on the
BPU cleanly. The stock head, after the `cv2`/`cv3` convs, **decodes** inside the
graph: DFL softmax + integral over `reg_max`, anchor-grid generation
(meshgrid/arange), stride multiply, `dist2bbox`, a class sigmoid, concat across
scales, and NMS. For seg it also folds in `coef @ proto → sigmoid → crop`. Those
are reshape/gather/reduce/meshgrid/matmul ops — cheap on the CPU, but a mix of
poorly-quantised and non-BPU-native operators, and they bake a fixed anchor
layout into the binary.

`export_*.py` monkeypatches `head.forward` to stop at the **raw per-scale conv
outputs**, permuted to NHWC:

- **det** (`bpu_forward`): per scale, `cv3→cls [B,H,W,80]` and `cv2→box
  [B,H,W,64]` (64 = 4·reg_max, DFL).
- **seg** (`bpu_seg_forward`): the same, plus `cv5→mask_coef [B,H,W,32]` per
  scale and one `proto [B,160,160,32]`.

bcdl's `YoloLtrbDetector` does the DFL integral, anchor math, sigmoid and NMS on
the CPU (auto-detecting `reg_max` from the 64 box channels); the seg build then
builds only the **per-detection** masks. Net effect: the BPU runs pure
convolution (clean int8/int16), one `.hbm` layout serves, and the 32×160×160
mask stack is never materialised on the BPU. If you feed the *stock* export to
bcdl it will not decode — the split-head layout is the contract.

**2. nv12 input; normalisation is not in the graph.** The board feeds an nv12
frame. The compiler converts nv12→rgb and applies `scale 1/255` (mean 0) — that
`data_scale` is the entire normalisation, because the Ultralytics export does
**not** bake `/255`; the ONNX expects RGB already in `[0,1]`. `calib.py` hands
`calib_pack.pack()` raw 0–255 pixels and `pack()` applies the same `1/255` from
`config_*.yaml`, so calibration matches deployment and the scale lives in exactly
one place. Do not pre-divide by 255 in `calib.py` — that double-scales the
calibration set, and (per this repo's first rule) it will compile without a
warning and decode to garbage.

**3. opset 19 → IR10; HBDK rejects IR > 9.** `export_*.py` caps `ir_version`
to 9 after export. IR9 covers up to opset 20, so it is a header change, not an
operator downgrade.

**4. Licence (AGPL-3.0).** Copyleft — ship the recipe, not the `.hbm`. See below.

## Running it

```bash
# 1. weights -> BPU split-head ONNX (fixed COCO-80). Two separate graphs.
python export_det.py --weights yoloe-11s-seg.pt --output yoloe_11s_coco80_det_bpu.onnx
python export_seg.py --weights yoloe-11s-seg.pt --output yoloe_11s_coco80_seg_bpu.onnx

# 2. calibration (real COCO images; one set serves either build). Pick the
#    config so the manifest lands beside the right cal_data_dir.
python calib.py --config config_det.yaml --images <coco val images> --limit 64
python calib.py --config config_seg.yaml --images <coco val images> --limit 64

# 3. compile (--config picks det vs seg)
./compile.sh --config config_det.yaml --gpu <ampere-or-newer>
./compile.sh --config config_seg.yaml --gpu <ampere-or-newer>
```

`--gpu` is an **nvidia-smi index**, handed to docker as `--gpus device=N` (not
`CUDA_VISIBLE_DEVICES`, which numbers differently — see
[CONVERSION.md](../../CONVERSION.md#traps)). Ampere or newer. `jobs` lives in the
config, not on the command line. Both configs target `nash-m` (S100/S100P).

Calibrate on real COCO images — the vocabulary is COCO-80, so the domain match is
COCO. 20–100 images is plenty; domain match beats count.

## Status

**Recipe reconstructed, NOT re-run.** Only the two ONNX export scripts survived
salvage — no config, no calibration set, no compile log, no board numbers. The
head-surgery logic (freeze vocab via `fuse()`, split head keeping raw
cls/box[/coef/proto] convs, NHWC out) is faithful to the salvaged scripts.
Everything else was rebuilt to be consistent with the nv12/640 input and those
outputs:

- **calib.py** — letterbox to 640 (aspect-preserving + pad 114), RGB, routed
  through `common/calib_pack.py`.
- **config_det.yaml / config_seg.yaml** — nv12 rt, rgb train, `data_scale`
  1/255, `nash-m`.
- **compile.sh** — the las2 pattern with `--config` to pick det or seg.

`expected.json` records **nothing as measured** — `.hbm` size, quantisation
split, cosine, latency and mAP are all `null` pending a re-run. A few things to
confirm on that re-run: the exported input name (assumed `images`), that the
Ultralytics export really does not bake `/255`, and the pad value against bcdl's
actual YOLO preprocessing.

## Licence

Ultralytics YOLOE is **AGPL-3.0**. The weights, the exported ONNX, and any
`.hbm` compiled from them are all derivatives under that licence. This repo's
scripts are MIT, but **the model is not ours**:

- The compiled `.hbm` is **not committed or published** here.
- Redistributing an `.hbm` triggers AGPL obligations (source availability,
  same-licence).
- Commercial or closed-source use requires an **Ultralytics Enterprise licence**.

Rebuild locally from this recipe and treat the output as AGPL.
