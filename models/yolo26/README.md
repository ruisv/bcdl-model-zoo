# YOLO26s — object detection, 640×640

BCDL's main detector. Upstream: [Ultralytics](https://github.com/ultralytics/ultralytics)
YOLO26s.

| build | BCDL consumes | note |
|---|---|---|
| `yolo26s_det_nashm_640x640_nv12.hbm` | 6 raw split-head tensors; BCDL decodes | AGPL — **recipe only, no binary** |

**Licence: AGPL-3.0.** Hosting a compiled `.hbm` (a derivative of AGPL weights)
pulls the copyleft obligations onto whoever distributes it, so this repo ships the
**recipe only** — build your own `.hbm` from your own licensed weights. Commercial
use needs Ultralytics' Enterprise licence.

## The recipe: cut the decode, keep the convolutions

Same principle as the other YOLO-family heads here. The stock `Detect` head, after
its `cv2` (box) and `cv3` (class) convs, **decodes** on-graph: DFL softmax+integral
over `reg_max`, an anchor grid built with meshgrid/arange, stride multiply,
`dist2bbox`, class sigmoid, cross-scale concat and NMS. Those are
reshape/gather/reduce/meshgrid ops — a mix of poorly-quantised and non-BPU-native
operators, and they bake a fixed anchor layout into the compiled binary.

`export.py` replaces the head's forward with `bpu_forward`, which stops at the
**raw per-scale conv outputs** (`[cls, box]` per scale, permuted to NHWC to match
the BPU's native layout). BCDL then does the DFL integral, anchor math, sigmoid
and NMS on the CPU. The BPU runs pure convolution — clean int8 — and one `.hbm`
layout serves. The box channel count (`4*reg_max = 64`) lets BCDL's decoder
auto-detect `reg_max = 16`; nothing is hard-coded.

This is exactly the [YOLOE detection cut](../yoloe/) minus the open-vocabulary
text-embedding fuse: YOLO26 has a fixed class set, so its `cv3` convs already emit
class logits.

Normalisation is not baked: the graph expects RGB in `[0,1]`, Ultralytics does not
add `/255`, so the config's `data_scale 1/255` is the whole normalisation, applied
identically to the runtime nv12 path and (through `calib_pack`) to the calibration
set.

## Running it

```bash
# 1. checkpoint -> raw split-head ONNX
python export.py --weights yolo26s.pt --output yolo26s_det_bpu.onnx --imgsz 640

# 2. calibrate on general detection frames (COCO val or similar)
python calib.py --config config.yaml --images /path/to/coco_val --limit 64

# 3. compile
./compile.sh --config config.yaml --gpu 2
```

## Status

**Reconstructed, and the highest-risk recipe in this set.** This is the one model
whose `hb_compile` config was never saved — the original build passed its
parameters inline on the command line. `config.yaml` here is reconstructed from
the shipped filename, the known nv12/640 input contract, and the split-head
export; `export.py` mirrors the YOLOE detection cut, which *is* verified. Compile
it and compare against a board run before trusting the config. Not re-run through
the toolchain in this repo.
