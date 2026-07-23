# YOLOP — panoptic driving (detection + drivable area + lane lines), 640×640

One network, three road-scene heads: vehicle detection, drivable-area
segmentation, lane-line segmentation. Upstream:
[YOLOP](https://github.com/hustvl/YOLOP).

| build | BCDL consumes | note |
|---|---|---|
| `yolop_cut_nashm_640x640_nv12.hbm` | 3 raw det heads + 2 seg maps; BCDL does the anchor decode | the decode is **cut** off the graph — see below |

Upstream licence: not yet reviewed — check YOLOP's terms before redistribution.

## The one trap, and it is a silent, total failure

There is only one thing that goes wrong on this model, and it is the worst kind:
**the obvious build compiles perfectly and the detector sees nothing.**

YOLOP's published `yolop-640-640.onnx` bakes the detection **anchor decode** into
the graph with a `ScatterND`. Compile that ONNX as-is and every check stays
green — no warning, no error, a `.hbm` that loads and runs at full speed. But the
`ScatterND` that was supposed to write the decoded objectness and class columns
does not survive quantised lowering, so **those columns are never written**.
Every box comes back with zero confidence. The segmentation maps look fine, which
makes it more confusing: two of three heads work, and detection is silently dead.

**The fix is not calibration or precision. It is to cut the graph.** `export.py`
takes the downloaded parent ONNX (which is byte-identical to the release asset —
there is no re-export step) and re-emits it with its outputs set to the **three
raw detection heads** — the per-scale conv outputs, `[1,18,80,80]`,
`[1,18,40,40]`, `[1,18,20,20]` at strides 8/16/32 — plus the two segmentation
maps. The anchor decode then happens on the CPU inside BCDL, where it is a
handful of cheap ops on already-small tensors.

The three head tensors are found by their shape signature (channels
`3*(5+nc) = 18`, which no other layer here produces), so the cut survives a
re-download of the release ONNX. If a future release renames them, pass
`--det-heads a,b,c` with the names read off the graph in netron.

**The general lesson**, worth taking beyond YOLOP: when a head's decode does not
survive quantised lowering, don't fight the quantiser — keep the convolutional
trunk on the BPU and move the postprocess arithmetic to the host. The trunk is
where the compute is; the decode is cheap and exact in float on the CPU.

## Running it

```bash
# 1. download the parent ONNX from the upstream YOLOP release, then cut it:
python export.py --onnx yolop-640-640.onnx --out yolop_cut.onnx

# 2. calibrate on DRIVING frames (BDD100K or any dashcam set), 640x640
python calib.py --config config.yaml --images /path/to/driving_frames --limit 64

# 3. compile
./compile.sh --config config.yaml --gpu 2
```

`--gpu` is an nvidia-smi index handed to docker; the card must be Ampere or
newer. `jobs` lives in the config, not the command line.

Calibrate on road scenes, not COCO: all three heads key on driving-scene
statistics, and an object-centric or indoor calibration set moves the activation
ranges away from what the model runs on.

## Status

Recipe reconstructs the cut from the known YOLOP graph structure; the parent
ONNX is a direct download, so the only authored step is the surgery in
`export.py`. Not re-run through the toolchain in this repo — the shipped build is
board-validated inside BCDL's panoptic-driving example, and its latency/accuracy
live in BCDL's benchmark results rather than here.
