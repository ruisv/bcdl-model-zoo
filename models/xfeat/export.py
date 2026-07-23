#!/usr/bin/env python3
"""[1/3] Export XFeat -> BPU-friendly ONNX at a fixed 640x480.

Upstream: verlab/accelerated_features ("XFeat: Accelerated Features for
Lightweight Image Matching", Apache-2.0). Run on the convert host in a torch env
that can import the upstream `modules` package.

XFeat is a sparse local-feature network: a shared trunk feeds three heads —
a dense 64-d descriptor map (`feats`), a keypoint-logit map (`keypoints`) and a
reliability heatmap (`heatmap`). Keypoint selection, NMS and descriptor sampling
are NOT in the graph; they are CPU post-processing that BCDL owns. This script
exports only the three maps.

Two edits to the graph, both verified numerically here rather than assumed:

  1. InstanceNorm2d comes OUT of the graph and moves to CPU preprocessing. It is
     a per-image, data-dependent statistic — exactly the kind of thing that
     quantises badly — and lifting it out also puts the model input in a
     standardised domain. The exported model therefore takes an ALREADY
     grayscale + instance-normalised [1,1,H,W] tensor. That is why config.yaml
     declares a single-channel `featuremap` input with `no_preprocess`: the
     board feeds a pre-standardised tensor, not pixels.

  2. `_unfold2d` (the keypoint head's window gather) becomes F.pixel_unshuffle,
     which exports to a single SpaceToDepth. For the C=1 input the two are the
     same permutation: both put window offset (dy,dx) at channel dy*8+dx. This
     is verified with torch.equal below before anything is exported — the same
     technique LAS2 uses for its unfold rewrite (replace an operator torch.onnx
     cannot lower with an equivalent standard-primitive one, then prove
     equivalence numerically instead of trusting it).

The compiler config consumes the *slimmed* graph (xfeat_640x480_slim.onnx). Run
onnxslim (or onnx-simplifier) on the file this writes before compiling:

    onnxslim xfeat_640x480.onnx xfeat_640x480_slim.onnx

Slimming folds the interpolate/space-to-depth constant shapes; the un-slimmed
graph also compiles but the slimmed one is what was shipped.

Usage:
    python export.py --repo /path/to/accelerated_features \\
                     --weights /path/to/accelerated_features/weights/xfeat.pt \\
                     --hw 480 640 --out xfeat_640x480.onnx
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser(description="Export XFeat to BPU-friendly ONNX")
    ap.add_argument("--repo", required=True, help="accelerated_features checkout")
    ap.add_argument("--weights", default=None,
                    help="xfeat.pt (default: <repo>/weights/xfeat.pt)")
    ap.add_argument("--hw", nargs=2, type=int, default=[480, 640], metavar=("H", "W"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--ir", type=int, default=9,
                    help="cap IR version; HBDK 4.x rejects IR > 9")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    H, W = a.hw
    sys.path.insert(0, a.repo)
    from modules.model import XFeatModel  # noqa: E402

    net = XFeatModel().eval()
    weights = a.weights or os.path.join(a.repo, "weights", "xfeat.pt")
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict(sd)

    # --- 1. is pixel_unshuffle really the same permutation as _unfold2d? ------
    probe = torch.randn(1, 1, 32, 40)
    assert torch.equal(net._unfold2d(probe, ws=8), F.pixel_unshuffle(probe, 8)), \
        "pixel_unshuffle is NOT the same permutation as _unfold2d"
    print("unfold2d == pixel_unshuffle: OK")

    class XFeatBpu(torch.nn.Module):
        """XFeat with the normalisation lifted out and unfold as pixel_unshuffle."""

        def __init__(self, net):
            super().__init__()
            self.n = net

        def forward(self, x):                   # x: [1,1,H,W], already normalised
            n = self.n
            x1 = n.block1(x)
            x2 = n.block2(x1 + n.skip1(x))
            x3 = n.block3(x2)
            x4 = n.block4(x3)
            x5 = n.block5(x4)
            x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
            x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
            feats = n.block_fusion(x3 + x4 + x5)
            heatmap = n.heatmap_head(feats)
            keypoints = n.keypoint_head(F.pixel_unshuffle(x, 8))
            return feats, keypoints, heatmap

    bpu = XFeatBpu(net).eval()

    # --- 2. does the rewritten module match the original end to end? ----------
    raw = torch.randn(1, 3, H, W)
    with torch.no_grad():
        ref = net(raw)
        pre = net.norm(raw.mean(dim=1, keepdim=True))   # the part moving to CPU
        got = bpu(pre)
    for name, x, y in zip(("feats", "keypoints", "heatmap"), ref, got):
        err = (x - y).abs().max().item()
        print("  %-9s max|diff| %.3e" % (name, err))
        assert err < 1e-4, f"{name} diverged"
    print("rewrite == original: OK")

    # --- 3. export ------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            bpu, torch.randn(1, 1, H, W), a.out,
            input_names=["image"], output_names=["feats", "keypoints", "heatmap"],
            opset_version=a.opset)

    import onnx  # noqa: E402
    m = onnx.load(a.out)
    m.ir_version = a.ir      # HBDK 4.x rejects IR > 9; header change, not a downgrade
    onnx.save(m, a.out)

    print("ops:", dict(collections.Counter(n.op_type for n in m.graph.node)))

    def shape(v):
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in v.type.tensor_type.shape.dim]

    print("in :", [(v.name, shape(v)) for v in m.graph.input])
    print("out:", [(v.name, shape(v)) for v in m.graph.output])
    print(f"[export] OK -> {a.out}  (XFeat {H}x{W}, opset{a.opset}, IR{a.ir})")
    print("[export] now slim it:  onnxslim %s xfeat_%dx%d_slim.onnx" % (a.out, W, H))

    # Optional: confirm the saved ONNX matches torch on the CPU runtime.
    try:
        import onnxruntime as ort  # noqa: E402
        sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
        o = sess.run(None, {"image": pre.numpy()})
        for name, x, y in zip(("feats", "keypoints", "heatmap"), ref, o):
            print("  onnx %-9s max|diff| vs torch %.3e"
                  % (name, np.abs(x.numpy() - y).max()))
    except ImportError:
        print("[export] onnxruntime not present; skipped ORT cross-check")


if __name__ == "__main__":
    main()
