#!/usr/bin/env python3
"""[1/3] Export NeuFlow v2 to a static ONNX graph at 512x384.

    python export.py --ckpt neuflow_mixed.pth --out neuflow_512x384.onnx
    onnxslim neuflow_512x384.onnx neuflow_512x384_slim.onnx

Three things must be copied from upstream's own `infer.py` or the export is
wrong in ways that still produce a graph:

**1. Fuse Conv+BN before exporting.** The released checkpoints are NOT fused.
Skipping this leaves the BN as separate ops — it still runs, just slower and
with more quantisation surface than necessary.

**2. Call `init_bhwd()` BEFORE tracing.** It precomputes the correlation
lookup's `linspace`/`meshgrid` into constants for one fixed resolution. Trace
without it and those become live `Range`/`Shape` ops, so the graph carries
dynamic shapes and the compiler has nothing static to work with.

**3. Export in fp32.** The demo runs the model in half precision; tracing it
that way exports a half graph the toolchain will not take.

The iteration counts are export-time constants (1 at 1/16 scale, 8 at 1/8), so
the refinement loops come out **fully unrolled** — the graph has no control flow
at all, which is what makes this compilable in the first place.

Operator profile after slimming: 890 nodes, opset 17, no Shape/Range/Sqrt left.
The only red-flag operator is **GridSample x9** (the correlation lookup), and
that one is natively supported on this BPU. The 4 LayerNorms belong to the
1/16-scale cross-attention and are the place to watch bit width.
"""
import argparse

import torch


def fuse_conv_bn(model):
    """Fold every BatchNorm into the convolution before it.

    Upstream ships unfused checkpoints and fuses at load time in `infer.py`; an
    export that skips this is not the network anybody measured.
    """
    import torch.nn as nn

    for module in model.modules():
        prev = None
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d) and isinstance(prev[1] if prev else None,
                                                                nn.Conv2d):
                fused = torch.nn.utils.fusion.fuse_conv_bn_eval(prev[1], child)
                setattr(module, prev[0], fused)
                setattr(module, name, nn.Identity())
            prev = (name, child)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="neuflow_mixed.pth",
                    help="upstream checkpoint; `mixed` is the one the demo uses")
    ap.add_argument("--repo", default=".", help="a NeuFlow_v2 checkout, for the model class")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--out", default="neuflow_512x384.onnx")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, a.repo)
    from NeuFlow.neuflow import NeuFlow                       # noqa: E402

    model = NeuFlow().eval()
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    fuse_conv_bn(model)

    # BEFORE tracing: turns the correlation lookup's coordinate grids into
    # constants for this exact resolution.
    model.init_bhwd(1, a.height, a.width, "cpu")

    x = torch.zeros(1, 3, a.height, a.width, dtype=torch.float32)
    torch.onnx.export(model, (x, x.clone()), a.out,
                      input_names=["img0", "img1"], output_names=["flow"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print("wrote", a.out)
    print("now run:  onnxslim %s %s" % (a.out, a.out.replace(".onnx", "_slim.onnx")))


if __name__ == "__main__":
    main()
