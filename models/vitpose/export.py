#!/usr/bin/env python3
"""[1/3] Export ViTPose-S whole-body (133 keypoints) -> BPU-friendly ONNX, 256x192.

RECONSTRUCTED. Read this before trusting it.

  The original export script for this model did NOT survive. What survived is the
  downstream salvage: a prep step that static-shapes an ALREADY-EXPORTED
  `vitpose-s-wholebody.onnx` (batch dim -> 1, then onnxslim), and a float
  reference runner. Neither one produces the ONNX from the checkpoint, so this
  file rebuilds that missing first step from (a) the input/output contract the
  salvage pins down and (b) the standard easy_ViTPose export path. It is faithful
  to the METHOD, not copied from the lost original — treat the exact config module
  name, output-tensor name and head variant as reconstruction, and verify the
  emitted graph's I/O signature against `expected.json` before compiling.

What the salvage DOES pin down (these are not guesses):
  * input  : 1x3x256x192 NCHW, name "input_0"  (config.yaml, vitpose_prep.py)
  * preprocessing the calibration/runtime path uses (easy_ViTPose convention):
    pad the person box to 3:4, resize to 192x256, /255, ImageNet z-score. The
    graph is fed `featuremap` with `no_preprocess`, so normalisation lives
    OUTSIDE the ONNX (see calib.py / the on-board preprocessor), not in it.
  * task : COCO-WholeBody, 133 keypoints (17 body + 6 foot + 68 face + 42 hand).

What is reconstructed (flag as uncertain):
  * The head emits top-down heatmaps at input/4 = 64x48, i.e. output [1,133,64,48].
    64x48 is the standard ViTPose classic/simple-decoder resolution at 256x192;
    the salvaged reference runner saves the heatmap but never prints its dims, so
    confirm the exact HxW from the exported graph.
  * output tensor name "heatmaps" — cosmetic; rename to match your decoder.

ViT-on-BPU consideration (this is a transformer, unlike the CNNs in this repo):
  The ViT backbone is LayerNorm-heavy. On the Nash BPU, LayerNorm is the operator
  whose int8 quantisation hurts most, and the compiler already knows this: with no
  `optimization` directive it PROMOTES the LayerNorm-adjacent layers to int16 on
  its own (the same behaviour observed on the PP-OCRv6 recogniser in this repo).
  So config.yaml deliberately does NOT force a global bit width — read
  `Output Data Type` in `<prefix>_node_info.csv` to see the split the compiler
  chose, and only hand-write a mixed config to OVERRIDE it, never to discover it.
  Do not reflexively reach for `set_all_nodes_int16`: measure first.

Heatmap decode is NOT in this graph. Argmax / soft-argmax over the 64x48 maps and
the crop->image coordinate un-mapping happen on the CPU downstream (the salvaged
reference runner keeps them out of the compare so a preprocessing difference and a
quantisation difference stay distinguishable). Keep it that way.

Runs on the convert host (torch env with the easy_ViTPose checkout importable).

Usage:
    python export.py --repo /path/to/easy_ViTPose \\
                     --cfg  /path/to/easy_ViTPose/configs/ViTPose_small_wholebody_256x192.py \\
                     --ckpt /path/to/vitpose-s-wholebody.pth \\
                     --out  vitpose_s_wholebody_static.onnx
"""
import argparse
import importlib.util
import os
import sys

import onnx
import torch

IN_H, IN_W = 256, 192          # NCHW: matches config.yaml input_shape 1x3x256x192
NUM_JOINTS = 133               # COCO-WholeBody


def load_model_cfg(cfg_path):
    """easy_ViTPose keeps its model definition in a python config module that
    exposes a `model` dict (backbone + keypoint_head). Import it by file path so
    this does not depend on the checkout being on sys.path as a package."""
    spec = importlib.util.spec_from_file_location("vitpose_cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "model"):
        sys.exit(f"{cfg_path}: expected a top-level `model` dict (easy_ViTPose config)")
    return mod.model


class Heatmaps(torch.nn.Module):
    """Fix the forward to the one thing the runtime consumes: the raw heatmaps.

    easy_ViTPose's ViTPose.forward already returns the head's heatmap tensor in
    eval mode; this wrapper only pins that so the exported graph has a single,
    named output and no train-time branches leak in."""

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        return self.model(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="easy_ViTPose checkout (for imports)")
    ap.add_argument("--cfg", required=True, help="ViTPose_small_wholebody config .py")
    ap.add_argument("--ckpt", required=True, help="ViTPose-S wholebody checkpoint (.pth)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--ir", type=int, default=9,
                    help="cap IR at 9: HBDK 4.x rejects IR10 (header change, not a downgrade)")
    ap.add_argument("--slim", action="store_true", default=True,
                    help="run onnxslim to concretise symbolic Shape/Gather output dims")
    ap.add_argument("--no-slim", dest="slim", action="store_false")
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    # easy_ViTPose's model class. Path has shifted between revisions of the repo;
    # try the two locations seen in the wild rather than pinning one.
    try:
        from easy_ViTPose.vit_models.model import ViTPose            # newer layout
    except ImportError:
        from vit_models.model import ViTPose                         # flat checkout

    model = ViTPose(load_model_cfg(a.cfg))
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    # strict=True on purpose: a silently-partial load exports a half-random head
    # that still emits plausible-looking heatmaps.
    model.load_state_dict(state, strict=True)
    model.eval()

    net = Heatmaps(model).eval()
    dummy = torch.zeros(1, 3, IN_H, IN_W)
    with torch.no_grad():
        hm = net(dummy)
    print(f"eval forward -> heatmaps {tuple(hm.shape)}")
    # The channel count is the contract; the 64x48 spatial size is reconstructed.
    assert hm.ndim == 4 and hm.shape[1] == NUM_JOINTS, \
        f"expected [1,{NUM_JOINTS},H,W], got {tuple(hm.shape)}"
    if tuple(hm.shape[2:]) != (64, 48):
        print(f"  NOTE: heatmap spatial size {tuple(hm.shape[2:])} != reconstructed "
              f"(64,48). Update expected.json output_shape to match this graph.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            net, dummy, a.out,
            input_names=["input_0"], output_names=["heatmaps"],
            opset_version=a.opset, do_constant_folding=True,
        )

    # onnxslim resolves the symbolic output dims the opset Shape/Gather arithmetic
    # leaves behind; hb_compile's shape inference needs them concrete. (In the
    # salvage this was a separate prep step; folded in here so export.py alone
    # yields the compile-ready static graph.)
    if a.slim:
        tmp = a.out + ".slim.onnx"
        if os.system(f"onnxslim {a.out} {tmp}") == 0 and os.path.exists(tmp):
            os.replace(tmp, a.out)
        else:
            print("  WARNING: onnxslim did not run/produce output; leaving graph un-slimmed")

    # IR must be <= 9 or HBDK rejects it. onnxslim/exporters emit IR10 by default;
    # this is a header rewrite (IR9 covers up to opset 20), not an operator downgrade.
    m = onnx.load(a.out)
    if m.ir_version > a.ir:
        m.ir_version = a.ir
        onnx.save(m, a.out)

    out = onnx.load(a.out)

    def shape(v):
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in v.type.tensor_type.shape.dim]

    print("static in :", [(v.name, shape(v)) for v in out.graph.input])
    print("static out:", [(v.name, shape(v)) for v in out.graph.output])
    print(f"[export] OK -> {a.out}  (ViTPose-S wholebody {IN_H}x{IN_W}, "
          f"opset{a.opset}, IR{out.ir_version})")


if __name__ == "__main__":
    main()
