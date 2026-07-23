#!/usr/bin/env python3
"""[1/3] PP-OCRv5 (det + rec + cls) -> static-shape ONNX, plus the rec dictionary.

Unlike v6 (which publishes ONNX directly), **v5 only ships Paddle inference
models**, so every sub-model has to go through `paddle2onnx` first. That step is
exactly where v5's original recipe rotted, and this script exists to get it
right. Two independent bugs were baked into the surviving export scripts:

  1. WRONG MODEL. `export_onnx_det.sh` pointed at `PP-OCRv4_server_seal_det`
     -- a *seal* detector, and the wrong major version -- instead of the general
     text detector `PP-OCRv5_server_det`. paddle2onnx converts it without a
     murmur: you get a valid ONNX of the wrong network, which then compiles,
     loads, and detects nothing on ordinary document text.

  2. SHAPE-FIX COMMENTED OUT. The `paddle2onnx.optimize --input_shape_dict`
     call that pins the graph to a static shape (det 1x3x960x960, rec
     1x3x48x320) was commented out in the det and rec scripts. A naive re-run
     produces a *dynamic-shape* ONNX, which the BPU compiler will not take.

So this script hard-codes the correct model directory per task and always runs
the shape fix. Only the cls script (`PP-LCNet_x1_0_textline_ori`, 80x160) was
clean in the salvage; the det/rec ones were not.

Pipeline per sub-model:

    paddle2onnx            Paddle inference model  ->  raw ONNX (dynamic)
    paddle2onnx.optimize   pin input shape         ->  static ONNX
    onnxslim               simplify + constant-fold ->  <prefix>.onnx
    ir cap                 IR version -> 9 (HBDK rejects IR > 9)

Shapes after the fix:

    det : 1x3x960x960  ->  1x1x960x960     (DB probability map)
    rec : 1x3x48x320   ->  1xTx<classes>   (CTC logits; classes below)
    cls : 1x3x80x160   ->  1x2             (textline orientation: 0 deg / 180 deg)

The rec dictionary is extracted from the same `inference.yml` that ships beside
the weights, so it cannot drift from the model it decodes. Getting this pairing
wrong does not crash -- it prints plausible-looking wrong characters. BCDL's CTC
decoder indexes `dict[argmax]` with blank at index 0, so the file is written as:

    line 0            'blank'   (placeholder; never emitted)
    lines 1..N        characters, in upstream order
    last line         ' '       (space)

which makes the line count equal the model's class count exactly. PaddleOCR
stores only the N characters and adds blank and space itself; the +2 is applied
here and asserted against the model's real output width.

Usage:
    python export.py --src <dir of PP-OCRv5 *_infer models> --task det
    python export.py --src <dir of PP-OCRv5 *_infer models> --task rec
    python export.py --src <dir of PP-OCRv5 *_infer models> --task cls
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import onnx
import yaml

# Correct upstream Paddle inference-model directories. The det entry is the
# whole point: the salvaged script named a seal detector here.
MODEL_DIR = {
    "det": "PP-OCRv5_server_det_infer",
    "rec": "PP-OCRv5_server_rec_infer",
    "cls": "PP-LCNet_x1_0_textline_ori_infer",
}

DEFAULT_HW = {"det": (960, 960), "rec": (48, 320), "cls": (80, 160)}

# Output prefix == what the .hbm will be called. Shape is part of the identity
# because the compiled model is shape-specific and its size scales with area.
PREFIX = {
    "det": "ppocrv5_server_det_960x960",
    "rec": "ppocrv5_server_rec_48x320",
    "cls": "ppocrv5_lcnet_cls_80x160",
}


def run(cmd: list) -> None:
    print("[export] $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def static_shapes(model_path: str) -> dict:
    m = onnx.load(model_path)
    out = {}
    for group in (m.graph.input, m.graph.output):
        for v in group:
            out[v.name] = [
                (d.dim_value if d.dim_value else (d.dim_param or "?"))
                for d in v.type.tensor_type.shape.dim
            ]
    return out


def write_dict(src_dir: str, out_path: str, model_classes: int) -> int:
    """Extract the charset from inference.yml into BCDL's dictionary format."""
    with open(os.path.join(src_dir, "inference.yml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pp = cfg.get("PostProcess") or {}
    if pp.get("name") != "CTCLabelDecode":
        raise SystemExit(
            f"expected CTCLabelDecode post-processing, got {pp.get('name')!r}. "
            f"BCDL's recogniser is a greedy CTC decoder and does not implement "
            f"anything else."
        )
    chars = pp.get("character_dict")
    if not chars:
        raise SystemExit(f"no character_dict in {src_dir}/inference.yml")

    # blank at 0, characters, then space -> exactly the model's class count.
    lines = ["blank"] + list(chars) + [" "]
    if len(lines) != model_classes:
        raise SystemExit(
            f"dictionary/model mismatch: built {len(lines)} entries "
            f"({len(chars)} chars + blank + space) but the model emits "
            f"{model_classes} classes. Decoding would be silently offset -- "
            f"the v5 server charset should be ~18385; check the model."
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="directory holding the PP-OCRv5 *_infer Paddle models")
    ap.add_argument("--task", required=True, choices=["det", "rec", "cls"])
    ap.add_argument("--hw", nargs=2, type=int, metavar=("H", "W"))
    ap.add_argument("--out", help="output .onnx (default derived from the task)")
    ap.add_argument("--dict-out", help="rec only: where to write the dictionary")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--keep-intermediate", action="store_true")
    a = ap.parse_args()

    h, w = a.hw or DEFAULT_HW[a.task]
    src_dir = os.path.join(a.src, MODEL_DIR[a.task])
    if not os.path.isdir(src_dir):
        raise SystemExit(f"missing Paddle model dir: {src_dir}")
    # Guard against the exact salvage bug: a seal detector where the general
    # text detector belongs.
    if a.task == "det" and "seal" in src_dir.lower():
        raise SystemExit(
            f"{src_dir} looks like a SEAL detector. v5 det must be the general "
            f"text detector PP-OCRv5_server_det, not *_seal_det -- this is the "
            f"trap the original script fell into.")

    out = a.out or f"{PREFIX[a.task]}.onnx"
    raw = f"_{a.task}_raw.onnx"
    shaped = f"_{a.task}_shaped.onnx"

    # 1. Paddle inference model -> raw (dynamic-shape) ONNX.
    run(["paddle2onnx",
         "--model_dir", src_dir,
         "--model_filename", "inference.json",
         "--params_filename", "inference.pdiparams",
         "--save_file", raw,
         "--opset_version", str(a.opset),
         "--enable_onnx_checker", "True"])

    # 2. Pin the input shape. THIS is the step that was commented out for det
    #    and rec; without it the graph stays dynamic and the BPU compiler
    #    rejects it.
    run([sys.executable, "-m", "paddle2onnx.optimize",
         "--input_model", raw,
         "--output_model", shaped,
         "--input_shape_dict", "{'x':[1,3,%d,%d]}" % (h, w)])

    # 3. Simplify + constant-fold, and re-assert the static shape.
    run(["onnxslim", shaped, out, "--input-shapes", f"x:1,3,{h},{w}"])

    # 4. HBDK 4.x rejects ONNX IR > 9. onnxslim preserves the source IR, so cap
    #    it afterwards. IR9 covers opset 20, so this is a header change only.
    m = onnx.load(out)
    if m.ir_version > 9:
        m.ir_version = 9
        onnx.save(m, out)
        print("[export] capped IR version -> 9")

    shapes = static_shapes(out)
    for name, dims in shapes.items():
        if any(isinstance(d, str) for d in dims):
            raise SystemExit(
                f"{name} is still dynamic after slimming: {dims}. The BPU "
                f"compiler needs fully static shapes -- was the shape fix run?")
    print(f"[export] {a.task}: {src_dir}  ->  {out}  ({h}x{w})")
    print(f"[export] shapes {shapes}")

    if not a.keep_intermediate:
        for f in (raw, shaped):
            if os.path.exists(f):
                os.remove(f)

    if a.task == "rec":
        classes = int(list(shapes.values())[-1][-1])
        dict_out = a.dict_out or f"ppocr_keys_v5_{classes}.txt"
        n = write_dict(src_dir, dict_out, classes)
        print(f"[export] dictionary {n} entries -> {dict_out}")
        print("[export] pair this file with THIS model: a dict of the wrong "
              "class count decodes to confident nonsense.")


if __name__ == "__main__":
    main()
