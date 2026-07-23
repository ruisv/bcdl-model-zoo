#!/usr/bin/env python3
"""[1/3] PP-OCRv6 -> static-shape ONNX, plus the character dictionary.

PP-OCRv6 publishes ONNX directly, so unlike v5 there is **no paddle2onnx step**
— and that removes the trap v5 had, where the export script pointed at a
different model (a seal detector) and the shape-fixing call was commented out.
Here the upstream ONNX is the input and the only work is fixing the shapes.

That work is necessary: the published models are fully dynamic
(`x: [DynamicDimension.0, 3, DynamicDimension.1, DynamicDimension.2]`), and the
BPU compiler needs static shapes. onnxslim resolves them and constant-folds:

    det : 1x3x960x960  ->  1x1x960x960     (DB probability map)
    rec : 1x3x48x320   ->  1x40x18710      (CTC logits: 40 steps, 18710 classes)

The dictionary is extracted from the same `inference.yml` that ships beside the
weights, so it cannot drift from the model it decodes. Getting this pairing
wrong does not crash — it prints plausible-looking wrong characters.

BCDL's CTC decoder indexes `dict[argmax]` with blank at index 0, so the file is
written as:

    line 0            'blank'   (placeholder; never emitted)
    lines 1..N        characters, in upstream order
    last line         ' '       (space)

which makes the line count equal the model's class count exactly. PaddleOCR
stores only the N characters and adds blank and space itself; the +2 is applied
here and asserted against the model's real output width.

Usage:
    python export.py --src <PP-OCRv6 huggingface dir> --task det --size medium
    python export.py --src <PP-OCRv6 huggingface dir> --task rec --size medium
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import onnx
import yaml

# Sizes that share a dictionary. tiny is a DIFFERENT charset (6906 classes, no
# emoji tail), so a tiny model decoded with the medium dictionary produces
# confident nonsense. Keyed by the model's real output width, checked below.
DEFAULT_HW = {"det": (960, 960), "rec": (48, 320)}


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
            f"{model_classes} classes. Decoding would be silently offset — "
            f"check whether this size uses a different charset."
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="directory holding the PP-OCRv6_* huggingface repos")
    ap.add_argument("--task", required=True, choices=["det", "rec"])
    ap.add_argument("--size", default="medium", choices=["medium", "small", "tiny"])
    ap.add_argument("--hw", nargs=2, type=int, metavar=("H", "W"))
    ap.add_argument("--out", help="output .onnx (default derived from the above)")
    ap.add_argument("--dict-out", help="rec only: where to write the dictionary")
    a = ap.parse_args()

    h, w = a.hw or DEFAULT_HW[a.task]
    src_dir = os.path.join(a.src, f"PP-OCRv6_{a.size}_{a.task}_onnx")
    src = os.path.join(src_dir, "inference.onnx")
    if not os.path.exists(src):
        raise SystemExit(f"missing upstream ONNX: {src}")

    out = a.out or f"ppocrv6_{a.size}_{a.task}_{h}x{w}.onnx"

    # Name the file after what it is: the shape is part of the identity because
    # the compiled .hbm is shape-specific and its size scales with input area.
    print(f"[export] {a.size} {a.task}: {src}  ->  {out}  ({h}x{w})")
    subprocess.run(
        ["onnxslim", src, out, "--input-shapes", f"x:1,3,{h},{w}"],
        check=True, stdout=subprocess.DEVNULL,
    )

    # The OE toolchain (HBDK 4.x) rejects ONNX IR > 9: "The ir version of the
    # model is 10, which is greater than the maximum supported ir version of 9."
    # The v6 detector exports at IR10, so cap it. IR9 covers opset 14, and
    # onnxslim preserves the original IR, so this has to be done after it.
    m = onnx.load(out)
    if m.ir_version > 9:
        m.ir_version = 9
        onnx.save(m, out)
        print(f"[export] capped IR version -> 9")

    shapes = static_shapes(out)
    for name, dims in shapes.items():
        if any(isinstance(d, str) for d in dims):
            raise SystemExit(
                f"{name} is still dynamic after slimming: {dims}. The BPU "
                f"compiler needs fully static shapes."
            )
    print(f"[export] shapes {shapes}")

    if a.task == "rec":
        classes = list(shapes.values())[-1][-1]
        dict_out = a.dict_out or f"ppocr_keys_v6_{classes}.txt"
        n = write_dict(src_dir, dict_out, classes)
        print(f"[export] dictionary {n} entries -> {dict_out}")
        print("[export] pair this file with THIS model: the tiny size uses a "
              "different charset and mixing them decodes to nonsense.")


if __name__ == "__main__":
    main()
