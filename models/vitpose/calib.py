#!/usr/bin/env python3
"""[2/3] Build a domain-matched PTQ calibration set for ViTPose-S whole-body.

Based on the salvaged `vitpose_prep.py` preprocessing, re-routed through the one
calibration entry point (common/calib_pack.py) like every model in this repo.

This is a TOP-DOWN pose model: it sees a single person cropped and letterboxed to
its aspect ratio, never a whole scene. So the calibration data must be person
crops, preprocessed EXACTLY as the runtime does, or the activation statistics are
gathered on a distribution the deployed model never sees. The preprocessing here
mirrors easy_ViTPose:

    detect people  ->  widen each box by PAD_BBOX px  ->  pad the crop to 3:4
    ->  resize to 192x256  ->  /255  ->  ImageNet z-score  ->  CHW float32.

The graph is compiled `featuremap` + `no_preprocess` (config.yaml): normalisation
is NOT done by the compiler, so it is done HERE and the already-normalised arrays
are handed to pack(). That is the featuremap contract — same as LAS2 in this repo.
calib_pack still owns the write (npy layout, dtype, manifest), so there remains
exactly one way calibration data reaches disk.

Domain match beats sample count: 20-100 crops from images resembling the
deployment scene are worth more than a large out-of-domain pile. Person crops from
COCO val are a reasonable public default.

Usage:
    python calib.py --config config.yaml \\
                    --images /path/to/person_images \\
                    --detector /path/to/yolo_person.pt --limit 64
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402

IN_W, IN_H = 192, 256                                   # (W,H); NCHW output is 1x3x256x192
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PAD_BBOX = 10          # easy_ViTPose widens the detector box by 10 px per side
ASPECT = IN_W / IN_H   # 0.75


def pad_to_aspect(img, aspect=ASPECT):
    """Zero-pad an RGB crop to the model's 3:4 aspect, centred. Mirrors
    easy_ViTPose.vit_utils.inference.pad_image."""
    h, w = img.shape[:2]
    if w / h < aspect:
        tw = int(aspect * h)
        left = (tw - w) // 2
        return np.pad(img, ((0, 0), (left, tw - w - left), (0, 0)))
    th = int(w / aspect)
    top = (th - h) // 2
    return np.pad(img, ((top, th - h - top), (0, 0), (0, 0)))


def preprocess(crop_rgb):
    """RGB crop -> normalised CHW float32, the exact tensor the ONNX consumes.
    easy_ViTPose feeds the pose model RGB (it only flips to BGR for the detector)."""
    x = cv2.resize(crop_rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1).astype(np.float32))


def person_crops(a):
    """Yield preprocessed person crops from the image directory."""
    from ultralytics import YOLO
    det = YOLO(a.detector)
    n = 0
    for name in sorted(os.listdir(a.images)):
        img = cv2.imread(os.path.join(a.images, name))   # BGR
        if img is None:
            continue
        r = det.predict(img, classes=[0], conf=a.conf, verbose=False)[0]
        for b in r.boxes.xyxy.cpu().numpy().round().astype(int):
            x1, y1, x2, y2 = b
            x1 = max(0, x1 - PAD_BBOX); x2 = min(img.shape[1], x2 + PAD_BBOX)
            y1 = max(0, y1 - PAD_BBOX); y2 = min(img.shape[0], y2 + PAD_BBOX)
            if x2 - x1 < 24 or y2 - y1 < 48:            # skip crops too small to be a person
                continue
            crop = pad_to_aspect(img[y1:y2, x1:x2, ::-1])  # BGR -> RGB
            yield preprocess(crop), name
            n += 1
            if a.limit and n >= a.limit:
                return


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config.yaml")
    ap.add_argument("--images", required=True, help="directory of scene images with people")
    ap.add_argument("--detector", required=True, help="person detector weights (e.g. a YOLO .pt)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=64, help="max crops (0 = all)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if len(cfg.inputs) != 1:
        sys.exit(f"{a.config}: expected 1 input, got {len(cfg.inputs)}")
    spec = cfg.inputs[0]
    # featuremap + no_preprocess: pack() will NOT re-normalise, which is why the
    # z-score is applied above. Guard against someone flipping the config to a
    # mean/scale norm_type, which would then double-normalise the crops.
    if spec.norm_type not in (None, "", "no_preprocess"):
        sys.exit(f"{spec.name}: norm_type={spec.norm_type!r}, but calib.py already "
                 f"applies the ImageNet z-score. Keep the graph no_preprocess.")

    arrays, sources = [], []
    for i, (arr, src) in enumerate(person_crops(a)):
        arrays.append(arr)
        sources.append({"index": i, "source": src})
    if not arrays:
        sys.exit("no person crops produced — check --images / --detector / --conf")

    print(f"[calib] {len(arrays)} person crops, {IN_H}x{IN_W}, ImageNet-normalised")
    pack(spec, arrays, fmt="npy", sources=sources)


if __name__ == "__main__":
    main()
