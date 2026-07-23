#!/usr/bin/env python3
"""Calibration set for the REJECTED int8 PTQ build (config.yaml).

This exists to REPRODUCE THE FAILURE, not to ship a model. int8 PTQ on OSNet
compiles without a warning and returns well-formed unit vectors whose
Market-1501 Rank-1 is ~51% against the float model's ~85% (see README). The
shipped model is the QAT one (export.py -> crops.py -> qat.py -> deploy.py); this
path is here so you can watch int8 collapse for yourself and believe the QAT
apparatus is necessary.

WHAT THE MODEL SEES at runtime is a person box cut out of a frame and squashed to
128x256 — so that is what calibration sees too. Crops are drawn from TWO domains
(COCO people + KITTI pedestrians) for the same reason crops.py is: content
coverage matters more than frame count.

The arrays are FULLY PREPROCESSED here (RGB, /255, ImageNet mean/std, NCHW)
because config.yaml is `featuremap` / `no_preprocess`: there is no mean/scale in
the config for the packer to apply, so normalization is intrinsic to the input
and belongs in this script — exactly as it does for the other featuremap models.
The write still goes through common/calib_pack.pack(), which stays the one and
only way calibration files are produced, and records the manifest.

Usage:
    python calib.py --config config.yaml \\
                    --coco /path/to/coco --kitti /path/to/kitti
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from calib_pack import load_config, pack  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
H, W = 256, 128


def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    """BGR crop -> normalized 3x256x128 float32, exactly as the runtime does."""
    resized = cv2.resize(crop_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normed = (rgb - MEAN) / STD
    return np.ascontiguousarray(normed.transpose(2, 0, 1))


def _crop(img, x, y, w, h, min_h):
    ih, iw = img.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
    if x2 - x1 < 16 or y2 - y1 < min_h:
        return None
    return img[y1:y2, x1:x2]


def coco_crops(root: Path, want: int, min_h: int, rng: random.Random):
    data = json.loads((root / "annotations" / "instances_val2017.json").read_text())
    by_image = {im["id"]: im["file_name"] for im in data["images"]}
    people = [a for a in data["annotations"]
              if a["category_id"] == 1 and not a.get("iscrowd", 0) and a["bbox"][3] >= min_h]
    rng.shuffle(people)

    out, used = [], set()
    for a in people:
        if len(out) >= want:
            break
        # One crop per image: 30 crops from 30 scenes beat 30 of the same corner.
        if a["image_id"] in used:
            continue
        img = cv2.imread(str(root / "val2017" / by_image[a["image_id"]]))
        if img is None:
            continue
        c = _crop(img, *a["bbox"], min_h=min_h)
        if c is None:
            continue
        used.add(a["image_id"])
        out.append(("coco", by_image[a["image_id"]], c))
    return out


def kitti_crops(root: Path, want: int, min_h: int, rng: random.Random):
    label_dir, image_dir = root / "training" / "label_2", root / "training" / "image_2"
    files = sorted(label_dir.glob("*.txt"))
    rng.shuffle(files)

    out = []
    for lf in files:
        if len(out) >= want:
            break
        boxes = []
        for line in lf.read_text().splitlines():
            f = line.split()
            # occluded 0 = fully visible, 1 = partly; 2/3 are largely hidden.
            if f[0] not in ("Pedestrian", "Person_sitting") or int(f[2]) > 1:
                continue
            x1, y1, x2, y2 = (float(v) for v in f[4:8])
            if y2 - y1 >= min_h:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        if not boxes:
            continue
        img = cv2.imread(str(image_dir / (lf.stem + ".png")))
        if img is None:
            continue
        c = _crop(img, *rng.choice(boxes), min_h=min_h)
        if c is not None:
            out.append(("kitti", lf.stem + ".png", c))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--kitti", type=Path, required=True)
    ap.add_argument("--n-coco", type=int, default=150)
    ap.add_argument("--n-kitti", type=int, default=50)
    ap.add_argument("--min-height", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if len(cfg.inputs) != 1:
        sys.exit(f"{args.config}: expected a single input, got {len(cfg.inputs)}")
    spec = cfg.inputs[0]

    rng = random.Random(args.seed)
    crops = coco_crops(args.coco, args.n_coco, args.min_height, rng)
    crops += kitti_crops(args.kitti, args.n_kitti, args.min_height, rng)
    print(f"collected {len(crops)} crops "
          f"(coco={sum(1 for s, _, _ in crops if s == 'coco')}, "
          f"kitti={sum(1 for s, _, _ in crops if s == 'kitti')})")

    arrays = [preprocess(c) for _, _, c in crops]
    sources = [{"index": i, "domain": s, "source": name}
               for i, (s, name, _) in enumerate(crops)]
    pack(spec, arrays, sources=sources)


if __name__ == "__main__":
    main()
