#!/usr/bin/env python3
"""[2/4] Cut the unlabelled person-crop set that QAT self-distillation trains on.

NO LABELS ARE NEEDED. The student is trained to reproduce the FP32 teacher's
embedding, not to identify anyone, so any pile of person crops will do. That is
what makes this hours rather than days: no ReID training set, no identity
annotations, no triplet mining. See qat.py for why the defect is a
function-matching problem and not an identity-learning one.

Market-1501 is deliberately NOT a source here. It is the held-out benchmark
(market.py), and training on it — even without labels — would leave the final
Rank-1 unable to distinguish "the quantization was repaired" from "the model saw
the test domain".

Crops are written as JPEG rather than preprocessed .npy: 15k normalized float
arrays would be ~6 GB, while the jpgs are a couple hundred MB and qat.py's loader
preprocesses on the fly anyway.

Two domains on purpose: COCO people (indoor/outdoor, every scale and pose, often
partial) and KITTI pedestrians (street-level, small, uniform camera). A
single-domain crop set has cost this project real accuracy before.

Usage:
    python crops.py --coco /path/to/coco --kitti /path/to/kitti --out crops/
"""

import argparse
import json
import random
from pathlib import Path

import cv2

MIN_H = 48  # below this a crop carries no appearance worth matching


def coco_person_crops(root: Path, out: Path, limit: int, rng: random.Random) -> int:
    data = json.loads((root / "annotations" / "instances_val2017.json").read_text())
    by_image = {im["id"]: im["file_name"] for im in data["images"]}

    people = [a for a in data["annotations"]
              if a["category_id"] == 1 and not a.get("iscrowd", 0) and a["bbox"][3] >= MIN_H]
    rng.shuffle(people)

    # Group by image so each source jpg is decoded once, not once per person.
    per_image = {}
    for a in people[:limit]:
        per_image.setdefault(a["image_id"], []).append(a["bbox"])

    n = 0
    for image_id, boxes in per_image.items():
        img = cv2.imread(str(root / "val2017" / by_image[image_id]))
        if img is None:
            continue
        h, w = img.shape[:2]
        for j, (bx, by, bw, bh) in enumerate(boxes):
            x1, y1 = max(0, int(bx)), max(0, int(by))
            x2, y2 = min(w, int(bx + bw)), min(h, int(by + bh))
            if x2 - x1 < 16 or y2 - y1 < MIN_H:
                continue
            cv2.imwrite(str(out / f"coco_{image_id}_{j}.jpg"), img[y1:y2, x1:x2])
            n += 1
    return n


def kitti_person_crops(root: Path, out: Path, limit: int) -> int:
    label_dir, image_dir = root / "training" / "label_2", root / "training" / "image_2"
    n = 0
    for lf in sorted(label_dir.glob("*.txt")):
        if n >= limit:
            break
        boxes = []
        for line in lf.read_text().splitlines():
            f = line.split()
            if f[0] not in ("Pedestrian", "Person_sitting", "Cyclist") or int(f[2]) > 1:
                continue
            x1, y1, x2, y2 = (float(v) for v in f[4:8])
            if y2 - y1 >= MIN_H:
                boxes.append((x1, y1, x2, y2))
        if not boxes:
            continue
        img = cv2.imread(str(image_dir / (lf.stem + ".png")))
        if img is None:
            continue
        h, w = img.shape[:2]
        for j, (x1, y1, x2, y2) in enumerate(boxes):
            xa, ya = max(0, int(x1)), max(0, int(y1))
            xb, yb = min(w, int(x2)), min(h, int(y2))
            if xb - xa < 16 or yb - ya < MIN_H:
                continue
            cv2.imwrite(str(out / f"kitti_{lf.stem}_{j}.jpg"), img[ya:yb, xa:xb])
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--kitti", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-coco", type=int, default=12000)
    ap.add_argument("--n-kitti", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.out.glob("*.jpg"):
        f.unlink()

    rng = random.Random(args.seed)
    n_coco = coco_person_crops(args.coco, args.out, args.n_coco, rng)
    n_kitti = kitti_person_crops(args.kitti, args.out, args.n_kitti)
    print(f"wrote {n_coco + n_kitti} crops to {args.out} "
          f"(coco={n_coco}, kitti={n_kitti})")


if __name__ == "__main__":
    main()
