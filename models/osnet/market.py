#!/usr/bin/env python3
"""Market-1501 retrieval evaluation — the TASK METRIC, plus the board bundle.

WHY A RETRIEVAL METRIC AND NOT JUST COSINE. This is the model the three-layer
cosine check cannot fully guard: a quantized embedding tower that has genuinely
degraded still emits plausible unit vectors, so its cosine against the float
reference can look fine while ranking has gotten worse. int8 PTQ here scores
cosine that does not scream "broken" yet ranks at Rank-1 51% against the float
model's 85%. Rank-1 / mAP over a labelled gallery is the number that says whether
the model still tells people apart — the only thing tracking cares about. This is
why expected.json declares a task_metric and not just a cosine gate.

This is a CROSS-DOMAIN evaluation on purpose: the weights were trained on MSMT17
and are scored here on Market-1501, which is the situation a tracker is actually
in — it never gets to train on the scene it runs in.

Protocol is the standard single-query one: for each query, gallery images of the
same identity from the SAME camera are excluded (they are trivially similar) and
identity -1 / 0000 (distractors and junk) are dropped.

Point it at the float ONNX (export.py) to establish the baseline, or at any ONNX
to score it. `--bundle` writes a few-MB .npz of the query/gallery file names,
their identity/camera labels, and this model's float embeddings, so the board can
reproduce the identical subset from the same jpgs and be compared like for like.

Usage:
    python market.py --root /path/to/Market-1501-v15.09.15 \\
                     --onnx osnet_ain_x1_0_256x128.onnx --n-ids 100 \\
                     --bundle market_ref.npz
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
NAME_RE = re.compile(r"^(-?\d+)_c(\d+)")


def preprocess(path: Path, width=128, height=256) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1))


def scan(folder: Path):
    """(path, pid, camid) for every valid crop in a Market-1501 folder."""
    out = []
    for p in sorted(folder.glob("*.jpg")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        pid, cam = int(m.group(1)), int(m.group(2))
        if pid in (-1, 0):  # distractors / junk
            continue
        out.append((p, pid, cam))
    return out


def subsample(items, n_ids: int, rng: np.random.Generator):
    """Keep the first `n_ids` identities (by a fixed shuffle) — the board runs
    a subset, and both sides must run the SAME subset for the comparison to
    mean anything."""
    if n_ids <= 0:
        return items
    pids = sorted({pid for _, pid, _ in items})
    rng.shuffle(pids)
    keep = set(pids[:n_ids])
    return [it for it in items if it[1] in keep]


def embed_all(sess, arrays: np.ndarray) -> np.ndarray:
    """Run the ONNX model one crop at a time (the graph is batch-1 by design)."""
    name = sess.get_inputs()[0].name
    # The graph emits [1,512,1,1] (4-D on purpose — see export.py), so flatten
    # rather than indexing a fixed rank.
    out = np.stack([sess.run(None, {name: a[None]})[0].reshape(-1) for a in arrays])
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def evaluate(q_feat, q_pid, q_cam, g_feat, g_pid, g_cam):
    """Standard single-query Market-1501 Rank-1 / Rank-5 / mAP."""
    sim = q_feat @ g_feat.T
    order = np.argsort(-sim, axis=1)

    r1 = r5 = 0
    aps = []
    for i in range(len(q_pid)):
        idx = order[i]
        # Same identity seen by the same camera is not a re-identification.
        valid = ~((g_pid[idx] == q_pid[i]) & (g_cam[idx] == q_cam[i]))
        idx = idx[valid]
        hit = g_pid[idx] == q_pid[i]
        if not hit.any():
            continue
        r1 += int(hit[0])
        r5 += int(hit[:5].any())
        ranks = np.flatnonzero(hit) + 1
        precision = (np.arange(len(ranks)) + 1) / ranks
        aps.append(precision.mean())

    n = len(aps)
    return dict(rank1=r1 / n, rank5=r5 / n, mAP=float(np.mean(aps)), queries=n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Market-1501-v15.09.15 directory")
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--n-ids", type=int, default=100,
                    help="identities to keep (0 = the full test set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bundle", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    query = subsample(scan(args.root / "query"), args.n_ids, rng)
    rng = np.random.default_rng(args.seed)  # same shuffle -> same identities
    gallery = subsample(scan(args.root / "bounding_box_test"), args.n_ids, rng)
    print(f"query={len(query)} gallery={len(gallery)} "
          f"ids={len({p for _, p, _ in query})}")

    q_arr = np.stack([preprocess(p) for p, _, _ in query])
    g_arr = np.stack([preprocess(p) for p, _, _ in gallery])
    q_pid = np.array([p for _, p, _ in query])
    q_cam = np.array([c for _, _, c in query])
    g_pid = np.array([p for _, p, _ in gallery])
    g_cam = np.array([c for _, _, c in gallery])

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    q_feat = embed_all(sess, q_arr)
    g_feat = embed_all(sess, g_arr)

    m = evaluate(q_feat, q_pid, q_cam, g_feat, g_pid, g_cam)
    print(f"float ONNX  Rank-1 {m['rank1']:.4f}  Rank-5 {m['rank5']:.4f}  "
          f"mAP {m['mAP']:.4f}  ({m['queries']} queries)")

    if args.bundle:
        # Deliberately WITHOUT the preprocessed arrays. The board re-derives them
        # from the same jpgs with the same code, which keeps the bundle at a few
        # MB and puts the board's own preprocessing inside the thing being
        # validated instead of bypassing it.
        np.savez_compressed(
            args.bundle,
            q_name=np.array([p.name for p, _, _ in query]),
            g_name=np.array([p.name for p, _, _ in gallery]),
            q_pid=q_pid, q_cam=q_cam, g_pid=g_pid, g_cam=g_cam,
            q_feat=q_feat, g_feat=g_feat)
        print(f"wrote {args.bundle} "
              f"({args.bundle.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
