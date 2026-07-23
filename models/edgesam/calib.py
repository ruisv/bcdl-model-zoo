#!/usr/bin/env python3
"""[2/3] PTQ calibration for EdgeSAM: the encoder, and the two multi-input decoders.

Two calibration jobs, selected by --part:

  --part encoder
      SAM-preprocessed 1024x1024 images, as ONE `featuremap` input (raw_image).
      Letterbox = resize longest side to 1024, then pad to 1024x1024. Normalisation
      is BAKED into the encoder graph (see export.py), so the calibration tensor is
      raw pixels in [0,255] and pack() passes it through unchanged. If you exported
      with --no-bake-norm, add --cpu-normalize here so calibration matches what the
      board will feed.

  --part decoder --config config_decoder_{sp1,bp2}.yaml
      The interesting case: THREE inputs, and they are NOT independent.
        image_embeddings [1,256,64,64]  <- run the real encoder on each calib image
        point_coords     [1,N,2]        <- prompts sampled in the 1024 frame
        point_labels     [1,N]          <- fixed per build (sp1 {1}; bp2 {2,3})

      image_embeddings MUST come from the encoder, not random noise. The decoder's
      whole job is to attend prompts against a real embedding manifold; calibrating
      it on random tensors gathers activation statistics on a distribution the
      deployed decoder never sees, and — as everywhere in this repo — it compiles
      clean, runs at full speed, and segments to garbage. So this path loads the
      exported encoder ONNX, runs it on the same calibration images, and feeds its
      real outputs as the embedding calibration set. The prompts are sampled inside
      each image's valid (non-padded) region so the coordinate statistics are
      realistic too.

Prompt arity is read from the config filename (sp1 -> 1 point, labels {1};
bp2 -> 2 points, labels {2,3}); override with --prompt.

All three inputs are written through common/calib_pack.py: it is the one
calibration-writing path and it records the manifest. These are `no_preprocess`
featuremap inputs so pack() applies no normalisation; it also adds a leading batch
axis on save, which is harmless — hb_compile reshapes each .npy to the declared
input shape by ELEMENT COUNT, so the on-disk rank does not matter as long as the
count matches (256*64*64 for the embedding, 2N for coords, N for labels).

Usage:
    # encoder
    python calib.py --config config_encoder.yaml --part encoder \\
                    --images /path/to/images --limit 64

    # decoders (needs the exported encoder ONNX to produce real embeddings)
    python calib.py --config config_decoder_sp1.yaml --part decoder \\
                    --images /path/to/images --encoder-onnx edge_sam_encoder_bpu.onnx
    python calib.py --config config_decoder_bp2.yaml --part decoder \\
                    --images /path/to/images --encoder-onnx edge_sam_encoder_bpu.onnx
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402

IMG_SIZE = 1024
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# SAM point-label convention, per build.
PROMPTS = {
    "sp1": {"npoints": 1, "labels": [1.0]},         # one foreground point
    "bp2": {"npoints": 2, "labels": [2.0, 3.0]},    # box: top-left, bottom-right
}


def collect_images(pat, limit):
    files = (sorted(glob.glob(os.path.join(pat, "**", "*"), recursive=True))
             if os.path.isdir(pat) else sorted(glob.glob(pat)))
    files = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    if limit:
        files = files[:limit]
    if not files:
        sys.exit(f"no images found under {pat!r}")
    return files


def sam_letterbox(bgr, cpu_normalize):
    """Resize longest side to 1024, pad to 1024x1024. Returns (chw, new_h, new_w).

    chw is RGB CHW float32. Raw [0,255] by default (encoder graph bakes the
    normalisation); --cpu-normalize applies SAM mean/std here instead.
    """
    h, w = bgr.shape[:2]
    scale = IMG_SIZE / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    canvas[:nh, :nw] = resized.astype(np.float32)
    rgb = canvas[:, :, ::-1]                       # BGR -> RGB
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32)
    if cpu_normalize:
        chw = (chw - PIXEL_MEAN) / PIXEL_STD
    return chw, nh, nw


def sample_prompt(nh, nw, npoints, labels, rng):
    """Sample a plausible prompt inside the valid (non-padded) region, 1024 frame.

    sp1: one point anywhere in the image. bp2: two corners of a random box, in
    (top-left, bottom-right) order to match labels {2,3}.
    """
    if npoints == 1:
        pts = np.array([[rng.uniform(0, nw), rng.uniform(0, nh)]], dtype=np.float32)
    elif npoints == 2:
        x0, x1 = sorted(rng.uniform(0, nw, size=2))
        y0, y1 = sorted(rng.uniform(0, nh, size=2))
        # keep a minimum box so degenerate zero-area boxes do not dominate
        x1 = max(x1, min(x0 + 0.1 * nw, nw))
        y1 = max(y1, min(y0 + 0.1 * nh, nh))
        pts = np.array([[x0, y0], [x1, y1]], dtype=np.float32)
    else:
        sys.exit(f"unsupported npoints {npoints}")
    coords = pts[None]                             # [1, N, 2]
    labs = np.array([labels], dtype=np.float32)    # [1, N]
    return coords, labs


def calib_encoder(cfg, a):
    if len(cfg.inputs) != 1:
        sys.exit(f"{a.config}: encoder expects 1 input, got {len(cfg.inputs)}")
    spec = cfg.inputs[0]
    files = collect_images(a.images, a.limit)
    arrays, srcs = [], []
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        chw, _, _ = sam_letterbox(img, a.cpu_normalize)
        arrays.append(chw)
        srcs.append({"index": len(arrays) - 1, "source": os.path.basename(f)})
    if not arrays:
        sys.exit("no readable calibration images")
    print(f"[calib] encoder: {len(arrays)} x 3x1024x1024  "
          f"({'CPU-normalised' if a.cpu_normalize else 'raw pixels, norm baked in graph'})")
    pack(spec, arrays, fmt="npy", sources=srcs)


def calib_decoder(cfg, a):
    if len(cfg.inputs) != 3:
        sys.exit(f"{a.config}: decoder expects 3 inputs "
                 f"(image_embeddings;point_coords;point_labels), got {len(cfg.inputs)}")
    emb_spec, coord_spec, label_spec = cfg.inputs
    if not a.encoder_onnx:
        sys.exit("--encoder-onnx is required for --part decoder: the embeddings "
                 "MUST come from the real encoder, not random tensors")

    prompt = a.prompt or infer_prompt(a.config)
    if prompt not in PROMPTS:
        sys.exit(f"unknown prompt {prompt!r}; expected one of {list(PROMPTS)}")
    npoints, labels = PROMPTS[prompt]["npoints"], PROMPTS[prompt]["labels"]

    import onnxruntime as ort
    sess = ort.InferenceSession(a.encoder_onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    files = collect_images(a.images, a.limit)
    rng = np.random.default_rng(a.seed)
    embs, coords_list, labels_list, srcs = [], [], [], []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        chw, nh, nw = sam_letterbox(img, a.cpu_normalize)
        emb = sess.run(None, {in_name: chw[None].astype(np.float32)})[0]  # [1,256,64,64]
        coords, labs = sample_prompt(nh, nw, npoints, labels, rng)
        embs.append(np.ascontiguousarray(emb[0]))          # [256,64,64]
        coords_list.append(np.ascontiguousarray(coords))   # [1,N,2]
        labels_list.append(np.ascontiguousarray(labs[..., None]))  # [1,N,1], count=N
        srcs.append({"index": len(embs) - 1, "source": os.path.basename(f),
                     "prompt": prompt})
    if not embs:
        sys.exit("no readable calibration images")

    print(f"[calib] decoder {prompt}: {len(embs)} samples, "
          f"embeddings from {os.path.basename(a.encoder_onnx)}, "
          f"npoints={npoints}, labels={labels}")
    pack(emb_spec, embs, fmt="npy", sources=srcs)
    pack(coord_spec, coords_list, fmt="npy", sources=srcs)
    pack(label_spec, labels_list, fmt="npy", sources=srcs)


def infer_prompt(config_path):
    name = os.path.basename(config_path).lower()
    for key in PROMPTS:
        if key in name:
            return key
    sys.exit(f"cannot infer prompt from {config_path!r}; pass --prompt sp1|bp2")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", required=True, choices=["encoder", "decoder"])
    ap.add_argument("--images", required=True, help="image dir or glob")
    ap.add_argument("--encoder-onnx", help="exported encoder ONNX (decoder part only)")
    ap.add_argument("--prompt", choices=list(PROMPTS),
                    help="override the sp1/bp2 arity (default: infer from --config)")
    ap.add_argument("--limit", type=int, default=64, help="max images (0 = all)")
    ap.add_argument("--cpu-normalize", action="store_true",
                    help="apply SAM mean/std here (use iff exported with --no-bake-norm)")
    ap.add_argument("--seed", type=int, default=0, help="prompt sampling seed")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.part == "encoder":
        calib_encoder(cfg, a)
    else:
        calib_decoder(cfg, a)


if __name__ == "__main__":
    main()
