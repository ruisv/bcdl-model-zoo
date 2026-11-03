#!/usr/bin/env python3
"""[2/3] PTQ calibration data for both Matcha graphs, from real sentences.

    python calib.py --ids calib_ids.json

Reads `config_encoder.yaml` and `config_decoder.yaml` and writes five
directories — the encoder's (x_emb, x_mask) and the decoder's (mu_y, y_mask, z)
— with sample *i* in each coming from the same sentence. hb_compile matches
multi-input calibration sets positionally, so the counts and order must agree;
one pass writes all five so they cannot drift apart.

**The decoder is calibrated on what the encoder actually produces.** mu_y is the
float encoder's mu_x, length-regulated on the CPU exactly as at run time
(durations `ceil(exp(logw))`, length_scale 1), zero-padded to 512 frames with
y_mask marking the valid frames. Random tensors would size the thresholds for a
distribution the decoder never sees.

**z is N(0, 1)** — the run-time starting noise at noise_scale 1.0, the default
in both sherpa-onnx and BCDL. Seeded, so the set is reproducible.

The sentences come from `frontend_ids.py` (sherpa-onnx's own front end). The 40
texts in calib_texts.txt are short robot replies, disjoint from the 60 the build
was scored on; they split into 61 sentences, all under 128 tokens and 512 frames.
A sentence that does not fit is skipped in all five sets together.

Every input is a `featuremap` with `no_preprocess`: the arrays are exactly what
the board feeds, so pack() writes them unchanged. pack() adds a leading axis on
save ([1, 1, 80, 512] on disk); hb_compile reshapes each sample to the declared
input shape by element count, so that is harmless.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="token ids JSON from frontend_ids.py")
    ap.add_argument("--encoder-config", default=os.path.join(HERE, "config_encoder.yaml"))
    ap.add_argument("--decoder-config", default=os.path.join(HERE, "config_decoder.yaml"))
    ap.add_argument("--encoder-onnx", default="matcha_enc_L128_slim.onnx",
                    help="the FLOAT encoder, to produce the decoder's inputs")
    ap.add_argument("--embedding", default="matcha_zh_en_embedding.npy")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    import onnxruntime as ort
    enc_cfg, dec_cfg = load_config(a.encoder_config), load_config(a.decoder_config)
    if [s.name for s in enc_cfg.inputs] != ["x_emb", "x_mask"] or \
       [s.name for s in dec_cfg.inputs] != ["mu_y", "y_mask", "z"]:
        sys.exit("unexpected input names in the configs")

    E = ort.InferenceSession(a.encoder_onnx, providers=["CPUExecutionProvider"])
    L = E.get_inputs()[0].shape[2]
    T = 512
    table = np.load(a.embedding)
    rng = np.random.default_rng(a.seed)

    x_emb, x_mask, mu_y, y_mask, z, srcs = [], [], [], [], [], []
    skipped = 0
    for ti, sents in enumerate(json.load(open(a.ids))):
        for si, ids in enumerate(sents):
            n = len(ids)
            if n > L:
                skipped += 1
                continue
            xe = np.zeros((1, table.shape[1], L), np.float32)
            xe[0, :, :n] = table[np.asarray(ids)].T
            xm = np.zeros((1, 1, L), np.float32)
            xm[..., :n] = 1
            mu_x, logw = E.run(None, {"x_emb": xe, "x_mask": xm})
            d = np.ceil(np.exp(logw[0, 0, :n])).astype(int)
            F = int(d.sum())
            if F > T:
                skipped += 1
                continue
            mu = np.zeros((1, 80, T), np.float32)
            mu[0, :, :F] = np.repeat(mu_x[0, :, :n], d, axis=1)
            ym = np.zeros((1, 1, T), np.float32)
            ym[..., :F] = 1
            x_emb.append(xe)
            x_mask.append(xm)
            mu_y.append(mu)
            y_mask.append(ym)
            z.append(rng.standard_normal((1, 80, T)).astype(np.float32))
            srcs.append({"index": len(srcs), "text": ti, "sentence": si,
                         "tokens": n, "frames": F})
    if not srcs:
        sys.exit("no sentence fits the graphs")
    frames = [s["frames"] for s in srcs]
    print("[calib] %d sentences (%d skipped), tokens max %d, frames min/median/max %d/%d/%d"
          % (len(srcs), skipped, max(s["tokens"] for s in srcs),
             min(frames), int(np.median(frames)), max(frames)))

    pack(enc_cfg.inputs[0], x_emb, fmt="npy", sources=srcs)
    pack(enc_cfg.inputs[1], x_mask, fmt="npy", sources=srcs)
    pack(dec_cfg.inputs[0], mu_y, fmt="npy", sources=srcs)
    pack(dec_cfg.inputs[1], y_mask, fmt="npy", sources=srcs)
    pack(dec_cfg.inputs[2], z, fmt="npy", sources=srcs)


if __name__ == "__main__":
    main()
