#!/usr/bin/env python3
"""[1/3] Fix the published Vocos ONNX to 256 frames and cut it at the head.

    python export.py --onnx vocos-16khz-univ.onnx --out vocos16k_t256_head.onnx
    python export.py --onnx vocos-16khz-univ.onnx --out vocos16k_t256_head.onnx \\
                     --check /path/to/mels            # optional, see below

The input is the vocoder published with Matcha-TTS zh-en
(`vocos-16khz-univ.onnx`, the sherpa-onnx `vocoder-models` release or the
ModelScope model page; md5 2b43a6235c158cb70480334c48c1c8b9). There is no
PyTorch checkpoint to export from — the published ONNX is the upstream.

Three edits, in this order:

**1. Fix the shape to [1, 80, 256].** The published graph has a dynamic batch
and time axis. The BPU needs a static shape, and 256 frames (4.1 s) is the
window the runtime chunks by. Every output is fixed to [1, 513, 256] too.

**2. onnxslim.** Folds the one `Shape -> Gather` that computed the head's split
point and turns the 1x1 `MatMul`s into `Gemm`.

**3. Cut at the head's linear output.** The published graph ends in
`clip(exp(log_mag), max=100)`, `cos(phase)` and `sin(phase)`. Those four nodes
are removed and the two `Slice` outputs that fed them become the graph outputs,
renamed `log_mag` and `phase`. This is the cut that matters: left in, the
compiler lowers `cos`/`sin` to a quantised range reduction over a raw phase
that spans about +/-42 rad, and the phase comes out at cosine 0.58 against
float at int8 and 0.97-0.99 at int16 — audible either way. `exp`, the clip,
`cos`/`sin` and the iSTFT run on the CPU instead (BCDL's `VocosVocoder`).

`--check DIR` runs a directory of `[80, frames]` float32 mel `.npy` files (the
layout Matcha emits) three ways and prints the differences:

  - the cut graph followed by exp/clip/cos/sin, against the published graph at
    the same 256-frame window — the cut changes nothing;
  - the runtime's chunked synthesis (256-frame windows, 32 frames of overlap per
    side, -16 past either end) against the published graph run on the whole
    utterance, as audio after the same iSTFT — chunking changes nothing but the
    edges;
  - optionally `--ref` another head graph on the same windows (to compare a
    rebuild against the one a shipped `.hbm` was compiled from).
"""
import argparse
import glob
import os

import numpy as np
import onnx

T = 256                # mel frames per compiled window
BINS = 513             # n_fft 1024 -> 513 bins
CONTEXT = 32           # runtime overlap per side; receptive field is 27
PAD = -16.0            # runtime log-mel fill past either end of the utterance
N_FFT, HOP = 1024, 256


def fix_shape(model, frames):
    d = model.graph.input[0].type.tensor_type.shape.dim
    d[0].Clear(); d[0].dim_value = 1
    d[2].Clear(); d[2].dim_value = frames
    for o in model.graph.output:
        for k, dd in enumerate(o.type.tensor_type.shape.dim):
            dd.Clear()
            dd.dim_value = (1, BINS, frames)[k]
    return model


def cut_head(model):
    """Drop Exp/Clip/Cos/Sin; expose the two Slice outputs as log_mag, phase."""
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    clip = prod["mag"]
    exp = prod[clip.input[0]]
    cos, sin = prod["x"], prod["y"]
    assert (clip.op_type, exp.op_type, cos.op_type, sin.op_type) == \
        ("Clip", "Exp", "Cos", "Sin"), "unexpected tail; is this the published Vocos?"
    assert cos.input[0] == sin.input[0]
    clip_max = None
    if len(clip.input) > 2 and clip.input[2]:
        init = {i.name: i for i in g.initializer}
        c = prod.get(clip.input[2])
        if clip.input[2] in init:
            clip_max = float(onnx.numpy_helper.to_array(init[clip.input[2]]))
        elif c is not None and c.op_type == "Constant":
            clip_max = float(onnx.numpy_helper.to_array(c.attribute[0].t))
    print(f"[export] removed tail: clip(exp(.), max={clip_max}), cos, sin "
          f"-- these run on the CPU (BCDL VocosVocoder.max_magnitude = 100)")

    rename = {exp.input[0]: "log_mag", cos.input[0]: "phase"}
    drop = {id(n) for n in (clip, exp, cos, sin)}
    keep = [n for n in g.node if id(n) not in drop]
    for n in keep:
        for i, o in enumerate(n.output):
            if o in rename:
                n.output[i] = rename[o]
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    del g.node[:]
    g.node.extend(keep)
    # Constants only the removed nodes used (the clip bound) go with them.
    used = {x for n in g.node for x in n.input}
    for n in [n for n in g.node if n.op_type == "Constant" and n.output[0] not in used]:
        g.node.remove(n)
    for i in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(i)

    frames = g.output[0].type.tensor_type.shape.dim[2].dim_value
    del g.output[:]
    for name in ("log_mag", "phase"):
        g.output.append(onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, [1, BINS, frames]))
    return model


# ---------------------------------------------------------------- --check

def istft(mag, cos, sin):
    """Centred, periodic-Hann iSTFT, n_fft 1024 / hop 256 — what BCDL's Istft does."""
    spec = mag * (cos + 1j * sin)                        # [513, F]
    frames = np.fft.irfft(spec, n=N_FFT, axis=0)          # [1024, F]
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    f = spec.shape[1]
    n = N_FFT + HOP * (f - 1)
    y = np.zeros(n); wsum = np.zeros(n)
    for t in range(f):
        y[t * HOP:t * HOP + N_FFT] += frames[:, t] * win
        wsum[t * HOP:t * HOP + N_FFT] += win * win
    y = y[N_FFT // 2:n - N_FFT // 2]
    return y / np.maximum(wsum[N_FFT // 2:n - N_FFT // 2], 1e-11)


def polar(log_mag, phase):
    return np.minimum(np.exp(log_mag), 100.0), np.cos(phase), np.sin(phase)


def windows(mel):
    """The runtime's chunking: [b-C, b-C+T) windows, PAD past either end."""
    n = mel.shape[1]
    body = T - 2 * CONTEXT
    for b in range(0, n, body):
        idx = np.arange(b - CONTEXT, b - CONTEXT + T)
        w = np.full((mel.shape[0], T), PAD, np.float32)
        ok = (idx >= 0) & (idx < n)
        w[:, ok] = mel[:, idx[ok]]
        yield w, min(body, n - b)


def check(head_path, upstream_path, mel_dir, ref_path=None):
    import onnxruntime as ort

    def sess(p):
        return ort.InferenceSession(p, providers=["CPUExecutionProvider"])

    head, up = sess(head_path), sess(upstream_path)
    ref = sess(ref_path) if ref_path else None
    files = sorted(glob.glob(os.path.join(mel_dir, "*.npy")))
    if not files:
        raise SystemExit(f"no .npy mels in {mel_dir}")
    cut_err, mid_err = [0.0] * 3, [0.0] * 3      # mag, cos, sin
    ref_err = 0.0
    snr_all, snr_mid = [], []
    for f in files:
        mel = np.load(f).astype(np.float32)
        if mel.ndim == 3:
            mel = mel[0]
        parts = [[], [], []]
        for w, keep in windows(mel):
            lm, ph = head.run(None, {"mels": w[None]})
            got = polar(lm[0], ph[0])
            want = up.run(None, {"mels": w[None]})
            for j, (a, b) in enumerate(zip(got, want)):
                cut_err[j] = max(cut_err[j], float(np.abs(a - b[0]).max()))
            if ref is not None:
                r = ref.run(None, {"mels": w[None]})
                ref_err = max(ref_err, float(np.abs(lm - r[0]).max()),
                              float(np.abs(ph - r[1]).max()))
            for j in range(3):
                parts[j].append(got[j][:, CONTEXT:CONTEXT + keep])
        chunked = [np.concatenate(p, 1) for p in parts]
        whole = [o[0] for o in up.run(None, {"mels": mel[None]})]
        # Interior frames (>= 27 from either end) see no padding at all, so the
        # chunked result must equal the whole-utterance one there.
        e = 27
        if mel.shape[1] > 2 * e:
            for j, (a, b) in enumerate(zip(chunked, whole)):
                mid_err[j] = max(mid_err[j], float(np.abs(a[:, e:-e] - b[:, e:-e]).max()))
        wa, wb = istft(*whole), istft(*chunked)
        snr_all.append(10 * np.log10(np.sum(wa ** 2) / max(np.sum((wa - wb) ** 2), 1e-30)))
        s = 40 * HOP
        if len(wa) > 2 * s:
            m = wa[s:-s]; d = m - wb[s:-s]
            snr_mid.append(10 * np.log10(np.sum(m ** 2) / max(np.sum(d ** 2), 1e-30)))
    print(f"[check] {len(files)} utterances")
    fmt = lambda e: "mag %.1e  cos %.1e  sin %.1e" % tuple(e)  # noqa: E731
    print(f"  cut graph + CPU exp/clip/cos/sin vs published graph, same window, "
          f"max|diff|: {fmt(cut_err)}")
    if ref is not None:
        print(f"  this head vs {os.path.basename(ref_path)}: max|diff| {ref_err:.3e}")
    print(f"  chunked vs whole utterance, interior frames, max|diff|: {fmt(mid_err)}")
    print(f"  chunked vs whole utterance, audio SNR: min {min(snr_all):.1f} dB, "
          f"median {np.median(snr_all):.1f} dB (edges padded with {PAD})")
    if snr_mid:
        print(f"  ... excluding 40 frames at each end: min {min(snr_mid):.1f} dB")
    try:  # the iSTFT above is the reference one, if torch is around to say so
        import torch
        mag, c, s_ = whole
        tw = torch.istft(torch.tensor(mag * c) + 1j * torch.tensor(mag * s_), N_FFT, HOP,
                         N_FFT, torch.hann_window(N_FFT), center=True).numpy()
        print(f"  numpy iSTFT here vs torch.istft: max|diff| "
              f"{np.abs(tw - istft(mag, c, s_)).max():.3e}")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="vocos-16khz-univ.onnx",
                    help="the published Vocos 16 kHz ONNX")
    ap.add_argument("--frames", type=int, default=T)
    ap.add_argument("--out", default="vocos16k_t256_head.onnx")
    ap.add_argument("--check", metavar="MEL_DIR",
                    help="dir of [80, frames] float32 mel .npy files to verify against")
    ap.add_argument("--ref", help="another head ONNX to compare against under --check")
    a = ap.parse_args()

    import onnxslim

    m = fix_shape(onnx.load(a.onnx), a.frames)
    m = onnxslim.slim(m)
    m = cut_head(m)
    # HBDK 4.x takes IR <= 9; the published graph is IR7, so this is a guard.
    m.ir_version = min(m.ir_version, 9)
    onnx.checker.check_model(m)
    onnx.save(m, a.out)
    from collections import Counter
    ops = Counter(n.op_type for n in m.graph.node)
    print(f"[export] wrote {a.out}: {len(m.graph.node)} nodes, IR {m.ir_version}, "
          f"opset {m.opset_import[0].version}")
    print(f"[export] ops: {dict(ops)}")
    left = {"Exp", "Cos", "Sin", "Clip", "Shape"} & set(ops)
    if left:
        raise SystemExit(f"[export] unexpected ops left in the graph: {left}")

    if a.check:
        if a.frames != T:
            raise SystemExit("--check emulates the 256-frame runtime chunking")
        check(a.out, a.onnx, a.check, a.ref)


if __name__ == "__main__":
    main()
