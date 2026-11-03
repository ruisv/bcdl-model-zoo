#!/usr/bin/env python3
"""[2/3] PTQ calibration data for Silero VAD: REAL streaming states.

    python calib.py --config config.yaml --onnx silero_vad_16k_mag_slim.onnx \\
                    --aishell /path/to/data_aishell/wav --musan /path/to/musan/noise

The graph is one step of a recurrent model, so a calibration sample is not just
a spectrum: it is a spectrum **plus the LSTM state that chunk actually arrives
with**. Zero states describe only the first chunk after a reset; every other
call sees a state the network produced itself, and the thresholds for `h`, `c`
and the gate arithmetic have to be sized for those. So this script runs the
float graph over whole recordings, chunk by chunk with the state carried, and
snapshots (mag, h, c) along the way — three positionally-matched directories.

**The audio is noisy speech plus noise alone**, which is what a VAD is for:

- sessions of ~30 s: AISHELL-1 utterances (OpenSLR 33), trimmed to their
  speech, separated by 0.3-3 s of silence, mixed with MUSAN noise (OpenSLR 17,
  the `noise/` part) at 10 dB and at 0 dB SNR, measured over the speech;
- three minutes of MUSAN noise alone, at the level the 10 dB sessions carry.

Noise clips alternate between the two uses, so the noise-only minutes are not
the same recordings as the session backgrounds.

**245 samples was enough.** The shipped build snapshots every 37th chunk and
keeps every third of those. Taking all of them (733) made calibration take ~20
minutes and changed nothing measurable.

Inputs are `featuremap` / `no_preprocess`: pack() writes them unchanged and adds
a leading axis, which is harmless — hb_compile reshapes each .npy to the
declared input shape by element count.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, HERE)
from calib_pack import load_config, pack  # noqa: E402
from export import CHUNK, CONTEXT, SR, frontend  # noqa: E402

EVERY, OFFSET, THIN = 37, 5, 3


def read16k(path):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(1)
    if sr != SR:
        raise SystemExit(f"{path}: {sr} Hz; resample to 16 kHz first")
    return a


def audio_files(root):
    return sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True) +
                  glob.glob(os.path.join(root, "**", "*.flac"), recursive=True))


def trim(a):
    """Cut an utterance to its speech, plus 30 ms either side.

    AISHELL recordings have only ~25 dB between peak and floor, so a peak-relative
    trim removes nothing. Threshold on frame energy relative to the recording's
    own floor instead: max(floor + 12 dB, loud - 25 dB).
    """
    n = (len(a) - 400) // 160 + 1
    if n < 2:
        return a
    idx = np.arange(400)[None, :] + 160 * np.arange(n)[:, None]
    db = 20 * np.log10(np.sqrt(np.mean(a[idx] ** 2, axis=1)) + 1e-9)
    thr = max(np.percentile(db, 10) + 12, np.percentile(db, 99) - 25)
    act = np.where(db > thr)[0]
    if len(act) == 0:
        return a
    return a[max(0, act[0] * 160 - 480):min(len(a), act[-1] * 160 + 400 + 480)]


def noise_seg(noise, n, rng):
    x = noise[int(rng.integers(len(noise)))]
    if len(x) < n:
        x = np.tile(x, n // len(x) + 1)
    s = int(rng.integers(0, len(x) - n + 1))
    return x[s:s + n]


def build_audio(aishell, musan, sessions, noise_minutes, seed):
    rng = np.random.default_rng(seed)
    noise = [x for x in (read16k(f) for f in audio_files(musan)) if len(x) > 3 * SR]
    if len(noise) < 2:
        raise SystemExit(f"no usable MUSAN noise (>3 s, 16 kHz) under {musan}")
    nz_sess, nz_pure = noise[0::2], noise[1::2]
    files = audio_files(aishell)
    if not files:
        raise SystemExit(f"no audio under {aishell}")
    pick = rng.choice(len(files), size=min(len(files), 40 * sessions), replace=False)
    utts = [trim(read16k(files[i])) for i in sorted(pick)]

    out = []
    for si in range(sessions):
        parts = [np.zeros(int(rng.uniform(0.3, 3.0) * SR), np.float32)]
        speech = []
        while sum(len(p) for p in parts) < 27 * SR:
            u = utts[int(rng.integers(len(utts)))]
            parts += [u, np.zeros(int(rng.uniform(0.3, 3.0) * SR), np.float32)]
            speech.append(u)
        parts.append(np.zeros(3 * SR, np.float32))
        x = np.concatenate(parts)
        p_sp = float(np.mean(np.concatenate(speech) ** 2))
        nz = noise_seg(nz_sess, len(x), rng)
        for snr in (10, 0):
            g = np.sqrt(p_sp / ((np.mean(nz ** 2) + 1e-12) * 10 ** (snr / 10)))
            y = x + nz * g
            y = (y / max(1.0, np.abs(y).max() / 0.99)).astype(np.float32)
            out.append((f"session{si:02d}_snr{snr}", y))

    # Noise alone, at the level a 10 dB session's background has.
    rms_sp = float(np.median([np.sqrt(np.mean(u ** 2)) for u in utts]))
    tgt = rms_sp / 10 ** (10 / 20)
    buf, n = [], 0
    for i in rng.permutation(len(nz_pure)):
        c = nz_pure[i][:30 * SR]
        buf.append(np.clip(c * tgt / (np.sqrt(np.mean(c ** 2)) + 1e-9), -0.99, 0.99))
        n += len(c)
        if n >= noise_minutes * 60 * SR:
            break
    y = np.concatenate(buf).astype(np.float32)
    for k in range(noise_minutes):
        seg = y[k * 60 * SR:(k + 1) * 60 * SR]
        if len(seg) >= CHUNK:
            out.append((f"noise{k:02d}", seg))
    return out


def snapshots(sess, recordings):
    """Run the float graph chunk by chunk; snapshot the inputs every EVERY chunks."""
    mags, hs, cs, srcs = [], [], [], []
    for name, a in recordings:
        h = np.zeros((1, 128), np.float32)
        c = np.zeros((1, 128), np.float32)
        ctx = np.zeros(CONTEXT, np.float32)
        for i in range(len(a) // CHUNK):
            _, mag, ctx = frontend(a[i * CHUNK:(i + 1) * CHUNK], ctx)
            if i % EVERY == OFFSET:
                mags.append(mag[None])           # [1, 129, 4]
                hs.append(h.copy()[None])        # [1, 1, 128]
                cs.append(c.copy()[None])
                srcs.append(f"{name}@{i * CHUNK / SR:.2f}s")
            _, h, c = sess.run(None, {"mag": mag[None], "h": h, "c": c})
    return mags[::THIN], hs[::THIN], cs[::THIN], srcs[::THIN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--onnx", default="silero_vad_16k_mag_slim.onnx",
                    help="the float graph being compiled; it produces the states")
    ap.add_argument("--aishell", required=True,
                    help="AISHELL-1 wav root (OpenSLR 33, untarred), searched recursively")
    ap.add_argument("--musan", required=True,
                    help="MUSAN noise root (OpenSLR 17, musan/noise), searched recursively")
    ap.add_argument("--sessions", type=int, default=10,
                    help="~30 s sessions; each is used at 10 dB and at 0 dB")
    ap.add_argument("--noise-minutes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()

    import onnxruntime as ort

    cfg = load_config(a.config)
    names = [s.name for s in cfg.inputs]
    if names != ["mag", "h", "c"]:
        sys.exit(f"expected inputs mag;h;c, the config declares {names}")
    recs = build_audio(a.aishell, a.musan, a.sessions, a.noise_minutes, a.seed)
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    mags, hs, cs, names = snapshots(sess, recs)
    print("[calib] %d recordings (%.1f min) -> %d samples of (mag, h, c)"
          % (len(recs), sum(len(r) for _, r in recs) / SR / 60, len(mags)))
    srcs = [{"index": i, "source": n} for i, n in enumerate(names)]
    for spec, arrs in zip(cfg.inputs, (mags, hs, cs)):
        pack(spec, arrs, fmt="npy", sources=srcs)


if __name__ == "__main__":
    main()
