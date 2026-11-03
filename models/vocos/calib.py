#!/usr/bin/env python3
"""[2/3] PTQ calibration data for Vocos: Matcha-TTS mels, chunked as the runtime does.

    python calib.py --config config.yaml --matcha /path/to/matcha-icefall-zh-en \\
                    --vocoder vocos-16khz-univ.onnx --texts calib_texts.txt \\
                    --save-mels mels_calib

The vocoder is calibrated on what it is fed in deployment: mel spectrograms from
the Matcha-TTS zh-en acoustic model it is published with. Those come from the
**float** Matcha in sherpa-onnx's packaging (`matcha-icefall-zh-en` from the
sherpa-onnx `tts-models` release: `model-steps-3.onnx`, lexicon, tokens, rule
FSTs), so this needs the `sherpa_onnx` and `onnxruntime` Python packages and
nothing compiled.

**Use sherpa-onnx's own front end, not a hand-written one.** Text -> token ids
is where a home-made front end goes wrong quietly: one that maps characters
through the lexicon and drops punctuation produced mels with no pauses, and
round-trip CER of 4.2% against 2.0% for the real front end. sherpa-onnx does not
expose its token ids, so they are read back from its debug log ("final IDs"),
one sentence at a time, and the acoustic model is then run on them directly to
get the mel. That log line is not an API; if a sherpa-onnx release changes it,
`ids_for` fails loudly rather than returning nothing.

**noise_scale 1.0, length_scale 1.0** — sherpa-onnx's defaults for this model,
so the mels have the variance the runtime sees. Matcha's decoder draws its
starting noise with `RandomNormalLike`, which has no seed in the published
graph, so unseeded runs never produce the same mels twice. `--seed` (default 0)
writes a seed into an in-memory copy of the graph so this script is repeatable;
the shipped build was calibrated on an unseeded draw of the same 40 texts, so a
rebuild matches it statistically, not byte for byte.

**Chunk exactly as the runtime does.** BCDL's `VocosVocoder` runs fixed
256-frame windows that overlap by 32 frames per side, filling frames past either
end of the utterance with -16 (log-mel silence in these mels sits near -22).
Every window the board will see is a calibration sample, edges and padding
included; 40 texts -> 61 sentences -> 65 windows.
"""
import argparse
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402

T, CONTEXT, PAD = 256, 32, -16.0
IDS = re.compile(r"final IDs \(\d+\): \[([^\]]*)\]")


def sherpa_tts(matcha, vocoder):
    import sherpa_onnx as so

    m = lambda f: os.path.join(matcha, f)  # noqa: E731
    fsts = [m(f) for f in ("date-zh.fst", "phone-zh.fst", "number-zh.fst") if os.path.exists(m(f))]
    cfg = so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(
            matcha=so.OfflineTtsMatchaModelConfig(
                acoustic_model=m("model-steps-3.onnx"), vocoder=vocoder,
                lexicon=m("lexicon.txt"), tokens=m("tokens.txt"),
                data_dir=m("espeak-ng-data")),
            num_threads=4, debug=True),
        rule_fsts=",".join(fsts),
        # One sentence per generate pass, so each "final IDs" line is a sentence.
        max_num_sentences=1)
    mc = cfg.model.matcha
    return so.OfflineTts(cfg), float(mc.noise_scale), float(mc.length_scale)


def ids_for(tts, text):
    """Token ids per sentence, as sherpa-onnx's front end produced them."""
    fd, log = tempfile.mkstemp()
    saved = os.dup(2)
    os.dup2(fd, 2)                # the debug log goes to stderr from C++
    try:
        tts.generate(text)
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(fd)
    with open(log, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    os.remove(log)
    sents = [[int(v) for v in s.split(",") if v.strip()] for s in IDS.findall(txt)]
    if not sents:
        raise SystemExit("no 'final IDs' line in sherpa-onnx's debug log for %r — "
                         "has the log format changed in this release?" % text)
    return sents


def acoustic(matcha, seed):
    import onnx
    import onnxruntime as ort

    g = onnx.load(os.path.join(matcha, "model-steps-3.onnx"))
    for n in g.graph.node:
        if n.op_type in ("RandomNormalLike", "RandomNormal") and seed is not None:
            n.attribute.append(onnx.helper.make_attribute("seed", float(seed)))
    return ort.InferenceSession(g.SerializeToString(), providers=["CPUExecutionProvider"])


def windows(mel):
    """Same windows as BCDL's VocosVocoder (and export.py --check)."""
    n = mel.shape[1]
    body = T - 2 * CONTEXT
    for b in range(0, n, body):
        idx = np.arange(b - CONTEXT, b - CONTEXT + T)
        w = np.full((mel.shape[0], T), PAD, np.float32)
        ok = (idx >= 0) & (idx < n)
        w[:, ok] = mel[:, idx[ok]]
        yield w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--matcha", required=True,
                    help="sherpa-onnx matcha-icefall-zh-en directory")
    ap.add_argument("--vocoder", required=True,
                    help="vocos-16khz-univ.onnx (sherpa-onnx needs one to load the model)")
    ap.add_argument("--texts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "calib_texts.txt"))
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for Matcha's decoder noise; -1 leaves it unseeded")
    ap.add_argument("--save-mels", help="also write the per-sentence mels here "
                                        "(input for export.py --check)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if len(cfg.inputs) != 1:
        sys.exit("this model has one input; the config declares %d" % len(cfg.inputs))

    tts, noise, length = sherpa_tts(a.matcha, a.vocoder)
    am = acoustic(a.matcha, None if a.seed < 0 else a.seed)
    print(f"[calib] noise_scale {noise}  length_scale {length}  seed {a.seed}")

    texts = [l.strip() for l in open(a.texts, encoding="utf-8") if l.strip()]
    if a.save_mels:
        os.makedirs(a.save_mels, exist_ok=True)
    chunks, srcs, lens = [], [], []
    for i, text in enumerate(texts):
        for j, ids in enumerate(ids_for(tts, text)):
            x = np.array([ids], np.int64)
            mel = am.run(None, {"x": x, "x_length": np.array([x.shape[1]], np.int64),
                                "noise_scale": np.array([noise], np.float32),
                                "length_scale": np.array([length], np.float32)})[0][0]
            mel = mel.astype(np.float32)
            lens.append(mel.shape[1])
            if a.save_mels:
                np.save(os.path.join(a.save_mels, f"{i:02d}_{j}.npy"), mel)
            for k, w in enumerate(windows(mel)):
                chunks.append(w[None])            # [1, 80, 256] as "CHW"
                srcs.append({"index": len(srcs), "source": f"text {i} sentence {j} window {k}"})
    print(f"[calib] {len(texts)} texts, {len(lens)} sentences, mel frames "
          f"min/median/max {min(lens)}/{int(np.median(lens))}/{max(lens)} "
          f"-> {len(chunks)} windows of {T}")
    pack(cfg.inputs[0], chunks, fmt="npy", sources=srcs)


if __name__ == "__main__":
    main()
