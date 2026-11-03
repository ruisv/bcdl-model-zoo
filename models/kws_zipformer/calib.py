#!/usr/bin/env python3
"""[2/3] PTQ calibration data for the KWS encoder: real streaming states.

    python calib.py --config config.yaml \\
        --audio /path/to/wakeword_clips --audio /path/to/aishell1_wavs:8 \\
        --audio /path/to/noisy_speech:6 --audio /path/to/musan_noise:1

The encoder has 39 inputs: the fbank window `x`, `pad_mask`, and 37 caches
(attention keys/values, conv states, embedding state) that the previous call
wrote. **Calibrating on zeros or random tensors is calibrating a different
model.** The caches are what the encoder carries across 160 ms chunks; their
distribution only exists in a stream. So this script runs the float graph
(export.py's output, onnxruntime) over real audio exactly as BCDL does on the
board — same fbank, same pad_mask — and records every input of a call.

Which calls: every chunk of the first 8 of each clip (the stretch where
`pad_mask` is still partly 1 and the caches are partly empty, which BCDL hits
after every reset) and then every 5th chunk. Clips are cut at --max-seconds.
If that yields more than --limit states, an even stride keeps --limit of them.

Audio: any 16 kHz speech, mono or first channel, .wav or .flac. What it should
contain is what the device will hear. The shipped build used 39 clips — 12
wake-word clips clean and 12 with noise mixed in, 8 confusable phrases, 6 noisy
speech sessions (AISHELL-1 speech mixed with MUSAN noise at 10 and 0 dB SNR)
and one pure-noise recording — giving 576 states, of which 144 were kept.
Public sources for the non-wake-word part: AISHELL-1 (OpenSLR 33) and MUSAN
(OpenSLR 17). Record your own wake word, or synthesise it with any TTS.

`DIR:N` takes N clips from a directory (evenly spaced over its sorted list);
plain `DIR` takes all of them. Directories are processed in the order given.

**Calibration time is the cost to plan for.** hb_compile spends 15-33 s per
state on this graph: 144 states took 50-62 minutes, so all 576 would take hours.

Every input is `no_preprocess` featuremap and pack() writes it unchanged. pack()
writes CHW / 1CHW arrays; states of lower rank are written with leading 1s added
and hb_compile reshapes each .npy to the input's declared shape on load, so the
extra unit dimensions are harmless.

Needs: numpy, onnxruntime, kaldi-native-fbank, soundfile, pyyaml.
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
from calib_pack import load_config, pack  # noqa: E402
from kws_ref import StreamingEncoder, fbank, read_audio  # noqa: E402

FIRST_CHUNKS, EVERY = 8, 5


def clips(spec):
    d, n = spec, ""
    if not os.path.isdir(spec):
        head, _, tail = spec.rpartition(":")
        if head and tail.isdigit():
            d, n = head, tail
    fs =sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True) +
                glob.glob(os.path.join(d, "**", "*.flac"), recursive=True))
    if not fs:
        raise SystemExit(f"no .wav/.flac under {d}")
    if n:
        n = int(n)
        fs = [fs[int(i * len(fs) / n)] for i in range(min(n, len(fs)))]
    return fs


def as_4d(a):
    return a.reshape((1,) * (4 - a.ndim) + a.shape) if a.ndim < 4 else a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--audio", action="append", required=True,
                    help="DIR or DIR:N of 16 kHz clips (repeatable, processed in order)")
    ap.add_argument("--limit", type=int, default=144, help="states to keep (default 144)")
    ap.add_argument("--max-seconds", type=float, default=20.0, help="cut each clip here")
    ap.add_argument("--encoder", help="float graph to run (default: the config's onnx_model)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    import yaml
    onnx_model = yaml.safe_load(open(a.config))["model_parameters"]["onnx_model"]
    enc = StreamingEncoder(a.encoder or os.path.join(os.path.dirname(os.path.abspath(a.config)),
                                                     onnx_model))
    graph_inputs = [i.name for i in enc.inputs]
    if [s.name for s in cfg.inputs] != graph_inputs:
        raise SystemExit("config input_name does not match the graph's inputs, in order:\n  "
                         + ";".join(graph_inputs))

    files = [f for spec in a.audio for f in clips(spec)]
    states, srcs = [], []
    for path in files:
        enc.reset()
        for c, x in enumerate(enc.chunks(fbank(read_audio(path, a.max_seconds)))):
            if c < FIRST_CHUNKS or c % EVERY == 0:
                states.append({k: v.astype(np.float32) for k, v in enc.feed_dict(x).items()})
                srcs.append({"source": os.path.basename(path), "chunk": c})
            enc.step(x)
    if not states:
        raise SystemExit("no states: clips shorter than one 290 ms window?")
    stride = max(1, math.ceil(len(states) / a.limit))
    keep = list(range(0, len(states), stride))
    print(f"[calib] {len(files)} clips -> {len(states)} streaming states, "
          f"keeping every {stride}: {len(keep)}")
    srcs = [dict(srcs[i], index=j) for j, i in enumerate(keep)]
    for spec in cfg.inputs:
        pack(spec, [as_4d(states[i][spec.name]) for i in keep], fmt="npy", sources=srcs)


if __name__ == "__main__":
    main()
