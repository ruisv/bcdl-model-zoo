#!/usr/bin/env python3
"""Prove export.py's output is the same model as upstream, in float, before
anything is quantised.

    python check_float.py --model-dir /path/to/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 \\
        --keywords keywords.txt --audio /path/to/clips [--audio /path/to/more ...]

Three checks, each against something it did not produce itself:

1. **Decoder/joiner table** against the upstream decoder and joiner ONNX, on
   every (y0, y1) context the decoder can see and on random joiner inputs.
   Should agree to float rounding (~1e-6).
2. **Encoder**: the rewritten graph (pad_mask) and the upstream graph
   (processed_lens) streamed side by side over every clip, each with its own
   caches. The rewrites are exact in real arithmetic; expect float-rounding
   differences around 1e-4 on encoder_out.
3. **Detections**: sherpa_onnx.KeywordSpotter on the upstream ONNX, against
   kws_ref.Spotter driven by the rewritten encoder and the table — i.e. the
   arithmetic BCDL runs, minus quantisation. Same keywords file, threshold 0.25,
   boost 1.0. Clips get 0.5 s of silence appended so a keyword at the very end is
   decoded by both.

A clip passes when the two detection lists are identical. If they are not,
re-run with --orig: if the port on the *upstream* encoder also disagrees with
sherpa, the difference is in the search or the features, not in the rewrite.

The keywords file is sherpa-onnx's format (`n ǐ h ǎo x iǎo ch ē @你好小车`);
sherpa-onnx's `text2token` writes it from plain text.

Needs: numpy, onnxruntime, kaldi-native-fbank, soundfile, sherpa-onnx.
"""
import argparse
import glob
import os

import numpy as np

from kws_ref import (OnnxDecoderJoiner, Spotter, StreamingEncoder, TableDecoderJoiner,
                     fbank, load_tokens, read_audio)

TAG = "epoch-13-avg-2-chunk-8-left-64"
NAME = "kws_zipformer_zh_en_3m"
HERE = os.path.dirname(os.path.abspath(__file__))


def audio_files(dirs, limit):
    files = []
    for d in dirs:
        fs = sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True) +
                    glob.glob(os.path.join(d, "**", "*.flac"), recursive=True))
        files += fs[:limit] if limit else fs
    return files


def check_table(dj_onnx, dj_table, V):
    ys = np.array([(a, b) for a in range(-1, V) for b in range(V)], np.int64)
    d0, d1 = dj_onnx.decoder(ys), dj_table.decoder(ys)
    rng = np.random.default_rng(0)
    enc = rng.standard_normal((512, d0.shape[1])).astype(np.float32)
    dec = d0[rng.integers(0, len(d0), 512)]
    j0, j1 = dj_onnx.joiner(enc, dec), dj_table.joiner(enc, dec)
    print(f"[table]   decoder max |diff| {np.abs(d0 - d1).max():.2e} over {len(ys)} contexts, "
          f"joiner max |diff| {np.abs(j0 - j1).max():.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="the extracted upstream archive")
    ap.add_argument("--keywords", required=True, help="sherpa-onnx keywords file")
    ap.add_argument("--audio", action="append", required=True,
                    help="directory of 16 kHz clips (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="clips per directory (0 = all)")
    ap.add_argument("--encoder", default=os.path.join(HERE, f"{NAME}_c16_bpu.onnx"))
    ap.add_argument("--table", default=os.path.join(HERE, f"{NAME}_decoder_joiner.npy"))
    ap.add_argument("--orig", action="store_true",
                    help="also run the port on the upstream encoder (to localise a mismatch)")
    a = ap.parse_args()

    import sherpa_onnx

    md = a.model_dir
    up = {k: os.path.join(md, f"{k}-{TAG}.onnx") for k in ("encoder", "decoder", "joiner")}
    tokens = load_tokens(os.path.join(md, "tokens.txt"))
    dj_onnx = OnnxDecoderJoiner(up["decoder"], up["joiner"])
    dj_table = TableDecoderJoiner(a.table, len(tokens))
    check_table(dj_onnx, dj_table, len(tokens))

    kws = sherpa_onnx.KeywordSpotter(
        tokens=os.path.join(md, "tokens.txt"), encoder=up["encoder"], decoder=up["decoder"],
        joiner=up["joiner"], keywords_file=a.keywords, num_threads=1,
        keywords_threshold=0.25, keywords_score=1.0)

    def sherpa(samples):
        s = kws.create_stream()
        s.accept_waveform(16000, samples)
        s.input_finished()
        hits = []
        while kws.is_ready(s):
            kws.decode_stream(s)
            r = kws.get_result(s)
            if r:
                hits.append(r)
                kws.reset_stream(s)
        return hits

    enc_up, enc_bpu = StreamingEncoder(up["encoder"]), StreamingEncoder(a.encoder)
    port = Spotter(enc_bpu, dj_table, a.keywords, tokens)
    port_up = Spotter(StreamingEncoder(up["encoder"]), dj_onnx, a.keywords, tokens) if a.orig else None

    files = audio_files(a.audio, a.limit)
    if not files:
        raise SystemExit("no clips found")
    worst, agree, agree_up, fired = 0.0, 0, 0, 0
    for f in files:
        x = np.concatenate([read_audio(f), np.zeros(8000, np.float32)])
        # 2. encoder, side by side
        feats = fbank(x)
        enc_up.reset()
        enc_bpu.reset()
        for c in enc_up.chunks(feats):
            worst = max(worst, float(np.abs(enc_up.step(c) - enc_bpu.step(c)).max()))
        # 3. detections
        ref = sherpa(x)
        got = [p for p, _, _ in port.run(x)[0]]
        fired += bool(ref)
        agree += int(ref == got)
        if ref != got:
            print(f"  differs: {os.path.basename(f)}  sherpa {ref}  port {got}")
        if port_up:
            agree_up += int(ref == [p for p, _, _ in port_up.run(x)[0]])
    print(f"[encoder] {len(files)} clips: max |encoder_out rewritten - upstream| {worst:.2e}")
    print(f"[detect]  {agree}/{len(files)} clips identical to sherpa-onnx "
          f"(sherpa fired on {fired})")
    if port_up:
        print(f"[detect]  port on the upstream encoder: {agree_up}/{len(files)} identical")
    if agree != len(files):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
