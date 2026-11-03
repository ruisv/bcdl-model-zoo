#!/usr/bin/env python3
"""[0/3] Token ids for a text file, from sherpa-onnx's own Matcha zh-en front end.

    python frontend_ids.py --sherpa-dir /path/to/matcha-icefall-zh-en \\
                           --vocoder /path/to/vocos-16khz-univ.onnx \\
                           --texts calib_texts.txt --out calib_ids.json

Output: a JSON list with one entry per line of text, each a list of sentences,
each a list of token ids — exactly what the acoustic model is fed at run time.

**Do not write your own front end for this.** sherpa-onnx's front end splits
sentences, normalises numbers/dates/phone numbers with its FSTs, and keeps
punctuation as tokens; the punctuation is what makes the model pause. A
character-by-character front end without punctuation was measured at a
round-trip CER of 4.2% against 2.0% for the real one on the same sentences —
calibrating on its ids describes inputs the model is never given. (BCDL's
`MatchaFrontend` reproduces this one id for id.)

sherpa-onnx has no API that returns the ids, so this runs a full synthesis with
`debug=True` and reads them from the "final IDs" lines it logs to stderr. The
vocoder is only there because the TTS object will not build without one.
`--sherpa-dir` is the sherpa-onnx packaging of this model (`matcha-icefall-zh-en`,
from the sherpa-onnx tts-models release); its tokens.txt and lexicon.txt are the
ones BCDL ships as matcha_zh_en_tokens.txt / matcha_zh_en_lexicon.txt.
"""
import argparse
import json
import os
import re
import tempfile

PATTERN = re.compile(r"final IDs \(\d+\): \[([^\]]*)\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sherpa-dir", required=True,
                    help="matcha-icefall-zh-en: model-steps-3.onnx, tokens.txt, lexicon.txt, ...")
    ap.add_argument("--vocoder", required=True, help="vocos-16khz-univ.onnx")
    ap.add_argument("--texts", required=True, help="UTF-8, one text per line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import sherpa_onnx as so
    d = a.sherpa_dir
    cfg = so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(
            matcha=so.OfflineTtsMatchaModelConfig(
                acoustic_model=os.path.join(d, "model-steps-3.onnx"), vocoder=a.vocoder,
                lexicon=os.path.join(d, "lexicon.txt"), tokens=os.path.join(d, "tokens.txt"),
                data_dir=os.path.join(d, "espeak-ng-data")),
            num_threads=4, debug=True),
        rule_fsts=",".join(os.path.join(d, f)
                           for f in ("date-zh.fst", "phone-zh.fst", "number-zh.fst")),
        max_num_sentences=1)        # one "final IDs" line per sentence
    tts = so.OfflineTts(cfg)

    texts = [line.strip() for line in open(a.texts, encoding="utf-8") if line.strip()]
    out = []
    for t in texts:
        fd, log = tempfile.mkstemp()
        saved = os.dup(2)
        os.dup2(fd, 2)               # the ids are logged by the C++ side, on fd 2
        try:
            tts.generate(t)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            os.close(fd)
        with open(log, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        os.remove(log)
        sents = [[int(v) for v in m.split(",") if v.strip()] for m in PATTERN.findall(txt)]
        if not sents:
            raise SystemExit("no ids logged for %r — is this sherpa-onnx build logging debug?" % t)
        out.append(sents)
    with open(a.out, "w") as f:
        json.dump(out, f)
    n = sum(len(s) for s in out)
    print("%d texts, %d sentences -> %s" % (len(out), n, a.out))


if __name__ == "__main__":
    main()
