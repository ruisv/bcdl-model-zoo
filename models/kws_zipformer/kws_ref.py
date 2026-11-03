#!/usr/bin/env python3
"""Reference KWS pipeline in Python, step for step what BCDL runs on the board.

Shared by calib.py (which needs real streaming encoder states) and
check_float.py (which proves the rewritten graph is the same model). Three parts:

**Features.** kaldi-native-fbank exactly as sherpa-onnx configures it for icefall
models: 80 mel bins, samples in [-1, 1] (NOT scaled by 32768), Povey window,
low_freq 20, high_freq -400 (i.e. Nyquist - 400 Hz), dither 0, snip_edges=false.
Get the scaling wrong and the encoder still produces confident-looking output;
an earlier port that multiplied by 32768 agreed with sherpa-onnx on 37 of 87
clips. BCDL's C++ fbank has no snip_edges=false mode and reproduces it by
prepending a 120-sample reflection of the stream's start, which is the same
framing.

**The streaming encoder.** 29 fbank frames in per call, the window advancing by
16 (160 ms): 4 encoder frames out at 40 ms each. Either graph can be driven:

  - upstream (`encoder-*.onnx`): an int64 `processed_lens` input, the number of
    frames already in the attention cache, counted at the EMBEDDING rate, i.e.
    +8 per chunk — not the encoder-output rate (+4). Counting at the wrong rate
    still runs, and drives the encoder output off by tens.
  - rewritten (export.py's output): a float `pad_mask` of length 72 = 64 cached
    frames + 8 frames of this chunk. 1 marks a cache slot not yet filled:
    `pad_mask = [1] * (64 - q) + [0] * (q + 8)` with `q = min(processed, 64)`.
    This is what BCDL computes on the CPU.

**The keyword search.** A port of sherpa-onnx's KeywordSpotter: modified beam
search (4 active paths) over an Aho-Corasick context graph in strict mode, with
the per-keyword score / threshold and the trailing-blank rule. BCDL's C++ is a
line-by-line port of the same code.

Needs: numpy, onnxruntime, kaldi-native-fbank.
"""
import math

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
BLANK, UNK = 0, 2


def fbank(samples: np.ndarray) -> np.ndarray:
    """(N,) float32 in [-1, 1] at 16 kHz -> (frames, 80) log-mel."""
    import kaldi_native_fbank as knf

    o = knf.FbankOptions()
    o.frame_opts.dither = 0
    o.frame_opts.snip_edges = False
    o.frame_opts.samp_freq = SAMPLE_RATE
    o.mel_opts.num_bins = 80
    o.mel_opts.low_freq = 20
    o.mel_opts.high_freq = -400
    f = knf.OnlineFbank(o)
    f.accept_waveform(SAMPLE_RATE, samples.astype(np.float32).tolist())
    f.input_finished()
    if f.num_frames_ready == 0:
        return np.zeros((0, 80), np.float32)
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)


def read_audio(path: str, max_seconds: float = 0) -> np.ndarray:
    """Mono float32 at 16 kHz; refuses other rates rather than resampling silently."""
    import soundfile as sf

    a, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{path}: {sr} Hz — this model is 16 kHz; resample first")
    a = a[:, 0]
    if max_seconds:
        a = a[: int(SAMPLE_RATE * max_seconds)]
    return np.ascontiguousarray(a)


def load_tokens(path: str) -> dict:
    t = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        s, i = line.rsplit(" ", 1)
        t[s] = int(i)
    return t


class StreamingEncoder:
    """Drives either encoder graph chunk by chunk; holds the caches."""

    def __init__(self, onnx_path: str, threads: int = 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.t_in = int(meta["T"])                    # 29 fbank frames per call
        self.shift = int(meta["decode_chunk_len"])    # 16 fbank frames = 160 ms
        self.inputs = self.sess.get_inputs()
        self.out_names = [o.name for o in self.sess.get_outputs()]
        names = [i.name for i in self.inputs]
        self.masked = "pad_mask" in names
        if self.masked:
            self.mask_len = int(next(i for i in self.inputs if i.name == "pad_mask").shape[1])
            self.left = self.mask_len - self.shift // 2
        else:
            self.left = int(meta["left_context_len"].split(",")[0])
        self.state_names = [n for n in names if n not in ("x", "pad_mask", "processed_lens")]
        self.reset()

    def reset(self):
        self.states = {}
        for i in self.inputs:
            if i.name in ("x", "pad_mask"):
                continue
            shp = [d if isinstance(d, int) else 1 for d in i.shape]
            dt = np.int64 if "int64" in i.type else np.float32
            self.states[i.name] = np.zeros(shp, dt)
        self.processed = 0   # cached frames at the EMBEDDING rate: +shift/2 per chunk

    def pad_mask(self) -> np.ndarray:
        q = min(self.processed, self.left)
        m = np.concatenate([np.ones(self.left - q), np.zeros(q + self.shift // 2)])
        return m.astype(np.float32)[None]

    def feed_dict(self, x: np.ndarray) -> dict:
        """Every input of the next call, as BCDL would supply them."""
        feed = {k: v for k, v in self.states.items()}
        feed["x"] = x[None].astype(np.float32)
        if self.masked:
            feed["pad_mask"] = self.pad_mask()
        return feed

    def step(self, x: np.ndarray) -> np.ndarray:
        """x: (t_in, 80) -> encoder_out (4, 320); advances the caches."""
        outs = self.sess.run(None, self.feed_dict(x))
        for n, v in zip(self.out_names[1:], outs[1:]):
            self.states[n[len("new_"):]] = v
        self.processed += self.shift // 2
        return outs[0][0]

    def chunks(self, feats: np.ndarray):
        """Yield the fbank window of each call, in order."""
        n = 0
        while n + self.t_in <= len(feats):
            yield feats[n:n + self.t_in]
            n += self.shift


# --------------------------------------------------------------------------
# Decoder + joiner: either the upstream ONNX or BCDL's weight table.


class OnnxDecoderJoiner:
    def __init__(self, decoder: str, joiner: str):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.dec = ort.InferenceSession(decoder, so, providers=["CPUExecutionProvider"])
        self.join = ort.InferenceSession(joiner, so, providers=["CPUExecutionProvider"])

    def decoder(self, y: np.ndarray) -> np.ndarray:          # (N, 2) int64 -> (N, D)
        return self.dec.run(None, {"y": y.astype(np.int64)})[0]

    def joiner(self, enc: np.ndarray, dec: np.ndarray) -> np.ndarray:
        return self.join.run(None, {"encoder_out": enc.astype(np.float32),
                                    "decoder_out": dec.astype(np.float32)})[0]


class TableDecoderJoiner:
    """The arithmetic BCDL does on the CPU, from the [2V+2D, D+1] table
    (layout in export.py's `decoder_joiner_table`)."""

    def __init__(self, table_path: str, vocab: int):
        t = np.load(table_path)
        V = vocab
        D = t.shape[1] - 1
        assert t.shape[0] == 2 * V + 2 * D, (t.shape, V)
        self.V, self.D = V, D
        self.emb = t[:V, :D]
        self.conv = t[V:V + D, :8].reshape(D, 4, 2)
        self.proj_w, self.proj_b = t[V + D:V + 2 * D, :D], t[V + D:V + 2 * D, D]
        self.out_w, self.out_b = t[V + 2 * D:, :D], t[V + 2 * D:, D]

    def decoder(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, np.int64)
        e = np.where((y >= 0)[..., None], self.emb[np.clip(y, 0, None)], 0.0)  # (N, 2, D)
        D = self.D
        g = np.arange(D) // 4                            # group of each output channel
        idx = g[:, None] * 4 + np.arange(4)[None]        # (D, 4) input channels
        e0, e1 = e[:, 0][:, idx], e[:, 1][:, idx]        # (N, D, 4)
        r = (self.conv[None, :, :, 0] * e0 + self.conv[None, :, :, 1] * e1).sum(-1)
        r = np.maximum(r, 0.0)
        return (r @ self.proj_w.T + self.proj_b).astype(np.float32)

    def joiner(self, enc: np.ndarray, dec: np.ndarray) -> np.ndarray:
        return (np.tanh(enc + dec) @ self.out_w.T + self.out_b).astype(np.float32)


# --------------------------------------------------------------------------
# Keyword search: sherpa-onnx's ContextGraph (strict) + modified beam search.


class _Node:
    __slots__ = ("token", "token_score", "node_score", "output_score", "level",
                 "ac_threshold", "is_end", "phrase", "next", "fail", "output")

    def __init__(self, token, token_score, node_score, output_score, level,
                 ac_threshold, is_end, phrase):
        self.token, self.token_score = token, token_score
        self.node_score, self.output_score = node_score, output_score
        self.level, self.ac_threshold = level, ac_threshold
        self.is_end, self.phrase = is_end, phrase
        self.next, self.fail, self.output = {}, None, None


class ContextGraph:
    def __init__(self, lines):
        """lines: [(token_ids, score, threshold, phrase)]"""
        self.root = _Node(-1, 0, 0, 0, 0, 0, False, "")
        self.root.fail = self.root
        for ids, score, thr, ph in lines:
            node = self.root
            for j, t in enumerate(ids):
                last = j == len(ids) - 1
                if t not in node.next:
                    nd = _Node(t, score, node.node_score + score,
                               node.node_score + score if last else 0, j + 1,
                               thr if last else 0, last, ph if last else "")
                    node.next[t] = nd
                else:
                    nd = node.next[t]
                    nd.token_score = max(score, nd.token_score)
                    nd.node_score = node.node_score + nd.token_score
                    nd.is_end = last or nd.is_end
                    nd.output_score = nd.node_score if nd.is_end else 0
                    if last:
                        nd.phrase, nd.ac_threshold = ph, thr
                node = nd
        queue = []
        for c in self.root.next.values():
            c.fail = self.root
            queue.append(c)
        while queue:
            u = queue.pop(0)
            for t, v in u.next.items():
                f = u.fail
                while t not in f.next:
                    if f is self.root:
                        break
                    f = f.fail
                v.fail = f.next[t] if t in f.next and f.next[t] is not v else self.root
                o = v.fail
                while not o.is_end:
                    if o is self.root:
                        o = None
                        break
                    o = o.fail
                v.output = o
                v.output_score += o.output_score if o else 0
                queue.append(v)

    def forward(self, st, t):
        if t in st.next:
            node = st.next[t]
            score = node.token_score
        else:
            node = st.fail
            while t not in node.next:
                if node is self.root:
                    break
                node = node.fail
            if t in node.next:
                node = node.next[t]
            score = node.node_score - st.node_score
        return score + node.output_score, node

    def matched(self, st):
        return st if st.is_end else st.output


class _Hyp:
    __slots__ = ("ys", "log_prob", "timestamps", "probs", "state", "trailing")

    def __init__(self, state):
        self.ys, self.log_prob, self.timestamps, self.probs = [-1, BLANK], 0.0, [], []
        self.state, self.trailing = state, 0

    def copy(self):
        h = _Hyp(self.state)
        h.ys, h.log_prob = list(self.ys), self.log_prob
        h.timestamps, h.probs, h.trailing = list(self.timestamps), list(self.probs), self.trailing
        return h


def _log_add(a, b):
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


def parse_keywords(path: str, tokens: dict, score=1.0, threshold=0.25):
    """sherpa-onnx keywords file: `tok tok ... [:boost] [#threshold] @phrase`."""
    lines = []
    for raw in open(path, encoding="utf-8"):
        raw = raw.strip()
        if not raw:
            continue
        toks, _, phrase = raw.partition("@")
        ids, s, th = [], score, threshold
        for t in toks.split():
            if t.startswith(":") and len(t) > 1:
                s = float(t[1:])
            elif t.startswith("#") and len(t) > 1:
                th = float(t[1:])
            else:
                ids.append(tokens[t])
        lines.append((ids, s, th, phrase.strip() or " ".join(toks.split())))
    return lines


class Spotter:
    def __init__(self, encoder: StreamingEncoder, dj, keywords_path: str, tokens: dict,
                 num_trailing_blanks=1, max_active_paths=4):
        self.enc, self.dj = encoder, dj
        self.graph = ContextGraph(parse_keywords(keywords_path, tokens))
        self.trailing, self.paths = num_trailing_blanks, max_active_paths

    def run(self, samples: np.ndarray):
        """-> (detections [(phrase, first_frame, last_frame)], encoder_out (frames, D))"""
        self.enc.reset()
        hyps = [_Hyp(self.graph.root)]
        hits, encs, t_off = [], [], 0
        for x in self.enc.chunks(fbank(samples)):
            e = self.enc.step(x)
            encs.append(e)
            for t in range(e.shape[0]):
                y = np.array([h.ys[-2:] for h in hyps], np.int64)
                logit = self.dj.joiner(np.repeat(e[t:t + 1], len(hyps), 0), self.dj.decoder(y))
                lp = logit - logit.max(1, keepdims=True)
                lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
                sc = lp + np.array([h.log_prob for h in hyps])[:, None]
                flat, V = sc.ravel(), sc.shape[1]
                top = np.argsort(-flat, kind="stable")[:self.paths]
                new = {}
                for k in top:
                    hi, tok = divmod(int(k), V)
                    h, cs = hyps[hi].copy(), 0.0
                    if tok not in (BLANK, UNK):
                        h.ys.append(tok)
                        h.timestamps.append(t + t_off)
                        h.probs.append(math.exp(lp[hi, tok]))
                        h.trailing = 0
                        cs, h.state = self.graph.forward(h.state, tok)
                        if h.state.token == -1:      # fell back to the root: start over
                            h.ys, h.timestamps, h.probs = [-1, BLANK], [], []
                    else:
                        h.trailing += 1
                    h.log_prob = float(flat[k]) + cs
                    key = tuple(h.ys)
                    if key in new:
                        new[key].log_prob = _log_add(new[key].log_prob, h.log_prob)
                    else:
                        new[key] = h
                hyps = list(new.values())
                best = max(hyps, key=lambda h: h.log_prob)
                m = self.graph.matched(best.state)
                if m is not None:
                    avg = sum(best.probs[:m.level]) / m.level
                    if best.trailing > self.trailing and avg >= m.ac_threshold:
                        ts = best.timestamps[-m.level:]
                        hits.append((m.phrase, ts[0], ts[-1]))
                        hyps = [_Hyp(self.graph.root)]
            t_off += e.shape[0]
        enc = np.concatenate(encs) if encs else np.zeros((0, 320), np.float32)
        return hits, enc
