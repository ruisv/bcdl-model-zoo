#!/usr/bin/env python3
"""[1/3] sherpa-onnx KWS zipformer2 (zh-en, 3M) -> a BPU-compilable encoder,
plus the decoder/joiner weight table and token list BCDL reads on the CPU.

    python export.py --model-dir /path/to/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20

Writes, next to this script unless --out-dir says otherwise:

    kws_zipformer_zh_en_3m_c16_bpu.onnx            the encoder hb_compile takes
    kws_zipformer_zh_en_3m_decoder_joiner.npy      decoder + joiner, one table
    kws_zipformer_zh_en_3m_tokens.txt              the archive's tokens.txt

There is no torch export here: upstream publishes the streaming encoder as ONNX
already, and this script edits that graph. Use the `chunk-8-left-64` files. The
"8" counts frames at the embedding rate (after the first 2x subsampling); in
fbank frames that is a 16-frame / 160 ms chunk, which is the `decode_chunk_len:
16` in the graph's own metadata and the `c16` in the build name. The archive's
`chunk-16` files are a different, 320 ms model.

Step 0 is onnxsim with every batch dimension fixed to 1. Then four rewrites,
each checked against the upstream graph in float (check_float.py: encoder
output within ~1.5e-4, identical detections). Without any one of them, the
compiler either refuses the graph or builds one that is wrong:

1. **int64 `processed_lens` -> float `pad_mask` [1, 72].** Upstream turns a
   frame counter into a boolean attention mask inside the graph (`arange`,
   `<=`, `flip`, then `Where` on the scores). The BPU takes no int64 and no
   boolean `Where`. The CPU now computes the mask (1 = cache slot not yet
   filled; see kws_ref.py) and every `Where(mask, c, x)` becomes
   `x * (1 - m) + c * m`, exact for a 0/1 mask. `new_processed_lens` is dropped;
   the runtime counts.

2. **Swoosh's softplus `log(1 + exp(s))` -> `relu(s) + log(1 + exp(-|s|))`.**
   Mathematically identical. In the original form `exp(s)` spans many orders of
   magnitude and no fixed-point scale holds it: that `Exp` quantised to cosine
   0, which took the whole encoder output to cosine -0.21 against float — a
   build that compiles and loads and is worthless. In the rewritten form `exp`
   stays in (0, 1] and `log` in [log 1, log 2]. The `Where(isinf(...))` overflow
   guard around it goes too: there is no inf in fixed point.

3. **Relative-position shift: 6 `GatherElements` -> per-query `Slice`s.** The
   indices are constants, and each row is a contiguous run whose offset depends
   only on the query position. So each gather becomes T slices of the 4-D score
   tensor concatenated back, and the `Reshape [H,1,T,W] -> [H*T,W]` pair that
   wrapped it disappears (that pair is also what the optimiser pass disabled in
   hbc.py trips over).

4. **10 depthwise convs cropped to their 2L-1 centre taps.** The conv modules'
   kernel is 15 with padding 7, but deep in the encoder the time axis is only
   1-4 frames long. The code generator cannot lower a pad longer than its input
   ("invalid storage"). On L input frames only the 2L-1 centre taps ever touch
   data — the rest only ever multiply padding zeros — so kernel 2L-1 with pad
   L-1 is exactly the same convolution.

A final onnxsim pass removes the dead `processed_lens` chain.

The decoder and joiner stay on the CPU (one embedding lookup, a grouped conv
and three small matmuls per hypothesis — not worth a BPU round trip). Their
weights are packed into one float32 table of shape [2V + 2D, D + 1], V = 263
tokens, D = 320; see `decoder_joiner_table` for the row layout BCDL reads.

Needs: onnx, onnxsim, numpy (onnxruntime for onnxsim's checks).
"""
import argparse
import os
import shutil
from collections import Counter

import numpy as np
import onnx
import onnxsim
from onnx import TensorProto, helper, numpy_helper

TAG = "epoch-13-avg-2-chunk-8-left-64"
NAME = "kws_zipformer_zh_en_3m"


# --------------------------------------------------------------------------
# step 0


def simplify_batch1(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fix every symbolic dimension (the batch, `N`) to 1 and simplify."""
    shapes = {}
    for i in model.graph.input:
        shapes[i.name] = [d.dim_value if d.dim_value > 0 else 1
                          for d in i.type.tensor_type.shape.dim]
    m, ok = onnxsim.simplify(model, overwrite_input_shapes=shapes)
    assert ok, "onnxsim could not validate the batch-1 graph"
    return m


# --------------------------------------------------------------------------
# rewrites


def _const(g, name):
    for x in g.initializer:
        if x.name == name:
            return numpy_helper.to_array(x)
    for n in g.node:
        if name in n.output and n.op_type == "Constant":
            return numpy_helper.to_array(n.attribute[0].t)
    return None


def _find_mask_concat(g):
    """The Concat that assembles the attention padding mask:
    cat([processed_mask (cached frames), chunk_mask (this chunk)], dim=1).
    It is the first Concat downstream of `processed_lens`."""
    tainted, frontier = {"processed_lens"}, True
    while frontier:
        frontier = False
        for n in g.node:
            if any(i in tainted for i in n.input):
                if n.op_type == "Concat":
                    return n
                for o in n.output:
                    if o not in tainted:
                        tainted.add(o)
                        frontier = True
    raise SystemExit("no Concat downstream of processed_lens — is this the "
                     "streaming zipformer2 encoder?")


def rewrite_pad_mask(m):
    """1. processed_lens -> pad_mask; Where(mask, c, x) -> x*(1-m) + c*m."""
    g = m.graph
    concat = _find_mask_concat(g)
    cat_out = concat.output[0]
    shp = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
           for v in onnx.shape_inference.infer_shapes(m).graph.value_info}
    mask_len = shp[cat_out][1]
    for n in g.node:
        for k, i in enumerate(n.input):
            if i == cat_out:
                n.input[k] = "pad_mask"
    # follow the mask through type-agnostic ops; its Casts become Identity
    mask_t, changed = {"pad_mask"}, True
    while changed:
        changed = False
        for n in g.node:
            if (n.op_type in ("Slice", "Unsqueeze", "Expand", "Cast", "Reshape", "Squeeze", "Concat")
                    and n.input and n.input[0] in mask_t and n.output[0] not in mask_t):
                if n.op_type == "Concat" and not all(i in mask_t for i in n.input):
                    continue
                if n.op_type == "Cast":
                    n.op_type = "Identity"
                    del n.attribute[:]
                mask_t.add(n.output[0])
                changed = True
    nodes, count = [], 0
    for n in g.node:
        if n.output[0] == cat_out:
            continue
        if n.op_type == "Where" and n.input[0] in mask_t:
            mk, c, x = n.input
            cv = _const(g, c)
            assert cv is not None and cv.size == 1, (n.name, c)
            cval = float(cv.ravel()[0])
            p = n.name
            g.initializer.extend([
                numpy_helper.from_array(np.array([1.0], np.float32), p + "_one"),
                numpy_helper.from_array(np.array([cval], np.float32), p + "_c")])
            nodes += [helper.make_node("Sub", [p + "_one", mk], [p + "_onem"], name=p + "_onem"),
                      helper.make_node("Mul", [x, p + "_onem"], [p + "_xm"], name=p + "_xm")]
            if cval != 0.0:
                nodes += [helper.make_node("Mul", [mk, p + "_c"], [p + "_cm"], name=p + "_cm"),
                          helper.make_node("Add", [p + "_xm", p + "_cm"], [n.output[0]], name=p)]
            else:
                nodes.append(helper.make_node("Identity", [p + "_xm"], [n.output[0]], name=p))
            count += 1
            continue
        nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    ins = [i for i in g.input if i.name != "processed_lens"]
    del g.input[:]
    g.input.extend(ins + [helper.make_tensor_value_info("pad_mask", TensorProto.FLOAT, [1, mask_len])])
    outs = [o for o in g.output if o.name != "new_processed_lens"]
    del g.output[:]
    g.output.extend(outs)
    print(f"  1. pad_mask [1, {mask_len}] replaces {cat_out}: {count} mask Where rewritten")
    return mask_len


def rewrite_softplus(m):
    """2. drop the inf guard, then log(1+exp(s)) -> relu(s) + log(1+exp(-|s|))."""
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    nodes, guards = [], 0
    for n in g.node:
        if n.op_type == "Where" and n.input[0] in prod and prod[n.input[0]].op_type == "Equal":
            eq = prod[n.input[0]]
            v = _const(g, eq.input[1])
            if eq.input[0] == n.input[2] and v is not None and np.isinf(v).all():
                nodes.append(helper.make_node("Identity", [n.input[2]], [n.output[0]], name=n.name))
                guards += 1
                continue
        nodes.append(n)
    prod = {o: n for n in nodes for o in n.output}
    out, count = [], 0
    for n in nodes:
        if n.op_type == "Log" and n.input[0] in prod and prod[n.input[0]].op_type == "Add":
            ad = prod[n.input[0]]
            ex = [prod.get(i) for i in ad.input]
            k = [i for i, e in enumerate(ex) if e is not None and e.op_type == "Exp"]
            if len(k) == 1:
                one = _const(g, ad.input[1 - k[0]])
                if one is not None and one.size == 1 and abs(float(one.ravel()[0]) - 1.0) < 1e-6:
                    s, p = ex[k[0]].input[0], n.name + "_sp"
                    out += [helper.make_node("Abs", [s], [p + "_abs"], name=p + "_abs"),
                            helper.make_node("Neg", [p + "_abs"], [p + "_neg"], name=p + "_neg"),
                            helper.make_node("Exp", [p + "_neg"], [p + "_exp"], name=p + "_exp"),
                            helper.make_node("Add", [p + "_exp", ad.input[1 - k[0]]], [p + "_add"],
                                             name=p + "_add"),
                            helper.make_node("Log", [p + "_add"], [p + "_log"], name=p + "_log"),
                            helper.make_node("Relu", [s], [p + "_relu"], name=p + "_relu"),
                            helper.make_node("Add", [p + "_relu", p + "_log"], [n.output[0]], name=n.name)]
                    count += 1
                    continue
        out.append(n)
    del g.node[:]
    g.node.extend(out)
    print(f"  2. softplus: {guards} inf guards removed, {count} rewritten")


def rewrite_rel_shift(m):
    """3. constant-index GatherElements (one contiguous run per row) -> Slices."""
    g = m.graph
    init = {x.name: x for x in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def c(name, arr):
        g.initializer.append(numpy_helper.from_array(np.array(arr, np.int64), name))
        return name

    drop, add_before, count = set(), {}, 0
    for n in g.node:
        if n.op_type != "GatherElements" or n.input[1] not in init:
            continue
        idx = numpy_helper.to_array(init[n.input[1]])
        R, Wd = idx.shape
        off = idx[:, 0]
        assert (idx - off[:, None] == np.arange(Wd)[None]).all(), n.name
        pre, post = prod[n.input[0]], cons[n.output[0]]
        assert pre.op_type == "Reshape" and len(post) == 1 and post[0].op_type == "Reshape", n.name
        T = int(np.argmax(off == off.min())) + 1          # query positions
        assert R % T == 0 and all(off[r] == off[r % T] for r in range(R)), n.name
        src, p, parts, new = pre.input[0], n.name + "_rs", [], []
        for q in range(T):
            o = int(off[q])
            new.append(helper.make_node(
                "Slice", [src, c(f"{p}_b{q}", [q, o]), c(f"{p}_e{q}", [q + 1, o + Wd]),
                          c(f"{p}_a{q}", [2, 3])], [f"{p}_q{q}"], name=f"{p}_q{q}"))
            parts.append(f"{p}_q{q}")
        new.append(helper.make_node("Concat", parts, [post[0].output[0]], name=post[0].name, axis=2))
        drop |= {n.name, post[0].name}
        if len(cons[pre.output[0]]) == 1:
            drop.add(pre.name)
        add_before[n.name] = new
        count += 1
    out = []
    for n in g.node:
        out += add_before.get(n.name, [])
        if n.name not in drop:
            out.append(n)
    del g.node[:]
    g.node.extend(out)
    print(f"  3. rel-shift: {count} GatherElements -> per-query slices")


def crop_depthwise(m):
    """4. Conv1d with pad P >= L on L frames -> kernel 2L-1, pad L-1 (exact)."""
    g = m.graph
    vi = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
          for v in list(onnx.shape_inference.infer_shapes(m).graph.value_info) + list(g.input)}
    init = {x.name: x for x in g.initializer}
    count = 0
    for n in g.node:
        if n.op_type != "Conv" or n.input[1] not in init:
            continue
        at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        shp = vi.get(n.input[0])
        if not shp or len(shp) != 3 or len(at.get("pads", [])) != 2:
            continue
        L, K, P = shp[2], at["kernel_shape"][0], at["pads"][0]
        if (at["pads"][1] != P or 2 * P + 1 != K or P <= L - 1
                or list(at.get("strides", [1])) != [1] or list(at.get("dilations", [1])) != [1]):
            continue
        w = numpy_helper.to_array(init[n.input[1]])
        wname = n.input[1] + f"_crop{L}"
        g.initializer.append(numpy_helper.from_array(w[:, :, P - (L - 1):P + L].copy(), wname))
        n.input[1] = wname
        for a in n.attribute:
            if a.name == "kernel_shape":
                a.ints[:] = [2 * L - 1]
            if a.name == "pads":
                a.ints[:] = [L - 1, L - 1]
        count += 1
    print(f"  4. depthwise convs cropped to 2L-1 taps: {count}")


def dead_code(m):
    g = m.graph
    while True:
        used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
        avail = {i.name for i in g.input} | {x.name for x in g.initializer}
        keep = [n for n in g.node if any(o in used for o in n.output)]
        outs = {o for n in keep for o in n.output}
        keep = [n for n in keep if all(i == "" or i in avail or i in outs for i in n.input)]
        if len(keep) == len(g.node):
            return
        del g.node[:]
        g.node.extend(keep)


def export_encoder(src, dst):
    m = onnx.load(src)
    meta = {p.key: p.value for p in m.metadata_props}
    assert meta.get("model_type") == "zipformer2", meta
    print(f"[export] {os.path.basename(src)}: T={meta['T']} decode_chunk_len={meta['decode_chunk_len']}")
    m = simplify_batch1(m)
    print(f"  0. onnxsim, batch 1: {len(m.graph.node)} nodes")
    rewrite_pad_mask(m)
    rewrite_softplus(m)
    rewrite_rel_shift(m)
    crop_depthwise(m)
    dead_code(m)
    del m.graph.value_info[:]
    m, ok = onnxsim.simplify(m)
    assert ok, "onnxsim could not validate the rewritten graph"
    ops = Counter(n.op_type for n in m.graph.node)
    for bad in ("GatherElements", "Where", "Equal", "LessOrEqual"):
        assert bad not in ops, f"{bad} survived the rewrite"
    # HBDK rejects IR > 9; upstream is IR 7 today, cap it in case that changes.
    m.ir_version = min(m.ir_version, 9)
    onnx.save(m, dst)
    print(f"  -> {dst}: {len(m.graph.node)} nodes, {len(m.graph.input)} inputs, "
          f"{len(m.graph.output)} outputs")


# --------------------------------------------------------------------------
# decoder + joiner


def decoder_joiner_table(decoder_path, joiner_path):
    """Pack the decoder and joiner into one float32 [2V + 2D, D + 1] table.

    The decoder is: embedding (y = -1 embeds to zeros) -> grouped Conv1d over
    the last 2 tokens (groups D/4, kernel 2, no bias) -> ReLU -> Linear
    (decoder_proj). The joiner is: tanh(encoder_out + decoder_out) -> Linear.
    encoder_proj is already inside the encoder graph, so encoder_out is in the
    joiner's space. Rows, with W = D + 1 columns each:

      [0, V)            embedding row v in columns [0, D); column D is 0
      [V, V + D)        conv weight of output channel o: column i*2 + k is
                        weight[o, i, k] (input channel (o // 4) * 4 + i, tap k
                        where k = 0 multiplies the older token); columns 8.. are 0
      [V + D, V + 2D)   decoder_proj: weight row j in [0, D), bias in column D
      [V + 2D, 2V + 2D) joiner output_linear: weight row v in [0, D), bias in column D

    BCDL checks the shape against the token list and the encoder's output width.
    """
    dec, joi = onnx.load(decoder_path), onnx.load(joiner_path)
    w = {x.name: numpy_helper.to_array(x) for x in list(dec.graph.initializer) + list(joi.graph.initializer)}
    emb, conv = w["decoder.embedding.weight"], w["decoder.conv.weight"]
    pw, pb = w["decoder_proj.weight"], w["decoder_proj.bias"]
    ow, ob = w["output_linear.weight"], w["output_linear.bias"]
    V, D = emb.shape
    conv_node = next(n for n in dec.graph.node if n.op_type == "Conv")
    group = next(a.i for a in conv_node.attribute if a.name == "group")
    assert conv.shape == (D, 4, 2) and group == D // 4, (conv.shape, group)
    assert pw.shape == (D, D) and ow.shape == (V, D), (pw.shape, ow.shape)
    assert not any(n.op_type == "Conv" and len(n.input) > 2 for n in dec.graph.node), "conv has a bias"
    t = np.zeros((2 * V + 2 * D, D + 1), np.float32)
    t[:V, :D] = emb
    t[V:V + D, :8] = conv.reshape(D, 8)
    t[V + D:V + 2 * D, :D], t[V + D:V + 2 * D, D] = pw, pb
    t[V + 2 * D:, :D], t[V + 2 * D:, D] = ow, ob
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="the extracted sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 archive")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    md, od = a.model_dir, a.out_dir
    os.makedirs(od, exist_ok=True)
    export_encoder(os.path.join(md, f"encoder-{TAG}.onnx"),
                   os.path.join(od, f"{NAME}_c16_bpu.onnx"))

    t = decoder_joiner_table(os.path.join(md, f"decoder-{TAG}.onnx"),
                             os.path.join(md, f"joiner-{TAG}.onnx"))
    np.save(os.path.join(od, f"{NAME}_decoder_joiner.npy"), t)
    print(f"[export] decoder+joiner table {t.shape} -> {NAME}_decoder_joiner.npy")

    tokens = os.path.join(od, f"{NAME}_tokens.txt")
    shutil.copyfile(os.path.join(md, "tokens.txt"), tokens)
    n = sum(1 for line in open(tokens, encoding="utf-8") if line.strip())
    assert 2 * n + 2 * (t.shape[1] - 1) == t.shape[0], "tokens.txt does not match the joiner"
    print(f"[export] {n} tokens -> {NAME}_tokens.txt")


if __name__ == "__main__":
    main()
