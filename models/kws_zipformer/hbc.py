#!/usr/bin/env python3
"""hb_compile with one optimiser check disabled. Runs INSIDE the toolchain
container (compile.sh calls `python3 hbc.py -c config.yaml`).

hmct's `move_reshape_node_back` pass asks, for each candidate Reshape, whether
the arithmetic node next to it is a supported two-feature op, and answers by
building a throwaway onnxruntime session on a small subgraph. On this encoder
one of those subgraphs segfaults onnxruntime. hb_compile logs "Start to
optimize the onnx model." and the container exits with status 139 (SIGSEGV) —
on export.py's final graph too, so the rewrites do not remove it. Making `_is_supported_two_features_arithmetic` return False
only means the pass leaves those Reshapes where they are; nothing else in the
optimiser or the quantiser changes. Verified on OpenExplorer 3.7.0 (hmct 2.6.5).
"""
import sys

import hmct.converter.optimizer.move.move_reshape_node_back as mr

patched = 0
for name in dir(mr):
    obj = getattr(mr, name)
    if isinstance(obj, type) and hasattr(obj, "_is_supported_two_features_arithmetic"):
        obj._is_supported_two_features_arithmetic = lambda self, node: False
        patched += 1
if not patched:
    sys.exit("hbc.py: _is_supported_two_features_arithmetic not found — the toolchain "
             "changed; check whether the crash is still there before removing this")

from horizon_tc_ui.hb_compile import main  # noqa: E402

sys.argv = ["hb_compile"] + sys.argv[1:]
main()
