#!/usr/bin/env python3
"""Paper C ablation, produce a static K quantised file without an importance matrix.

The community files evaluated in the main experiment were quantised with an importance matrix
computed on a calibration text that is almost entirely in Latin script. To ask whether that
calibration choice contributes to the Greek loss, we requantise the eight bit Q8_0 file to the
same K quantisation types with no importance matrix at all. Requantising from Q8_0 adds a small
extra rounding error on top of the eight bit step, so if anything this disadvantages the static
files relative to files made from the original sixteen bit weights.

usage  python3 requantise_static.py SRC.gguf DST.gguf {Q4_K_M,Q2_K}
"""
import ctypes
import os
import sys
import time

import llama_cpp

FTYPES = {"Q4_K_M": llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M, "Q2_K": llama_cpp.LLAMA_FTYPE_MOSTLY_Q2_K}


def main():
    src, dst, qtype = sys.argv[1], sys.argv[2], sys.argv[3]
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = FTYPES[qtype]
    params.nthread = int(os.environ.get("THREADS", "2"))
    params.allow_requantize = True          # the source is already Q8_0
    t0 = time.time()
    rc = llama_cpp.llama_model_quantize(src.encode(), dst.encode(), ctypes.byref(params))
    if rc != 0:
        sys.exit(f"quantisation failed with code {rc}")
    print(f"{dst} written, {os.path.getsize(dst) / 1e9:.3f} GB, {time.time() - t0:.0f} s, static {qtype} from Q8_0")


if __name__ == "__main__":
    main()
