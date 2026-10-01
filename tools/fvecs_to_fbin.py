#!/usr/bin/env python3
"""Convert .fvecs / .ivecs to the .fbin / .ibin this pipeline reads.

The public ANN benchmark sets (SIFT, GIST, Deep, ...) ship as .fvecs, which
repeats the dimension in front of every row:

    .fvecs   [int32 d][d float32]  [int32 d][d float32]  ...
    .fbin    [int32 n][int32 d]    [n*d float32]

Both hold the same numbers in the same order; only the header differs. The
conversion is a reshape, so it streams in blocks and needs no more memory than
one block regardless of file size.

Usage:
    python fvecs_to_fbin.py input.fvecs output.fbin
    python fvecs_to_fbin.py --limit 100000 base.fvecs base_100k.fbin

  --limit N   write only the first N rows, e.g. to carve a small set out of a
              large one for a first run
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

BLOCK_ROWS = 1 << 16


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="input .fvecs (float32) or .ivecs (int32)")
    ap.add_argument("dst", help="output .fbin / .ibin")
    ap.add_argument("--limit", type=int, default=None,
                    help="write only the first N rows")
    args = ap.parse_args()

    itemsize = 4  # both float32 and int32
    dtype = np.int32 if args.src.endswith(".ivecs") else np.float32

    with open(args.src, "rb") as f:
        d = int(np.fromfile(f, dtype=np.int32, count=1)[0])
    if d <= 0:
        print(f"[FATAL] {args.src} does not start with a positive dimension; "
              f"is it really .fvecs/.ivecs?", file=sys.stderr)
        return 1

    row_bytes = (d + 1) * itemsize
    total = os.path.getsize(args.src)
    if total % row_bytes:
        print(f"[FATAL] {args.src} is {total} bytes, not a whole number of "
              f"{row_bytes}-byte rows for d={d}", file=sys.stderr)
        return 1
    n = total // row_bytes
    if args.limit is not None:
        n = min(n, args.limit)

    print(f"{args.src}: n={n:,} d={d} -> {args.dst}", flush=True)
    with open(args.src, "rb") as fin, open(args.dst, "wb") as fout:
        np.array([n, d], dtype=np.int32).tofile(fout)
        written = 0
        while written < n:
            rows = min(BLOCK_ROWS, n - written)
            block = np.fromfile(fin, dtype=dtype, count=rows * (d + 1))
            # drop the per-row dimension prefix, keep the payload
            block.reshape(rows, d + 1)[:, 1:].tofile(fout)
            written += rows
    print(f"wrote {args.dst} ({8 + n * d * itemsize:,} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
