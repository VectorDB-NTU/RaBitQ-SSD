#!/usr/bin/env python3
"""Prepare a DINO-10B prefix for the RaBitQ-SSD pipeline.

DINO-10B (Meta, dino_vitl_10B) holds 10 billion uint8 vectors of 1024 dimensions,
searched under L2, with 100,000 queries and an exact top-10 ground truth shipped
for each of its standard sizes. A size-N subset is the first N base vectors, and
the ground truth for N is defined on exactly that prefix.

This writes, into --out-dir:

    train.u8bin   the first N base vectors: [int32 N][int32 1024] then N*1024 uint8
    test.u8bin    the 100,000 queries, same layout
    gt10          the shipped ground truth for N, ids only:
                  [int32 nq][int32 10] then nq*10 uint32
    gt10.nb       N, the size gt10 belongs to; a run for another size rewrites gt10

N is one of the sizes the dataset ships a ground truth for, written as a count or
with a K/M/B suffix: 100K 200K 500K 1M 2M 5M 10M 20M 50M 100M 200M 500M 1B 2B
5B 10B. 5B and 10B do not fit a .u8bin header or 32-bit ids, so for them

    train.bvecs   the base in its original layout, [int32 1024][1024 uint8] per
                  row. With a local base.bvecs (see below) it is a symbolic
                  link to that file, which may hold more than N rows; the
                  pipeline then indexes its first N rows (N_ENV).
    gt10          [u64 magic "DINOGT64"][u64 1][u64 nq][u64 10] then nq*10
                  uint64 ids

Usage:
    python tools/dino_prepare.py 1M --out-dir $DATA_ROOT/dino1m
    python tools/dino_prepare.py 1M --out-dir DIR --source /path/to/dino_vitl_10B

--source is the dataset's public URL by default. The base is fetched with HTTP
range requests, so only the first N vectors are downloaded (about 1.03 GB per
million), and an interrupted download resumes where it stopped. A local copy
works too: --source can name a directory laid out like the download
(chunked_base_10B/chunk_XXXX.bvecs, queries_clean.bvecs, gts/), or holding one
base.bvecs with all chunks concatenated in place of chunked_base_10B/.
--base-bvecs, --queries-bvecs and --gt-npy override single files.

Every output is written under a .part name and renamed when complete; a run
whose outputs already exist with the expected size skips them.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np

URL = "https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B"
DIM = 1024
REC = 4 + DIM                 # a .bvecs record: [int32 d][d uint8]
CHUNK_ROWS = 200_000_000      # base vectors per chunk_XXXX.bvecs
NQ = 100_000
K = 10
SHIPPED = [100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000,
           20_000_000, 50_000_000, 100_000_000, 200_000_000, 500_000_000,
           1_000_000_000, 2_000_000_000, 5_000_000_000, 10_000_000_000]
MAX_ROWS = 2**31 - 1          # the most rows a .u8bin header can hold
GT64_MAGIC = 0x343654474F4E4944


def parse_size(s: str) -> int:
    m = re.fullmatch(r"(\d+)([KkMmBb]?)", s.strip())
    if not m:
        raise SystemExit(f"cannot parse size '{s}'; use e.g. 1000000 or 1M")
    mult = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[m.group(2).lower()]
    n = int(m.group(1)) * mult
    if n not in SHIPPED:
        raise SystemExit(f"{n:,} is not a size DINO-10B ships a ground truth for: "
                         + " ".join(f"{x:,}" for x in SHIPPED))
    return n


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1000 or unit == "TB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1000
    return ""


# ---------------------------------------------------------------- byte sources
def stream_url(url: str, start: int, length: int, sink, retries: int = 30) -> None:
    """Feed bytes [start, start+length) of url to sink, retrying from where a
    dropped connection stopped."""
    got = 0
    failures = 0
    while got < length:
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start + got}-{start + length - 1}",
                          "User-Agent": "rabitq-ssd/dino_prepare"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status != 206:
                    raise SystemExit(f"{url}: the server ignored the range request "
                                     f"(HTTP {resp.status}); cannot fetch a prefix")
                while got < length:
                    buf = resp.read(min(1 << 22, length - got))
                    if not buf:
                        break
                    sink(buf)
                    got += len(buf)
                    failures = 0
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            failures += 1
            if failures > retries:
                raise SystemExit(f"{url}: giving up after {retries} failed attempts "
                                 f"({exc}); rerun to resume")
            wait = min(60, 2 ** min(failures, 6))
            print(f"  [retry] {exc} -- resuming at byte {start + got:,} in {wait}s",
                  flush=True)
            time.sleep(wait)


def stream_file(path: str, start: int, length: int, sink) -> None:
    with open(path, "rb") as f:
        f.seek(start)
        left = length
        while left > 0:
            buf = f.read(min(1 << 24, left))
            if not buf:
                raise SystemExit(f"{path}: ended {left:,} bytes early")
            sink(buf)
            left -= len(buf)


class RecordSink:
    """Checks a stream of .bvecs records and appends them to out: as packed
    uint8 rows, or unchanged when keep_records is set."""

    def __init__(self, out, total_rows: int, done_rows: int, label: str,
                 keep_records: bool = False):
        self.out = out
        self.total = total_rows
        self.rows = done_rows
        self.carry = b""
        self.label = label
        self.keep = keep_records
        self.t0 = time.time()
        self.rows0 = done_rows
        self.last = 0.0

    def __call__(self, buf: bytes) -> None:
        data = self.carry + buf
        n = len(data) // REC
        self.carry = data[n * REC:]
        if n == 0:
            return
        rec = np.frombuffer(data, dtype=np.uint8, count=n * REC).reshape(n, REC)
        dims = rec[:, :4].copy().view("<i4").ravel()
        if not np.all(dims == DIM):
            bad = int(np.flatnonzero(dims != DIM)[0])
            raise SystemExit(f"{self.label}: record {self.rows + bad:,} has dimension "
                             f"{int(dims[bad])}, expected {DIM}; the source is not DINO-10B")
        if self.keep:
            self.out.write(data[: n * REC])
        else:
            self.out.write(np.ascontiguousarray(rec[:, 4:]).tobytes())
        self.rows += n
        now = time.time()
        if now - self.last > 10 or self.rows == self.total:
            self.last = now
            rate = (self.rows - self.rows0) * REC / max(now - self.t0, 1e-9)
            eta = (self.total - self.rows) * REC / max(rate, 1e-9)
            print(f"  {self.label}: {self.rows:,}/{self.total:,} rows "
                  f"({100 * self.rows / self.total:.1f}%), {human(rate)}/s, "
                  f"ETA {eta / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- outputs
def u8bin_size(rows: int) -> int:
    return 8 + rows * DIM


def write_vectors(dst: str, rows: int, pieces, label: str) -> None:
    """Write `rows` rows from pieces = [(fetch, start, nbytes)], the byte ranges
    of .bvecs sources holding those rows in order: as a .u8bin, or unchanged
    when dst ends in .bvecs. Resumes a .part."""
    bvecs = dst.endswith(".bvecs")
    head = 0 if bvecs else 8
    row_bytes = REC if bvecs else DIM
    if os.path.exists(dst) and os.path.getsize(dst) == head + rows * row_bytes:
        print(f"[skip] {dst} exists ({rows:,} rows)")
        return
    part = dst + ".part"
    done = 0
    if os.path.exists(part):
        done = max(0, (os.path.getsize(part) - head) // row_bytes)
        done = min(done, rows)
    mode = "r+b" if os.path.exists(part) else "wb"
    with open(part, mode) as out:
        out.seek(0)
        if not bvecs:
            np.array([rows, DIM], dtype="<i4").tofile(out)
        out.truncate(head + done * row_bytes)
        out.seek(head + done * row_bytes)
        if done:
            print(f"  resuming {label} at row {done:,}")
        sink = RecordSink(out, rows, done, label, keep_records=bvecs)
        skip = done * REC          # bytes of the source already converted
        for fetch, start, nbytes in pieces:
            if skip >= nbytes:
                skip -= nbytes
                continue
            fetch(start + skip, nbytes - skip, sink)
            skip = 0
        if sink.carry or sink.rows != rows:
            raise SystemExit(f"{label}: got {sink.rows:,} rows, expected {rows:,}")
    os.replace(part, dst)
    print(f"[ok]   {dst} ({rows:,} x {DIM} uint8, {human(head + rows * row_bytes)})")


def local_base_bvecs(args):
    """The local file holding the whole base in one piece, if there is one."""
    if not args.base_bvecs and args.local:
        whole = os.path.join(args.source, "base.bvecs")
        if (not os.path.isdir(os.path.join(args.source, "chunked_base_10B"))
                and os.path.exists(whole)):
            args.base_bvecs = whole
    return args.base_bvecs


def link_base(n: int, path: str, dst: str) -> None:
    """Point dst at a local .bvecs holding at least the first n vectors."""
    size = os.path.getsize(path)
    head = np.fromfile(path, dtype="<i4", count=1)
    if size % REC or head.size != 1 or int(head[0]) != DIM:
        raise SystemExit(f"{path} is not a {DIM}-dimensional .bvecs file")
    with open(path, "rb") as f:
        f.seek(size - REC)
        if int(np.frombuffer(f.read(4), dtype="<i4")[0]) != DIM:
            raise SystemExit(f"{path}: its last record is not {DIM}-dimensional")
    if size // REC < n:
        raise SystemExit(f"{path} holds {size // REC:,} vectors, fewer than {n:,}")
    target = os.path.abspath(path)
    if os.path.islink(dst) and os.readlink(dst) == target:
        print(f"[skip] {dst} -> {target}")
        return
    if os.path.lexists(dst):
        raise SystemExit(f"{dst} exists and is not a link to {target}; remove it first")
    os.symlink(target, dst)
    print(f"[ok]   {dst} -> {target} ({size // REC:,} vectors; the first {n:,} "
          f"are indexed)")


def base_pieces(n: int, args):
    """Byte ranges covering the first n base records, chunk by chunk."""
    pieces = []
    if local_base_bvecs(args):
        path = args.base_bvecs
        have = os.path.getsize(path) // REC
        if have < n:
            raise SystemExit(f"{path} holds {have:,} vectors, fewer than {n:,}")
        return [(lambda s, l, k, p=path: stream_file(p, s, l, k), 0, n * REC)]
    left, chunk = n, 0
    while left > 0:
        rows = min(left, CHUNK_ROWS)
        name = f"chunked_base_10B/chunk_{chunk:04d}.bvecs"
        if args.local:
            path = os.path.join(args.source, name)
            if not os.path.exists(path):
                raise SystemExit(f"missing {path}; pass --base-bvecs for another layout")
            pieces.append((lambda s, l, k, p=path: stream_file(p, s, l, k), 0, rows * REC))
        else:
            url = f"{args.source}/{name}"
            pieces.append((lambda s, l, k, u=url: stream_url(u, s, l, k), 0, rows * REC))
        left -= rows
        chunk += 1
    return pieces


def fetch_small(url: str, dst: str) -> None:
    """Download a small file (the ground truth) in one piece."""
    part = dst + ".part"
    size = int(urllib.request.urlopen(urllib.request.Request(
        url, method="HEAD"), timeout=120).headers["Content-Length"])
    with open(part, "wb") as g:
        stream_url(url, 0, size, g.write)
    os.replace(part, dst)


def gt_size(n: int) -> int:
    return 32 + NQ * K * 8 if n > MAX_ROWS else 8 + NQ * K * 4


def gt_is_for(dst: str, n: int) -> bool:
    """True when dst is the ground truth this script wrote for the first n vectors.
    The file alone cannot tell: every size up to 2B gives it the same length."""
    try:
        with open(dst + ".nb") as f:
            recorded = int(f.read().strip())
    except (OSError, ValueError):
        return False
    return recorded == n and os.path.exists(dst) and os.path.getsize(dst) == gt_size(n)


def write_gt(npy: str, dst: str, n: int) -> None:
    gt = np.load(npy)
    if gt.ndim != 2 or gt.shape[1] < K or gt.shape[0] != NQ:
        raise SystemExit(f"{npy}: shape {gt.shape}, expected ({NQ}, {K})")
    gt = gt[:, :K]
    if gt.min() < 0 or gt.max() >= n:
        raise SystemExit(f"{npy}: ids span [{gt.min()}, {gt.max()}], outside [0, {n}); "
                         f"it is not the ground truth for the first {n:,} vectors")
    part = dst + ".part"
    with open(part, "wb") as f:
        if n > MAX_ROWS:
            np.array([GT64_MAGIC, 1, gt.shape[0], K], dtype="<u8").tofile(f)
            gt.astype("<u8").tofile(f)
        else:
            np.array([gt.shape[0], K], dtype="<i4").tofile(f)
            gt.astype("<u4").tofile(f)
    if os.path.exists(dst + ".nb"):
        os.remove(dst + ".nb")
    os.replace(part, dst)
    with open(dst + ".nb", "w") as f:
        f.write(f"{n}\n")
    print(f"[ok]   {dst} ({gt.shape[0]:,} queries x {K} ids, from {os.path.basename(npy)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("size", help="number of base vectors, e.g. 1M or 1000000")
    ap.add_argument("--out-dir", required=True, help="dataset directory to write into")
    ap.add_argument("--source", default=URL,
                    help="dataset URL (default) or a local directory with the same layout")
    ap.add_argument("--base-bvecs", help="local .bvecs holding at least the first N vectors")
    ap.add_argument("--queries-bvecs", help="local queries_clean.bvecs")
    ap.add_argument("--gt-npy", help="local gts_dino_patch_<N>_k10.npy")
    args = ap.parse_args()

    n = parse_size(args.size)
    args.local = not re.match(r"https?://", args.source)
    args.source = args.source.rstrip("/")
    os.makedirs(args.out_dir, exist_ok=True)
    where = args.source if args.local else f"{args.source} (HTTP range requests)"
    print(f"DINO-10B prefix: N={n:,}  out={args.out_dir}  source={where}")

    # Ground truth first: it is small and fails fast on a wrong size or source.
    gt_out = os.path.join(args.out_dir, "gt10")
    if gt_is_for(gt_out, n):
        print(f"[skip] {gt_out} exists ({n:,} vectors)")
    else:
        npy = args.gt_npy
        name = f"gts/gts_dino_patch_{n}_k10.npy"
        if npy is None and args.local:
            npy = os.path.join(args.source, name)
        elif npy is None:
            npy = os.path.join(args.out_dir, os.path.basename(name))
            fetch_small(f"{args.source}/{name}", npy)
        write_gt(npy, gt_out, n)

    q_out = os.path.join(args.out_dir, "test.u8bin")
    if args.queries_bvecs:
        q_src = args.queries_bvecs
        q_pieces = [(lambda s, l, k, p=q_src: stream_file(p, s, l, k), 0, NQ * REC)]
    elif args.local:
        q_src = os.path.join(args.source, "queries_clean.bvecs")
        q_pieces = [(lambda s, l, k, p=q_src: stream_file(p, s, l, k), 0, NQ * REC)]
    else:
        q_src = f"{args.source}/queries_clean.bvecs"
        q_pieces = [(lambda s, l, k, u=q_src: stream_url(u, s, l, k), 0, NQ * REC)]
    write_vectors(q_out, NQ, q_pieces, "queries")

    if n <= MAX_ROWS:
        write_vectors(os.path.join(args.out_dir, "train.u8bin"), n,
                      base_pieces(n, args), "base")
    elif local_base_bvecs(args):
        link_base(n, args.base_bvecs, os.path.join(args.out_dir, "train.bvecs"))
    else:
        write_vectors(os.path.join(args.out_dir, "train.bvecs"), n,
                      base_pieces(n, args), "base")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
