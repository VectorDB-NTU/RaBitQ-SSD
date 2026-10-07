#!/usr/bin/env python3
"""Two-level k-means on GPU over a uint8 base (DINO).

Trains C centroids on a sample of the first N base vectors and writes them in
first-level-cell order, with the two-level structure gpu_assign.py needs:

    <out-dir>/trained_centroids_<C>.fvecs    C x d float32
    <out-dir>/trained_centroids_<C>.l1meta   the nc1 first-level centroids and
                                             how many of the C belong to each

The sample takes every (N / n_train)-th row from a random start in one
sequential pass, n_train = pts_per_centroid * C, and is held in RAM as uint8.
A first-level k-means splits it into nc1 cells (at most l1_cap_ppc points per
centroid are used); the C centroids are shared out among the cells in
proportion to their sizes, and each cell runs its own k-means (at most
sub_cap_ppc points per centroid). All k-means run on every GPU faiss sees.

The base is .bvecs ([u32 d][d x uint8] per row) or .u8bin ([u32 n][u32 d]
then n*d uint8). Needs numpy and faiss with GPU support.

.l1meta layout (little-endian): [u64 magic "DNL1META"][u64 version = 1]
[u64 nc1][u64 C][u64 d][nc1 x u32 centroids per cell][nc1 x d float32].
"""
import argparse
import math
import os
import queue
import threading
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import faiss

L1META_MAGIC = 0x4154454D314C4E44
L1META_VERSION = 1


def fatal(msg):
    raise SystemExit(f"[FATAL] {msg}")


class U8Base:
    """A uint8 vector file: rows, dimension and where row i's values start."""

    def __init__(self, path):
        ext = os.path.splitext(path)[1].lower()
        size = os.path.getsize(path)
        if ext == ".bvecs":
            d = int(np.fromfile(path, dtype="<u4", count=1)[0])
            self.stride, self.hdr, self.data_off = 4 + d, 4, 0
            if d <= 0 or size % self.stride:
                fatal(f"{path} is not a whole number of {d}-dim .bvecs rows")
            rows = size // self.stride
        elif ext == ".u8bin":
            rows, d = (int(v) for v in np.fromfile(path, dtype="<u4", count=2))
            self.stride, self.hdr, self.data_off = d, 0, 8
            if d <= 0 or size < 8 + rows * d:
                fatal(f"{path} is shorter than its header's {rows} x {d}")
        else:
            fatal(f"{path}: expected a .bvecs or .u8bin file")
        self.path, self.d, self.rows = path, d, rows
        self.fd = os.open(path, os.O_RDONLY)

    def pread(self, buf, row0, count):
        """Read rows [row0, row0 + count) as stored into buf[:count]."""
        mv = memoryview(buf[:count]).cast("B")
        want, got = mv.nbytes, 0
        off = self.data_off + row0 * self.stride
        while got < want:
            r = os.preadv(self.fd, [mv[got:]], off + got)
            if r == 0:
                fatal(f"{self.path}: unexpected end of file at row {row0:,}")
            got += r
        if self.hdr:
            heads = buf[:count, :4].copy().view("<u4")
            if (heads[[0, -1]] != self.d).any():
                fatal(f"{self.path}: a row near {row0:,} does not start with d={self.d}")


class SlabReader:
    """Rows [0, stop) of a U8Base in slabs, one slab read ahead by a thread."""

    def __init__(self, base, stop, slab_rows, nbuf=3):
        self.base, self.stop, self.slab_rows = base, int(stop), int(slab_rows)
        self.free, self.full = queue.Queue(), queue.Queue()
        for _ in range(nbuf):
            self.free.put(np.empty((self.slab_rows, base.stride), dtype=np.uint8))
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _work(self):
        try:
            for row0 in range(0, self.stop, self.slab_rows):
                cnt = min(self.slab_rows, self.stop - row0)
                buf = self.free.get()
                self.base.pread(buf, row0, cnt)
                self.full.put((row0, cnt, buf))
            self.full.put(None)
        except BaseException as e:
            self.full.put(e)

    def __iter__(self):
        while True:
            item = self.full.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            row0, cnt, buf = item
            yield row0, buf[:cnt, self.base.hdr:]
            self.free.put(buf)


def systematic_sample(base, n_base, n_train, seed, slab_rows):
    """Rows floor(phase + i * n_base / n_train), i < n_train, phase random."""
    step = n_base / n_train
    phase = np.random.default_rng(seed).uniform(0.0, step)
    rows = (phase + np.arange(n_train, dtype=np.float64) * step).astype(np.int64)
    np.minimum(rows, n_base - 1, out=rows)
    xt = np.empty((n_train, base.d), dtype=np.uint8)
    t0 = last = time.time()
    for row0, slab in SlabReader(base, n_base, slab_rows):
        lo = np.searchsorted(rows, row0)
        hi = np.searchsorted(rows, row0 + slab.shape[0])
        xt[lo:hi] = slab[rows[lo:hi] - row0]
        if time.time() - last > 30:
            last = time.time()
            print(f"[sample] {row0 + slab.shape[0]:,}/{n_base:,} rows read "
                  f"({(row0 + slab.shape[0]) * base.stride / (last - t0) / 1e9:.2f} GB/s)",
                  flush=True)
    print(f"[sample] {n_train:,} of {n_base:,} rows in {time.time() - t0:.0f} s",
          flush=True)
    return xt


def widen_f32(src, out=None, pool=None, nthreads=1):
    """float32 copy of a uint8 block, split across threads."""
    n = src.shape[0]
    if out is None or out.shape[0] < n:
        out = np.empty((n, src.shape[1]), dtype=np.float32)
    dst = out[:n]
    if pool is None or nthreads <= 1 or n < 16384:
        np.copyto(dst, src, casting="unsafe")
        return dst
    edges = np.linspace(0, n, nthreads + 1).astype(np.int64)

    def work(t):
        a, b = int(edges[t]), int(edges[t + 1])
        if b > a:
            np.copyto(dst[a:b], src[a:b], casting="unsafe")

    pool.map(work, range(nthreads))
    return dst


def subsample_rows(n, cap, rng):
    """Sorted indices of cap rows out of n, or None when n <= cap."""
    if cap <= 0 or n <= cap:
        return None
    sel = rng.choice(n, size=cap, replace=False)
    sel.sort()
    return sel


def chunked_assign(xt, centroids, index, batch, pool, nthreads):
    """Nearest centroid of every row of xt, widening one batch at a time."""
    n, d = xt.shape
    index.reset()
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    out = np.empty(n, dtype=np.int64)
    buf = np.empty((min(batch, n), d), dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        _, I = index.search(widen_f32(xt[s:e], out=buf, pool=pool, nthreads=nthreads), 1)
        out[s:e] = I[:, 0]
    if int(out.min()) < 0:
        fatal("first-level assignment found no neighbor for some rows")
    return out


def two_level_clustering(xt, nc1, nc2, niter, l1_cap_ppc, sub_cap_ppc, assign_batch,
                         pool, nthreads, ngpu, seed):
    """Returns (centroids in cell order, first-level centroids, centroids per cell)."""
    n, d = xt.shape
    rng = np.random.default_rng(seed)
    kmargs = dict(gpu=ngpu, seed=seed, verbose=True, min_points_per_centroid=5,
                  spherical=False)

    t0 = time.time()
    l1_mppc = l1_cap_ppc if l1_cap_ppc > 0 else (1 << 30)
    km = faiss.Kmeans(d, nc1, niter=niter, max_points_per_centroid=l1_mppc, **kmargs)
    sel = subsample_rows(n, l1_cap_ppc * nc1, rng) if l1_cap_ppc > 0 else None
    x1 = widen_f32(xt if sel is None else xt[sel], pool=pool, nthreads=nthreads)
    km.index.reset()
    km.train(x1)
    del x1
    centroids1 = np.ascontiguousarray(km.centroids, dtype=np.float32)
    assign1 = chunked_assign(xt, centroids1, km.index, assign_batch, pool, nthreads)
    bc = np.bincount(assign1, minlength=nc1)
    o = assign1.argsort()
    del km
    print(f"[l1] {nc1} cells in {time.time() - t0:.0f} s, sizes {bc.min()}-{bc.max()}",
          flush=True)

    bc_sum = np.cumsum(bc)
    all_nc2 = bc_sum * nc2 // bc_sum[-1]
    all_nc2[1:] -= all_nc2[:-1]
    bad = [c for c in range(nc1) if all_nc2[c] <= 0 or bc[c] < all_nc2[c]]
    if bad:
        fatal(f"{len(bad)} of {nc1} first-level cells have fewer training points than "
              f"centroids; lower --nc1 or raise --pts-per-centroid")

    sub_args = dict(kmargs, verbose=False)
    if sub_cap_ppc <= 0:
        sub_args["max_points_per_centroid"] = 1 << 30
    elif sub_cap_ppc != 256:
        sub_args["max_points_per_centroid"] = sub_cap_ppc
    km = faiss.Kmeans(d, 1, **sub_args)
    t0 = time.time()
    i0 = 0
    c2 = []
    xsub = None
    for c1 in range(nc1):
        k2 = int(all_nc2[c1])
        i1 = i0 + int(bc[c1])
        subset = o[i0:i1]
        cap = subsample_rows(subset.shape[0], sub_cap_ppc * k2, rng) if sub_cap_ppc > 0 else None
        if cap is not None:
            subset = np.sort(subset[cap])
        ns = int(subset.shape[0])
        if xsub is None or xsub.shape[0] < ns:
            xsub = np.empty((ns, d), dtype=np.float32)
        widen_f32(xt[subset], out=xsub, pool=pool, nthreads=nthreads)
        km.reset(k2)
        km.index.reset()
        km.train(xsub[:ns])
        c2.append(np.ascontiguousarray(km.centroids, dtype=np.float32))
        i0 = i1
        if c1 % max(1, nc1 // 20) == 0:
            print(f"[l2] cell {c1 + 1}/{nc1}, {time.time() - t0:.0f} s", flush=True)
    out = np.vstack(c2)
    if out.shape != (nc2, d):
        fatal(f"clustering produced {out.shape}, expected ({nc2}, {d})")
    return out, centroids1, np.asarray(all_nc2, dtype=np.int64)


def write_fvecs(path, m, block_rows=1 << 20):
    m = np.ascontiguousarray(m, dtype=np.float32)
    k, d = m.shape
    with open(path, "wb") as f:
        out = np.empty((min(block_rows, k), d + 1), dtype=np.int32)
        out[:, 0] = d
        for s in range(0, k, block_rows):
            e = min(s + block_rows, k)
            out[: e - s, 1:] = m[s:e].view(np.int32)
            out[: e - s].tofile(f)


def write_l1meta(path, centroids1, counts):
    c1 = np.ascontiguousarray(centroids1, dtype=np.float32)
    cnt = np.ascontiguousarray(counts, dtype=np.uint32)
    nc1, d = c1.shape
    with open(path, "wb") as f:
        np.array([L1META_MAGIC, L1META_VERSION, nc1, int(cnt.sum()), d],
                 dtype=np.uint64).tofile(f)
        cnt.tofile(f)
        c1.tofile(f)


def mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Two-level k-means on GPU over a uint8 .bvecs/.u8bin base.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", required=True, help=".bvecs or .u8bin base vectors")
    ap.add_argument("--n-base", type=int, default=None,
                    help="use the first N rows (default: all)")
    ap.add_argument("--C", type=int, required=True, help="number of centroids")
    ap.add_argument("--nc1", type=int, default=None,
                    help="first-level cells (default: round(sqrt(C)))")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pts-per-centroid", type=int, default=128,
                    help="training rows per centroid")
    ap.add_argument("--niter", type=int, default=25, help="first-level iterations")
    ap.add_argument("--l1-cap-ppc", type=int, default=2000,
                    help="points per centroid used by the first-level k-means")
    ap.add_argument("--sub-cap-ppc", type=int, default=256,
                    help="points per centroid used by each cell's k-means")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed (default: drawn at random and printed)")
    ap.add_argument("--ngpu", type=int, default=None,
                    help="GPUs to use (default: every GPU faiss sees)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--scan-slab-mb", type=int, default=256, help="read size of the scan")
    ap.add_argument("--assign-batch", type=int, default=1_000_000,
                    help="rows per batch of the first-level assignment")
    args = ap.parse_args()

    base = U8Base(args.base)
    n_base = args.n_base or base.rows
    if n_base > base.rows:
        fatal(f"--n-base {n_base:,} but {args.base} holds {base.rows:,} rows")
    d, C = base.d, args.C
    nc1 = args.nc1 or int(round(math.sqrt(C)))
    n_train = min(args.pts_per_centroid * C, n_base)
    if n_train < C:
        fatal(f"{n_train:,} training rows for {C:,} centroids")
    ngpu = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    if args.ngpu is not None:
        ngpu = min(ngpu, args.ngpu)
    if ngpu <= 0:
        fatal("faiss sees no GPU; this tool needs faiss with GPU support")
    need = n_train * (d + 16) + 8 * C * d
    avail = mem_available()
    if avail is not None and need > avail:
        fatal(f"the training needs about {need / 2**30:,.0f} GiB of RAM, "
              f"{avail / 2**30:,.0f} GiB are available; lower --pts-per-centroid")
    seed = args.seed
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    os.makedirs(args.out_dir, exist_ok=True)
    faiss.omp_set_num_threads(args.threads)
    print(f"[gpu_kmeans] base={args.base} rows={n_base:,} d={d} C={C:,} nc1={nc1} "
          f"n_train={n_train:,} gpus={ngpu} seed={seed}", flush=True)

    t0 = time.time()
    slab_rows = max(1, (args.scan_slab_mb << 20) // base.stride)
    xt = systematic_sample(base, n_base, n_train, seed, slab_rows)
    pool = ThreadPool(args.threads)
    cen, cen1, counts = two_level_clustering(
        xt, nc1, C, args.niter, args.l1_cap_ppc, args.sub_cap_ppc, args.assign_batch,
        pool, args.threads, ngpu, seed)
    del xt
    pool.close()
    trained = os.path.join(args.out_dir, f"trained_centroids_{C}.fvecs")
    l1meta = os.path.join(args.out_dir, f"trained_centroids_{C}.l1meta")
    write_fvecs(trained + ".part", cen)
    write_l1meta(l1meta + ".part", cen1, counts)
    os.replace(l1meta + ".part", l1meta)
    os.replace(trained + ".part", trained)
    print(f"[gpu_kmeans] wrote {trained} and its .l1meta in {time.time() - t0:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
