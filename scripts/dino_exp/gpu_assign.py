#!/usr/bin/env python3
"""Assign every base vector to its nearest trained centroid on GPU (DINO).

Reads the output of gpu_kmeans.py and the first N base vectors, and writes

    <out-dir>/clusterids_<C>.cid64   the cluster of every vector
    <out-dir>/centroids_<C>.fvecs    each cluster's mean (an empty cluster
                                     keeps its trained centroid)

Each vector is compared with the first-level centroids, and then with the
centroids of its `probes` nearest first-level cells; the nearest of those
wins. Distances use TF32 matrix products on vectors centred by the mean of
the trained centroids. The base is read once, in batches, and the batches are
spread over the GPUs.

The base is .bvecs ([u32 d][d x uint8] per row) or .u8bin ([u32 n][u32 d]
then n*d uint8). Needs numpy and PyTorch with CUDA.

.cid64 layout (little-endian): [u64 magic "DNDCID64"][u64 version = 1][u64 n]
[u64 C][n x u32 cluster ids].
"""
import argparse
import errno
import math
import mmap
import os
import queue
import sys
import threading
import time
import traceback

import numpy as np

L1META_MAGIC = 0x4154454D314C4E44
CID64_MAGIC = 0x3436444943444E44
PAD_SCORE = 1e30
ALIGN = 4096


def fatal(msg):
    raise SystemExit(f"[FATAL] {msg}")


def fvecs_open(path):
    """Memory-map a .fvecs: returns (rows x (d+1) int32 memmap, rows, d)."""
    d = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    size = os.path.getsize(path)
    if d <= 0 or size % (4 * (d + 1)):
        fatal(f"{path} is not a .fvecs file")
    n = size // (4 * (d + 1))
    return np.memmap(path, dtype=np.int32, mode="r", shape=(n, d + 1)), n, d


def fvecs_rows(mm, s, e):
    return np.array(mm[s:e, 1:].view(np.float32), dtype=np.float32, copy=True)


def read_l1meta(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.uint64, count=5)
        if hdr.size != 5 or int(hdr[0]) != L1META_MAGIC or int(hdr[1]) != 1:
            fatal(f"{path} is not a .l1meta file")
        nc1, C, d = (int(v) for v in hdr[2:])
        cnt = np.fromfile(f, dtype=np.uint32, count=nc1).astype(np.int64)
        c1 = np.fromfile(f, dtype=np.float32, count=nc1 * d).reshape(nc1, d)
    if c1.shape != (nc1, d) or int(cnt.sum()) != C or (cnt <= 0).any():
        fatal(f"{path} is truncated or inconsistent")
    return c1, cnt, C, d


class Model:
    """The trained centroids in cell order, padded per cell for the GPU."""

    def __init__(self, cen_path, l1_path, pad):
        self.mm, C, d = fvecs_open(cen_path)
        c1, counts, C1, d1 = read_l1meta(l1_path)
        if (C1, d1) != (C, d):
            fatal(f"{l1_path} describes {C1} x {d1}, {cen_path} holds {C} x {d}")
        self.C, self.d, self.nc1 = C, d, c1.shape[0]
        self.counts, self.c1 = counts, np.ascontiguousarray(c1, dtype=np.float32)
        self.off = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self.pad = pad
        self.k2p = ((counts + pad - 1) // pad) * pad
        self.poff = np.concatenate(([0], np.cumsum(self.k2p))).astype(np.int64)
        self.P = int(self.poff[-1])
        self.nc1p = int(((self.nc1 + pad - 1) // pad) * pad)
        acc = np.zeros(d, dtype=np.float64)
        blk = max(1, min(C, 1 << 16))
        for s in range(0, C, blk):
            acc += fvecs_rows(self.mm, s, min(s + blk, C)).astype(np.float64).sum(0)
        self.mu = (acc / C).astype(np.float32)
        # A padding slot scores PAD_SCORE; no real score can come close.
        mu64 = self.mu.astype(np.float64)
        x_norm = float(np.sqrt((np.maximum(mu64, 255.0 - mu64) ** 2).sum()))
        c_norm2 = 0.0
        for s in range(0, C, blk):
            r = fvecs_rows(self.mm, s, min(s + blk, C)).astype(np.float64) - mu64
            c_norm2 = max(c_norm2, float((r * r).sum(1).max()))
        c_norm2 = max(c_norm2, float(((c1.astype(np.float64) - mu64) ** 2).sum(1).max()))
        if not c_norm2 + 2.0 * x_norm * math.sqrt(c_norm2) < PAD_SCORE / 16:
            fatal("centroid magnitudes too large for the padding sentinel")


class GpuWorker(threading.Thread):
    def __init__(self, torch, dev_id, model, args, jobs, results, free, err):
        super().__init__(daemon=True)
        self.torch, self.m, self.a = torch, model, args
        self.dev = torch.device(f"cuda:{dev_id}")
        self.jobs, self.results, self.free, self.err = jobs, results, free, err
        self.widx = None
        torch.cuda.set_device(self.dev)
        m = model
        mu = torch.from_numpy(m.mu).to(self.dev).double()
        self.tbl = torch.zeros((m.P, m.d), dtype=torch.float32, device=self.dev)
        blk = max(1, min(m.C, 1 << 14))
        for j in range(m.nc1):
            a, b, pa = int(m.off[j]), int(m.off[j + 1]), int(m.poff[j])
            for s in range(a, b, blk):
                e = min(s + blk, b)
                host = torch.from_numpy(fvecs_rows(m.mm, s, e))
                self.tbl[pa + (s - a): pa + (e - a)].copy_(
                    (host.to(self.dev).double() - mu).to(torch.float32))
        self.bias = torch.full((m.P,), PAD_SCORE, dtype=torch.float32, device=self.dev)
        real = np.zeros(m.P, dtype=bool)
        for j in range(m.nc1):
            real[int(m.poff[j]): int(m.poff[j]) + int(m.counts[j])] = True
        real = torch.from_numpy(real).to(self.dev)
        for s in range(0, m.P, 1 << 16):
            e = min(s + (1 << 16), m.P)
            nb = (self.tbl[s:e].double() ** 2).sum(1)
            self.bias[s:e] = torch.where(real[s:e], nb.to(torch.float32), self.bias[s:e])
        self.c1 = torch.zeros((m.nc1p, m.d), dtype=torch.float32, device=self.dev)
        self.c1[: m.nc1] = (torch.from_numpy(m.c1).to(self.dev).double() - mu).to(torch.float32)
        self.c1bias = torch.full((m.nc1p,), PAD_SCORE, dtype=torch.float32, device=self.dev)
        self.c1bias[: m.nc1] = (self.c1[: m.nc1].double() ** 2).sum(1).to(torch.float32)
        self.mu_w = mu.to(torch.float32)
        self.sums = torch.zeros((m.C, m.d), dtype=torch.float64, device=self.dev)
        self.counts = torch.zeros(m.C, dtype=torch.int64, device=self.dev)
        self.ones = torch.ones(args.accum_chunk, dtype=torch.int64, device=self.dev)
        self.bufs = [torch.empty((args.batch, m.d), dtype=torch.uint8, device=self.dev)
                     for _ in range(args.dbuf)]
        self.copy_stream = torch.cuda.Stream(device=self.dev)
        self.rows_cache = {}
        torch.cuda.synchronize(self.dev)

    def scores(self, x, tbl, bias):
        return self.torch.addmm(bias, x, tbl.T, beta=1.0, alpha=-2.0)

    def l1(self, xw):
        torch = self.torch
        n = xw.shape[0]
        out = torch.empty((n, self.a.probes), dtype=torch.int32, device=self.dev)
        for s in range(0, n, self.a.l1_chunk):
            e = min(s + self.a.l1_chunk, n)
            sc = self.scores(xw[s:e], self.c1, self.c1bias)
            out[s:e] = torch.topk(sc, self.a.probes, dim=1, largest=False,
                                  sorted=False).indices.to(torch.int32)
            del sc
        return out

    def l2(self, xw):
        torch = self.torch
        n = xw.shape[0]
        cells_s, srt = torch.sort(self.l1(xw).reshape(-1))
        rows = self.rows_cache.get(n)
        if rows is None:
            rows = torch.arange(n, device=self.dev,
                                dtype=torch.int64).repeat_interleave(self.a.probes)
            self.rows_cache[n] = rows
        rows_s = rows[srt]
        uniq, cnts = torch.unique_consecutive(cells_s, return_counts=True)
        starts = torch.cumsum(cnts, 0) - cnts
        uniq_h, starts_h, cnts_h = uniq.tolist(), starts.tolist(), cnts.tolist()
        del cells_s, srt, uniq, cnts, starts
        if uniq_h and max(uniq_h) >= self.m.nc1:
            fatal("a padding cell was selected")
        best_v = torch.full((n,), float("inf"), dtype=torch.float32, device=self.dev)
        best_i = torch.zeros(n, dtype=torch.int64, device=self.dev)
        off, poff, k2p = self.m.off, self.m.poff, self.m.k2p
        for cell, st, ct in zip(uniq_h, starts_h, cnts_h):
            q = rows_s[st: st + ct]
            pa = int(poff[cell])
            pb = pa + int(k2p[cell])
            sc = self.scores(xw.index_select(0, q), self.tbl[pa:pb], self.bias[pa:pb])
            mv, am = sc.min(1)
            ov = best_v.index_select(0, q)
            better = mv < ov
            best_v[q] = torch.where(better, mv, ov)
            best_i[q] = torch.where(better, am + int(off[cell]), best_i.index_select(0, q))
            del sc, mv, am, ov, better
        return best_i

    def accum(self, xu, cid):
        for s in range(0, xu.shape[0], self.a.accum_chunk):
            e = min(s + self.a.accum_chunk, xu.shape[0])
            self.sums.index_add_(0, cid[s:e], xu[s:e].double())
            self.counts.index_add_(0, cid[s:e], self.ones[: e - s])

    def process(self, buf, n):
        torch = self.torch
        cid = torch.empty(n, dtype=torch.int32, device=self.dev)
        for s in range(0, n, self.a.gpu_chunk):
            e = min(s + self.a.gpu_chunk, n)
            xu = buf[s:e]
            xw = xu.to(torch.float32)
            xw -= self.mu_w
            best = self.l2(xw)
            cid[s:e] = best.to(torch.int32)
            self.accum(xu, best)
            del xw, best
        return cid.cpu().numpy()

    def run(self):
        try:
            self.torch.cuda.set_device(self.dev)
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                start, slot, n = job
                cid = self.process(self.bufs[slot], n)
                self.free.put((self.widx, slot))
                self.results.put((start, cid))
        except BaseException as exc:
            self.err.append(exc)
            self.results.put(None)
            self.free.put(None)   # wakes the main thread if it waits for a buffer


class BatchReader:
    """Rows [0, n_rows) of a .bvecs/.u8bin as (row0, rows x d uint8) batches in
    pinned memory, read by n_threads concurrent O_DIRECT streams. A batch is
    reused once the loop body that received it returns."""

    def __init__(self, path, n_rows, batch, n_threads, slots, torch, odirect=True,
                 chunk_mb=32):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".bvecs":
            d = int(np.fromfile(path, dtype="<u4", count=1)[0])
            self.stride, self.hdr, self.data_off = 4 + d, 4, 0
            file_rows = os.path.getsize(path) // self.stride
        elif ext == ".u8bin":
            file_rows, d = (int(v) for v in np.fromfile(path, dtype="<u4", count=2))
            self.stride, self.hdr, self.data_off = d, 0, 8
        else:
            fatal(f"{path}: expected a .bvecs or .u8bin file")
        if n_rows > file_rows:
            fatal(f"{path} holds {file_rows:,} rows, fewer than {n_rows:,}")
        self.path, self.d, self.n_rows, self.batch = path, d, n_rows, batch
        self.chunk_rows = max(1, min(batch, (chunk_mb << 20) // self.stride))
        self.scratch_bytes = (self.chunk_rows * self.stride + 2 * ALIGN + ALIGN - 1) // ALIGN * ALIGN
        self.slots = [torch.empty((batch, d), dtype=torch.uint8, pin_memory=True).numpy()
                      for _ in range(slots)]
        self.odirect = odirect and self._odirect_works()
        self.jobs, self.free, self.out = queue.Queue(), queue.Queue(), queue.Queue()
        for i in range(slots):
            self.free.put(i)
        self.stop = threading.Event()
        self.threads = [threading.Thread(target=self._worker, daemon=True)
                        for _ in range(n_threads)]
        self.threads.append(threading.Thread(target=self._drive, daemon=True))
        for t in self.threads:
            t.start()

    def _odirect_works(self):
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
        except OSError as e:
            if e.errno in (errno.EINVAL, errno.EOPNOTSUPP):
                return False
            raise
        try:
            buf = mmap.mmap(-1, ALIGN)
            os.preadv(fd, [memoryview(buf)], 0)
            return True
        except OSError:
            return False
        finally:
            os.close(fd)

    def _worker(self):
        try:
            fd = os.open(self.path, os.O_RDONLY | (os.O_DIRECT if self.odirect else 0))
            scratch = mmap.mmap(-1, self.scratch_bytes)
        except BaseException as e:
            # Answer every job with the error, so the batch fails instead of waiting.
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                job[0].put(e)
        snp = np.frombuffer(scratch, dtype=np.uint8)
        mv = memoryview(scratch)
        try:
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                done, dst, batch_row0, r0, r1 = job
                try:
                    b0 = self.data_off + r0 * self.stride
                    b1 = self.data_off + r1 * self.stride
                    a0 = b0 // ALIGN * ALIGN
                    a1 = (b1 + ALIGN - 1) // ALIGN * ALIGN
                    got = 0
                    while got < a1 - a0:
                        n = os.preadv(fd, [mv[got:a1 - a0]], a0 + got)
                        if n == 0:
                            break
                        got += n
                    if got < b1 - a0:
                        raise IOError(f"{self.path}: short read at row {r0:,}")
                    if not self.odirect:
                        os.posix_fadvise(fd, a0, got, os.POSIX_FADV_DONTNEED)
                    view = snp[b0 - a0: b1 - a0].reshape(r1 - r0, self.stride)
                    if self.hdr and (view[[0, -1], :4].copy().view("<u4") != self.d).any():
                        raise IOError(f"{self.path}: a row near {r0:,} does not start "
                                      f"with d={self.d}")
                    dst[r0 - batch_row0: r1 - batch_row0] = view[:, self.hdr:]
                    done.put(None)
                except BaseException as e:
                    done.put(e)
        finally:
            del snp, mv
            os.close(fd)

    def _drive(self):
        try:
            for pos in range(0, self.n_rows, self.batch):
                cnt = min(self.batch, self.n_rows - pos)
                slot = self.free.get()
                if slot is None or self.stop.is_set():
                    return
                done = queue.Queue()
                bounds = list(range(pos, pos + cnt, self.chunk_rows)) + [pos + cnt]
                for j in range(len(bounds) - 1):
                    self.jobs.put((done, self.slots[slot], pos, bounds[j], bounds[j + 1]))
                for _ in range(len(bounds) - 1):
                    e = done.get()
                    if e is not None:
                        raise e
                self.out.put((pos, cnt, slot))
            self.out.put(None)
        except BaseException as e:
            self.out.put(e)

    def __iter__(self):
        prev = None
        while True:
            if prev is not None:
                self.free.put(prev)
            item = self.out.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            pos, cnt, prev = item
            yield pos, self.slots[prev][:cnt]

    def close(self):
        self.stop.set()
        self.free.put(None)
        for _ in self.threads:
            self.jobs.put(None)


def assign(args, torch, model):
    C, d, n_base = model.C, model.d, args.n_base
    os.makedirs(args.out_dir, exist_ok=True)
    cid_path = os.path.join(args.out_dir, f"clusterids_{C}.cid64")
    cen_path = os.path.join(args.out_dir, f"centroids_{C}.fvecs")
    jobs = [queue.Queue() for _ in args.gpu_ids]
    results, free, err = queue.Queue(), queue.Queue(), []
    workers = []
    for k, g in enumerate(args.gpu_ids):
        w = GpuWorker(torch, g, model, args, jobs[k], results, free, err)
        w.widx = k
        workers.append(w)
        for slot in range(args.dbuf):
            free.put((k, slot))
    for w in workers:
        w.start()
    reader = BatchReader(args.base, n_base, args.batch, args.read_threads, args.read_slots,
                         torch, odirect=not args.no_odirect)
    if reader.d != d:
        fatal(f"{args.base} has d={reader.d}, the centroids d={d}")
    f = open(cid_path + ".part", "wb")
    np.array([CID64_MAGIC, 1, n_base, C], dtype="<u8").tofile(f)
    pending, written, issued, collected = {}, 0, 0, 0
    t0 = last = time.time()

    def collect(block):
        nonlocal written, collected
        try:
            item = results.get(block=block)
        except queue.Empty:
            return False
        if item is None:
            raise RuntimeError("a GPU worker failed") from (err[0] if err else None)
        collected += 1
        pending[item[0]] = item[1]
        while written in pending:
            blk = pending.pop(written)
            blk.astype("<u4").tofile(f)
            written += blk.size
        return True

    try:
        for start, arr in reader:
            while collect(False):
                pass
            item = free.get()
            if item is None:
                raise RuntimeError("a GPU worker failed") from (err[0] if err else None)
            k, slot = item
            w = workers[k]
            with torch.cuda.device(w.dev), torch.cuda.stream(w.copy_stream):
                w.bufs[slot][: arr.shape[0]].copy_(torch.from_numpy(arr), non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(w.copy_stream)
            ev.synchronize()
            jobs[k].put((start, slot, arr.shape[0]))
            issued += 1
            if time.time() - last > 30:
                last = time.time()
                print(f"[assign] {written:,}/{n_base:,} rows, "
                      f"{written / (last - t0) / 1e6:.2f} M rows/s", flush=True)
        while collected < issued:
            collect(True)
    finally:
        reader.close()
    f.close()
    if written != n_base:
        fatal(f"wrote {written:,} cluster ids, expected {n_base:,}")
    for q in jobs:
        q.put(None)

    counts = np.zeros(C, dtype=np.int64)
    for w in workers:
        counts += w.counts.cpu().numpy()
    if int(counts.sum()) != n_base:
        fatal("the per-cluster counts do not add up to the base size")
    with open(cen_path + ".part", "wb") as out:
        for s in range(0, C, 1 << 15):
            e = min(s + (1 << 15), C)
            acc = workers[0].sums[s:e].cpu().numpy()
            for w in workers[1:]:
                acc = acc + w.sums[s:e].cpu().numpy()
            cnt = counts[s:e]
            nz = cnt > 0
            blk = fvecs_rows(model.mm, s, e)
            if nz.any():
                blk[nz] = (acc[nz] / cnt[nz, None]).astype(np.float32)
            rows = np.empty((e - s, d + 1), dtype=np.int32)
            rows[:, 0] = d
            rows[:, 1:] = blk.view(np.int32)
            rows.tofile(out)
    os.replace(cid_path + ".part", cid_path)
    os.replace(cen_path + ".part", cen_path)
    print(f"[assign] {n_base:,} rows in {time.time() - t0:.0f} s; "
          f"{int((counts == 0).sum()):,} empty clusters kept their trained centroid",
          flush=True)
    print(f"[assign] wrote {cid_path} and {cen_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Two-level nearest-centroid assignment on GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", required=True, help=".bvecs or .u8bin base vectors")
    ap.add_argument("--n-base", type=int, default=None,
                    help="assign the first N rows (default: all)")
    ap.add_argument("--centroids", required=True,
                    help="trained_centroids_<C>.fvecs from gpu_kmeans.py")
    ap.add_argument("--l1meta", default=None,
                    help="its .l1meta (default: next to --centroids)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--probes", type=int, default=8, help="first-level cells per vector")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated CUDA device ids (default: all)")
    ap.add_argument("--batch", type=int, default=4_000_000, help="rows per read batch")
    ap.add_argument("--gpu-chunk", type=int, default=1_048_576, help="rows per GPU step")
    ap.add_argument("--l1-chunk", type=int, default=262_144,
                    help="rows per first-level product")
    ap.add_argument("--accum-chunk", type=int, default=262_144,
                    help="rows per accumulation step")
    ap.add_argument("--pad-multiple", type=int, default=64,
                    help="pad each cell's centroids to a multiple of this")
    ap.add_argument("--dbuf", type=int, default=2, help="batch buffers per GPU")
    ap.add_argument("--read-threads", type=int, default=8)
    ap.add_argument("--read-slots", type=int, default=4)
    ap.add_argument("--no-odirect", action="store_true", help="read through the page cache")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        fatal("PyTorch sees no CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    except Exception:
        pass
    if args.l1meta is None:
        args.l1meta = os.path.splitext(args.centroids)[0] + ".l1meta"
    args.gpu_ids = ([int(v) for v in args.gpus.split(",") if v.strip()] if args.gpus
                    else list(range(torch.cuda.device_count())))
    model = Model(args.centroids, args.l1meta, args.pad_multiple)
    args.probes = min(args.probes, model.nc1)
    if args.n_base is None:
        ext = os.path.splitext(args.base)[1].lower()
        args.n_base = (os.path.getsize(args.base) // (4 + model.d) if ext == ".bvecs"
                       else int(np.fromfile(args.base, dtype="<u4", count=1)[0]))
    # A base smaller than one batch is read as one batch of its own size.
    args.batch = min(args.batch, args.n_base)
    print(f"[gpu_assign] base={args.base} rows={args.n_base:,} C={model.C:,} "
          f"nc1={model.nc1} probes={args.probes} gpus={args.gpu_ids}", flush=True)
    # Leave with os._exit either way: worker threads and CUDA state can stall or
    # abort an ordinary interpreter exit once something has failed.
    code = 0
    try:
        assign(args, torch, model)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
