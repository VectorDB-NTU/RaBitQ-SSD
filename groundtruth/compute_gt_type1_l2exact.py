#!/usr/bin/env python3
"""EXACT L2 KNN ground truth, written as DiskANN type-1 (ids + squared-L2 dists),
depth `maxk`.

Candidates are retrieved by EXACT L2 (faiss.knn_gpu / faiss.knn with METRIC_L2), which is
correct for non-normalized as well as near-normalized data. Retrieval by INNER PRODUCT
is exact only for unit-normalized data: where norms vary at all, L2 order and IP order
disagree, and the true L2 tail can fall outside an IP top-M. This script streams the
base in shards,
keeping the running top-M smallest-L2 candidates per query, then re-ranks those M with
float64 squared-L2 -> true top-maxk.

Two things make that exact rather than merely close:

  * Retrieval runs on CENTERED copies. faiss computes ||q||^2 + ||x||^2 - 2<q,x> in
    fp32; for data far from the origin those two large terms cancel and the fp32
    error grows with ||x||^2 while the distances stay small, which destroys the
    ranking -- and the fp64 re-rank cannot repair it, because by then the wrong
    candidates have been chosen. Squared L2 is translation-invariant, so a common
    offset changes no distance while restoring the conditioning.
  * The kept candidates are re-scored in float64 against the ORIGINAL vectors, so
    the emitted distances are the true ones, not the centered ones.

The run prints the fp32 cancellation floor and the candidate margin dist[M]-dist[K],
and warns when the margin is not comfortably above the floor.

Output: for EACH name in --out-names, a depth-`maxk` type-1 file
    [int32 nq][int32 maxk][nq*maxk uint32 ids][nq*maxk float32 squared-L2 dists]
All output files share identical depth-`maxk` content; gt10/gt100/gt1000 are byte-identical
so DiskANN-style tie extension has depth to capture ties at d_K for any K<=maxk.
"""
import os
import time
import shutil
import argparse
import numpy as np
import faiss


def read_fbin_meta(p):
    n, d = np.fromfile(p, dtype=np.int32, count=2)
    return int(n), int(d)


def vec_dtype(p):
    """Value type of a .fbin-layout file, from its extension: .u8bin is uint8 (e.g.
    DINO-10B), anything else float32. Both share the [int32 n][int32 d] header."""
    ext = os.path.splitext(p)[1].lower()
    if ext == ".i8bin":
        raise SystemExit(f"{p}: int8 vectors (.i8bin) are not supported")
    return np.uint8 if ext == ".u8bin" else np.float32


def write_type1(path, ids, dists):
    nq, k = ids.shape
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        np.array([nq, k], dtype=np.int32).tofile(f)
        np.ascontiguousarray(ids, dtype=np.uint32).tofile(f)
        np.ascontiguousarray(dists, dtype=np.float32).tofile(f)
    os.rename(tmp, path)
    print(f"  wrote {path}  (nq={nq} k={k}, type-1 ids+dist, {os.path.getsize(path)} B)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-names", default="gt10,gt100,gt1000")
    ap.add_argument("--maxk", type=int, default=2000, help="GT depth written to every out file")
    ap.add_argument("--rerank-m", type=int, default=2048, help="exact-L2 candidates/query for fp64 re-rank")
    ap.add_argument("--shard", type=int, default=10_000_000,
                    help="base rows per knn shard; lower it only if a shard does not fit in GPU memory")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cpu", action="store_true",
                    help="use faiss CPU brute-force knn (BLAS) instead of GPU; exact, no CUDA needed")
    ap.add_argument("--threads", type=int, default=0, help="OMP threads for --cpu (0=faiss default)")
    ap.add_argument("--backup-dir", default=None)
    args = ap.parse_args()

    nb, d = read_fbin_meta(args.base)
    nq, dq = read_fbin_meta(args.queries)
    assert dq == d, (dq, d)
    # A uint8 base stays uint8 on disk; every use below widens what it reads.
    base = np.memmap(args.base, dtype=vec_dtype(args.base), mode="r", offset=8, shape=(nb, d))
    q = np.ascontiguousarray(
        np.memmap(args.queries, dtype=vec_dtype(args.queries), mode="r", offset=8,
                  shape=(nq, d)), dtype=np.float32)
    K = args.maxk
    M = min(max(args.rerank_m, K), 2048)
    # Retrieval keeps M candidates and the re-rank picks the best K of them, so
    # K > M is unsatisfiable. The cap is 2048 because that is faiss's GPU knn
    # limit. Fail here rather than after the whole retrieval pass.
    if K > M:
        raise SystemExit(
            f"--maxk {K} exceeds the retrieval depth {M}: the candidate set is "
            f"capped at 2048, so no deeper ground truth can be produced. "
            f"Lower --maxk to {M} or less.")
    out_names = [s for s in args.out_names.split(",") if s.strip()]
    print(f"[gt1-l2] base nb={nb:,} d={d}  queries nq={nq}  maxk={K}  rerank_m={M}  shard={args.shard:,} "
          f"gpu={args.gpu}  out={out_names}  (EXACT L2 retrieval)", flush=True)
    bn = np.linalg.norm(base[:2000].astype(np.float64), axis=1)
    qn = np.linalg.norm(q[:2000].astype(np.float64), axis=1)
    print(f"[gt1-l2] base L2 norms: min={bn.min():.4f} mean={bn.mean():.4f} max={bn.max():.4f}; "
          f"query: min={qn.min():.4f} mean={qn.mean():.4f} max={qn.max():.4f} "
          f"(exact-L2 path -> normalization irrelevant to correctness)", flush=True)

    use_cpu = args.cpu
    res = None
    if use_cpu:
        if args.threads > 0:
            faiss.omp_set_num_threads(args.threads)
        print(f"[gt1-l2] backend=CPU faiss.knn (BLAS), threads={faiss.omp_get_max_threads()}", flush=True)
    else:
        res = faiss.StandardGpuResources()
        # Cap the temp pool (default is ~18% of GPU mem) so init is robust to transient
        # GPU contention; knn_gpu just tiles smaller scratch, still exact.
        res.setTempMemory(4 * 1024 * 1024 * 1024)  # 4 GiB
        print(f"[gt1-l2] backend=GPU device={args.gpu}", flush=True)

    def shard_knn(xb):
        if use_cpu:
            D, I = faiss.knn(q_ret, xb, M, metric=faiss.METRIC_L2)  # CPU exact, D=squared L2
            return D, I
        return faiss.knn_gpu(res, q_ret, xb, M, metric=faiss.METRIC_L2, device=args.gpu)

    # ---- centering: conditioning only, not a change of metric ----------------
    # faiss's brute force computes ||q||^2 + ||x||^2 - 2<q,x> in fp32. When the
    # vectors sit far from the origin, those two large terms cancel and the fp32
    # absolute error grows with ||x||^2 while the distance itself may stay small;
    # past ||x||^2 * 2^-24 ~ d_typical the ranking is destroyed, and the fp64
    # re-rank cannot repair it because the candidates are already the wrong ones.
    # Squared L2 is translation-invariant, so subtracting a common offset from
    # base and queries alike changes no distance (exactly: the subtraction of
    # nearby floats is itself exact) while shrinking ||x||^2 to the data's own
    # spread. The offset need not be the true mean -- any nearby point works.
    off_v = np.asarray(base[:min(nb, 500_000)].mean(axis=0, dtype=np.float64),
                       dtype=np.float32)
    r_raw = float(np.linalg.norm(base[:min(nb, 10_000)].astype(np.float64), axis=1).max())
    r_cen = float(np.linalg.norm((base[:min(nb, 10_000)] - off_v).astype(np.float64), axis=1).max())
    print(f"[gt1-l2] centering for retrieval: max||x|| {r_raw:.4g} -> {r_cen:.4g} "
          f"(fp32 cancellation floor {r_raw**2 * 2**-24:.3g} -> {r_cen**2 * 2**-24:.3g}); "
          f"distances are unchanged", flush=True)
    q_ret = np.ascontiguousarray(q - off_v)

    run_D = np.full((nq, M), np.inf, dtype=np.float32)   # smallest-L2 running set
    run_I = np.full((nq, M), -1, dtype=np.int64)
    t0 = time.time()
    for s in range(0, nb, args.shard):
        e = min(s + args.shard, nb)
        xb = np.ascontiguousarray(base[s:e]) - off_v
        D, I = shard_knn(xb)  # D=squared L2
        I = I.astype(np.int64) + s
        allD = np.concatenate([run_D, D.astype(np.float32)], axis=1)
        allI = np.concatenate([run_I, I], axis=1)
        sel = np.argpartition(allD, M - 1, axis=1)[:, :M]   # keep M SMALLEST
        run_D = np.take_along_axis(allD, sel, axis=1)
        run_I = np.take_along_axis(allI, sel, axis=1)
        del xb, D, I, allD, allI
        if (s // args.shard) % 5 == 0:
            el = time.time() - t0
            print(f"[gt1-l2] retrieve {e:,}/{nb:,} ({100*e/nb:.1f}%) {el:.0f}s "
                  f"ETA {(nb-e)/max(1,e)*el/60:.1f}min", flush=True)
    print(f"[gt1-l2] retrieval {time.time()-t0:.1f}s; float64 squared-L2 re-rank of {M} cand/query", flush=True)

    tr = time.time()
    cand = run_I
    uniq, inv = np.unique(cand.ravel(), return_inverse=True)
    inv = inv.reshape(nq, M)
    vecs = np.zeros((len(uniq), d), dtype=np.float32)
    gi = np.where(uniq >= 0)[0]
    for i in range(0, len(gi), 1_000_000):
        j = min(i + 1_000_000, len(gi))
        vecs[gi[i:j]] = base[uniq[gi[i:j]]]
    q64 = q.astype(np.float64)
    out_I = np.full((nq, K), -1, dtype=np.int64)
    out_D = np.zeros((nq, K), dtype=np.float32)
    nonmono = 0
    margins = np.full(nq, np.inf, dtype=np.float64)  # dist[M-th cand] - dist[K-th kept]
    for i in range(nq):
        cv = vecs[inv[i]].astype(np.float64)
        dd = ((q64[i] - cv) ** 2).sum(1)
        dd[cand[i] < 0] = np.inf
        order = np.argsort(dd, kind="stable")
        out_I[i] = cand[i][order[:K]]
        out_D[i] = dd[order[:K]].astype(np.float32)
        if M > K and np.isfinite(dd[order[M - 1]]):
            margins[i] = dd[order[M - 1]] - dd[order[K - 1]]
        if np.any(np.diff(out_D[i]) < -1e-6):
            nonmono += 1
    print(f"[gt1-l2] re-rank {time.time()-tr:.1f}s; rows not ascending (should be 0): {nonmono}", flush=True)

    # A true top-K neighbour is lost only if fp32 retrieval ranked it past M. The
    # headroom for that is dist[M] - dist[K]; the error that has to be crossed is
    # the cancellation floor printed above. Report the ratio so a pathological
    # dataset is visible instead of silently producing a wrong ground truth.
    floor = r_cen ** 2 * 2 ** -24
    if M > K:
        mn = float(np.min(margins))
        print(f"[gt1-l2] candidate margin dist[{M}]-dist[{K}]: min={mn:.6g} "
              f"median={float(np.median(margins)):.6g}  vs fp32 floor {floor:.3g} "
              f"-> {mn / max(floor, 1e-30):.3g}x", flush=True)
        if mn < 50 * floor:
            print(f"[gt1-l2] WARNING: only {mn / max(floor, 1e-30):.1f}x headroom over the fp32 "
                  f"retrieval error. Lower --maxk (so M > K by more) or check whether this data "
                  f"is extremely far from the origin; the deepest ranks may be wrong.", flush=True)
    else:
        print(f"[gt1-l2] NOTE: maxk == rerank depth ({M}), so the candidate set has no margin; "
              f"the K nearest are exact only where fp32 retrieval already ranked them within {M}. "
              f"fp32 floor here is {floor:.3g}.", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)
    ids32 = out_I.astype(np.uint32)
    for name in out_names:
        p = os.path.join(args.out_dir, name)
        if args.backup_dir and os.path.exists(p):
            bp = os.path.join(args.backup_dir, name)
            if not os.path.exists(bp):
                shutil.copy2(os.path.realpath(p), bp)
            if os.path.islink(p):
                os.unlink(p)
        write_type1(p, ids32, out_D)
    print("[gt1-l2] done", flush=True)


if __name__ == "__main__":
    main()
