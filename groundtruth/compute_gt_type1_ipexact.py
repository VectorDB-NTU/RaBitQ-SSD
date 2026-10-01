#!/usr/bin/env python3
"""
EXACT inner-product KNN ground truth, written as DiskANN type-1 (ids + distances),
depth `maxk`, for a NON-normalized dataset (e.g. DPR, ||v||~12.5, where IP-nearest
!= L2-nearest).

Counterpart of compute_gt_type1_l2exact.py, which emits a squared-L2 distance tail.
This one emits the index's own IP min-distance:

    dist = 1 - <q, x>          (dot_product_dis in rabitqlib/utils/space.hpp)

so rows are ascending-by-dist == descending-by-IP == nearest-first, matching the
"gt_dist ascending" convention of the L2 GT files. The querying binaries never
compare a search distance against the GT tail: count_correct_with_ties() scores by
id intersection and uses the tail only for the exact-equality tie walk
`gt_dist[t] == gt_dist[topk-1]`. So the two load-bearing invariants are
(1) rows emitted best-first, (2) tied neighbors carrying bit-identical float32.

Method: stream the base in GPU chunks, keep a running top-M candidate set per query by
fp32 inner product (TF32 disabled -> true fp32 accumulation), then re-rank those M
candidates with EXACT float64 inner product -> true top-maxk.

Backend: torch CUDA (the L2 counterpart uses faiss); requires a visible CUDA device.

Output: for EACH name in --out-names, a depth-`maxk` type-1 file
    [int32 nq][int32 maxk][nq*maxk uint32 ids][nq*maxk float32 (1-IP) dists]
All output files are byte-identical at depth `maxk` (gt10/gt100/gt1000 share content so
the tie-extension walk has depth to capture ties at d_K for any K <= maxk).
Existing files are backed up to --backup-dir first.
"""
import os
import time
import shutil
import argparse
import numpy as np
import torch


def read_fbin_meta(p):
    n, d = np.fromfile(p, dtype=np.int32, count=2)
    return int(n), int(d)


def write_type1(path, ids, dists):
    nq, k = ids.shape
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        np.array([nq, k], dtype=np.int32).tofile(f)
        np.ascontiguousarray(ids, dtype=np.uint32).tofile(f)
        np.ascontiguousarray(dists, dtype=np.float32).tofile(f)
    os.replace(tmp, path)
    print(f"  wrote {path}  (nq={nq} k={k}, type-1 ids+dist, {os.path.getsize(path)} B)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-names", default="gt10,gt100,gt1000")
    ap.add_argument("--maxk", type=int, default=2000, help="GT depth written to every out file")
    ap.add_argument("--rerank-m", type=int, default=2048, help="fp32-IP candidates/query for fp64 re-rank")
    ap.add_argument("--chunk", type=int, default=250_000, help="base rows per GPU GEMM chunk")
    ap.add_argument("--backup-dir", default=None, help="back up existing out files here before overwrite")
    args = ap.parse_args()

    # True fp32 accumulation: TF32 (10-bit mantissa) would corrupt the top-K boundary.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 must be off for exact fp32 GEMM"
    assert torch.cuda.is_available(), "no CUDA device visible (pin a healthy GPU via CUDA_VISIBLE_DEVICES)"
    dev = "cuda"

    nb, d = read_fbin_meta(args.base)
    nq, dq = read_fbin_meta(args.queries)
    assert dq == d, (dq, d)
    q = np.ascontiguousarray(
        np.memmap(args.queries, dtype=np.float32, mode="r", offset=8, shape=(nq, d)))
    K = args.maxk
    M = min(max(args.rerank_m, K), 2048)
    assert M >= K, (M, K)
    out_names = [s for s in args.out_names.split(",") if s.strip()]
    print(f"[gt1-ip] base nb={nb:,} d={d}  queries nq={nq}  maxk={K}  rerank_m={M}  "
          f"chunk={args.chunk:,}  gpu={torch.cuda.get_device_name(0)}  out={out_names}", flush=True)
    print(f"[gt1-ip] torch {torch.__version__}  tf32_matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"(exact fp32 candidate GEMM, float64 IP re-rank)", flush=True)
    qn = np.linalg.norm(q.astype(np.float64), axis=1)
    print(f"[gt1-ip] query L2 norms: min={qn.min():.4f} mean={qn.mean():.4f} max={qn.max():.4f} "
          f"(IP metric; normalization irrelevant to correctness)", flush=True)

    Qd = torch.from_numpy(q).to(dev)
    run_S = torch.full((nq, M), -float("inf"), device=dev, dtype=torch.float32)
    run_I = torch.full((nq, M), -1, device=dev, dtype=torch.long)

    t0 = time.time()
    nchunk = (nb + args.chunk - 1) // args.chunk
    with open(args.base, "rb") as f:
        f.seek(8)
        off = 0
        ci = 0
        while off < nb:
            m = min(args.chunk, nb - off)
            B = np.frombuffer(f.read(m * d * 4), dtype=np.float32).reshape(m, d)
            Bd = torch.from_numpy(B.copy()).to(dev)
            scores = Qd @ Bd.T                                    # (nq, m) fp32 IP
            kk = min(M, m)
            cv, cl = torch.topk(scores, kk, dim=1, largest=True, sorted=True)
            cg = cl.to(torch.long) + off
            cat_v = torch.cat([run_S, cv], dim=1)
            cat_g = torch.cat([run_I, cg], dim=1)
            run_S, sel = torch.topk(cat_v, M, dim=1, largest=True, sorted=True)
            run_I = torch.gather(cat_g, 1, sel)
            del Bd, scores, cv, cl, cg, cat_v, cat_g, sel
            off += m
            ci += 1
            if ci % 40 == 0 or off == nb:
                el = time.time() - t0
                print(f"[gt1-ip] retrieve {off:,}/{nb:,} ({100*off/nb:.1f}%) {el:.0f}s "
                      f"ETA {(nb-off)/max(1,off)*el/60:.1f}min  [chunk {ci}/{nchunk}]", flush=True)
    t_ret = time.time() - t0
    print(f"[gt1-ip] retrieval {t_ret:.1f}s; float64 IP re-rank of {M} cand/query", flush=True)

    cand = run_I.cpu().numpy()
    assert (cand >= 0).all(), "fewer than M candidates for some query"
    del Qd, run_S, run_I
    torch.cuda.empty_cache()

    tr = time.time()
    base = np.memmap(args.base, dtype=np.float32, mode="r", offset=8, shape=(nb, d))
    uniq, inv = np.unique(cand.ravel(), return_inverse=True)
    inv = inv.reshape(nq, M)
    print(f"[gt1-ip] gathering {len(uniq):,} unique candidate vectors "
          f"({len(uniq)*d*4/2**30:.1f} GiB)", flush=True)
    vecs = np.zeros((len(uniq), d), dtype=np.float32)
    for i in range(0, len(uniq), 1_000_000):
        j = min(i + 1_000_000, len(uniq))
        vecs[i:j] = base[uniq[i:j]]
    print(f"[gt1-ip] gather {time.time()-tr:.1f}s", flush=True)

    q64 = q.astype(np.float64)
    out_I = np.full((nq, K), -1, dtype=np.int64)
    out_D = np.zeros((nq, K), dtype=np.float32)
    nonmono = 0
    margins = np.empty(nq, dtype=np.float64)   # dist gap between the K-th kept and the M-th candidate
    for i in range(nq):
        cv = vecs[inv[i]].astype(np.float64)
        dd = 1.0 - cv.dot(q64[i])              # 1 - IP, the index's own IP min-distance
        dd[cand[i] < 0] = np.inf
        order = np.argsort(dd, kind="stable")
        out_I[i] = cand[i][order[:K]]
        out_D[i] = dd[order[:K]].astype(np.float32)
        margins[i] = dd[order[M - 1]] - dd[order[K - 1]]
        if np.any(np.diff(out_D[i]) < -1e-6):
            nonmono += 1
    print(f"[gt1-ip] re-rank {time.time()-tr:.1f}s; rows not ascending (should be 0): {nonmono}", flush=True)
    print(f"[gt1-ip] candidate safety margin dist[{M}] - dist[{K}]: "
          f"min={margins.min():.6g} p1={np.percentile(margins,1):.6g} "
          f"median={np.median(margins):.6g}  (>> fp32 GEMM error ~1e-3 at |IP|~150 => top-{K} is safe)",
          flush=True)
    assert nonmono == 0, "GT rows must be ascending in (1 - IP)"

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
                print(f"  backed up {name} -> {bp}", flush=True)
            if os.path.islink(p):
                os.unlink(p)
        write_type1(p, ids32, out_D)
    print("[gt1-ip] done", flush=True)


if __name__ == "__main__":
    main()
