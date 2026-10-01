#!/usr/bin/env python3
"""Measure the TRUE size of the exact-tie groups for GT rows that saturated at depth k.

validate_gt_type1.py flags a row as SATURATED when gt_dist[k-1] == gt_dist[topk-1]:
the tie group at the k-th neighbour runs past the stored depth, so the harness's forward
tie scan (count_correct_with_ties, apps/querying.cpp:76-99) cannot see all the points it
should credit. Rebuilding such a GT at a larger depth requires knowing the true group
size: a row's entire stored depth can be a single tie value, so the group can be far
larger than the stored depth suggests.

Method (one streaming pass over the base):
  threshold d*[i] = gt_dist[row_i][k-1]        (for a saturated row the whole tail is tied,
                                                so this is also gt_dist[topk-1])
  GPU pass : squared-L2 via the fp32 expansion ||q||^2 - 2<q,x> + ||x||^2, keep every base
             point with d <= d*[i] + TOL. The expansion is used ONLY to shortlist; it is
             never the reported distance.
  CPU check: for each shortlisted (query,id), recompute the EXACT float64 squared L2 the
             same way the GT does, cast to float32 once, and count how many compare
             bit-equal to d*[i]. That bit-equality is exactly what the harness tests.

Reports, per saturated row: |{d < d*}| (strictly closer), |{d == d*}| (the tie group), and
the depth needed to hold them all = n_closer + n_tied -- the minimum depth a rebuilt GT
must store.
"""
import os
import sys
import time
import json
import argparse
import numpy as np


def read_type1(path):
    fsz = os.path.getsize(path)
    nq, k = np.fromfile(path, dtype=np.int32, count=2)
    nq, k = int(nq), int(k)
    assert fsz == 8 + 2 * nq * k * 4, (fsz, nq, k)
    ids = np.fromfile(path, dtype=np.uint32, count=nq * k, offset=8).reshape(nq, k)
    dist = np.fromfile(path, dtype=np.float32, count=nq * k,
                       offset=8 + nq * k * 4).reshape(nq, k)
    return nq, k, ids, dist


def saturated_rows(dist, topks):
    """Union of rows where the tie group runs to the stored depth, for any topk."""
    k = dist.shape[1]
    rows = set()
    for t in topks:
        if t <= k:
            rows |= set(np.where(dist[:, k - 1] == dist[:, t - 1])[0].tolist())
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--topks", default="10,100,1000")
    ap.add_argument("--shard", type=int, default=4_000_000,
                help="rows per GPU shard; 4M*512*4 = 8.2GB, and the einsum below\n                      avoids the doubling temporary that OOMed at 20M")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="fp32-expansion shortlist half-width around d* (generous; the "
                         "exact fp64 check decides membership)")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    import torch
    dev = torch.device("cuda:0")
    print("[tie] torch %s on %s" % (torch.__version__, torch.cuda.get_device_name(0)), flush=True)

    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    nq_gt, k, gt_ids, gt_dist = read_type1(args.gt)
    rows = saturated_rows(gt_dist, topks)
    print("[tie] GT %s: nq=%d k=%d -> %d saturated rows: %s"
          % (args.gt, nq_gt, k, len(rows), rows), flush=True)
    if not rows:
        print("[tie] nothing to do."); return 0

    nb, d = np.fromfile(args.base, dtype=np.int32, count=2)
    nb, d = int(nb), int(d)
    nq, dq = np.fromfile(args.queries, dtype=np.int32, count=2)
    assert int(dq) == d and int(nq) == nq_gt
    base = np.memmap(args.base, dtype=np.float32, mode="r", offset=8, shape=(nb, d))
    Q = np.memmap(args.queries, dtype=np.float32, mode="r", offset=8, shape=(int(nq), d))

    qs = np.ascontiguousarray(Q[rows])                  # [R, d]
    R = len(rows)
    dstar = gt_dist[rows, k - 1].astype(np.float32)     # [R]
    print("[tie] thresholds d* = %s" % np.array2string(dstar, precision=7), flush=True)

    qg = torch.from_numpy(qs).to(dev)
    qn = torch.einsum('ij,ij->i', qg, qg)
    thr = torch.from_numpy(dstar + args.tol).to(dev)

    hits = [[] for _ in range(R)]
    t0 = time.time()
    for s in range(0, nb, args.shard):
        e = min(s + args.shard, nb)
        xb = torch.from_numpy(np.ascontiguousarray(base[s:e])).to(dev)
        bn = torch.einsum('ij,ij->i', xb, xb)   # no [shard,d] temporary
        D = qn[:, None] - 2.0 * (qg @ xb.T) + bn[None, :]     # [R, shard] fp32 expansion
        m = D <= thr[:, None]
        idx = m.nonzero()
        if idx.numel():
            ic = idx.cpu().numpy()
            for r in range(R):
                sel = ic[ic[:, 0] == r][:, 1]
                if len(sel):
                    hits[r].append(sel.astype(np.int64) + s)
        del xb, bn, D, m, idx
        if (s // args.shard) % 5 == 0:
            el = time.time() - t0
            print("[tie] scan %d/%d (%.1f%%) %.0fs ETA %.1fmin"
                  % (e, nb, 100 * e / nb, el, (nb - e) / max(1, e) * el / 60), flush=True)
    print("[tie] scan done %.1fs; exact fp64 verification" % (time.time() - t0), flush=True)

    out = {}
    for r in range(R):
        cand = np.concatenate(hits[r]) if hits[r] else np.zeros(0, dtype=np.int64)
        cand = np.unique(cand)
        q64 = qs[r].astype(np.float64)
        n_closer = n_tied = 0
        tied_ids = []
        for i in range(0, len(cand), 200_000):
            blk = cand[i:i + 200_000]
            v = base[blk].astype(np.float64)
            dd = ((q64[None, :] - v) ** 2).sum(1).astype(np.float32)
            n_closer += int((dd < dstar[r]).sum())
            eq = dd == dstar[r]
            n_tied += int(eq.sum())
            tied_ids.append(blk[eq])
        need = n_closer + n_tied
        row = rows[r]
        out[str(row)] = dict(d_star=float(dstar[r]), n_shortlisted=int(len(cand)),
                             n_closer=n_closer, n_tied=n_tied, depth_needed=need)
        print("[tie] row %-5d d*=%.7f  shortlist=%-9d closer=%-8d tied=%-8d "
              "=> depth needed = %d" % (row, dstar[r], len(cand), n_closer, n_tied, need),
              flush=True)

    needs = [v["depth_needed"] for v in out.values()]
    print("\n[tie] SUMMARY over %d saturated rows: max depth needed = %d, "
          "median = %d, total tied points = %d"
          % (R, max(needs), int(np.median(needs)), sum(v["n_tied"] for v in out.values())),
          flush=True)
    print("[tie] -> a rebuilt GT must store at least k = %d to close every tie window."
          % max(needs), flush=True)

    if args.out_json:
        json.dump(dict(gt=args.gt, k=k, topks=topks, rows=rows, per_row=out,
                       max_depth_needed=max(needs)), open(args.out_json, "w"), indent=2)
        print("[tie] wrote %s" % args.out_json, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
