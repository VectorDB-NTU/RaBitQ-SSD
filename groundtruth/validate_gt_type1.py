#!/usr/bin/env python3
"""Validate a DiskANN type-1 ground-truth file against what querying.cpp actually requires.

The harness (apps/querying.cpp) fails SILENTLY, so every check here guards a failure mode
that produces a plausible-but-wrong recall number rather than an error:

  1. FORMAT DETECTION IS SIZE-EQUALITY ONLY (load_gt_maybe_type1, querying.cpp:45-71).
     It reads nq,k from the first 8 bytes and requires
         filesize == 8 + 2*nq*k*4
     One stray byte and the file falls back to the ids-only path: tie-awareness silently
     off, no warning. -> check_size()

  2. THE TIE SCAN WALKS FORWARD ONLY (count_correct_with_ties, querying.cpp:76-99):
         tie_breaker = topk-1
         while tie_breaker < k and gt_dist[tie_breaker] == gt_dist[topk-1]: ++tie_breaker
     so every row MUST be sorted ASCENDING, or the tie set is silently truncated.
     Equality is EXACT float32 ==. -> check_ascending()

  3. SATURATION. The window can only widen over ids actually stored in the row. If the
     tie group at rank topk-1 runs past the stored depth k, the row is TRUNCATED and
     those queries get under-credited. This happens on data with large
     exact-duplicate groups. -> check_saturation()

  4. k >= topk is the only thing the harness itself validates (querying.cpp:606-611).

Writes a human-readable report to stdout and, with --log, to a file next to the GT.

Exit code 0 = every check passed. 1 = a hard failure (format/ordering). 2 = saturation
only -- the GT is loadable and correct as far as it goes, but some rows need more depth.
"""
import os
import sys
import argparse
import numpy as np


def read_type1(path):
    """Return (nq, k, ids[nq,k] uint32, dists[nq,k] float32). Raises on size mismatch."""
    fsz = os.path.getsize(path)
    nq, k = np.fromfile(path, dtype=np.int32, count=2)
    nq, k = int(nq), int(k)
    expect = 8 + 2 * nq * k * 4
    if nq <= 0 or k <= 0 or fsz != expect:
        raise ValueError(
            "NOT type-1: header=(nq=%d,k=%d) needs size %d but file is %d (delta %+d). "
            "querying.cpp would silently fall back to ids-only and disable tie-aware recall."
            % (nq, k, expect, fsz, fsz - expect))
    ids = np.fromfile(path, dtype=np.uint32, count=nq * k, offset=8).reshape(nq, k)
    dists = np.fromfile(path, dtype=np.float32, count=nq * k,
                        offset=8 + nq * k * 4).reshape(nq, k)
    return nq, k, ids, dists


def check_ascending(dists):
    """Rows must be non-decreasing. Returns (n_bad_rows, worst_drop, example_row)."""
    d = np.diff(dists, axis=1)
    bad = d < 0
    rows = np.where(bad.any(axis=1))[0]
    worst = float(d[bad].min()) if bad.any() else 0.0
    return len(rows), worst, (int(rows[0]) if len(rows) else -1)


def tie_window(dists, topk):
    """Replicate count_correct_with_ties' forward scan exactly, per row.

    Returns (window_end[nq]) where window_end is the exclusive index the scan stops at,
    i.e. tie_breaker. window_end == topk means no tie opened.
    """
    nq, k = dists.shape
    ref = dists[:, topk - 1][:, None]                  # gt_dist[topk-1]
    tail = dists[:, topk - 1:]                         # columns topk-1 .. k-1
    eq = (tail == ref)                                 # EXACT float32 equality
    # scan stops at the first False; count the leading run of True
    run = np.argmin(eq, axis=1)
    run = np.where(eq.all(axis=1), eq.shape[1], run)   # all-equal -> full tail
    return (topk - 1) + run


def check_saturation(dists, topks):
    """A row SATURATES at topk if the tie window reaches the stored depth k, i.e.
    gt_dist[k-1] == gt_dist[topk-1] -- the true tie group may extend past the stored depth."""
    nq, k = dists.shape
    out = {}
    for topk in topks:
        if topk > k:
            out[topk] = dict(applicable=False)
            continue
        end = tie_window(dists, topk)
        opened = end > topk                            # tie window widened past topk
        sat = dists[:, k - 1] == dists[:, topk - 1]    # ran to the end of stored depth
        win = end - topk                               # extra ids credited
        out[topk] = dict(applicable=True,
                         n_opened=int(opened.sum()),
                         pct_opened=100.0 * opened.mean(),
                         mean_window=float(win[opened].mean()) if opened.any() else 0.0,
                         max_window=int(win.max()),
                         n_saturated=int(sat.sum()),
                         saturated_rows=np.where(sat)[0][:50].tolist())
    return out


def tie_stats(ids, dists):
    """Duplicate/tie structure of the GT itself -- how tie-prone this dataset is."""
    flat = dists.ravel()
    # largest run of identical distances within any single row
    biggest = 0
    for r in range(dists.shape[0]):
        row = dists[r]
        chg = np.flatnonzero(np.diff(row)) + 1
        runs = np.diff(np.concatenate(([0], chg, [len(row)])))
        biggest = max(biggest, int(runs.max()))
    return dict(distinct_dists=int(len(np.unique(flat))),
                total_slots=int(flat.size),
                largest_tie_run_in_a_row=biggest,
                n_duplicate_ids=int(ids.size - len(np.unique(ids))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", help="path to the type-1 GT file")
    ap.add_argument("--topks", default="10,100,1000")
    ap.add_argument("--expect-nq", type=int, default=0)
    ap.add_argument("--log", default=None, help="also write the report here")
    ap.add_argument("--queries", default=None,
                    help="optional: .fbin whose nq must match the GT header")
    args = ap.parse_args()

    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 78)
    say("type-1 GT validation: %s" % args.gt)
    say("=" * 78)

    try:
        nq, k, ids, dists = read_type1(args.gt)
    except ValueError as e:
        say("[FAIL] %s" % e)
        if args.log:
            open(args.log, "w").write("\n".join(lines) + "\n")
        return 1

    fsz = os.path.getsize(args.gt)
    say("[ok  ] format: nq=%d k=%d  size=%d B == 8+2*nq*k*4  -> type-1, tie-aware ENABLED"
        % (nq, k, fsz))

    hard_fail = False

    if args.queries:
        qn, qd = np.fromfile(args.queries, dtype=np.int32, count=2)
        if int(qn) != nq:
            say("[FAIL] nq mismatch: GT says %d, %s says %d" % (nq, args.queries, int(qn)))
            hard_fail = True
        else:
            say("[ok  ] nq matches query file %s (nq=%d d=%d)" % (args.queries, int(qn), int(qd)))

    if args.expect_nq and nq != args.expect_nq:
        say("[FAIL] nq=%d but expected %d" % (nq, args.expect_nq))
        hard_fail = True

    bad_k = [t for t in topks if t > k]
    if bad_k:
        say("[FAIL] k=%d < topk %s -- querying.cpp hard-errors (querying.cpp:606-611)"
            % (k, bad_k))
        hard_fail = True
    else:
        say("[ok  ] k=%d >= max topk %d" % (k, max(topks)))

    n_bad, worst, ex = check_ascending(dists)
    if n_bad:
        say("[FAIL] %d rows NOT ascending (worst drop %.6g, e.g. row %d) -- the tie scan "
            "would truncate silently" % (n_bad, worst, ex))
        hard_fail = True
    else:
        say("[ok  ] all %d rows sorted ascending" % nq)

    id_range_bad = int((ids == np.uint32(0xFFFFFFFF)).sum())
    say("[info] ids: min=%d max=%d  (%d sentinel 0xFFFFFFFF slots)"
        % (int(ids.min()), int(ids.max()), id_range_bad))

    say()
    say("---- tie structure (how tie-prone this GT is) ----")
    ts = tie_stats(ids, dists)
    say("  distinct distance values : %d / %d slots" % (ts["distinct_dists"], ts["total_slots"]))
    say("  largest tie run in a row : %d" % ts["largest_tie_run_in_a_row"])
    say("  duplicate ids across GT  : %d" % ts["n_duplicate_ids"])

    say()
    say("---- tie window + SATURATION per topk ----")
    sat = check_saturation(dists, topks)
    saturated_any = False
    for t in topks:
        s = sat[t]
        if not s["applicable"]:
            say("  topk=%-5d n/a (k=%d < topk)" % (t, k))
            continue
        say("  topk=%-5d window opens on %6d rows (%5.2f%%)  mean=%7.1f  max=%d"
            % (t, s["n_opened"], s["pct_opened"], s["mean_window"], s["max_window"]))
        if s["n_saturated"]:
            saturated_any = True
            say("            [SATURATED] %d rows hit stored depth k=%d -- their tie group "
                "is TRUNCATED" % (s["n_saturated"], k))
            say("            rows (first 50): %s" % s["saturated_rows"])
        else:
            say("            [ok] no row reaches depth k -- every tie group fits")

    say()
    if hard_fail:
        say("VERDICT: FAIL -- do not ship this file.")
        rc = 1
    elif saturated_any:
        say("VERDICT: LOADABLE but INCOMPLETE -- some rows' tie groups are truncated at "
            "k=%d. Those queries will be under-credited. Extend depth for the saturated "
            "rows (see groundtruth/README.md)." % k)
        rc = 2
    else:
        say("VERDICT: PASS -- format, ordering, depth and tie windows all correct.")
        rc = 0

    if args.log:
        open(args.log, "w").write("\n".join(lines) + "\n")
        print("\nwrote %s" % args.log, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
