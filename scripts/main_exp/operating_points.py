#!/usr/bin/env python3
"""Print the operating points a search run measured.

usage: operating_points.py <query_results.csv> [--targets 0.90,0.95]

For every (topk, threads) cell of the CSV (OUTPUT_SCHEMA.md), one line per
operating point in nprobe order: recall, QPS, 4 KB pages read per query and
mean latency. For each recall target, the first operating point that reaches
it -- the smallest nprobe achieving that recall -- is marked.
"""
import argparse
import csv
import sys
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="query_results.csv from scripts/main_exp/run_main_exp.sh")
    ap.add_argument("--targets", default="0.90,0.95",
                    help="comma-separated recall targets to mark (default 0.90,0.95)")
    args = ap.parse_args()
    targets = [float(t) for t in args.targets.split(",") if t.strip()]

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{args.csv}: no operating points", file=sys.stderr)
        return 1

    cells = defaultdict(list)
    for r in rows:
        cells[(int(r["topk"]), int(r["threads"]))].append(r)

    for (topk, threads), pts in sorted(cells.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        pts.sort(key=lambda r: float(r["op_param"]))
        first = {}
        for t in targets:
            for i, r in enumerate(pts):
                if float(r["recall_at_topk"]) >= t:
                    first.setdefault(i, []).append(t)
                    break
        dataset = pts[0].get("dataset", "")
        print(f"{dataset}  topk={topk}  threads={threads}")
        print(f"  {'nprobe':>8} {'recall':>8} {'QPS':>10} {'pages/q':>9} {'lat_ms':>8}")
        for i, r in enumerate(pts):
            mark = "  <- first to reach " + ", ".join(f"{t:.2f}" for t in first[i]) \
                if i in first else ""
            print(f"  {int(float(r['op_param'])):>8} {float(r['recall_at_topk']):>8.4f} "
                  f"{float(r['qps']):>10.1f} {float(r['io_pages_mean']):>9.1f} "
                  f"{float(r['lat_mean_us']) / 1000:>8.2f}{mark}")
        missed = [t for t in targets if t not in sum(first.values(), [])]
        for t in missed:
            print(f"  recall {t:.2f} was not reached by any measured operating point")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
