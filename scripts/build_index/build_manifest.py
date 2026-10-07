#!/usr/bin/env python3
# Assemble the build-manifest CSV (one row per index); see OUTPUT_SCHEMA.md.
#
# Usage:
#   build_manifest.py <out.csv> <index_dir> [<index_dir> ...]
#
# Columns: index_id, system, dataset, build_params, build_s,
#                     build_peak_rss_mb, index_mem_bytes, index_ssd_bytes
#
# build_s        = build_invlist wall + build_coarse wall (the memdim/IRQ-
#                  specific steps; first-level + inner k-means are shared across
#                  memdim variants and amortized -> noted in build_params).
# build_peak_rss = max "Maximum resident set size" across the build logs (MiB).
# index_mem_bytes = base file size (the in-memory bincode part).
# index_ssd_bytes = .ssd file size (the on-SSD part).
import csv
import glob
import os
import re
import sys

SYSTEM = "RaBitQ-SSD"


def _read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def parse_elapsed_to_s(text):
    # /usr/bin/time -v: "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:23.45"
    m = re.search(r"Elapsed \(wall clock\) time[^:]*:\s*([0-9:.]+)", text)
    if not m:
        return None
    parts = m.group(1).split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


def max_rss_mib(text):
    # "Maximum resident set size (kbytes): N"  (KB -> MiB)
    vals = [int(x) for x in re.findall(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)]
    return max(vals) / 1024.0 if vals else None


def parse_index_dir(idx_dir):
    name = os.path.basename(os.path.normpath(idx_dir))
    # indexes/<dataset>/idx_... -> the parent dir IS the dataset name.
    dataset = os.path.basename(os.path.dirname(os.path.normpath(idx_dir)))
    # idx_C96422_cpu_B9_memdim1280_l2_ordered, or ..._cpu_raw_memdim... for an
    # index whose SSD records hold the vectors themselves (STORE_ENV=raw).
    g = lambda pat, d="": (re.search(pat, name).group(1) if re.search(pat, name) else d)
    C = g(r"_C(\d+)")
    B = g(r"_B(\d+)")
    raw = re.search(r"_cpu_raw_", name) is not None
    memdim = g(r"memdim(\d+)")
    metric = g(r"_(l2|ip|cosine)_")
    order = g(r"_(ordered|unordered)")

    inv_log = _read(os.path.join(idx_dir, "build_invlist.log"))
    # Sorted so a directory holding several coarse arms picks the same one
    # every time rather than whatever the filesystem happened to return first.
    coarse_log = ""
    for cg in sorted(glob.glob(os.path.join(idx_dir, "build_coarse*log"))):
        coarse_log = _read(cg)
        break

    # The build budget is whatever the runner actually passed, not a constant:
    # _build_lib.sh echoes the full command as the log's first line.
    m = re.search(r"mem_budget_gb=([0-9.]+)", inv_log)
    budget_gb = m.group(1) if m else ""

    # build_invlist wall: prefer time -v Elapsed, else "[timing] total ... min"
    inv_s = parse_elapsed_to_s(inv_log)
    if inv_s is None:
        m = re.search(r"total \(construct \+ save_base\):\s*([0-9.]+)\s*min", inv_log)
        inv_s = float(m.group(1)) * 60 if m else None
    # build_coarse wall: "Coarse-quantizer build time: N seconds."
    m = re.search(r"Coarse-quantizer build time:\s*([0-9.eE+]+)\s*seconds", coarse_log)
    coarse_s = float(m.group(1)) if m else None

    build_s = ""
    if inv_s is not None or coarse_s is not None:
        build_s = round((inv_s or 0.0) + (coarse_s or 0.0), 2)

    rss = max(filter(None, [max_rss_mib(inv_log), max_rss_mib(coarse_log)]), default=None)
    # fall back to any time_v*.txt in the dir
    if rss is None:
        for tv in glob.glob(os.path.join(idx_dir, "time_v*")):
            rss = max(filter(None, [rss, max_rss_mib(_read(tv))]), default=rss)
    build_peak_rss_mb = round(rss, 2) if rss is not None else ""

    def fsize(pat):
        fs = glob.glob(os.path.join(idx_dir, pat))
        return os.path.getsize(fs[0]) if fs else ""

    return {
        "index_id": name,
        "system": SYSTEM,
        "dataset": dataset,
        "build_params": f"C={C};{'store=raw' if raw else f'B={B}'};memdim={memdim};"
                        f"metric={metric};order={order};"
                        f"budget_gb={budget_gb};kmeans=shared/amortized",
        "build_s": build_s,
        "build_peak_rss_mb": build_peak_rss_mb,
        "index_mem_bytes": fsize("*.base"),
        "index_ssd_bytes": fsize("*.ssd"),
    }


def main():
    if len(sys.argv) < 3:
        print("usage: build_manifest.py <out.csv> <index_dir> [<index_dir> ...]", file=sys.stderr)
        sys.exit(1)
    out, dirs = sys.argv[1], sys.argv[2:]
    cols = ["index_id", "system", "dataset", "build_params", "build_s",
            "build_peak_rss_mb", "index_mem_bytes", "index_ssd_bytes"]
    rows = [parse_index_dir(d) for d in dirs if os.path.isdir(d)]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} row(s) to {out}")
    for r in rows:
        print(f"  {r['index_id']}: build_s={r['build_s']} peak_rss_mb={r['build_peak_rss_mb']} "
              f"mem={r['index_mem_bytes']} ssd={r['index_ssd_bytes']}")


if __name__ == "__main__":
    main()
