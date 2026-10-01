#!/usr/bin/env python3
"""Second-level k-means over the trained IVF centroids (the IRQ inner k-means).

Reads the first-level centroids file and runs another k-means whose target cluster
size defaults to 1000 first-level centroids per second-level cluster (overridable
via env var RABITQ_IRQ_INNER_SIZE).

Outputs:
  inner_centroids_path : .fvecs  (num_inner_clusters x dim)  -- kmeans means
  inner_clusterids_path: .ivecs  (num_first_level_centroids x 1) -- assignments

Usage:
  python ivf_centroids.py <centroids.fvecs> <inner_centroids.fvecs>
                          <inner_clusterids.ivecs> [metric=l2|ip] [scheme=mean]

The optional 5th positional arg selects the inner-centroid scheme; "mean" is the
only accepted value. Requires only faiss (CPU) and numpy.
"""
from __future__ import annotations

import os
import sys
from time import time

import faiss
import numpy as np

DEFAULT_INNER_SIZE = 1000


# --------------------------------------------------------------------------- #
# .fvecs / .ivecs I/O. Row layout: [int32 d][d payload values], i.e.
# .fvecs row = [int32 d, d*float32]; .ivecs row = [int32 d, d*int32].
# --------------------------------------------------------------------------- #
def read_fvecs(path):
    """Read a .fvecs ([int32 d, d*float32] per row) -> (n, d) float32 array."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    d = int(raw[0])
    arr = raw.reshape(-1, d + 1)[:, 1:].view(np.float32)
    return np.ascontiguousarray(arr)


def write_fvecs(filename, m):
    """Each row is [int32 d, d * float32]."""
    m = np.ascontiguousarray(m, dtype=np.float32)
    k, d = m.shape
    out = np.empty((k, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = m.view(np.int32)
    out.tofile(filename)
    print(f"\t{filename} wrote ({k}x{d})", flush=True)


def write_ivecs(filename, m):
    """Each row is [int32 d, d * int32]. 1-D input is treated as (n, 1)."""
    m = np.ascontiguousarray(m, dtype=np.int32)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    n, d = m.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = m
    out.tofile(filename)
    print(f"\t{filename} wrote ({n}x{d})", flush=True)


def _parse_metric(arg: str) -> int:
    s = arg.lower()
    if s == "l2":
        return faiss.METRIC_L2
    if s in ("ip", "innerproduct"):
        return faiss.METRIC_INNER_PRODUCT
    raise ValueError(f"Unsupported metric: {arg}")


def _parse_scheme(arg: str) -> str:
    s = arg.lower()
    if s in ("mean", "synthetic"):
        return "mean"
    raise ValueError(
        f"Unsupported scheme: {arg} (only 'mean' is supported)")


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: ivf_centroids.py <centroids.fvecs> <inner_centroids.fvecs> "
            "<inner_clusterids.ivecs> [metric=l2|ip] [scheme=mean]",
            file=sys.stderr,
        )
        return 1

    centroids_path = sys.argv[1]
    inner_centroids_path = sys.argv[2]
    inner_cids_path = sys.argv[3]
    metric = _parse_metric(sys.argv[4]) if len(sys.argv) > 4 else faiss.METRIC_L2
    if len(sys.argv) > 5:
        _parse_scheme(sys.argv[5])  # call-shape compatibility; 'mean' only

    inner_size = int(os.environ.get("RABITQ_IRQ_INNER_SIZE", DEFAULT_INNER_SIZE))
    if inner_size <= 0:
        raise ValueError(f"RABITQ_IRQ_INNER_SIZE must be > 0 (got {inner_size})")

    centroids = read_fvecs(centroids_path).astype(np.float32, copy=False)
    num_centroids, dim = centroids.shape
    num_inner_clusters = max(1, round(num_centroids / inner_size))
    if num_inner_clusters > num_centroids:
        num_inner_clusters = num_centroids

    print(
        f"second-level k-means: num_centroids={num_centroids} dim={dim} "
        f"inner_size={inner_size} num_inner_clusters={num_inner_clusters} "
        f"metric={'IP' if metric == faiss.METRIC_INNER_PRODUCT else 'L2'}"
    )

    t0 = time()
    index = faiss.index_factory(dim, f"IVF{num_inner_clusters},Flat", metric)
    index.verbose = True
    index.train(centroids)
    print(f"  train time: {time() - t0:.2f}s")

    means = index.quantizer.reconstruct_n(0, index.nlist)
    _, inner_cids = index.quantizer.search(centroids, 1)

    write_fvecs(inner_centroids_path, means)
    write_ivecs(inner_cids_path, inner_cids)
    print(f"wrote {inner_centroids_path} ({means.shape})")
    print(f"wrote {inner_cids_path} ({inner_cids.shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
