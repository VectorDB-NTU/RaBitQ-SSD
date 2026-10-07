#!/usr/bin/env python3
"""Second-level k-means over the IVF centroids (the IRQ inner k-means), on GPU.

Same inputs and outputs as clustering/ivf_centroids.py: about
RABITQ_IRQ_INNER_SIZE (default 1000) first-level centroids per inner cluster.

  python gpu_inner_kmeans.py <centroids.fvecs> <inner_centroids.fvecs>
                             <inner_clusterids.ivecs> [l2|ip] [mean]

Needs numpy and faiss with GPU support.
"""
import os
import sys
from time import time

import numpy as np
import faiss


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return np.ascontiguousarray(raw.reshape(-1, d + 1)[:, 1:].view(np.float32))


def write_vecs(path, m, dtype):
    m = np.ascontiguousarray(m, dtype=dtype)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    out = np.empty((m.shape[0], m.shape[1] + 1), dtype=np.int32)
    out[:, 0] = m.shape[1]
    out[:, 1:] = m.view(np.int32)
    out.tofile(path)


def main():
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 1
    cen_path, inner_cen_path, inner_ids_path = sys.argv[1:4]
    metric = (sys.argv[4] if len(sys.argv) > 4 else "l2").lower()
    if metric not in ("l2", "ip"):
        print(f"unsupported metric {metric}", file=sys.stderr)
        return 1
    if len(sys.argv) > 5 and sys.argv[5].lower() != "mean":
        print("only the 'mean' scheme is supported", file=sys.stderr)
        return 1
    inner_size = int(os.environ.get("RABITQ_IRQ_INNER_SIZE", 1000))
    ngpu = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    if ngpu <= 0:
        print("faiss sees no GPU; use clustering/ivf_centroids.py", file=sys.stderr)
        return 1

    centroids = read_fvecs(cen_path)
    n, d = centroids.shape
    k = min(max(1, round(n / inner_size)), n)
    print(f"second-level k-means (GPU): num_centroids={n} dim={d} "
          f"num_inner_clusters={k} metric={metric.upper()} gpus={ngpu}", flush=True)
    t0 = time()
    km = faiss.Kmeans(d, k, niter=25, verbose=True, gpu=ngpu,
                      max_points_per_centroid=256, spherical=(metric == "ip"))
    km.train(centroids)
    means = np.ascontiguousarray(km.centroids.reshape(k, d), dtype=np.float32)
    print(f"  train time: {time() - t0:.2f}s", flush=True)

    quant = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
    quant.add(means)
    res = faiss.StandardGpuResources()
    gq = faiss.index_cpu_to_gpu(res, 0, quant)
    ids = np.empty((n, 1), dtype=np.int64)
    for s in range(0, n, 1 << 20):
        e = min(s + (1 << 20), n)
        _, ids[s:e] = gq.search(centroids[s:e], 1)
    del gq, res
    if ids.min() < 0 or ids.max() >= k:
        print("inner assignment out of range", file=sys.stderr)
        return 1
    write_vecs(inner_cen_path, means, np.float32)
    write_vecs(inner_ids_path, ids.astype(np.int32), np.int32)
    print(f"wrote {inner_cen_path} ({k}x{d}) and {inner_ids_path} ({n}x1)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
