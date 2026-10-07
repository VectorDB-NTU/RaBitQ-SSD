#!/usr/bin/env python3
"""
two_level_kmeans.py -- first-level IVF clustering driver (dataset-generic).
=====================================================================================
Trains C centroids with two-level k-means and assigns every base point to one of
them. Base, centroid count, metric and all knobs come from CLI flags. Two lines:

  * --line cpu (default) : CPU two-level clustering + APPROXIMATE HNSW assignment.
                          The tractable choice when N and C are both large, and
                          the only line the shipped pipeline uses.
  * --line gpu           : GPU two-level clustering + EXACT GPU flat assignment.
                          Needs a CUDA build of faiss.

The base is never loaded whole: n_train rows are reservoir-sampled for training and
the assignment pass streams the base from a read-only memmap, so a base far larger
than RAM runs inside a few hundred GB. The span `first base read -> centroids +
clusterids written` is wrapped in ONE wall-clock measurement, which is this stage's
indexing-time contribution; diagnostics stay outside it.

The base is a .fbin (float32) or a .u8bin (uint8, e.g. DINO-10B); uint8 rows are
widened to float32 as they are read, so everything downstream is unchanged.

Semantics:
  - faiss.contrib.clustering.two_level_clustering, cost ~ O(n_train * sqrt(C) * d)
    instead of the flat O(n_train * C * d)
  - stored centroids = per-cluster MEAN of the assigned raw vectors (un-normalized;
    what RaBitQ-SSD residual encoding `data - centroid` needs), empty clusters keep
    their trained centroid
  - centroids -> centroids_<C>.fvecs (.fvecs: [int32 d, d*f32] per row)
    clusterids -> clusterids_<C>.ivecs (.ivecs: [int32 1, int32 id] per row, N rows)
  These are the files downstream load_matrix_auto consumes.

Defaults: metric l2; n_train = pts_per_centroid * C with pts_per_centroid = 240
(faiss saturates near 256; 240 keeps the training sample's memory a little lower,
which matters because it is C * pts_per_centroid * d * 4 bytes).

USAGE -- --base and --out-dir are REQUIRED. clustering/run_kmeans.sh wraps this
with the defaults the rest of the pipeline expects; call it directly only to
override something the runner does not expose.
  # a timing estimate first: cap train/assign, keep C fixed, extrapolate linearly
  python two_level_kmeans.py --base /path/to/datasets/<dataset>/train.fbin \
         --out-dir /path/to/indexes/<dataset>/clustering \
         --calib-n-train 5000000 --calib-n-assign 20000000 --allow-partial --validate
  # the real build (no extra passes):
  python two_level_kmeans.py --base /path/to/datasets/<dataset>/train.fbin \
         --out-dir /path/to/indexes/<dataset>/clustering
"""
import os
import sys
import json
import math
import time
import argparse

import numpy as np
import faiss
from faiss.contrib.clustering import two_level_clustering


# --------------------------------------------------------------------------- #
# .fbin / .u8bin IO -- partial-file safe: a base still being written has a header row
# count larger than the rows on disk, so the memmap covers rows_present, not the header n.
# --------------------------------------------------------------------------- #
def base_dtype(path):
    """Value type of a base file, from its extension: .u8bin is uint8, anything else
    (.fbin) float32. .i8bin is int8, which the index cannot read, so it is refused."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".i8bin":
        raise SystemExit(f"[FATAL] {path}: int8 vectors (.i8bin) are not supported; "
                         f"use float32 (.fbin) or uint8 (.u8bin)")
    return np.uint8 if ext == ".u8bin" else np.float32


def safe_memmap_fbin(path):
    """Return (arr, n_header, d, rows_present). arr is a read-only memmap over the
    rows ACTUALLY present on disk (not the header n), so it never raises
    'mmap length is greater than file size' on a still-copying base. Its dtype is
    float32 for a .fbin and uint8 for a .u8bin; every reader below converts what it
    takes to float32."""
    dtype = base_dtype(path)
    n_header, d = (int(v) for v in np.fromfile(path, dtype=np.int32, count=2))
    size = os.path.getsize(path)
    payload = size - 8
    rowbytes = d * np.dtype(dtype).itemsize
    # floor to whole rows: a file still being written may end mid-row at any instant --
    # that trailing partial row is "not yet present", not corruption.
    rows_present = payload // rowbytes
    leftover = payload - rows_present * rowbytes
    if leftover:
        print(f"[io] {path}: {leftover}B trailing partial row (copy in progress) -> "
              f"using {rows_present:,} whole rows", flush=True)
    # memmap over whole rows only; the file only grows, so this never exceeds the file size.
    arr = np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(rows_present, d))
    return arr, n_header, d, rows_present


# --------------------------------------------------------------------------- #
# .fvecs / .ivecs writers. The row layouts below are what the index's
# load_matrix_auto expects; they are part of the on-disk contract.
# --------------------------------------------------------------------------- #
def write_fvecs_fast(filename, m):
    """Each row is [int32 d, d * float32]."""
    m = np.ascontiguousarray(m, dtype=np.float32)
    k, d = m.shape
    out = np.empty((k, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = m.view(np.int32)
    out.tofile(filename)
    print(f"\t{filename} wrote ({k}x{d})", flush=True)


def write_ivecs_fast(filename, ids):
    """Each row is [int32 1, int32 id] (d=1 per row, N rows)."""
    ids = np.ascontiguousarray(ids, dtype=np.int32).reshape(-1)
    n = ids.shape[0]
    out = np.empty((n, 2), dtype=np.int32)
    out[:, 0] = 1
    out[:, 1] = ids
    out.tofile(filename)
    print(f"\t{filename} wrote ({n} rows)", flush=True)


def read_fvecs(path):
    """Read a .fvecs ([int32 d, d*float32] per row) -> contiguous (n, d) float32 array."""
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    rows = raw.reshape(-1, d + 1)
    return np.ascontiguousarray(rows[:, 1:]).view(np.float32)


# --------------------------------------------------------------------------- #
# Reservoir subsample (sorted indices -> monotonic memmap read on the SSD).
# --------------------------------------------------------------------------- #
def reservoir_sample(arr, n_pool, n_train, seed=None, gather_block=1_000_000,
                     verbose=True):
    """Draw n_train rows uniformly without replacement from arr[:n_pool] into a
    contiguous float32 RAM array. Indices are sorted before the gather so the
    memmap is read monotonically (good locality on a single SSD)."""
    assert n_train <= n_pool, (n_train, n_pool)
    d = arr.shape[1]
    t0 = time.time()
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_pool, size=n_train, replace=False)
    idx.sort()
    if verbose:
        print(f"[sample] selected {n_train}/{n_pool} indices in {time.time()-t0:.1f}s; "
              f"gathering ~{n_train*d*4/2**30:.0f} GiB", flush=True)
    xt = np.empty((n_train, d), dtype=np.float32)
    for s in range(0, n_train, gather_block):
        e = min(s + gather_block, n_train)
        xt[s:e] = arr[idx[s:e]]
        if verbose and (s // gather_block) % 25 == 0:
            print(f"[sample] gathered {e}/{n_train} ({100*e/n_train:.1f}%), "
                  f"{time.time()-t0:.1f}s", flush=True)
    if verbose:
        print(f"[sample] done in {time.time()-t0:.1f}s, xt={xt.shape}", flush=True)
    del idx
    return xt


# --------------------------------------------------------------------------- #
# GPU flat quantizer over the centroids, REPLICATED on each GPU (co.shard=False):
# the (C,d) index (~1.9 GiB) fits on every GPU and the query batch is split across
# GPUs -> ~ngpu throughput. use_float16 is fine for assignment (boundary ties
# irrelevant) and faster.
# --------------------------------------------------------------------------- #
def gpu_flat_index(centroids, metric, ngpu, use_float16):
    d = centroids.shape[1]
    quant = (faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT
             else faiss.IndexFlatL2(d))
    quant.add(np.ascontiguousarray(centroids, dtype=np.float32))
    co = faiss.GpuMultipleClonerOptions()
    co.shard = False
    co.useFloat16 = use_float16
    res = [faiss.StandardGpuResources() for _ in range(ngpu)]
    vres, vdev = faiss.GpuResourcesVector(), faiss.Int32Vector()
    for i in range(ngpu):
        vdev.push_back(i)
        vres.push_back(res[i])
    return faiss.index_cpu_to_gpu_multiple(vres, vdev, quant, co)


# --------------------------------------------------------------------------- #
# Assign ALL n_assign base points to the C centroids on GPU, streaming from the
# memmap, AND accumulate per-cluster sums so the OUTPUT centroid is the mean of
# each cluster's raw members. Optional disk checkpoint for crash-resume.
# --------------------------------------------------------------------------- #
def assign_and_accumulate(arr, n_assign, gpu_index, C, d, batch, ckpt_path=None,
                          ckpt_every_pts=200_000_000, resume=False, verbose=True):
    cluster_id = np.empty(n_assign, dtype=np.int32)
    sums = np.zeros((C, d), dtype=np.float64)   # C * d * 8 bytes
    counts = np.zeros(C, dtype=np.int64)
    start = 0

    if resume and ckpt_path and os.path.exists(ckpt_path):
        ck = np.load(ckpt_path)
        start = int(ck["cursor"])
        sums[:] = ck["sums"]
        counts[:] = ck["counts"]
        cluster_id[:start] = ck["cluster_id_prefix"]
        print(f"[assign] resume from cursor={start}/{n_assign}", flush=True)

    t0 = time.time()
    last_ckpt = start
    for i in range(start, n_assign, batch):
        j = min(i + batch, n_assign)
        xb = np.ascontiguousarray(arr[i:j], dtype=np.float32)
        _, I = gpu_index.search(xb, 1)
        cb = I[:, 0]
        cluster_id[i:j] = cb.astype(np.int32)
        # per-cluster sum within this batch (sort + reduceat; collision-free scatter)
        order = np.argsort(cb, kind="stable")
        cs = cb[order]
        bnd = np.concatenate(([0], np.nonzero(np.diff(cs))[0] + 1))
        gids = cs[bnd]
        sums[gids] += np.add.reduceat(xb[order], bnd, axis=0)
        counts[gids] += np.diff(np.append(bnd, cs.shape[0]))
        if verbose and (i // batch) % 20 == 0:
            el = time.time() - t0
            pps = (j - start) / max(1e-9, el)
            eta = (n_assign - j) / max(1.0, pps)
            print(f"[assign] {j}/{n_assign} ({100.0*j/n_assign:.1f}%) "
                  f"{el:.1f}s  {pps/1e6:.1f} Mpts/s  ETA {eta/60:.1f} min", flush=True)
        if ckpt_path and (j - last_ckpt) >= ckpt_every_pts and j < n_assign:
            np.savez(ckpt_path + ".new", cursor=j, sums=sums, counts=counts,
                     cluster_id_prefix=cluster_id[:j])
            os.replace(ckpt_path + ".new.npz", ckpt_path)
            last_ckpt = j
            print(f"[assign] checkpoint @ {j}", flush=True)
    print(f"[assign] done {time.time()-t0:.1f}s", flush=True)
    return cluster_id, sums, counts


def assign_hnsw_accumulate(arr, n_assign, centroids, C, d, M, efc, efs, batch,
                           metric=faiss.METRIC_L2, verbose=True):
    """CPU-line assignment: build an HNSW graph over the centroids and search nearest-1
    for every base point (APPROXIMATE; efSearch trades recall@1 vs speed). Accumulates
    per-cluster sums in the same pass so the output centroid is the per-cluster mean."""
    cen = np.ascontiguousarray(centroids, dtype=np.float32)
    t0 = time.time()
    index = faiss.IndexHNSWFlat(d, M, metric)
    index.hnsw.efConstruction = efc
    index.add(cen)
    index.hnsw.efSearch = efs
    print(f"[hnsw] built over {C} centroids in {time.time()-t0:.1f}s "
          f"(M={M} efC={efc} efS={efs})", flush=True)
    cluster_id = np.empty(n_assign, dtype=np.int32)
    sums = np.zeros((C, d), dtype=np.float64)
    counts = np.zeros(C, dtype=np.int64)
    t0 = time.time()
    for i in range(0, n_assign, batch):
        j = min(i + batch, n_assign)
        xb = np.ascontiguousarray(arr[i:j], dtype=np.float32)
        _, I = index.search(xb, 1)
        cb = I[:, 0]
        cluster_id[i:j] = cb.astype(np.int32)
        order = np.argsort(cb, kind="stable")
        cs = cb[order]
        bnd = np.concatenate(([0], np.nonzero(np.diff(cs))[0] + 1))
        gids = cs[bnd]
        sums[gids] += np.add.reduceat(xb[order], bnd, axis=0)
        counts[gids] += np.diff(np.append(bnd, cs.shape[0]))
        if verbose and (i // batch) % 20 == 0:
            el = time.time() - t0
            pps = j / max(1e-9, el)
            print(f"[hnsw] {j}/{n_assign} ({100.0*j/n_assign:.1f}%) {el:.1f}s "
                  f"{pps/1e6:.2f} Mpts/s ETA {(n_assign-j)/max(1.0,pps)/60:.1f} min", flush=True)
    print(f"[hnsw] assign done {time.time()-t0:.1f}s", flush=True)
    return cluster_id, sums, counts


def run_efs_sweep(arr, n_pool, centroids, C, d, efs_list, sweep_n, metric, out_dir,
                  M=32, efc=200, seed=None, suffix=""):
    """Pick efSearch empirically: build one HNSW over the trained centroids, then for each
    efSearch measure recall@1 (vs EXACT flat) and search throughput on a base-point sample.
    Throughput is HNSW-compute-bound (sample held in RAM) so it ~= the full-assign rate.
    NOT part of any timed span. Writes efs_sweep_C<C>.json."""
    cen = np.ascontiguousarray(centroids, dtype=np.float32)
    rng = np.random.default_rng(seed)
    smp = np.sort(rng.choice(n_pool, size=min(sweep_n, n_pool), replace=False))
    x = np.ascontiguousarray(arr[smp], dtype=np.float32)
    flat = (faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d))
    flat.add(cen)
    _, Itrue = flat.search(x, 1)
    Itrue = Itrue.ravel()
    index = faiss.IndexHNSWFlat(d, M, metric)
    index.hnsw.efConstruction = efc
    t0 = time.time()
    index.add(cen)
    print(f"[sweep] HNSW built over {C} centroids in {time.time()-t0:.1f}s "
          f"(M={M} efC={efc}); sweeping efSearch on {len(x):,} sample points", flush=True)
    results = []
    for efs in efs_list:
        index.hnsw.efSearch = int(efs)
        index.search(x[:min(20000, len(x))], 1)            # warm
        t = time.time(); _, I = index.search(x, 1); dt = time.time() - t
        rec = float((I.ravel() == Itrue).mean())
        mpts = len(x) / dt / 1e6
        full_h = (1e9 / (mpts * 1e6)) / 3600.0
        results.append({"efSearch": int(efs), "recall@1": rec,
                        "Mpts_s": mpts, "full_1B_assign_hours": full_h})
        print(f"[sweep]  efS={int(efs):4d}  recall@1={rec:.4f}  {mpts:.3f} Mpts/s  "
              f"-> full 1B assign ~{full_h:.1f} h", flush=True)
    path = os.path.join(out_dir, f"efs_sweep_C{C}{suffix}.json")
    with open(path, "w") as f:
        json.dump({"C": C, "M": M, "efC": efc, "sweep_n": int(len(x)),
                   "results": results}, f, indent=2)
    print(f"[sweep] -> {path}", flush=True)
    return results


def imbalance_factor(cluster_id, C):
    bc = np.bincount(cluster_id, minlength=C).astype(np.float64)
    imb = C * (bc ** 2).sum() / (bc.sum() ** 2)
    return {
        "imbalance": float(imb), "empty": int((bc == 0).sum()),
        "min": int(bc.min()), "max": int(bc.max()),
        "mean": float(bc.mean()), "std": float(bc.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="path to the dataset's base/train .fbin or .u8bin (REQUIRED)")
    ap.add_argument("--out-dir", required=True,
                    help="directory that receives centroids_<C>.fvecs / clusterids_<C>.ivecs "
                         "and the build report (REQUIRED; created if missing)")
    ap.add_argument("--name-suffix", default=None,
                    help="appended to every output file name, so runs with different "
                         "settings can share --out-dir (default: _ip for --metric ip, "
                         "nothing for l2, the names the build scripts look for)")
    ap.add_argument("--dataset", default=None,
                    help="dataset name used in log prefixes and the build report's 'dataset' "
                         "field (default: inferred from --base's parent directory name, e.g. "
                         "'.../datasets/<dataset>/train.fbin' -> '<dataset>')")
    ap.add_argument("--C", type=int, default=None,
                    help="num centroids (default ceil(N/1024), as in scripts/_params.sh)")
    ap.add_argument("--nc1", type=int, default=None, help="L1 coarse cells (default round(sqrt(C)))")
    ap.add_argument("--pts-per-centroid", type=int, default=240,
                    help="n_train = this * C; the sample needs n_train * d * 4 bytes "
                         "(256 is where faiss saturates)")
    ap.add_argument("--n-train", type=int, default=None, help="override n_train explicitly")
    ap.add_argument("--line", choices=["gpu", "cpu"], default="cpu",
                    help="gpu = GPU two-level cluster + EXACT GPU flat assign; "
                         "cpu = CPU two-level cluster + APPROXIMATE HNSW assign")
    ap.add_argument("--hnsw-M", type=int, default=32, help="cpu: HNSW graph degree")
    ap.add_argument("--hnsw-efc", type=int, default=200, help="cpu: HNSW efConstruction")
    ap.add_argument("--hnsw-efs", type=int, default=256,
                    help="cpu: HNSW efSearch (recall@1 vs speed knob for assignment)")
    ap.add_argument("--metric", choices=["l2", "ip"], default="l2")
    ap.add_argument("--niter", type=int, default=25, help="L1 (coarse) k-means iterations")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed of the sampling and the k-means (default: drawn "
                         "at random and printed, so a run can be repeated)")
    ap.add_argument("--ngpu", type=int,
                    default=faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 1)
    ap.add_argument("--use-float16", action="store_true",
                    help="fp16 distances on GPU for assignment (fine for unit-norm data, faster)")
    ap.add_argument("--assign-batch", type=int, default=2_000_000)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--calib-n-train", type=int, default=None,
                    help="timing estimate: cap #training points (KEEP C fixed) -> linear extrapolation")
    ap.add_argument("--calib-n-assign", type=int, default=None,
                    help="timing estimate: assign only the first K base points (KEEP C fixed)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="permit running while the base is still copying (sample/assign only "
                         "within the rows present on disk). For timing estimates only.")
    ap.add_argument("--checkpoint", action="store_true",
                    help="disk-backed resume for the assignment pass (adds IO; OFF for the lean "
                         "real run). Pair with --resume to continue.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--centroids-from", default=None,
                    help="load TRAINED centroids from this .fvecs and SKIP subsample+cluster "
                         "(iterate the assignment cheaply without re-clustering)")
    ap.add_argument("--cluster-only", action="store_true",
                    help="stop after training, before assigning the N base points (the expensive "
                         "step): write trained_centroids_<C>.fvecs and exit. Those are NOT the "
                         "centroids the index is built from -- centroids_<C>.fvecs is the "
                         "per-cluster mean, and only a full run produces it.")
    ap.add_argument("--efs-sweep", default=None,
                    help="comma list of efSearch to sweep, e.g. '16,32,64,128,256'. "
                         "REQUIRES --cluster-only: the sweep runs on the freshly trained "
                         "centroids, in the gap where the full assignment would otherwise "
                         "start. Measures recall@1 against an exact assignment. Not timed.")
    ap.add_argument("--sweep-n", type=int, default=500_000,
                    help="#base points for the efSearch sweep (recall + throughput estimate). "
                         "Only read when --efs-sweep is given.")
    ap.add_argument("--validate", action="store_true",
                    help="sanity recall@1 of assignment vs exact (assignment is exact -> ~1.0). "
                         "Calibration only; adds an extra pass.")
    args = ap.parse_args()

    # dataset tag for logs/report: explicit --dataset, else the base file's parent dir name
    # (e.g. ".../datasets/<dataset>/train.fbin" -> "<dataset>").
    dataset = args.dataset or os.path.basename(os.path.dirname(os.path.abspath(args.base))) or "dataset"

    faiss.omp_set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    metric = faiss.METRIC_INNER_PRODUCT if args.metric == "ip" else faiss.METRIC_L2
    spherical = (args.metric == "ip")
    if args.efs_sweep and not args.cluster_only:
        raise SystemExit(
            "--efs-sweep requires --cluster-only: the sweep runs on the freshly "
            "trained centroids, in the gap before the full assignment starts. "
            "Re-run as: --cluster-only --efs-sweep " + args.efs_sweep)

    mode = "calibration" if (args.calib_n_train or args.calib_n_assign) else "full"

    if args.line == "gpu" and args.ngpu <= 0:
        raise RuntimeError("--line gpu needs CUDA faiss, but faiss reports 0 GPUs")
    if args.validate and mode == "full":
        print("[WARN] --validate adds an extra exact-search pass inside the timed span; "
              "omit it when the timing matters.", flush=True)

    # ==================== TIMED SPAN -- this stage's indexing time ====================
    # first base read -> centroids written -> every point's cluster id written.
    # Nothing else (imbalance, validation, report) lives inside this span.
    t_total = time.time()
    timings = {}

    arr, N_header, d, rows_present = safe_memmap_fbin(args.base)
    C = args.C if args.C else -(-N_header // 1024)  # ceil(N/1024); ALWAYS from full header N (fixed in calib)
    nc1 = args.nc1 if args.nc1 else int(round(math.sqrt(C)))
    n_train = args.n_train if args.n_train else args.pts_per_centroid * C
    n_train = min(n_train, N_header)

    complete = (rows_present >= N_header)
    if not complete and not args.allow_partial:
        raise SystemExit(
            f"[FATAL] base not fully copied: {rows_present:,}/{N_header:,} rows present "
            f"({100*rows_present/N_header:.1f}%). A real build needs the complete file. "
            f"Re-run when the copy finishes, or pass --allow-partial for a timing estimate.")
    pool = N_header if complete else rows_present  # sample/assign domain

    n_train_eff = min(n_train, args.calib_n_train) if args.calib_n_train else n_train
    n_train_eff = min(n_train_eff, pool)
    n_assign = min(pool, args.calib_n_assign) if args.calib_n_assign else pool

    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    print(f"[{dataset}] mode={mode} N_header={N_header:,} rows_present={rows_present:,} "
          f"d={d} C={C} nc1={nc1} n_train={n_train_eff:,} n_assign={n_assign:,} "
          f"metric={args.metric} ngpu={args.ngpu} fp16={args.use_float16} "
          f"pts/centroid={n_train_eff/C:.1f} seed={args.seed}", flush=True)

    sfx = args.name_suffix
    if sfx is None:
        sfx = "_ip" if args.metric == "ip" else ""
    trained_path = os.path.join(args.out_dir, f"trained_centroids_{C}{sfx}.fvecs")

    # ---- 1+2. trained (routing) centroids: load a checkpoint, OR subsample + cluster ----
    if args.centroids_from:
        t = time.time()
        cen_trained = read_fvecs(args.centroids_from)
        assert cen_trained.shape[1] == d, cen_trained.shape
        C = cen_trained.shape[0]
        timings["subsample_io_s"] = 0.0
        timings["cluster_compute_s"] = 0.0
        timings["load_trained_s"] = time.time() - t
        print(f"[{dataset}] loaded {C} TRAINED centroids from {args.centroids_from} in "
              f"{timings['load_trained_s']:.1f}s (skipped subsample+cluster)", flush=True)
    else:
        t = time.time()
        xt = reservoir_sample(arr, pool, n_train_eff, seed=args.seed)
        timings["subsample_io_s"] = time.time() - t

        t = time.time()
        cluster_gpu = args.ngpu if args.line == "gpu" else 0   # the cpu line clusters on CPU
        cen_trained, _stats = two_level_clustering(
            xt, nc1=nc1, nc2=C, rebalance=True, clustering_niter=args.niter,
            gpu=cluster_gpu, seed=args.seed, verbose=True, min_points_per_centroid=5,
            spherical=spherical)
        cen_trained = np.ascontiguousarray(cen_trained, dtype=np.float32)
        assert cen_trained.shape == (C, d), cen_trained.shape
        timings["cluster_compute_s"] = time.time() - t
        del xt                                      # free the training sample before assignment

        # checkpoint the TRAINED (routing) centroids -- these are k-means centroids over the
        # training SAMPLE, used only to route points during assignment. They are DISTINCT from
        # the final centroids_<C>.fvecs (per-cluster MEAN over all assigned points, written
        # after assignment). Saving them allows the assignment pass to be re-run via
        # --centroids-from without re-clustering.
        t = time.time(); write_fvecs_fast(trained_path, cen_trained)
        timings["write_trained_s"] = time.time() - t

    # ---- optional: stop after clustering (checkpoint only; assignment iterated separately) ----
    if args.cluster_only:
        timings["indexing_time_s"] = time.time() - t_total
        print(f"\n[{dataset}] CLUSTER-ONLY done; TRAINED centroids -> {trained_path}\n"
              f"  cluster phase (read -> trained centroids) = {timings['indexing_time_s']:.1f} s "
              f"= {timings['indexing_time_s']/60:.1f} min", flush=True)
        if args.efs_sweep:
            efs_list = [int(v) for v in args.efs_sweep.split(",") if v.strip()]
            run_efs_sweep(arr, pool, cen_trained, C, d, efs_list, args.sweep_n,
                          metric, args.out_dir, M=args.hnsw_M, efc=args.hnsw_efc,
                          seed=args.seed, suffix=sfx)
        rep = {"dataset": dataset, "line": args.line, "mode": "cluster_only",
               "C": int(C), "nc1": int(nc1), "n_train": int(n_train_eff),
               "pts_per_centroid": float(n_train_eff / C),
               "trained_centroids_file": trained_path, "timings_s": timings}
        with open(os.path.join(args.out_dir, f"cluster_report_C{C}_{args.line}{sfx}.json"), "w") as f:
            json.dump(rep, f, indent=2)
        print(f"[{dataset}] cluster report -> "
              f"{os.path.join(args.out_dir, f'cluster_report_C{C}_{args.line}{sfx}.json')}", flush=True)
        return

    # ---- 3. assign all N to the trained centroids + accumulate per-cluster sums ----
    t = time.time()
    if args.line == "gpu":                                     # EXACT GPU flat
        gpu_index = gpu_flat_index(cen_trained, metric, args.ngpu, args.use_float16)
        cluster_id, sums, counts = assign_and_accumulate(
            arr, n_assign, gpu_index, C, d, args.assign_batch,
            ckpt_path=(os.path.join(args.out_dir, f"clusterids_{C}{sfx}.ckpt.npz")
                       if args.checkpoint else None),
            resume=args.resume)
    else:                                                      # APPROXIMATE CPU HNSW
        cluster_id, sums, counts = assign_hnsw_accumulate(
            arr, n_assign, cen_trained, C, d, args.hnsw_M, args.hnsw_efc, args.hnsw_efs,
            args.assign_batch, metric=metric)
    timings["assign_s"] = time.time() - t

    # ---- 4. output centroids = per-cluster MEAN (un-normalized); empties keep trained ----
    nonempty = counts > 0
    out_centroids = np.array(cen_trained, dtype=np.float32)
    out_centroids[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)
    n_empty = int((~nonempty).sum())

    cen_path = os.path.join(args.out_dir, f"centroids_{C}{sfx}.fvecs")
    cid_path = os.path.join(args.out_dir, f"clusterids_{C}{sfx}.ivecs")
    t = time.time(); write_fvecs_fast(cen_path, out_centroids); timings["write_centroids_s"] = time.time() - t
    t = time.time(); write_ivecs_fast(cid_path, cluster_id); timings["write_assign_s"] = time.time() - t

    timings["indexing_time_s"] = time.time() - t_total   # read -> write wall clock
    # =============================== END TIMED SPAN ===============================

    print(f"\n[{dataset}] INDEXING TIME (read -> write) = "
          f"{timings['indexing_time_s']:.1f} s = {timings['indexing_time_s']/60:.1f} min", flush=True)

    # ---- diagnostics & report (NOT part of the timed span) ----
    cn = np.linalg.norm(out_centroids[nonempty], axis=1)
    report = {
        "dataset": dataset, "line": args.line,
        "pipeline": ("gpu-cluster+exact-flat" if args.line == "gpu"
                     else "cpu-cluster+hnsw-approx"), "mode": mode,
        "N_header": int(N_header), "rows_present": int(rows_present), "complete": bool(complete),
        "d": int(d), "C": int(C), "nc1": int(nc1),
        "n_train": int(n_train_eff), "n_assign": int(n_assign),
        "pts_per_centroid": float(n_train_eff / C),
        "metric": args.metric, "ngpu": args.ngpu, "use_float16": args.use_float16,
        "centroid_semantics": "per_cluster_mean_unnormalized",
        "empty_clusters_kept_trained": n_empty,
        "centroid_norms": {"min": float(cn.min()), "mean": float(cn.mean()), "max": float(cn.max())},
        "timings_s": timings,
        "balance": imbalance_factor(cluster_id, C),
        "centroids_file": cen_path, "clusterids_file": cid_path,
    }
    # CPU/HNSW assignment is approximate -> always report recall@1 (vs exact). For the GPU
    # line (exact) it's a ~1.0 sanity check, only when --validate. Runs AFTER the timed span.
    if args.validate or args.line == "cpu":
        rng = np.random.default_rng(args.seed)
        smp = np.sort(rng.choice(n_assign, size=min(100_000, n_assign), replace=False))
        x = np.ascontiguousarray(arr[smp], dtype=np.float32)
        flat = (faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(d))
        flat.add(cen_trained)
        _, It = flat.search(x, 1)
        report["assign_recall@1"] = float((It.ravel() == cluster_id[smp]).mean())
        print(f"[{dataset}] assign_recall@1 = {report['assign_recall@1']:.5f}", flush=True)
    if mode == "calibration":
        fc, fa = n_train / max(1, n_train_eff), N_header / max(1, n_assign)
        report["extrapolation"] = {
            "note": "timing estimate, not a full build; C kept full. cluster/assign extrapolate "
                    "linearly; the read->write total also has IO that does not.",
            "predicted_full_cluster_compute_s": timings["cluster_compute_s"] * fc,
            "predicted_full_assign_s": timings["assign_s"] * fa,
            "cluster_factor": fc, "assign_factor": fa,
        }
        print(f"[{dataset}] timing estimate -- predicted full: "
              f"cluster~{timings['cluster_compute_s']*fc:.0f}s (x{fc:.1f}), "
              f"assign~{timings['assign_s']*fa:.0f}s (x{fa:.1f})", flush=True)

    rep_path = os.path.join(args.out_dir,
                            f"build_report_C{C}_{args.line}_{mode}{sfx}.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== {dataset} build report ===")
    print(json.dumps(report, indent=2), flush=True)
    print(f"\n[{dataset}] report -> {rep_path}", flush=True)


if __name__ == "__main__":
    main()
