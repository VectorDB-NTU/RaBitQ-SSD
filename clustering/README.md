# Clustering

Stage 1 of the pipeline: partition the base vectors into `C` clusters. This
produces the centroids the index is built against, and the cluster assignment
of every base vector.

```
run_kmeans.sh        first-level k-means: ./run_kmeans.sh <dataset>
two_level_kmeans.py  the driver run_kmeans.sh calls
ivf_centroids.py     second-level k-means over the centroids (input to the IRQ
                     coarse quantizer; build_index.sh runs it for you)
```

Needs Python with faiss (the CPU build is enough) and numpy.

## Running it

```sh
DATA_ENV=/path/to/datasets/<dataset>/train.fbin \
OUTDIR_ENV=/path/to/indexes/<dataset>/clustering \
  ./run_kmeans.sh <dataset>
```

Both variables are required, and the script exits with an instruction naming
whichever is missing. `PYTHON_ENV` selects the interpreter (default `python3`).
The base may be a float32 `train.fbin` or a uint8 `train.u8bin`; uint8 rows are
converted to float32 as they are read.

Outputs land in `OUTDIR_ENV`:

| file | what |
|---|---|
| `centroids_<C>.fvecs` | the routing table: per-cluster mean of the assigned vectors |
| `clusterids_<C>.ivecs` | cluster id of every base vector, in base order |
| `trained_centroids_<C>.fvecs` | the pre-assignment centroids, kept for reference |
| `build_report_C<C>_cpu_full.json` | per-phase timings, `assign_recall@1`, cluster imbalance |

With `METRIC_ENV=ip` every name ends in `_ip`, as in `centroids_<C>_ip.fvecs`,
so an l2 and an ip clustering of one dataset can share `OUTDIR_ENV`.

`assign_recall@1` is how often the approximate assignment picked the true
nearest centroid; `imbalance` is the cluster size-balance factor, 1.0 being
perfectly balanced. Both are worth a glance before building on the result.

Re-running with the same `C` skips the work if the outputs already exist.

## What it does, and why

Assigning `N` points to `C` centroids exhaustively costs `O(N·C·d)`, which is
out of reach once both are large. Two things bring it down:

- **Two-level k-means** trains through a cascade of about `√C` coarse cells,
  each split into about `√C` again, turning `C` in the cost from a product into
  a sum — `O(N·√C·d)`.
- **HNSW assignment** routes each base point to its nearest centroid through a
  graph over the centroids rather than a full scan. The result is approximate;
  `efSearch` trades that accuracy against speed. Searching more than one cluster
  at query time absorbs the occasional mis-assignment.

The base is never loaded whole — the training sample is drawn by reservoir
sampling and the assignment pass streams a read-only memmap — so peak memory is
set by the training sample, `pts_per_centroid · C · d · 4` bytes, not by the
dataset.

## Parameters

`C` defaults to `ceil(N/1024)` and the fan-out follows as `round(√C)`; both are
derived from the base file, so nothing needs configuring per dataset. See
[`scripts/_params.sh`](../scripts/_params.sh) for the derivation and the
`C_ENV` / `METRIC_ENV` overrides.

The remaining knobs are fixed in the runner: 240 points per centroid, HNSW
efSearch 64, 25 iterations. The sample and the k-means draw a random seed per
run and print it; `two_level_kmeans.py --seed` repeats a run exactly.

To change something the runner does not expose, call the driver directly;
`two_level_kmeans.py --help` lists every flag.

`--line gpu` clusters and assigns on GPU — exact assignment, needs a CUDA build
of faiss. The pipeline uses the CPU line; the GPU line is not exercised by the
scripts.

## Tuning efSearch

The stage has two halves, and they cost very differently. Training the `C`
centroids reads only the sample; assigning all `N` base points reads everything.
On a 98.7M-vector run the split was about 9 minutes to train and 43 to assign.

Three flags exist to exploit that asymmetry, and they only make sense together:

| flag | what it does |
|---|---|
| `--cluster-only` | stop after training, before the assignment pass, and exit |
| `--efs-sweep 16,32,64,128,256` | in that gap, assign a sample at each efSearch and score recall@1 against an exact assignment |
| `--centroids-from FILE` | skip training and reuse centroids you already have |

So the loop is: train once, sweep to find the knee, then do the real run.

```sh
# 1. train, and measure what each efSearch would buy
python two_level_kmeans.py --base .../train.fbin --out-dir DIR \
    --cluster-only --efs-sweep 16,32,64,128,256      # -> efs_sweep_C<C>.json

# 2. the real run, with the efSearch you picked
DATA_ENV=... OUTDIR_ENV=DIR ./run_kmeans.sh <dataset>
```

**`--efs-sweep` requires `--cluster-only`** — the sweep runs on the freshly
trained centroids, in exactly the gap where the full assignment would otherwise
begin, so there is nowhere else to put it. Passing it alone now stops with that
message rather than being silently ignored.

**`--cluster-only` does not produce a usable clustering.** It writes
`trained_centroids_<C>.fvecs`; the index is built from `centroids_<C>.fvecs`,
which is the per-cluster *mean* of the assigned vectors and therefore only
exists after a full run. `build_index.sh` looks for the latter.

The runner's default of 64 was chosen this way: recall@1 climbs steeply to about
64 and then flattens while assignment time keeps growing.

## Second-level k-means

The IRQ coarse quantizer needs the `C` centroids themselves grouped, into
`round(C / inner_size)` inner clusters. `build_index.sh` runs this when the
files are missing, so you do not normally call it:

```sh
RABITQ_IRQ_INNER_SIZE=1000 python ivf_centroids.py \
    centroids_<C>.fvecs \
    inner_centroids_C<C>_innersize1000.fvecs \
    inner_clusterids_C<C>_innersize1000.ivecs \
    l2 mean
```

`inner_size` defaults to 1000. The metric must match the one the index is built
with (`ip` uses spherical k-means and IP routing), and for `ip` `build_index.sh`
reads `centroids_<C>_ip.fvecs` and ends both output names in `_ip`. The trailing
argument selects the inner-centroid scheme; `mean` is the only accepted value.
