# Getting started

A complete run on **YFCC-Images** — download, cluster, build, search — in the
order you would do it.

YFCC-Images is 98.7M images embedded by a CLIP model into 1280-dimensional
vectors, distributed by
[big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/dataset_preparation/yfccimages_dataset.md).
The full set is about 471 GiB, and the same page publishes 1M and 10M random
subsets. This walkthrough uses the **1M subset**, downloads about 5.6 GB of
base and query vectors, and generates ground truth for that exact base.
[Scaling up](#5-scale-up) uses the same commands with a larger base file.

## 0. Build the binaries

```sh
sudo apt install cmake g++ libaio-dev time wget python3-venv
git clone https://github.com/VectorDB-NTU/RaBitQ-SSD.git
cd RaBitQ-SSD
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
```

Every command below runs from the repository root as well.

The clustering, ground-truth and index-build stages each run a Python script
that imports `faiss` and `numpy`. You can use an existing Python environment
with these packages, or create one:

```sh
python3 -m venv "$HOME/.venvs/rabitq-ssd"
. "$HOME/.venvs/rabitq-ssd/bin/activate"
python -m pip install faiss-cpu numpy
export PYTHON_ENV="$(command -v python)"
```

When using an existing environment, set `PYTHON_ENV` to its Python executable.
Export it again in each new terminal used for the workflow. The ground-truth
command below also uses this interpreter. The ground-truth script defaults to
**GPU** faiss (`--gpu 0`); with `faiss-cpu`, or with a faiss built for an older
CUDA than your card, use `--cpu --threads N` instead — equally exact, slower.

## 1. Get the data

```sh
export DATA_ROOT=$PWD/data INDEX_ROOT=$PWD/indexes
mkdir -p "$DATA_ROOT/yfcc1m" "$INDEX_ROOT"
cd "$DATA_ROOT/yfcc1m"

URL=https://comp21storage.z5.web.core.windows.net/yfcc100m_images
wget -O train.fbin "$URL/yfcc100m_vecs_sampled_1m.fbin"     # 5.12 GB, 1M x 1280
wget -O test.fbin  "$URL/yfcc100m_query_vecs.fbin"           # 512 MB, 100k queries
cd -
```

If downloads are slow, a parallel downloader such as `aria2c` can use several
connections. Use the same URLs and output filenames as above.

Every stage looks for `train.fbin` (the vectors to index) and `test.fbin` (the
queries) inside `$DATA_ROOT/<dataset>/`, so the directory name — `yfcc1m` here
— is the only name you pass around. Ground truth sits beside them as
`gt<topk>`.

**Generate ground truth for the downloaded 1M base.** Ground truth from a
different subset is not valid even when the queries are identical. This
walkthrough generates separate `gt10`, `gt100`, and `gt1000` files:

```sh
"${PYTHON_ENV:-python3}" groundtruth/compute_gt_type1_l2exact.py \
    --base "$DATA_ROOT/yfcc1m/train.fbin" \
    --queries "$DATA_ROOT/yfcc1m/test.fbin" \
    --out-dir "$DATA_ROOT/yfcc1m" --out-names gt10,gt100,gt1000 \
    --maxk 1024 --rerank-m 2048 --cpu --threads 64
```

These files contain **distances as well as ids**, enabling tie-aware recall:
a result counts as correct if it ties with the k-th neighbour. The search step
prints `Groundtruth: type-1 (ids+distances)` when it detects this format.

Working with your own data instead? Write it as `.fbin` — `[int32 n][int32 d]`
followed by `n·d` float32 values, row by row — or convert from `.fvecs` with
[`tools/fvecs_to_fbin.py`](../tools/fvecs_to_fbin.py). Then generate ground
truth yourself; see [groundtruth/README.md](../groundtruth/README.md).

Already have the vectors somewhere on this machine? A stage only ever reads
`train.fbin` and `test.fbin`, so link them in rather than copying:

```sh
mkdir -p $DATA_ROOT/<dataset>
ln -s /existing/path/base.fbin    $DATA_ROOT/<dataset>/train.fbin
ln -s /existing/path/queries.fbin $DATA_ROOT/<dataset>/test.fbin
```

Everything generated afterwards — ground truth, the index, the results — is
written under `$DATA_ROOT/<dataset>/`, `$INDEX_ROOT` and `results/`, so the
original files are never modified.

## 2. Cluster the vectors

```sh
DATA_ENV=$DATA_ROOT/yfcc1m/train.fbin \
OUTDIR_ENV=$INDEX_ROOT/yfcc1m/clustering \
  clustering/run_kmeans.sh yfcc1m
```

This produces `centroids_<C>.fvecs` and `clusterids_<C>.ivecs`. The number of
clusters defaults to `ceil(N/1024)` — 977 here — and every later stage rederives
it from the same file, so there is nothing to write down. At 100M scale this
stage becomes the long pole, mostly in the assignment pass.

## 3. Build the index

```sh
scripts/build_index/build_index.sh yfcc1m
```

Three phases, each skipped if its output already exists: a second-level k-means
over the centroids (the coarse quantizer's own index), then the inverted list,
then the coarse quantizer itself.

**Disk.** The runner always builds in streaming mode, so the build always
reorders every vector into a scratch directory first and always needs **as much
free space as the dataset itself** on top of the index — about 5.12 GB for this
walkthrough. The scratch sits next to `train.fbin` and is deleted when the build
finishes. If that volume has no room, put it elsewhere:

```sh
RABITQ_BUILD_SCRATCH_DIR=/path/with/room \
  scripts/build_index/build_index.sh yfcc1m
```

The build checks that space before starting, so an undersized volume fails in
seconds rather than hours in.

By default the full 1-bit code stays in RAM, while the extra-precision codes
remain on SSD. To retain only one quarter of the 1-bit code in RAM and store
its remaining dimensions on SSD:

```sh
MEMDIM_ENV=320 scripts/build_index/build_index.sh yfcc1m
```

`memdim` controls how many of the 1280 dimensions of the 1-bit code stay
resident. Each value is a separate index, so you can build several and compare —
which is why **`MEMDIM_ENV` has to be repeated at search time**. It appears in
the index filename, so searching without it looks for the default index instead:
either the run stops on a missing file, or, if you also built the default, it
quietly benchmarks the wrong one.

## 4. Search

Start with a single cell — one topk, one thread count:

```sh
TOPKS_ENV=10 TS_ENV=8 scripts/main_exp/run_main_exp.sh yfcc1m
```

Both variables matter. The grid otherwise defaults to `topk = 10, 100, 1000`
crossed with `threads = 64, 32, 16, 8, 4, 1` — eighteen cells, each searching
for several operating points and timing every one over all 100,000 queries,
which is hours rather than minutes. The ground-truth command above generates
all three files needed by that grid. If you instead have only a published
`gt100`, use `TOPKS_ENV=100`. To include top-10, make a separate copy named
`gt10`; top-1000 requires a separately generated `gt1000`.

Widen it once a cell has run end to end:

```sh
TOPKS_ENV="10 100" TS_ENV="64 8" scripts/main_exp/run_main_exp.sh yfcc1m
```

For each `topk` and thread count the sweep searches for the operating points
that reach a range of recall targets, then times each one. Output goes to
`results/main_exp/`, one CSV per (topk, threads) cell plus, under `merged/`,
one CSV joining every cell of that configuration; the columns are defined in
[OUTPUT_SCHEMA.md](../OUTPUT_SCHEMA.md). Each cell's
log also carries a per-operating-point table, of this shape:

```
nprobe         QPS   QPSnoCoarse      Recall     AvgIOReqs      AvgIOPages   ...
    16      ...           ...        0.905           ...             ...
    32      ...           ...        0.951           ...             ...
    64      ...           ...        0.982           ...             ...
```

Reading it: `nprobe` is how many clusters the search visited. Increasing it
adds search work and typically improves recall. Pick a row that reaches the
recall you need and read its
QPS — that pair is what you compare against another system. `QPSnoCoarse`
excludes the coarse-quantizer step, which tells you how much time went into
picking clusters rather than reading and re-ranking data; `AvgIOPages` is the
4 KiB SSD reads per query, the cost a smaller `memdim` raises.

Absolute numbers depend on your SSD and CPU, so run benchmarks on an idle
machine — see the [notes in the README](../README.md#notes) for why competing
load moves recall and not just throughput.

## 5. Scale up

The three published sizes differ only in which base file you download; the
query set and the commands are identical, and `C` follows from `N`:

| dataset | base file | size (GiB, approximate) | `C` |
|---|---|---|---|
| 1M | `yfcc100m_vecs_sampled_1m.fbin` | 4.8 | 977 |
| 10M | `yfcc100m_vecs_sampled_10m.fbin` | 48 | 9766 |
| 98.7M | `yfcc100m_vecs.fbin` | 471 | 96422 |

Take the matching ground truth — `yfcc100m_query_gt100_sampled_10m.bin` or
`yfcc100m_query_gt100.bin` — since neighbours are only exact with respect to
the base set they were computed over. Store it as `gt100`. For top-10 search,
make a separate copy named `gt10` in the same dataset directory; a top-100
file also supports evaluating top-10.

Two things to set at the larger sizes: `BUDGET_GB_ENV` to the DRAM the build
may use (default 128), and `INDEX_ROOT` to a device with room for the index.
The build streams the dataset within that budget, so it does not need to hold
the vectors in memory.

Expect the clustering and the index build to be the long poles: on a 64-core
machine the full 98.7M set took about 50 minutes to cluster and 30 minutes to
build, with the search sweep on top of that.

## Where to go next

- Trade memory for SSD: rebuild with a smaller `MEMDIM_ENV` and compare curves.
- Compare coarse quantizers: `COARSE_ENV=flat` or `hnsw`, in both the build and
  the search step. They share the index, so only the coarse phase is rebuilt.
- Inner-product data: repeat `METRIC_ENV=ip` for clustering, build, and search. (YFCC
  vectors are normalized, so its L2 ground truth stays valid under
  inner-product search.)
- The full parameter list is in the [README](../README.md), and every binary
  under `bin/` prints its arguments when run with none.
