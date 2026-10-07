# DINO-10B prefixes

DINO-10B holds 10 billion image-patch descriptors from a DINO ViT-L model:
1024-dimensional **uint8** vectors, searched under L2, with 100,000 queries
and an exact top-10 ground truth for each of its standard sizes. A size-N
subset is simply the first N base vectors, so the dataset scales from 100K to
10B without changing anything but N. It is distributed by Meta; see
`DatasetDINO10B` in
[faiss/contrib/datasets.py](https://github.com/facebookresearch/faiss/blob/main/contrib/datasets.py).

Section 5.4 of the paper runs RaBitQ-SSD on prefixes of 100K, 1M, 10M, 100M,
1B and 10B vectors (Figure 9(b)). This page runs those prefixes with one
command per size.

## How the index differs from the default

The SSD records hold the **original uint8 vectors** instead of ex-codes
(`STORE_ENV=raw`). A record is 1024 bytes, four to a 4 KB page, and because the
stored bytes are the source bytes, the final re-ranking is exact. DRAM keeps a
1-bit code of 256 of the 1024 dimensions per vector, as in the paper, which
has to fit 10 billion of them.

| setting | value | why |
|---|---|---|
| SSD store | `STORE_ENV=raw` | uint8 vectors are already compact; re-ranking is exact |
| in-RAM dims `memdim` | 256, a quarter of D | the 10B index had to fit in DRAM |
| clusters `C` | N/4096, rounded down | about 4096 points per cluster, four times the default |
| clustering | on GPU, [below](#clustering) | |
| `k` | 10 | the dataset ships ground truth at k = 10 only |
| threads `T` | 64 | |
| recall sweep | 0.80 to 0.96 | covers the 0.90 and 0.95 points the report marks |

Everything else is the default configuration: the IRQ coarse quantizer, a
drain threshold of 0.4 and 8 in-flight reads per thread at T = 64.

## Clustering

The base is clustered on GPU, in three steps that write into
`$INDEX_ROOT/dino<size>/clustering/`:

1. **train**: [`gpu_kmeans.py`](../scripts/dino_exp/gpu_kmeans.py) trains the
   C centroids with faiss. It samples 128 base vectors per centroid, evenly
   spaced from a random start, splits the sample into nc1 first-level cells
   with k-means (nc1 = round(√C), and 3072 at 10B), and gives each cell its
   own k-means with a share of the C centroids proportional to its size.
2. **assign**: [`gpu_assign.py`](../scripts/dino_exp/gpu_assign.py) assigns
   every base vector with PyTorch: it compares the vector with the nc1
   first-level centroids, then with the centroids of its 8 nearest cells,
   using TF32 matrix products, and keeps the nearest. Each cluster's mean
   becomes its centroid.
3. **inner**: [`gpu_inner_kmeans.py`](../scripts/dino_exp/gpu_inner_kmeans.py)
   groups the C centroids, about 1000 to a group, for the IRQ coarse
   quantizer.

The steps need faiss with GPU support and PyTorch with CUDA, and use every GPU
they see (`CUDA_VISIBLE_DEVICES` restricts them). Without a GPU,
`DINO_CLUSTER_ENV=cpu` clusters with `clustering/run_kmeans.sh`, as for the
other datasets, for sizes up to 2B.

## Running it

```sh
export DATA_ROOT=$PWD/data INDEX_ROOT=$PWD/indexes
export PYTHON_ENV=/path/to/python         # numpy, faiss with GPU support
export TORCH_PYTHON_ENV=/path/to/python   # numpy, PyTorch with CUDA, if another one
scripts/dino_exp/run_dino.sh 1M
```

The size is 100K, 1M, 10M, 100M, 1B or 10B, the sizes in the paper, or any
other size the dataset ships a ground truth for. 5B and 10B need the build
described [below](#5b-and-10b). The driver runs five stages, each skipped when
its outputs already exist:

1. **data**: [`tools/dino_prepare.py`](../tools/dino_prepare.py) writes
   `train.u8bin` (the first N base vectors), `test.u8bin` (the queries) and
   `gt10` (the shipped ground truth for N) into `$DATA_ROOT/dino1m/`.
2. **cluster**: the three [clustering](#clustering) steps with C = N/4096.
3. **build**: `scripts/build_index/build_index.sh` with `STORE_ENV=raw MEMDIM_ENV=256`.
4. **search**: `scripts/main_exp/run_main_exp.sh` with `TOPKS_ENV=10 TS_ENV=64`.
5. **report**: the operating points the search measured, from
   [`scripts/main_exp/operating_points.py`](../scripts/main_exp/operating_points.py).

The report lists every measured operating point and marks the first one that
reaches recall 0.90 and 0.95, i.e. the smallest `nprobe` achieving that
recall:

```
dino1m  topk=10  threads=64
    nprobe   recall        QPS   pages/q   lat_ms
         1      ...        ...       ...      ...
       ...
         9      ...        ...       ...      ...  <- first to reach 0.90
       ...
        43      ...        ...       ...      ...  <- first to reach 0.95
       ...
```

Absolute numbers depend on your SSD and CPU, so run benchmarks on an idle
machine; see the [notes in the README](../README.md#notes).

`TS_ENV="64 16"` adds thread counts, and `TS_ENV`, `BUDGET_GB_ENV`,
`RECALL_HI_ENV` and `RECALL_PLATEAU_ENV` pass through to the stages.

## Getting the data

By default stage 1 downloads from the dataset's public URL with HTTP range
requests, so a 1M prefix fetches about 1.1 GB: the first million base
vectors, the queries and one ground-truth file. An interrupted download
resumes where it stopped.

With the dataset already on disk, point the driver at it:

```sh
DINO_SOURCE_ENV=/path/to/dino_vitl_10B scripts/dino_exp/run_dino.sh 1M
```

The directory may be laid out as the download (`chunked_base_10B/`,
`queries_clean.bvecs`, `gts/`), or hold one `base.bvecs` with all chunks
concatenated. `tools/dino_prepare.py --help` lists overrides for single files.

## Resources per size

| size | C | download | on SSD (.ssd) | in DRAM (.base) |
|---|---|---|---|---|
| 100K | 24 | 0.2 GB | 0.1 GB | 5 MB |
| 1M | 244 | 1.1 GB | 1.0 GB | 49 MB |
| 10M | 2,441 | 10.4 GB | 10.2 GB | 0.5 GB |
| 100M | 24,414 | 103 GB | 102 GB | 4.9 GB |
| 1B | 244,140 | 1.02 TB | 1.02 TB | 49 GB |
| 10B | 2,441,406 | 10.28 TB | 10.24 TB | 0.53 TB |

The build streams the data, but its reorder scratch temporarily takes as much
disk as the base; `RABITQ_BUILD_SCRATCH_DIR` moves it. The driver sets the
build's memory budget from N (`BUDGET_GB_ENV` overrides it), and before the
build it checks that the budget is available and that `INDEX_ROOT` has room
for the index.

## 5B and 10B

Beyond 2^32 - 1 vectors the point ids need 64 bits. Build a second set of
binaries once, next to the default ones:

```sh
cmake -S . -B build64 -DCMAKE_BUILD_TYPE=Release -DRABITQ_PID64=ON
cmake --build build64 --parallel 8
```

They land in `bin64/`, and the driver uses them for 5B and 10B. An index
built by `bin64/` is read only by `bin64/`, and one built by `bin/` only by
`bin/`.

A `.u8bin` cannot hold more than 2^31 - 1 vectors, so stage 1 writes the base
as `train.bvecs`, in the dataset's own layout, and the ground truth with
64-bit ids. If the source is a local `base.bvecs`, `train.bvecs` is a link to
it and nothing is copied; the stages then index its first N vectors
(`N_ENV=N`). Otherwise stage 1 concatenates the first N vectors from the
chunks, 10.28 TB at 10B.

At 10B the machine needs:

- **RAM**: about 345 GB for the clustering's training sample, a build budget
  of about 700 GiB, and the 0.53 TB in-DRAM index while searching.
- **GPU**: about 46 GB of memory on each GPU for the assignment.
- **Disk**: 10.24 TB for the SSD file in `INDEX_ROOT`, plus as much again for
  the build's reorder scratch, which is released as the SSD file fills; when
  both share a file system, the build peaks at about one copy of the base.

## Running the stages by hand

The driver only fixes settings; the same index comes from the ordinary stages:

```sh
python tools/dino_prepare.py 1M --out-dir $DATA_ROOT/dino1m
CL=$INDEX_ROOT/dino1m/clustering
python scripts/dino_exp/gpu_kmeans.py --base $DATA_ROOT/dino1m/train.u8bin \
  --C 244 --out-dir $CL                                        # with faiss
python scripts/dino_exp/gpu_assign.py --base $DATA_ROOT/dino1m/train.u8bin \
  --centroids $CL/trained_centroids_244.fvecs --out-dir $CL    # with PyTorch
python scripts/dino_exp/gpu_inner_kmeans.py $CL/centroids_244.fvecs \
  $CL/inner_centroids_C244_innersize1000.fvecs \
  $CL/inner_clusterids_C244_innersize1000.ivecs l2 mean        # with faiss
export C_ENV=244 MEMDIM_ENV=256 STORE_ENV=raw
scripts/build_index/build_index.sh dino1m
TOPKS_ENV=10 TS_ENV=64 RECALL_HI_ENV=0.96 RECALL_PLATEAU_ENV=0.96 \
  scripts/main_exp/run_main_exp.sh dino1m
```

At 10B, `gpu_kmeans.py` also takes `--nc1 3072`, the base is `train.bvecs`,
and the build and search take `N_ENV=10000000000` and a larger
`BUDGET_GB_ENV`.

The raw store is not specific to DINO. `STORE_ENV=raw` works on any float32
or uint8 dataset, and keeps each vector in its own type: a 1280-dimensional
float32 YFCC record takes 5120 bytes, two pages.
