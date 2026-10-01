# RaBitQ-SSD

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for the paper **RaBitQ-SSD: Split Codes and Pipelined I/O for
SSD-Resident Vector Search**. See also the
[technical report](technical_report.pdf).

Billion-scale approximate nearest-neighbour search over vectors that do not fit
in memory.

RaBitQ-SSD extends IVF-RaBitQ to a DRAM–SSD hierarchy. Split-RaBitQ keeps a
configurable prefix of each 1-bit code in DRAM to estimate distances and prune
candidates using lower bounds. SSD records store the remaining code components
and extra-precision codes for re-ranking. An asynchronous search pipeline
coordinates pruning with page reads and overlaps computation with I/O.

The resident prefix length controls the memory budget. A shorter prefix saves
memory and scanning work, but can require more SSD reads because its distance
bounds are looser.

By default, an embedded IVF-RaBitQ index selects candidate clusters from the
coarse centroids. Flat and HNSW coarse quantizers are also available.

Supports L2 and inner-product metrics, and builds indexes larger than RAM by
streaming the dataset within a memory budget you set.

## Quick start

**[docs/getting_started.md](docs/getting_started.md)** walks through a complete
run — download, cluster, build, search — on a 1M-vector subset of YFCC-Images,
then scales the same commands to the full 98.7M. Start there.

To build the command-line tools:

```sh
sudo apt install cmake g++ libaio-dev time
git clone https://github.com/VectorDB-NTU/RaBitQ-SSD.git
cd RaBitQ-SSD
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
```

Subsequent workflow commands run from the repository root unless stated
otherwise. The current kernels require Linux and an x86-64 CPU with AVX2 or
AVX-512. The build uses `-march=native`, so compile on the machine where the
binaries will run.

Binaries land in `bin/`: `build_invlist`, `build_coarse`, `querying`. Each
prints its arguments when run with none.

The clustering, ground-truth **and index-build** stages each run a Python script
that imports `faiss` and `numpy`. If `python3` is not an interpreter that has
them, set `PYTHON_ENV=/path/to/that/python` on every one of those stages — the
build stage included, where the failure otherwise surfaces only as
`[FATAL] inner k-means failed` with the traceback buried in a log.

The search (`querying`) reads the SSD through Linux libaio.

## The pipeline

Four stages. `<dataset>` is the directory holding `train.fbin` and `test.fbin`
under `$DATA_ROOT`; the ground-truth stage is the one exception — it takes
explicit paths, and you can skip it when your dataset ships its own.

| stage | command | produces |
|---|---|---|
| 1. Cluster | `DATA_ENV=$DATA_ROOT/<ds>/train.fbin OUTDIR_ENV=$INDEX_ROOT/<ds>/clustering clustering/run_kmeans.sh <ds>` | centroids + per-point assignment |
| 2. Ground truth | `groundtruth/compute_gt_type1_l2exact.py --base … --queries … --out-dir $DATA_ROOT/<ds>` | `gt10`, `gt100`, `gt1000`, beside the vectors |
| 3. Build | `scripts/build_index/build_index.sh <ds>` | the in-RAM and on-SSD index |
| 4. Search | `scripts/main_exp/run_main_exp.sh <ds>` | recall/throughput CSVs |

Stage 1 takes its two paths explicitly, and `OUTDIR_ENV` must be
`$INDEX_ROOT/<dataset>/clustering` or stage 3 will not find the centroids.
Stage 2 writes `gt<topk>` next to `train.fbin`, which is where stage 4 looks.

```sh
export DATA_ROOT=/path/to/datasets    # one directory per dataset
export INDEX_ROOT=/path/to/indexes    # indexes go here; may be another device
```

Both are required — no path is baked into any script, and a stage stops with an
instruction if you have not set what it needs. They live only in the shell that
exported them, and the stages are hours apart, so put them in your profile or
re-export them in each new terminal.

**Disk.** The build reorders the whole dataset into a scratch directory before
quantizing, so it needs free space equal to the dataset on top of the index.
It checks before starting; `RABITQ_BUILD_SCRATCH_DIR` moves the scratch
elsewhere.

## Parameters

Nothing is configured per dataset. A `.fbin` begins with `[int32 n][int32 d]`,
so every stage reads those two numbers off the data and derives the rest the
same way — which is why the clustering, the index and the search agree without
a config file:

| parameter | default | override |
|---|---|---|
| clusters `C` | `ceil(N / 1024)` | `C_ENV` |
| in-RAM dims `memdim` | the full vector width, rounded up to a multiple of 64 | `MEMDIM_ENV` (a multiple of 64) |
| metric | `l2` | `METRIC_ENV=ip` |
| bits per dimension `B` | 9 — one sign bit plus 8 of extra precision | `B_ENV` (2–9) |
| cluster layout | `ordered` — points sorted by distance to their centroid | `ORDER_ENV=unordered` |

All five appear in the index directory and file names, so **an override must be
repeated across stages**. Omitting one can either produce a missing-file error
or select an existing index built with the defaults. They are derived in one place
([`scripts/_params.sh`](scripts/_params.sh)) precisely so the build and the
search cannot compose different names.

Lower `B` reduces the SSD footprint and re-ranking precision.

Other knobs, all optional:

| variable | default | effect |
|---|---|---|
| `COARSE_ENV` | `irq` | coarse quantizer: `irq`, `flat` or `hnsw` |
| `TOPKS_ENV`, `TS_ENV` | full grid | restrict the sweep, e.g. `TOPKS_ENV=10 TS_ENV=64` |
| `NPROBES_ENV` | unset | comma-separated nprobe values to time, instead of searching for operating points |
| `RECALL_HI_ENV`, `RECALL_PLATEAU_ENV` | 0.98 / 0.99 | cap the swept recall band, for data whose ground truth cannot reach the default |
| `DRAIN_ALPHA_ENV` | 0.4 | quantile threshold for drain-phase pruning; 1.0 uses the standard top-k threshold |
| `BUDGET_GB_ENV` | 128 | build-time DRAM ceiling, in GiB |
| `FASTER_ENV` | `false` | faster quantization at the build; trades code quality for build time. It is **not** part of the index path, so build two values into different `INDEX_ROOT`s or the second run will reuse the first index |
| `INNER_SIZE_ENV` | 1000 | how many first-level centroids go in each second-level cluster of the IRQ coarse quantizer. A size, unlike `C_ENV`: the inner cluster count follows as `round(C / it)`. A non-default value is part of the coarse quantizer's file name, so like the five parameters above it must be repeated at search time |
| `RABITQ_BUILD_SCRATCH_DIR` | next to `train.fbin` | where the streaming build puts its reorder scratch, which grows to the size of the dataset |
| `RABITQ_ROTATOR_SEED` | unset — a fresh random rotation each build | fixes the quantizer's random rotation. Reproducing an entire index also requires matching the other build settings and inputs |
| `MEM_HEADROOM_MB_ENV` | 1024 | working-set margin the search pre-flight requires beyond the index |
| `RABITQ_QD_PER_THREAD` | 8–16 by thread count | in-flight SSD reads per thread; set it to pin one value |
| `RABITQ_IRQ_OUTER_MULT` | 0.5 | coarse-search effort, relative to nprobe |
| `PYTHON_ENV` | `python3` | interpreter with faiss + numpy |

Calling the binaries directly instead of through the scripts?
**[docs/cli_reference.md](docs/cli_reference.md)** documents every argument and
environment variable of `build_invlist`, `build_coarse` and `querying`.

## Notes

- **Run benchmarks on an idle machine.** Competing SSD or CPU load changes not
  only throughput but recall: the pruning threshold applied while draining
  outstanding reads depends on how much re-ranking finished first, which
  depends on IO timing. Numbers from differently loaded machines are not
  comparable.
- Ground truth may carry ids only, or ids plus distances. With distances, a
  result counts as correct if it ties with the k-th neighbour — which is what
  makes recall meaningful on data containing duplicate vectors.
- A coarse quantizer records a fingerprint of the index it was built from and
  refuses to load against a different one.

## Repository layout

```
apps/                     build_invlist, build_coarse, querying
rabitqlib/, src/          the implementation (header-only) + vendored Eigen, hnswlib
clustering/               k-means over the vectors, and over the centroids
groundtruth/              exact-KNN ground truth: generation and validation
scripts/                  the build and search stages
tools/                    format conversion
docs/                     getting_started.md, cli_reference.md
```

`rabitqlib/` is a vendored copy of the RaBitQ-Library headers extended
with this project's SSD index, coarse quantizer and streaming build. The
library's other index types come along unused; being header-only templates,
they cost nothing.

## License

MIT — see [LICENSE](LICENSE). Vendored components keep their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
