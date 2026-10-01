# Command-line reference

The three binaries in `bin/`. The scripts under `scripts/` drive them for you —
this page is for calling them directly, or for understanding what a script did.

Every binary prints a usage summary when run with no arguments.

Vector files are `.fbin`: `[int32 n][int32 d]` then `n·d` float32 values, row by
row. `.fvecs` is auto-detected where noted; convert with
[`tools/fvecs_to_fbin.py`](../tools/fvecs_to_fbin.py).

---

## Building the inverted list — `bin/build_invlist`

```
bin/build_invlist <data> <centroids> <cluster_ids> <total_bits> <base_index> <ssd_index> [options...]
```

Six required positional arguments, then any number of optional ones in any
order.

1. **data**: the vectors to index, `.fbin` or `.fvecs` (auto-detected).
2. **centroids**: the cluster centroids, `.fvecs`, as produced by
   `clustering/run_kmeans.sh`.
3. **cluster_ids**: the cluster assignment of every base vector, `.ivecs`, from
   the same run. It must have been produced against the same centroids —
   nothing checks this, and a mismatch yields a silently wrong index.
4. **total_bits**: bits per dimension for the quantized code. The scripts use
   9: one sign bit plus 8 bits of extra precision. Fewer bits shrink the index
   and lower recall at a given `nprobe`.
5. **base_index**: output path for the in-RAM part of the index.
6. **ssd_index**: output path for the on-SSD part. Put it on the device you
   intend to search from; reads use `O_DIRECT` to bypass the Linux page cache.

Optional arguments, recognised by their form rather than their position:

7. **`l2`** or **`ip`** (default `l2`): the metric. Must match the metric the
   clustering used, and the one you search with.
8. **`true`** or **`false`** (default `false`): faster quantization. Trades a
   little code quality for build time.
9. **`unordered`** or **`ordered`** (default `unordered`): whether points
   within a cluster are stored in centroid-distance order. `ordered` is what
   the scripts build.
10. **a bare integer** (default: the padded vector width): `memdim`, how many
    leading dimensions of the 1-bit code stay resident in RAM. The remaining
    dimensions and the extra-precision codes are stored on SSD. Must be a
    multiple of 64. Lowering it reduces the in-RAM footprint and can increase
    SSD reads per query.
11. **`mem_budget_gb=<GiB>`** (default: unset): build within this DRAM ceiling.
    Unset loads the whole dataset and then quantizes, which needs room for the
    dataset in fp32. When set, the dataset is streamed instead and never held
    whole. **The resulting index is the same either way** — this only decides
    whether the build fits in memory. Keyed rather than positional
    because the bare-integer slot is already `memdim`.

The streaming path trades DRAM for disk: it reorders every vector into a
scratch directory first, so it needs **free disk equal to the dataset size**,
released once the build finishes. `RABITQ_BUILD_SCRATCH_DIR` chooses where that
goes; the default is beside the data file. The space is checked up front, so an
undersized volume fails immediately rather than hours in.

The build prints the metric, quantization mode, order mode and build mode it
resolved, so a run's log records what it actually did.

---

## Building the coarse quantizer — `bin/build_coarse`

```
bin/build_coarse <base_index> <coarse_index> <coarse_kind> [key=value...]
```

The coarse quantizer decides which clusters a query visits. It is a separate
artifact from the index, so you can build several over one index and compare
them without rebuilding anything else.

1. **base_index**: an existing base index from `build_invlist`.
2. **coarse_index**: output path.
3. **coarse_kind**: one of
   - `flat` — exhaustive scan of the centroids. Exact, and the cost grows
     linearly with the number of clusters.
   - `hnsw` — a graph over the centroids. Approximate; search effort scales
     with `nprobe`.
   - `irq` — an embedded IVF-RaBitQ index over the centroids. Approximate, and
     the cheapest of the three once the centroid count is large; this is what
     the scripts build by default.
   - `auto` — let the binary choose.
4. **`inner_centroids=PATH`**, **`inner_clusterids=PATH`**: required for
   `irq` only. A second-level k-means over the first-level centroids, produced
   by `clustering/ivf_centroids.py` (`build_index.sh` runs it for you when the
   files are missing).

The coarse quantizer records a fingerprint of the index it was built from and
refuses to load against a different one.

---

## Searching — `bin/querying`

```
bin/querying <base_index> <ssd_index> <coarse_index> <query> <gt> <threads> [key=value...]
```

1. **base_index**: the in-RAM part written by `build_invlist`.
2. **ssd_index**: the on-SSD part from the same run.
3. **coarse_index**: from `build_coarse`.
4. **query**: the queries, `.fbin` or `.fvecs`.
5. **gt**: ground truth. Either ids only, or ids plus distances
   ("type-1": `[int32 nq][int32 k][nq·k uint32 ids][nq·k float32 dists]`).
   With distances, a result counts as correct if it ties with the k-th
   neighbour, which is what makes recall meaningful when the data contains
   duplicate vectors. Must hold at least `topk` neighbours per row. The binary
   prints which form it detected.
6. **threads**: concurrent query threads. More threads can increase throughput
   until a resource saturates; beyond that point, additional concurrency can
   increase per-query latency without improving throughput.

Then any number of `key=value` options, in any order:

| key | default | effect |
|---|---|---|
| `topk=<K>` | 100 | neighbours to retrieve |
| `memdim=<D>` | the full vector width, rounded up to a multiple of 64 | must match what the index was built with |
| `use_hacc=<bool>` | `true` | high-accuracy fastscan for the in-RAM stage |
| `drain_alpha=<a>` | 1.0 | pruning in the drain phase only (see below) |
| `nprobes=v1,v2,...` | unset | time exactly these `nprobe` values instead of searching for operating points |
| `csv=<path>` | unset | write the result CSV ([OUTPUT_SCHEMA.md](../OUTPUT_SCHEMA.md)) |
| `method_csv=<path>` | unset | write this engine's scan-funnel counters; not part of the schema |
| `run_id=<str>` | derived | row identifier in the CSV |
| `system=<str>` | `RaBitQ-SSD` | system label in the CSV |
| `dataset=<name>` | base-index stem | dataset label in the CSV |
| `metric_space=<l2\|ip\|cosine>` | `l2` | recorded in the CSV |

`memdim` and `use_hacc` also accept a bare value: a bare integer is read as
`memdim` (not as `topk`), and a bare `true`/`false` as `use_hacc`.

**`drain_alpha`** changes the pruning threshold during the drain phase only.
While scanning the selected clusters, the search uses the standard top-k
threshold. Once those scans finish, a value below 1.0 uses an alpha-quantile
threshold to prune pending pages more aggressively. This can reduce SSD reads
at the cost of recall. A value of 1.0 retains the standard top-k threshold;
it does not disable ordinary lower-bound pruning. The scripts use 0.4.

**Without `nprobes=`** the binary does not time a fixed list. It first maps
recall against `nprobe`, then selects the operating points that reach a band of
recall targets, and times those. This is why two runs on one machine can differ
in the last decimal: which points get selected depends on a measured recall map.

### Environment variables

These are read by the library rather than parsed as arguments.

| variable | default | effect |
|---|---|---|
| `RABITQ_QD_PER_THREAD` | 16 | SSD reads in flight per thread |
| `RABITQ_IRQ_OUTER_MULT` | 0.5 | coarse-search effort, relative to `nprobe` |
| `RABITQ_IRQ_MODE` | `kfixedmult` | `kfixedmult` or `full` (scan every centroid) |
| `RABITQ_IRQ_REFINE` | `exbits` | `exbits` or `full_precision` |
| `RECALL_TARGET_LO` / `_HI` / `_STEP` | 0.80 / 0.98 / 0.01 | the recall band to find operating points for |
| `RECALL_PLATEAU` | 0.99 | recall above which the map stops climbing `nprobe` |
| `RECALL_HIGH_KEEP` / `RECALL_LOW_KEEP` | 2 / 4 | points kept above the plateau / below the band |

The `RABITQ_IRQ_*` variables apply to the `irq` coarse quantizer only; `flat`
and `hnsw` ignore them, and an unrecognised value for one of them is reported on
stderr before the default is used. The others fall back silently.
