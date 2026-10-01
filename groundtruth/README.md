# Ground truth (exact KNN)

Produces the `gt10 / gt100 / gt1000` files the query stage consumes. Run this
once per dataset, after `train.fbin` and `test.fbin` are in place and before
building the index.

## Format

DiskANN **type-1**: `[int32 nq][int32 k][nq·k uint32 ids][nq·k float32 dists
ascending]`. Generate at a depth `k` well above the largest topk you intend to
measure — 2048 is a good default for topk ≤ 1000.

The trailing distances enable **tie-aware recall**: `querying` counts a result
correct if it ties with the k-th neighbour, which is what makes recall
meaningful on datasets containing duplicate vectors. Ids-only ground truth also
loads, and then scores by plain id intersection.

The three filenames hold identical bytes; they exist so the query stage can
address one file per topk uniformly.

## Generating

L2 datasets, with faiss. `--gpu N` picks a card; `--cpu --threads N` runs
without one and is equally exact, just slower. Use `--cpu` also when faiss was
built for an older CUDA than the card supports — that combination dies inside
`faiss.StandardGpuResources()` with an opaque CUDA allocation error, not with a
message about the version.

```bash
python compute_gt_type1_l2exact.py \
    --base train.fbin --queries test.fbin --out-dir DIR \
    --out-names gt10,gt100,gt1000 --maxk 1024 --rerank-m 2048 --gpu 0
```

The base is streamed in shards of `--shard` rows (default 10,000,000); lower it
only if a shard does not fit in GPU memory.

**Keep `--maxk` below `--rerank-m`.** Retrieval keeps `rerank-m` candidates per
query (hard-capped at 2048) and the float64 re-rank then picks the best `maxk`
of them. With `maxk == 2048` there is no margin: the depth you keep is exactly
the depth retrieval proposed, so the run prints a NOTE saying so. At
`maxk = 1024` the re-rank chooses from twice the depth it emits, and the run
instead reports the margin it had.

Inner-product datasets (torch CUDA rather than faiss):

```bash
python compute_gt_type1_ipexact.py \
    --base train.fbin --queries test.fbin --out-dir DIR \
    --out-names gt10,gt100,gt1000 --maxk 2048
```

Then validate — this replays the exact loading and tie-scanning the query
binary performs:

```bash
python validate_gt_type1.py DIR/gt1000 --topks 10,100,1000 --queries test.fbin
```

Exit 0 means the file is sound. Exit 1 is a hard failure. **Exit 2** means at
least one query's tie group was truncated at the stored depth: the dataset
contains a duplicate group larger than `maxk`, so recall for that query cannot
reach 1.0. `measure_tie_groups.py` reports how large the true groups are.

Note that `--maxk` cannot be raised past 2048 — that is the retrieval cap — so
a duplicate group larger than that is a ceiling to accept, not to regenerate
away. Lower the `topk` you measure instead, or use a query set without such
groups.

## Contents

```
compute_gt_type1_l2exact.py   exact squared-L2 top-k (faiss GPU or CPU BLAS)
compute_gt_type1_ipexact.py   exact inner-product top-k (torch)
validate_gt_type1.py          validator; see the exit codes above
measure_tie_groups.py         true tie-group sizes, for diagnosing exit 2
```

Each takes `--help`.
