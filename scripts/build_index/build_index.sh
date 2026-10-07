#!/usr/bin/env bash
# build_index.sh <dataset> — build the on-disk index for one dataset.
#
#   ./build_index.sh mydata
#   MEMDIM_ENV=320 ./build_index.sh mydata       # a different RAM/SSD split
#   BUDGET_GB_ENV=256 ./build_index.sh mydata    # raise the build memory ceiling
#   COARSE_ENV=hnsw ./build_index.sh mydata      # comparison coarse quantizer
#   STORE_ENV=raw ./build_index.sh mydata        # SSD holds the vectors themselves
#
# <dataset> names the directory under DATA_ROOT and INDEX_ROOT; its base
# vectors are train.fbin (float32), or train.u8bin or train.bvecs (uint8).
# N_ENV indexes only the first N_ENV of them. Every dataset is
# built the same way (IRQ coarse quantizer); C, the metric, the total bit
# width, the SSD store, the cluster layout and the RAM/SSD split point are all
# derived from the base vectors by scripts/_params.sh. The build phases, the
# variable contract and the remaining env overrides are documented in
# _build_lib.sh.
#
# COARSE_ENV picks the coarse quantizer: irq (default) or the flat / hnsw
# comparison arms. Since split storage keeps the coarse quantizer a separate
# artifact, building flat or hnsw over an existing index skips straight to
# phase D and leaves base+ssd untouched, so all three arms share one index.
# Query them with COARSE_ENV=flat|hnsw scripts/main_exp/run_main_exp.sh.
#
# MEMDIM_ENV selects the split point between the in-RAM prefix and the on-SSD
# remainder; it defaults to the full vector width. All memdims for a dataset
# reuse the SAME first-level k-means and inner k-means, so only build_invlist
# and build_coarse have to be re-run to sweep it.
#
# First-level k-means is NOT run here — produce CEN/CIDS once per dataset with
# clustering/run_kmeans.sh (see clustering/README.md).
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../_params.sh"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <dataset>"
    echo "  <dataset> names the directory under DATA_ROOT / INDEX_ROOT."
    exit 1
fi
DATASET=$1

# Which coarse quantizer phase D builds. flat/hnsw reuse an existing base+ssd
# (split storage keeps the coarse quantizer separate), so building one of them
# over an already-built index only runs phase D.
COARSE_KIND=${COARSE_ENV:-irq}
# The build's DRAM ceiling is a property of the machine, not the dataset.
BUDGET_GB=${BUDGET_GB_ENV:-128}

[[ -n "${DATA_ENV:-}" ]] || : "${DATA_ROOT:?set DATA_ROOT to the directory holding the per-dataset folders, e.g. export DATA_ROOT=/path/to/datasets}"
: "${INDEX_ROOT:?set INDEX_ROOT to the directory holding the per-dataset index trees, e.g. export INDEX_ROOT=/path/to/indexes}"

if [[ -n "${DATA_ENV:-}" ]]; then
    DATA=${DATA_ENV}
else
    DATA=$(resolve_vectors "${DATA_ROOT}/${DATASET}" train) || exit 1
fi
derive_params "${DATA}" || exit 1
# INDEX_ROOT may point at a different device from DATA_ROOT (the indexes are
# large); a dataset keeping data and index in one tree points both at it.
IDXROOT=${INDEX_ROOT}
CL=${CL_ENV:-${IDXROOT}/${DATASET}/clustering}
# METRIC_TAG is _ip for METRIC_ENV=ip and empty for l2, as in the clustering
# file names.
CEN=${CL}/centroids_${C}${METRIC_TAG}.fvecs
# Cluster ids are .ivecs (clustering/run_kmeans.sh) or .cid64 (the DINO GPU
# clustering, scripts/dino_exp/).
CIDS=${CL}/clusterids_${C}${METRIC_TAG}.ivecs
[[ ! -e "${CIDS}" && -e "${CL}/clusterids_${C}${METRIC_TAG}.cid64" ]] && CIDS=${CL}/clusterids_${C}${METRIC_TAG}.cid64

# STORE_TAG is B<bits> for the default ex-code store and raw for the raw store.
# N_TAG is _n<N> when N_ENV indexes a prefix of the file, so two prefixes of
# one file never share an index; otherwise it is empty.
OUTDIR=${IDXROOT}/${DATASET}/idx_C${C}${N_TAG}_cpu_${STORE_TAG}_memdim${MEMDIM}_${METRIC}_${ORDER}
STEM=ss_${DATASET}_C${C}${N_TAG}_cpu_${STORE_TAG}_memdim${MEMDIM}_${METRIC}_${ORDER}

source "${SCRIPT_DIR}/_build_lib.sh"
build_index
