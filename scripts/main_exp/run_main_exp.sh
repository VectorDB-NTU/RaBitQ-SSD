#!/usr/bin/env bash
# run_main_exp.sh <dataset> — sweep recall against throughput for one dataset.
#
#   ./run_main_exp.sh yfcc1m
#   MEMDIM_ENV=320 ./run_main_exp.sh yfcc1m         # a different RAM/SSD split
#   TOPKS_ENV=10 TS_ENV=64 ./run_main_exp.sh yfcc1m # one cell only
#   COARSE_ENV=hnsw ./run_main_exp.sh yfcc1m        # another coarse quantizer
#   STORE_ENV=raw ./run_main_exp.sh yfcc1m          # an index built with STORE_ENV=raw
#
# <dataset> names the directory under DATA_ROOT and INDEX_ROOT. Every dataset
# uses the same configuration (IRQ coarse, kfixedmult 0.5, exbits refine,
# batched scan, drain_alpha 0.4, adaptive recall-targeted nprobe, an explicit
# per-thread queue depth); C, the metric, the SSD store and the RAM/SSD split
# point are derived from the base vectors by scripts/_params.sh. The base and
# queries are train/test.fbin (float32) or train/test.u8bin (uint8). The grid
# mechanics and the full list of env overrides are documented in _lib.sh.
#
# COARSE_ENV selects which coarse quantizer to query: irq (default) or the
# flat / hnsw alternatives. base and ssd are shared --
# split storage keeps the coarse quantizer a separate artifact -- so all three
# arms read the exact same index and the coarse quantizer is the only variable.
# Build the alternative quantizers with COARSE_ENV=flat|hnsw build_index.sh.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../_params.sh"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <dataset>"
    echo "  <dataset> names the directory under DATA_ROOT / INDEX_ROOT."
    exit 1
fi
DATASET=$1

COARSE_KIND=${COARSE_ENV:-irq}

: "${DATA_ROOT:?set DATA_ROOT to the directory holding the per-dataset folders, e.g. export DATA_ROOT=/path/to/datasets}"
[[ -n "${IDX_DIR_ENV:-}" ]] || : "${INDEX_ROOT:?set INDEX_ROOT to the directory holding the per-dataset index trees, e.g. export INDEX_ROOT=/path/to/indexes}"

DDIR=${DDIR_ENV:-${DATA_ROOT}/${DATASET}}
# The same derivation the build used, from the same file, so the two agree.
TRAIN_FILE=$(resolve_vectors "${DDIR}" train) || exit 1
QUERY_FILE=$(resolve_vectors "${DDIR}" test) || exit 1
derive_params "${TRAIN_FILE}" || exit 1
IDXROOT=${INDEX_ROOT:-}
IDX_NAME=idx_C${C}${N_TAG}_cpu_${STORE_TAG}_memdim${MEMDIM}_${METRIC}_${ORDER}
IDXDIR=${IDX_DIR_ENV:-${IDXROOT}/${DATASET}/${IDX_NAME}}
TAG=${IDX_TAG_ENV:-ss_${DATASET}_C${C}${N_TAG}_cpu_${STORE_TAG}_memdim${MEMDIM}_${METRIC}_${ORDER}}

# Keep the metric out of the tag when it is the default L2, so an l2 run and an
# ip run over the same dataset land in separate result trees. B, the SSD store
# and the cluster order follow the same rule: each selects a different index,
# so a non-default value must not reuse the cells of the default run.
CFG_METRIC=""
[[ ${METRIC} != l2 ]] && CFG_METRIC=_${METRIC}
CFG_B=""
if [[ ${STORE} == raw ]]; then
    CFG_B=_raw
elif [[ ${B} != 9 ]]; then
    CFG_B=_B${B}
fi
CFG_ORDER=""
[[ ${ORDER} != ordered ]] && CFG_ORDER=_${ORDER}
# A non-default IRQ inner size names its own coarse file at build time
# (_build_lib.sh); _lib.sh loads the file by the same suffix.
INNER_SUFFIX=""
[[ ${COARSE_KIND} == irq && ${INNER_SIZE_ENV:-1000} != 1000 ]] && INNER_SUFFIX=_inner${INNER_SIZE_ENV}
# The IRQ arm is tagged by its search config (the outer multiplier _lib.sh
# passes as RABITQ_IRQ_OUTER_MULT) and its inner size; the comparison arms by
# their coarse quantizer, so the three never share a result tree.
if [[ ${COARSE_KIND} == irq ]]; then
    CFG_COARSE=kfm${RABITQ_IRQ_OUTER_MULT:-0.5}${INNER_SUFFIX}
else
    CFG_COARSE=coarse${COARSE_KIND^^}
fi
# Set here rather than in _lib.sh because the tag below names it.
DRAIN_ALPHA=${DRAIN_ALPHA_ENV:-0.4}
CFG_TAG=${DATASET}_C${C}${N_TAG}${CFG_B}_memdim${MEMDIM}${CFG_METRIC}${CFG_ORDER}_${CFG_COARSE}_drainA${DRAIN_ALPHA}

source "${SCRIPT_DIR}/_lib.sh"
run_grid
