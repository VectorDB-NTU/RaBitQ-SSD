#!/usr/bin/env bash
# run_dino.sh <size> — the DINO-10B scaling experiment (Section 5.4 and
# Figure 9(b) of the paper) on one prefix of the dataset, end to end.
#
#   scripts/dino_exp/run_dino.sh 1M
#   DINO_SOURCE_ENV=/path/to/dino_vitl_10B scripts/dino_exp/run_dino.sh 10M
#   TS_ENV="64 16" scripts/dino_exp/run_dino.sh 1M
#
# <size> is the number of base vectors. 100K, 1M, 10M, 100M, 1B and 10B are the
# sizes in the paper; every other size the dataset ships a ground truth for
# runs the same way. The dataset directory is dino<size> in lower case.
#
# Each stage is the ordinary pipeline stage run with the paper's DINO settings,
# and each is skipped when its outputs exist:
#   1. data     tools/dino_prepare.py writes $DATA_ROOT/dino<size>/: the base
#               (train.u8bin, or train.bvecs from 5B on), the queries
#               (test.u8bin) and the shipped ground truth for N (gt10)
#   2. cluster  C = N/4096, rounded down. By default on GPU, in this directory:
#               gpu_kmeans.py trains the centroids, gpu_assign.py assigns every
#               base vector, and gpu_inner_kmeans.py clusters the centroids for
#               the IRQ coarse quantizer. DINO_CLUSTER_ENV=cpu runs
#               clustering/run_kmeans.sh instead (up to 2B).
#   3. build    scripts/build_index/build_index.sh with STORE_ENV=raw and
#               MEMDIM_ENV=256: the SSD holds the original uint8 vectors, which
#               are re-ranked exactly, and 256 of the 1024 dimensions of the
#               1-bit code stay in DRAM
#   4. search   scripts/main_exp/run_main_exp.sh at k = 10 (the depth of the
#               shipped ground truth) and T = 64, with the recall sweep capped
#               at 0.96
#   5. report   the operating points the search measured; the first to reach
#               recall 0.90 and 0.95 is marked
#
# DATA_ROOT and INDEX_ROOT are required, as for every stage. PYTHON_ENV names a
# Python with numpy and faiss (with GPU support for the default clustering);
# TORCH_PYTHON_ENV a Python with numpy and PyTorch with CUDA, if not the same
# one. DINO_SOURCE_ENV is where stage 1 reads the dataset: its public URL by
# default, which downloads only the first N base vectors, or a local copy.
# TS_ENV, BUDGET_GB_ENV, RECALL_HI_ENV and RECALL_PLATEAU_ENV pass through to
# the stages and override the defaults here. Above 2^32 - 1 vectors the build
# and search use bin64/, the build with 64-bit point ids (README).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "${HERE}/../.." && pwd)

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <size>    e.g. $0 1M  (100K 1M 10M 100M 1B 10B, or another shipped size)"
    exit 1
fi
: "${DATA_ROOT:?set DATA_ROOT to the directory holding the per-dataset folders, e.g. export DATA_ROOT=/path/to/datasets}"
: "${INDEX_ROOT:?set INDEX_ROOT to the directory holding the per-dataset index trees, e.g. export INDEX_ROOT=/path/to/indexes}"
PY=${PYTHON_ENV:-python3}
TORCH_PY=${TORCH_PYTHON_ENV:-${PY}}
CLUSTER=${DINO_CLUSTER_ENV:-gpu}

# 1M -> N=1000000; the directory name uses the shortest suffix form.
SIZE=$1
case ${SIZE} in
    *[kK]) N=$(( ${SIZE%[kK]} * 1000 )) ;;
    *[mM]) N=$(( ${SIZE%[mM]} * 1000000 )) ;;
    *[bB]) N=$(( ${SIZE%[bB]} * 1000000000 )) ;;
    *[!0-9]*|'') echo "[FATAL] cannot read size '${SIZE}'; use e.g. 1M or 1000000"; exit 1 ;;
    *) N=${SIZE} ;;
esac
if (( N % 1000000000 == 0 )); then TAG=$(( N / 1000000000 ))b
elif (( N % 1000000 == 0 )); then TAG=$(( N / 1000000 ))m
else TAG=$(( N / 1000 ))k
fi
DATASET=dino${TAG}
DDIR=${DATA_ROOT}/${DATASET}
CL=${INDEX_ROOT}/${DATASET}/clustering
case ${CLUSTER} in
    gpu|cpu) ;;
    *) echo "[FATAL] DINO_CLUSTER_ENV must be gpu or cpu, got '${CLUSTER}'"; exit 1 ;;
esac
# A .u8bin header holds at most 2^31 - 1 rows; larger prefixes stay .bvecs.
if (( N > 2147483647 )); then
    TRAIN=${DDIR}/train.bvecs
    if [[ ${CLUSTER} == cpu ]]; then
        echo "[FATAL] DINO_CLUSTER_ENV=cpu supports up to 2B vectors; use the GPU clustering"
        exit 1
    fi
else
    TRAIN=${DDIR}/train.u8bin
fi

# The paper's DINO configuration. C keeps about 4096 points per cluster, four
# times the default, because the in-memory index at 10B had to fit in DRAM.
C=$(( N / 4096 ))
export C_ENV=${C}
export N_ENV=${N}
export MEMDIM_ENV=256
export METRIC_ENV=l2
export STORE_ENV=raw
unset B_ENV   # raw records hold the vectors themselves; B does not apply
export TOPKS_ENV=10
export TS_ENV=${TS_ENV:-64}
export RECALL_HI_ENV=${RECALL_HI_ENV:-0.96}
export RECALL_PLATEAU_ENV=${RECALL_PLATEAU_ENV:-0.96}
# First-level cells of the GPU clustering: round(sqrt(C)), and 3072 at 10B.
NC1=$(awk -v c="${C}" 'BEGIN{printf "%d", int(sqrt(c)+0.5)}')
(( N == 10000000000 )) && NC1=3072

# The build holds about 62 bytes per vector in DRAM (58 with 32-bit ids) and
# needs that within 85% of its budget, plus room to work.
PER_VEC=58
(( N > 4294967295 )) && PER_VEC=62
NEED_GB=$(( N * PER_VEC / 1073741824 * 100 / 85 + 16 ))
export BUDGET_GB_ENV=${BUDGET_GB_ENV:-$(( NEED_GB > 128 ? NEED_GB : 128 ))}
# Before an index is built, check that the machine can hold the build and the
# disk the index: about 1090 bytes per vector, 1024 of them on SSD. The index
# name follows scripts/_params.sh; a prefix of a longer base adds _n<N>.
N_TAG=""
if [[ -e ${TRAIN} ]]; then
    source "${REPO}/scripts/_params.sh"
    derive_params "${TRAIN}" || exit 1
fi
STEM=C${C}${N_TAG}_cpu_raw_memdim256_l2_ordered
mkdir -p "${INDEX_ROOT}/${DATASET}"
if [[ ! -s ${INDEX_ROOT}/${DATASET}/idx_${STEM}/ss_${DATASET}_${STEM}.base ]]; then
    AVAIL_GB=$(awk '/^MemAvailable:/{printf "%d", $2 / 1048576}' /proc/meminfo)
    if (( BUDGET_GB_ENV > AVAIL_GB )); then
        echo "[FATAL] the build needs a ${BUDGET_GB_ENV} GiB memory budget, but ${AVAIL_GB} GiB are available"
        exit 1
    fi
    NEED_DISK_GB=$(( N * 1090 / 1000000000 ))
    FREE_DISK_GB=$(df -P -B1000000000 "${INDEX_ROOT}/${DATASET}" | awk 'NR==2{print $4}')
    if (( NEED_DISK_GB > FREE_DISK_GB )); then
        echo "[FATAL] the index needs about ${NEED_DISK_GB} GB in ${INDEX_ROOT}, which has ${FREE_DISK_GB} GB free"
        exit 1
    fi
fi

echo "=============================================================================="
echo "  DINO-10B prefix ${DATASET}: N=${N} C=${C} memdim=${MEMDIM_ENV} store=raw"
echo "  data: ${DDIR}   indexes: ${INDEX_ROOT}/${DATASET}"
echo "  clustering: ${CLUSTER}   build budget: ${BUDGET_GB_ENV} GiB"
echo "  search: k=${TOPKS_ENV} T={${TS_ENV}}  recall sweep 0.80..${RECALL_HI_ENV}"
echo "=============================================================================="

echo "---- 1/5 data ($(date -Is))"
"${PY}" "${REPO}/tools/dino_prepare.py" "${N}" --out-dir "${DDIR}" \
    ${DINO_SOURCE_ENV:+--source "${DINO_SOURCE_ENV}"}

echo "---- 2/5 cluster ($(date -Is))"
if [[ ${CLUSTER} == cpu ]]; then
    DATA_ENV=${TRAIN} OUTDIR_ENV=${CL} "${REPO}/clustering/run_kmeans.sh" "${DATASET}"
else
    mkdir -p "${CL}"
    TRAINED=${CL}/trained_centroids_${C}.fvecs
    if [[ -s ${CL}/centroids_${C}.fvecs && -s ${CL}/clusterids_${C}.cid64 ]]; then
        echo "[cluster] reuse ${CL}/clusterids_${C}.cid64"
    else
        if [[ -s ${TRAINED} && -s ${TRAINED%.fvecs}.l1meta ]]; then
            echo "[cluster] reuse ${TRAINED}"
        else
            "${PY}" "${HERE}/gpu_kmeans.py" --base "${TRAIN}" --n-base "${N}" \
                --C "${C}" --nc1 "${NC1}" --out-dir "${CL}" \
                2>&1 | tee "${CL}/gpu_kmeans_C${C}.log"
        fi
        "${TORCH_PY}" "${HERE}/gpu_assign.py" --base "${TRAIN}" --n-base "${N}" \
            --centroids "${TRAINED}" --out-dir "${CL}" --probes 8 \
            2>&1 | tee "${CL}/gpu_assign_C${C}.log"
    fi
    INNER_CEN=${CL}/inner_centroids_C${C}_innersize1000.fvecs
    INNER_CIDS=${CL}/inner_clusterids_C${C}_innersize1000.ivecs
    if [[ -s ${INNER_CEN} && -s ${INNER_CIDS} ]]; then
        echo "[cluster] reuse ${INNER_CEN}"
    else
        RABITQ_IRQ_INNER_SIZE=1000 "${PY}" "${HERE}/gpu_inner_kmeans.py" \
            "${CL}/centroids_${C}.fvecs" "${INNER_CEN}" "${INNER_CIDS}" l2 mean \
            2>&1 | tee "${CL}/gpu_inner_kmeans_C${C}.log"
    fi
fi

echo "---- 3/5 build ($(date -Is))"
"${REPO}/scripts/build_index/build_index.sh" "${DATASET}"

echo "---- 4/5 search ($(date -Is))"
SEARCH_LOG=$(mktemp)
trap 'rm -f "${SEARCH_LOG}"' EXIT
"${REPO}/scripts/main_exp/run_main_exp.sh" "${DATASET}" | tee "${SEARCH_LOG}"

echo "---- 5/5 report ($(date -Is))"
MERGED=$(awk '/^  merged: /{print $2; exit}' "${SEARCH_LOG}")
if [[ -z "${MERGED}" || ! -s "${MERGED}" ]]; then
    echo "[FATAL] the search did not report its merged CSV; see the output above"
    exit 1
fi
echo "operating points measured on ${DATASET} (${MERGED}):"
"${PY}" "${REPO}/scripts/main_exp/operating_points.py" "${MERGED}"
