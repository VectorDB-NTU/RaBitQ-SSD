#!/usr/bin/env bash
# run_kmeans.sh <dataset> — two-level k-means over the base vectors, on CPU.
#
#   DATA_ENV=/path/to/datasets/<dataset>/train.fbin \
#   OUTDIR_ENV=/path/to/indexes/<dataset>/clustering ./run_kmeans.sh <dataset>
#
# DATA_ENV is the base: train.fbin (float32) or train.u8bin (uint8). Every row
# of it is clustered, so N_ENV, which indexes only a prefix of a longer file,
# is refused here.
#
#   reservoir subsample -> two-level cluster -> HNSW approx assign (efS=64)
#   -> per-cluster-mean centroids + clusterids, i.e.
#   centroids_<C>.fvecs + clusterids_<C>.ivecs in OUTDIR_ENV; with
#   METRIC_ENV=ip every output name ends in _ip instead, so an l2 and an ip
#   clustering of one dataset can share OUTDIR_ENV.
#
# CPU only: CUDA_VISIBLE_DEVICES="" hides any GPU, so no GPU kernel runs even on
# a CUDA build of faiss.
#
# Every dataset uses the same knobs (240 pts/centroid, HNSW efS=64,
# 25 coarse iterations). C is derived from the base vectors and the
# fan-out follows as round(sqrt(C)); see scripts/_params.sh for the defaults
# and their overrides.
#
# METRIC_ENV must match how the vectors are meant to be compared, and must be
# the same here as at build time: vectors whose norms vary (so that IP-nearest
# differs from L2-nearest) have to be clustered with ip end to end, which
# selects spherical k-means and IP routing.
#
# DATA_ENV and OUTDIR_ENV are per-run paths: the train file and the index tree
# routinely live on different volumes. PYTHON_ENV picks the interpreter (any
# python with faiss + numpy).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/../scripts/_params.sh"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <dataset>"
    echo "  <dataset> names the directory under DATA_ROOT / INDEX_ROOT."
    exit 1
fi
DATASET=$1

DRIVER=$HERE/two_level_kmeans.py
PY=${PYTHON_ENV:-python3}   # any python with faiss + numpy

# Locked knobs shared by every dataset: pts_per_centroid 240 sets n_train
# (240*C sampled rows), HNSW efS=64 is the approx-assign search width, and the
# coarse k-means runs 25 iterations.
PPC=240; EFS=64; NITER=25

: "${DATA_ENV:?set DATA_ENV to the ${DATASET} train.fbin or train.u8bin, e.g. export DATA_ENV=/path/to/datasets/${DATASET}/train.fbin}"
: "${OUTDIR_ENV:?set OUTDIR_ENV to the directory that receives the coarse-quantizer outputs, e.g. export OUTDIR_ENV=/path/to/indexes/${DATASET}/clustering}"
DATA=${DATA_ENV}
OUTDIR=${OUTDIR_ENV}
mkdir -p "$OUTDIR"

derive_params "${DATA}" || exit 1
case ${DATA} in
    *.bvecs|*.fvecs)
        echo "[FATAL] ${DATA}: this clustering reads .fbin or .u8bin files" \
             "(convert .fvecs with tools/fvecs_to_fbin.py)"
        exit 1 ;;
esac
if [[ -n ${ROWS_OPT} ]]; then
    echo "[FATAL] N_ENV=${N_ENV} indexes the first ${N} rows of ${DATA}, but this" \
         "clustering reads the whole file; give it a file holding exactly those rows"
    exit 1
fi
NC1=$(awk -v c="$C" 'BEGIN{printf "%d", int(sqrt(c)+0.5)}')

# Every output names the metric when it is not the default L2 (METRIC_TAG), so
# an ip run never reuses or overwrites an l2 clustering in the same directory.
LOG=$OUTDIR/kmeans_C${C}${METRIC_TAG}.log
TIMEV=$OUTDIR/time_v_kmeans${METRIC_TAG}.txt

[[ -s "$DATA" ]]   || { echo "[FATAL] missing train: $DATA"; exit 1; }
[[ -s "$DRIVER" ]] || { echo "[FATAL] missing driver: $DRIVER"; exit 1; }

if [[ -s "$OUTDIR/centroids_${C}${METRIC_TAG}.fvecs" && -s "$OUTDIR/clusterids_${C}${METRIC_TAG}.ivecs" ]]; then
    echo "[kmeans] reuse existing C=${C} (centroids+clusterids exist) -> skip"; exit 0
fi

echo "=== ${DATASET} CPU two-level k-means (N=$N d=$D C=$C nc1=$NC1 metric=$METRIC) ===" | tee "$LOG"
echo "  base=$DATA  out=$OUTDIR" | tee -a "$LOG"
echo "started: $(date -Is)" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="" OMP_PROC_BIND=close OMP_PLACES=cores \
/usr/bin/time -v -o "$TIMEV" \
  "$PY" "$DRIVER" --line cpu \
    --base "$DATA" --out-dir "$OUTDIR" --dataset "$DATASET" \
    --C "$C" --nc1 "$NC1" --pts-per-centroid "$PPC" \
    --metric "$METRIC" --hnsw-efs "$EFS" --niter "$NITER" --name-suffix "$METRIC_TAG" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

echo "finished: $(date -Is)  exit_code=$RC" | tee -a "$LOG"
echo "=== /usr/bin/time -v (wall + peak RSS) ===" | tee -a "$LOG"; cat "$TIMEV" | tee -a "$LOG"
echo "=== outputs ===" | tee -a "$LOG"; ls -la "$OUTDIR" | tee -a "$LOG"
exit $RC
