# _build_lib.sh — the build phases shared by the index-build driver.
# Sourced by build_index.sh AFTER its config block; the driver then calls
# build_index. Not executable on its own.
#
# Variable contract (set by the sourcing driver):
#   DATASET  dataset label (log only)
#   C B METRIC MEMDIM  index config (B = total bits, ex_bits = B-1)
#   METRIC_TAG         _ip for METRIC=ip, else empty; ends the clustering and
#                      inner k-means file names
#   STORE              SSD records: exbits (codes, B bits) or raw (the vectors
#                      as given, float32 or uint8; B does not apply)
#   BUDGET_GB          streaming-build RSS budget (mem_budget_gb)
#   DATA               train.fbin, train.u8bin or train.bvecs (the base
#                      vectors), read from fast storage; the streaming reorder
#                      scratch is created NEXT TO it unless
#                      RABITQ_BUILD_SCRATCH_DIR moves it
#   ROWS_OPT           rows=<N> to index only the first N rows of DATA, or empty
#   BIN_SUBDIR         bin, or bin64 for more than 2^32 - 1 points
#   CEN CIDS           first-level k-means outputs (clustering/ tools)
#   OUTDIR STEM        index output dir + file stem
#
# Optional:
#   INNER_SIZE_ENV  first-level centroids per second-level cluster, default
#                   1000. A SIZE, not a count -- the number of inner clusters
#                   follows as round(C / it), the opposite convention from C_ENV.
#                   A non-default size names its own IRQ coarse file (phase D),
#                   so changing it rebuilds the quantizer instead of reusing the
#                   old one; pass the same value to run_main_exp.sh.
#   FASTER_ENV      faster quantization, default false. NOT part of the index
#                   path, so two values of it overwrite one another in the same
#                   INDEX_ROOT; build them into different INDEX_ROOTs.
#
# Phases (each idempotent — skipped when its outputs exist):
#   B  inner (second-level) k-means over the first-level centroids
#      (clustering/ivf_centroids.py) — the IRQ coarse input; skipped for the
#      flat / hnsw quantizers, which need no second level
#   C  build_invlist (streaming, mem_budget_gb) -> <STEM>.base + <STEM>.ssd
#   D  build_coarse <kind> -> <STEM>.coarse_<kind>[_inner<size>] (+ sidecar for irq/hnsw)
#
# COARSE_KIND (irq by default) selects the quantizer built in phase D. Split
# storage keeps it a separate artifact, so building flat or hnsw over an
# existing index reuses base+ssd untouched and phase C is skipped.
#
# First-level k-means is NOT run here — produce CEN/CIDS once per dataset
# with clustering/run_kmeans.sh (see clustering/README.md).

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
INVLIST_BIN=${INVLIST_BIN_ENV:-${REPO}/${BIN_SUBDIR:-bin}/build_invlist}
COARSE_BIN=${COARSE_BIN_ENV:-${REPO}/${BIN_SUBDIR:-bin}/build_coarse}
IVF_CENTROIDS=${REPO}/clustering/ivf_centroids.py
# Python with faiss + numpy (see clustering/README.md).
PY=${PYTHON_ENV:-python3}

# ORDER, B and STORE come from derive_params (they appear in the index name).
FASTER=${FASTER_ENV:-false}
STORE=${STORE:-exbits}
# total_bits is build_invlist's 4th argument. The raw store has no
# extra-precision code, so it gets 1 there and ignores it.
TOTAL_BITS=${B}
[[ ${STORE} == raw ]] && TOTAL_BITS=1
# A size, not a count: ivf_centroids.py derives round(C / INNER) inner clusters.
INNER=${INNER_SIZE_ENV:-1000}

COARSE_KIND=${COARSE_KIND:-irq}
case ${COARSE_KIND} in
    irq)  COARSE_SIDECAR=.irq  ;;
    hnsw) COARSE_SIDECAR=.hnsw ;;
    flat) COARSE_SIDECAR=""    ;;   # flat writes no sidecar
    *) echo "[FATAL] COARSE_KIND must be irq|flat|hnsw, got '${COARSE_KIND}'"; exit 1 ;;
esac

# Only the IRQ quantizer is built from the inner k-means, so only its file
# carries a non-default inner size.
INNER_SUFFIX=""
[[ ${COARSE_KIND} == irq && ${INNER} != 1000 ]] && INNER_SUFFIX=_inner${INNER}

BASE=${OUTDIR}/${STEM}.base
SSD=${OUTDIR}/${STEM}.ssd
COARSE=${OUTDIR}/${STEM}.coarse_${COARSE_KIND}${INNER_SUFFIX}

export OMP_PROC_BIND=close OMP_PLACES=cores

build_index() {
    mkdir -p "${OUTDIR}"

    # ivf_centroids.py is only needed by the IRQ arm's second-level k-means.
    for f in "${INVLIST_BIN}" "${COARSE_BIN}" "${DATA}" "${CEN}" "${CIDS}" \
             $([[ ${COARSE_KIND} == irq ]] && echo "${IVF_CENTROIDS}"); do
        [[ -s "$f" ]] || { echo "[FATAL] missing prerequisite: $f"; exit 1; }
    done

    local STORE_DESC="B=${B} (ex_bits=$((B-1)))"
    [[ ${STORE} == raw ]] && STORE_DESC="store=raw"
    echo "================ build ${DATASET} C=${C} ${STORE_DESC%% *} memdim=${MEMDIM} metric=${METRIC} coarse=${COARSE_KIND} ================"
    echo "  start  : $(date -Is)"
    echo "  outdir : ${OUTDIR}"
    echo "  cfg    : ${STORE_DESC} metric=${METRIC} order=${ORDER} faster=${FASTER} memdim=${MEMDIM} budget=${BUDGET_GB}GiB inner=${INNER}"
    echo "==============================================================================="

    local LOG
    INNER_CEN=$(dirname "${CEN}")/inner_centroids_C${C}_innersize${INNER}${METRIC_TAG}.fvecs
    INNER_CIDS=$(dirname "${CEN}")/inner_clusterids_C${C}_innersize${INNER}${METRIC_TAG}.ivecs

    # ---- Phase B: inner (second-level) k-means for the IRQ coarse ----------
    if [[ ${COARSE_KIND} != irq ]]; then
        echo "[inner-B]  not needed for the ${COARSE_KIND} coarse quantizer -> skip"
    elif [[ -s "${INNER_CEN}" && -s "${INNER_CIDS}" ]]; then
        echo "[inner-B]  reuse (inner_centroids+inner_clusterids exist) -> skip"
    else
        echo "[inner-B]  CPU inner k-means innersize=${INNER} metric=${METRIC} $(date -Is)"
        LOG=${OUTDIR}/inner_kmeans_C${C}${INNER_SUFFIX}.log
        RABITQ_IRQ_INNER_SIZE="${INNER}" /usr/bin/time -v "${PY}" "${IVF_CENTROIDS}" \
            "${CEN}" "${INNER_CEN}" "${INNER_CIDS}" "${METRIC}" mean \
            > "${LOG}" 2>&1 || { echo "[FATAL] inner k-means failed (see ${LOG})"; exit 1; }
        echo "[inner-B]  done $(date -Is)"
    fi

    # ---- Phase C: build_invlist (streaming; the main cost) -----------------
    if [[ -s "${BASE}" && -s "${SSD}" ]]; then
        echo "[invlist-C] reuse (base+ssd exist) -> skip"
    else
        echo "[invlist-C] start $(date -Is)"
        LOG=${OUTDIR}/build_invlist.log
        echo "cmd: ${INVLIST_BIN} ${DATA} ${CEN} ${CIDS} ${TOTAL_BITS} ${BASE} ${SSD} ${METRIC} ${FASTER} ${ORDER} ${MEMDIM} mem_budget_gb=${BUDGET_GB} store=${STORE}${ROWS_OPT:+ ${ROWS_OPT}}" | tee "${LOG}"
        /usr/bin/time -v "${INVLIST_BIN}" \
            "${DATA}" "${CEN}" "${CIDS}" "${TOTAL_BITS}" "${BASE}" "${SSD}" \
            "${METRIC}" "${FASTER}" "${ORDER}" "${MEMDIM}" "mem_budget_gb=${BUDGET_GB}" \
            "store=${STORE}" ${ROWS_OPT:+"${ROWS_OPT}"} \
            >> "${LOG}" 2>&1 || { echo "[FATAL] build_invlist failed (see ${LOG})"; exit 1; }
        echo "[invlist-C] done $(date -Is)  base=$(stat -c %s "${BASE}") ssd=$(stat -c %s "${SSD}")"
    fi

    # ---- Phase D: build_coarse (IRQ is auto IP-aware from the base's metric) --
    # The IRQ quantizer consumes the phase-B inner k-means; flat and hnsw are
    # built from the base's centroids alone.
    local COARSE_ARGS=()
    [[ ${COARSE_KIND} == irq ]] && COARSE_ARGS=(
        "inner_centroids=${INNER_CEN}" "inner_clusterids=${INNER_CIDS}")
    LOG=${OUTDIR}/build_coarse_${COARSE_KIND}${INNER_SUFFIX}_C${C}.log
    if [[ -s "${COARSE}" ]] && { [[ -z "${COARSE_SIDECAR}" ]] || [[ -s "${COARSE}${COARSE_SIDECAR}" ]]; }; then
        echo "[coarse-D]  reuse (coarse_${COARSE_KIND} exists) -> skip"
    else
        echo "[coarse-D]  build ${COARSE_KIND} $(date -Is)"
        /usr/bin/time -v "${COARSE_BIN}" "${BASE}" "${COARSE}" "${COARSE_KIND}" \
            "${COARSE_ARGS[@]}" \
            > "${LOG}" 2>&1 \
            || { echo "[FATAL] build_coarse failed (see ${LOG})"; exit 1; }
        echo "[coarse-D]  done $(date -Is)  coarse=$(stat -c %s "${COARSE}")${COARSE_SIDECAR:+ sidecar=$(stat -c %s "${COARSE}${COARSE_SIDECAR}" 2>/dev/null)}"
    fi

    echo "================ ${DATASET} build finished: $(date -Is) ================"
    echo "  base   : ${BASE}"
    echo "  ssd    : ${SSD}"
    echo "  coarse : ${COARSE}${COARSE_SIDECAR:+ (+ ${COARSE_SIDECAR} sidecar)}"
}
