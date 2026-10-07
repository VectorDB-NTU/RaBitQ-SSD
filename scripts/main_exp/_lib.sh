# _lib.sh — the grid mechanics shared by the search drivers.
# Sourced by run_main_exp.sh AFTER its config block; the driver then calls
# run_grid. Not executable on its own.
#
# Variable contract (set by the sourcing driver):
#   DATASET   dataset label written to the CSV (e.g. yfcc100m)
#   METRIC    metric_space CSV label (l2|ip). The REAL metric is baked into
#             the .base at build time; this is only the label.
#   MEMDIM    in-RAM prefix dims for this run
#   CFG_TAG   per-run directory / run_id prefix
#   IDXDIR    directory holding <TAG>.base/.ssd/.coarse_<kind>[_inner<size>]
#   TAG       index file stem
#   DDIR      dataset dir holding gt<topk>
#   QUERY_FILE  the queries: DDIR/test.fbin or DDIR/test.u8bin
#   STORE     the SSD store the index was built with (exbits | raw)
#   COARSE_KIND  irq (default) | flat | hnsw -- which coarse quantizer to load
#   INNER_SUFFIX _inner<size> for a non-default IRQ inner size, else empty
#
# Env overrides accepted by every driver:
#   MEMDIM_ENV IDX_DIR_ENV IDX_TAG_ENV OUT_TAG_SUFFIX  index/point selection
#   STORE_ENV         exbits (default) | raw -- the SSD store the index holds
#   COARSE_ENV        irq (default) | flat | hnsw -- the coarse-quantizer arm
#   INNER_SIZE_ENV    the IRQ inner size the index was built with, if not 1000
#   RECALL_HI_ENV RECALL_PLATEAU_ENV   cap the swept recall band. Needed when
#                     the ground truth cannot reach the default plateau (a
#                     dataset whose duplicate groups were truncated at the GT
#                     depth) -- without it the search walks to nprobe = C.
#   TOPKS_ENV TS_ENV                                   grid ("10 100" / "64 1")
#   NPROBES_ENV       comma list: run these fixed operating points and SKIP the
#                     adaptive recall map (e.g. rerun one thread count at the
#                     operating points another run recorded in op_param)
#   QUERY_BIN_ENV     override the querying binary (default bin/querying, or
#                     bin64/querying above 2^32 - 1 points)
#   METHOD_EXT_ENV=1  also write this engine's internal scan-funnel counters
#                     (method_ext CSV); off by default -- research instrument,
#                     not part of the result schema
#
# Mechanics shared by every dataset: explicit per-thread queue depth, a
# /usr/bin/time -v wrapper filling the 4 driver-owned CSV columns, a [qd]
# guard that deletes the cell on mismatch, an idempotent skip of cells whose
# CSV is already filled, and header-once assembly of the merged CSVs.

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
QUERY_BIN=${QUERY_BIN_ENV:-${REPO}/${BIN_SUBDIR:-bin}/querying}

COARSE_KIND=${COARSE_KIND:-irq}
case ${COARSE_KIND} in
    irq)  COARSE_SIDECAR=.irq  ;;
    hnsw) COARSE_SIDECAR=.hnsw ;;
    flat) COARSE_SIDECAR=""    ;;   # flat writes no sidecar
    *) echo "[FATAL] COARSE_KIND must be irq|flat|hnsw, got '${COARSE_KIND}'"; exit 1 ;;
esac

BASE=${IDXDIR}/${TAG}.base
SSD=${IDXDIR}/${TAG}.ssd
COARSE=${IDXDIR}/${TAG}.coarse_${COARSE_KIND}${INNER_SUFFIX:-}
QUERY=${QUERY_FILE:-${DDIR}/test.fbin}
STORE=${STORE:-exbits}

SYSTEM=RaBitQ-SSD
USEHACC=true
DRAIN_ALPHA=${DRAIN_ALPHA:-0.4}
# IRQ coarse config; equal to the compiled defaults. Passed explicitly so each
# run records the coarse configuration it used regardless of binary defaults.
# Only meaningful for the IRQ arm; flat/hnsw ignore these env vars.
IRQ_MODE=kfixedmult
OUTER_MULT=${RABITQ_IRQ_OUTER_MULT:-0.5}

# Adaptive recall-targeted nprobe selection. RECALL_HI_ENV / RECALL_PLATEAU_ENV
# override the band for a dataset whose ground truth cannot reach the default.
#
# The flat and hnsw arms scan the centroid set far more expensively than IRQ,
# so their high-nprobe tail costs hours per cell. They stop the recall map at
# the first point above a lower plateau, which truncates only the top of the
# swept band -- every recorded recall value is still measured, not estimated.
RECALL_TARGET_LO=0.80
RECALL_TARGET_STEP=0.01
RECALL_LOW_KEEP=4
if [[ ${COARSE_KIND} == irq ]]; then
    RECALL_TARGET_HI=${RECALL_HI_ENV:-0.98}
    RECALL_PLATEAU=${RECALL_PLATEAU_ENV:-0.99}
    RECALL_HIGH_KEEP=2
else
    RECALL_TARGET_HI=${RECALL_HI_ENV:-0.96}
    RECALL_PLATEAU=${RECALL_PLATEAU_ENV:-0.97}
    RECALL_HIGH_KEEP=1
fi

# Per-thread queue depth. A deeper queue keeps more reads in flight per thread,
# which helps when few threads must saturate the device; at high thread counts
# the aggregate depth is already past what the SSD rewards, so it drops.
# RABITQ_QD_PER_THREAD overrides the policy for every thread count.
qd_for_T() {
    if [[ -n "${RABITQ_QD_PER_THREAD:-}" ]]; then echo "${RABITQ_QD_PER_THREAD}"; return; fi
    if (( $1 <= 16 )); then echo 16; else echo 8; fi
}

OUTDIR=${REPO}/results/main_exp/${CFG_TAG}${OUT_TAG_SUFFIX:-}
# The merge joins the cells of ONE configuration -- every topk x threads cell
# that shares an index (B, order, memdim, metric), a coarse quantizer with its
# search config, and drain_alpha. So it is keyed by exactly the same tag as the
# per-cell directory: assemble() truncates its output, so a key that dropped any
# of those would let a second arm silently erase the first one's merged CSV.
RESDIR=${REPO}/results/main_exp/merged/${CFG_TAG}${OUT_TAG_SUFFIX:-}

# The IRQ arm sweeps the full grid. The flat/hnsw comparison arms sweep the
# corners only: their coarse step dominates the query, so a single-thread cell
# can take hours.
if [[ ${COARSE_KIND} == irq ]]; then
    IFS=' ' read -ra TOPKS <<< "${TOPKS_ENV:-10 100 1000}"
    IFS=' ' read -ra TS    <<< "${TS_ENV:-64 32 16 8 4 1}"   # large -> small
else
    IFS=' ' read -ra TOPKS <<< "${TOPKS_ENV:-10 100}"
    IFS=' ' read -ra TS    <<< "${TS_ENV:-64 1}"             # fast cell first
fi

export OMP_PROC_BIND=close OMP_PLACES=cores

assemble() {  # $1=out  $2..=inputs (header once)
    local out=$1; shift; local first=1; : > "$out"
    for f in "$@"; do
        [[ -s "$f" ]] || continue
        if [[ $first -eq 1 ]]; then cat "$f" > "$out"; first=0; else tail -n +2 "$f" >> "$out"; fi
    done
}

run_grid() {
    mkdir -p "${OUTDIR}/csv" "${OUTDIR}/method_ext" "${OUTDIR}/logs" "${RESDIR}"

    # Every ground truth the grid will need is checked up front: at 100M scale a
    # cell is hours, and finding the missing gt only when the third cell starts
    # throws that away.
    local _gt
    for f in "${QUERY_BIN}" "${BASE}" "${SSD}" "${COARSE}" \
             ${COARSE_SIDECAR:+"${COARSE}${COARSE_SIDECAR}"} "${QUERY}" \
             $(for _gt in "${TOPKS[@]}"; do echo "${DDIR}/gt${_gt}"; done); do
        [[ -s "$f" ]] || { echo "[FATAL] missing prerequisite: $f"; exit 1; }
    done
    # The base index is resident for the whole run, so refuse to start without
    # room for it plus a working set. The margin scales with the index rather
    # than being a flat constant, so a small index stays runnable on a small
    # machine; MEM_HEADROOM_MB_ENV overrides the floor.
    local AVAIL_MB BASE_MB NEED_MB
    AVAIL_MB=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
    BASE_MB=$(( $(stat -c %s "${BASE}") / 1048576 ))
    NEED_MB=$(( BASE_MB + BASE_MB / 4 + ${MEM_HEADROOM_MB_ENV:-1024} ))
    if (( AVAIL_MB < NEED_MB )); then
        echo "[FATAL] only ${AVAIL_MB} MiB available; the base index is ${BASE_MB} MiB and"
        echo "        this run needs about ${NEED_MB} MiB. Free memory, or raise/lower"
        echo "        MEM_HEADROOM_MB_ENV (default 1024) if you know the working set."
        exit 1
    fi

    local INDEX_SSD_BYTES RESIDENT_INDEX_MB
    INDEX_SSD_BYTES=$(stat -c %s "${SSD}")
    RESIDENT_INDEX_MB=$(awk "BEGIN{printf \"%.3f\", $(stat -c %s "${BASE}")/1048576}")

    # The IO backend is a compile-time choice; the binary reports the real one
    # in its own log, so the banner does not guess.
    echo "================ ${CFG_TAG}${OUT_TAG_SUFFIX:-} ================"
    echo "  start  : $(date -Is)"
    echo "  index  : ${IDXDIR}  (SSD store: ${STORE})"
    echo "  grid   : topk={${TOPKS[*]}} T={${TS[*]}}"
    if [[ ${COARSE_KIND} == irq ]]; then
        echo "  coarse : irq ${IRQ_MODE} mult=${OUTER_MULT} | drain_alpha=${DRAIN_ALPHA} | metric_space=${METRIC}"
    else
        echo "  coarse : ${COARSE_KIND} | drain_alpha=${DRAIN_ALPHA} | metric_space=${METRIC}"
    fi
    if [[ -n "${NPROBES_ENV:-}" ]]; then
        echo "  nprobe : REUSED ${NPROBES_ENV} (recall-map SKIPPED)"
    else
        echo "  nprobe : adaptive recall [${RECALL_TARGET_LO},hi] step ${RECALL_TARGET_STEP} (hi/plateau per topk)"
    fi
    echo "  out    : ${OUTDIR}   merged -> ${RESDIR}"
    echo "=================================================================="

    local n_done=0 n_skip=0 n_fail=0
    local T QD TOPK RUN RUNID CSV MCSV LOG TV gt QDLINE CPU RSSKB PEAK_RSS_MB
    for T in "${TS[@]}"; do
        QD=$(qd_for_T "$T")
        for TOPK in "${TOPKS[@]}"; do
            RUN=topk${TOPK}_T${T}
            RUNID=${CFG_TAG}${OUT_TAG_SUFFIX:-}_${RUN}
            CSV=${OUTDIR}/csv/${RUN}.csv
            # method_ext holds this engine's internal scan-funnel counters. It
            # is a research instrument, not part of the result schema, so it is
            # off unless METHOD_EXT_ENV=1 asks for it.
            MCSV=""
            [[ -n "${METHOD_EXT_ENV:-}" ]] && MCSV=${OUTDIR}/method_ext/${RUN}.csv
            LOG=${OUTDIR}/logs/${RUN}.log
            TV=${OUTDIR}/logs/${RUN}.timev
            gt=${DDIR}/gt${TOPK}
            [[ -s "$gt" ]] || { echo "[FATAL] missing gt: $gt"; exit 1; }

            if [[ -s "${CSV}" ]]; then
                echo "[skip] ${RUN} (filled csv exists)"; n_skip=$((n_skip+1)); continue
            fi

            echo "[run ] ${RUN} qd=${QD}  $(date -Is)"
            RABITQ_QD_PER_THREAD=${QD} \
            RABITQ_IRQ_MODE=${IRQ_MODE} RABITQ_IRQ_OUTER_MULT=${OUTER_MULT} \
            RECALL_TARGET_LO=${RECALL_TARGET_LO} RECALL_TARGET_HI=${RECALL_TARGET_HI} \
            RECALL_TARGET_STEP=${RECALL_TARGET_STEP} RECALL_PLATEAU=${RECALL_PLATEAU} \
            RECALL_HIGH_KEEP=${RECALL_HIGH_KEEP} RECALL_LOW_KEEP=${RECALL_LOW_KEEP} \
            /usr/bin/time -v -o "${TV}" "${QUERY_BIN}" \
                "${BASE}" "${SSD}" "${COARSE}" "${QUERY}" "${gt}" "${T}" \
                "memdim=${MEMDIM}" "topk=${TOPK}" "use_hacc=${USEHACC}" \
                "drain_alpha=${DRAIN_ALPHA}" \
                "nprobes=${NPROBES_ENV:-}" \
                "system=${SYSTEM}" "dataset=${DATASET}" "metric_space=${METRIC}" \
                "run_id=${RUNID}" \
                "csv=${CSV}.raw" ${MCSV:+"method_csv=${MCSV}"} \
                > "${LOG}" 2>&1 \
                || { echo "[FAIL] ${RUN} (see ${LOG})"; rm -f "${CSV}.raw"; n_fail=$((n_fail+1)); continue; }

            # ---- guard: the binary must have loaded the coarse quantizer this
            # arm is about. The kind comes from the <coarse> file header, so a
            # mistyped path would otherwise silently compare the wrong arm. ----
            if ! grep -qi "Coarse quantizer kind: ${COARSE_KIND}" "${LOG}"; then
                echo "[FAIL] ${RUN}: expected coarse kind '${COARSE_KIND}', log says:" \
                     "$(grep -i -m1 'Coarse quantizer kind' "${LOG}" || echo '<no kind line>')"
                rm -f "${CSV}.raw" "${MCSV}"; n_fail=$((n_fail+1)); continue
            fi

            # ---- guard: the index must hold the SSD store this run is about.
            # The store is read from <base>, so a mistyped index path would
            # otherwise silently benchmark the wrong kind of index. ----
            if ! grep -q "SSD store: ${STORE}" "${LOG}"; then
                echo "[FAIL] ${RUN}: expected SSD store '${STORE}', log says:" \
                     "$(grep -m1 'SSD store:' "${LOG}" || echo '<no store line>')"
                rm -f "${CSV}.raw" "${MCSV}"; n_fail=$((n_fail+1)); continue
            fi

            # ---- guard: the qd the binary reports must match the requested one ----
            QDLINE=$(grep -m1 '^\[qd\]' "${LOG}" || true)
            if ! grep -q "per_thread_req=${QD} nthreads=${T} qd_cap=${QD}" <<< "${QDLINE}"; then
                echo "[FAIL] ${RUN}: qd guard failed -> '${QDLINE}'"
                rm -f "${CSV}.raw" "${MCSV}"; n_fail=$((n_fail+1)); continue
            fi

            # An adaptive map that never clears its plateau keeps climbing the
            # nprobe ladder toward nprobe = C, i.e. a brute-force scan. Report it
            # before the remaining cells are paid for; RECALL_PLATEAU_ENV is the
            # knob to lower when the ground truth cannot reach the default.
            local MAXNP
            # Column 12 is op_param (the nprobe); 13 is recall_at_topk.
            MAXNP=$(awk -F, 'NR>1 && $12+0>m {m=$12+0} END{printf "%d", m}' "${CSV}.raw" 2>/dev/null)
            if [[ -n "${MAXNP}" ]] && (( MAXNP > C / 2 )); then
                echo "[WARN] ${RUN}: recall map reached nprobe=${MAXNP}, over half of C=${C}; the plateau was"
                echo "       probably never cleared. Check the [map] lines in ${LOG} and consider"
                echo "       lowering RECALL_PLATEAU_ENV before the remaining cells."
            fi

            # ---- fill the 4 driver-owned CSV columns from time -v + stat ----
            CPU=$(awk -F': ' '/Percent of CPU/{gsub(/%/,"",$2); print $2}' "${TV}")
            RSSKB=$(awk -F': ' '/Maximum resident set size/{print $2}' "${TV}")
            PEAK_RSS_MB=$(awk "BEGIN{printf \"%.3f\", ${RSSKB:-0}/1024}")
            awk -v c="${CPU}" -v r="${PEAK_RSS_MB}" -v ri="${RESIDENT_INDEX_MB}" -v sb="${INDEX_SSD_BYTES}" \
                'NR==1{print;next}{sub(/,,,,$/, ","c","r","ri","sb); print}' \
                "${CSV}.raw" > "${CSV}" && rm -f "${CSV}.raw"

            echo "[ok  ] ${RUN}  $(date -Is)  (cpu%=${CPU} peak_rss_mb=${PEAK_RSS_MB})"
            n_done=$((n_done+1))
        done
    done

    # ---- assemble the merged CSVs (regenerated each run; header once) ----
    assemble "${RESDIR}/query_results.csv" "${OUTDIR}"/csv/*.csv
    [[ -n "${METHOD_EXT_ENV:-}" ]] && assemble "${RESDIR}/method_ext.csv" "${OUTDIR}"/method_ext/*.csv

    echo "=================================================================="
    echo "  grid: ${n_done} ran, ${n_skip} skipped, ${n_fail} failed"
    echo "  merged: ${RESDIR}/query_results.csv${METHOD_EXT_ENV:+  ${RESDIR}/method_ext.csv}"
    echo "  finished: $(date -Is)"
    echo "=================================================================="
    [[ ${n_fail} -eq 0 ]]
}
