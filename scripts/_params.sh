# _params.sh — derive the index parameters from the dataset itself.
# Sourced by the three stage runners; not executable on its own.
#
#   derive_params <train file>   sets N D C MEMDIM METRIC METRIC_TAG B ORDER STORE
#                                STORE_TAG ROWS_OPT N_TAG BIN_SUBDIR
#   resolve_vectors <dir> <stem> prints <dir>/<stem>.fbin, .u8bin or .bvecs
#
# There is no configuration file. A .fbin (float32) and a .u8bin (uint8) both
# begin with [int32 n][int32 d]; a .bvecs (uint8, used beyond 2^31 vectors)
# starts each row with d and has n = size / (4 + d). So the two numbers every
# parameter depends on are already in the data:
#
#   N       base vectors          default all of them; N_ENV indexes only the
#                                 first N_ENV (ROWS_OPT passes rows=N_ENV to
#                                 build_invlist), and such a prefix adds
#                                 _n<N> (N_TAG) to the index and result names.
#                                 Above 2^32 - 1 the binaries come from bin64/
#                                 (BIN_SUBDIR), the build with 64-bit point ids
#                                                                 override: N_ENV
#   C       clusters              default ceil(N / 1024), i.e. ~1024 points per
#                                 cluster                       override: C_ENV
#   MEMDIM  in-RAM prefix dims    default the full vector width, rounded up to
#                                 a multiple of 64. Lower it to trade memory
#                                 for SSD reads               override: MEMDIM_ENV
#   METRIC  l2 | ip               default l2; ip also appends _ip (METRIC_TAG)
#                                 to the clustering file names, so the two
#                                 metrics never share a clustering
#                                                              override: METRIC_ENV
#   B       total bits per dim    default 9 = 1 sign bit + 8 extra-precision
#                                 bits. Fewer bits shrink the index and lower
#                                 recall at a given nprobe        override: B_ENV
#   ORDER   ordered | unordered   default ordered: points within a cluster are
#                                 stored in centroid-distance order
#                                                              override: ORDER_ENV
#   STORE   exbits | raw          what the SSD records hold. exbits (default):
#                                 the rest of the 1-bit code plus (B-1)-bit
#                                 extra-precision codes. raw: the vectors as
#                                 given, float32 or uint8, re-ranked with exact
#                                 distances; B does not apply   override: STORE_ENV
#
# All of them appear in the index directory and file names, which is why they
# are derived here rather than in each driver: the build and the search then
# read the same values from the same rule. B and STORE share one name token,
# STORE_TAG: B9 for the default ex-code store, raw for the raw store.
#
# Every stage derives them the same way from the same file, so the clustering,
# the index and the query sweep agree without anything being written down. An
# override must therefore be repeated across stages -- and if it is not, the
# mismatch shows up immediately as a missing-file error naming the path that
# was expected, because all of them appear in the index filenames.

derive_params() {
    local train=$1
    if [[ ! -s "${train}" ]]; then
        echo "[FATAL] cannot read the base vectors at ${train}" >&2
        echo "        Point DATA_ROOT at the directory holding your dataset folders," >&2
        echo "        or DATA_ENV straight at a train.fbin or train.u8bin." >&2
        return 1
    fi
    if [[ ${train} == *.bvecs ]]; then
        D=$(od -An -tu4 -N4 "${train}" | awk '{print $1}')
        N=$(( $(stat -L -c %s "${train}") / (4 + D) ))
    else
        read -r N D < <(od -An -tu4 -N8 "${train}" | awk '{print $1, $2}')
    fi
    if (( N <= 0 || D <= 0 )); then
        echo "[FATAL] ${train} does not start with a plausible [int32 n][int32 d] header" >&2
        return 1
    fi
    ROWS_OPT=""
    N_TAG=""
    if [[ -n "${N_ENV:-}" ]]; then
        if [[ ! ${N_ENV} =~ ^[0-9]+$ ]] || (( N_ENV < 1 || N_ENV > N )); then
            echo "[FATAL] N_ENV must be an integer in [1, ${N}] (the rows of ${train}), got '${N_ENV}'" >&2
            return 1
        fi
        if (( N_ENV < N )); then
            ROWS_OPT=rows=${N_ENV}
            N_TAG=_n${N_ENV}
        fi
        N=${N_ENV}
    fi
    BIN_SUBDIR=bin
    (( N > 4294967295 )) && BIN_SUBDIR=bin64
    C=${C_ENV:-$(( (N + 1023) / 1024 ))}
    MEMDIM=${MEMDIM_ENV:-$(( (D + 63) / 64 * 64 ))}
    METRIC=${METRIC_ENV:-l2}
    B=${B_ENV:-9}
    ORDER=${ORDER_ENV:-ordered}
    STORE=${STORE_ENV:-exbits}
    case ${STORE} in
        exbits) STORE_TAG=B${B} ;;
        raw)
            if [[ -n "${B_ENV:-}" ]]; then
                echo "[FATAL] B_ENV does not apply to STORE_ENV=raw: raw records hold the vectors themselves" >&2
                return 1
            fi
            STORE_TAG=raw ;;
        *) echo "[FATAL] STORE_ENV must be exbits or raw, got '${STORE}'" >&2; return 1 ;;
    esac
    case ${METRIC} in
        l2|ip) ;;
        *) echo "[FATAL] METRIC_ENV must be l2 or ip, got '${METRIC}'" >&2; return 1 ;;
    esac
    METRIC_TAG=""
    if [[ ${METRIC} != l2 ]]; then
        METRIC_TAG=_${METRIC}
    fi
    case ${ORDER} in
        ordered|unordered) ;;
        *) echo "[FATAL] ORDER_ENV must be ordered or unordered, got '${ORDER}'" >&2; return 1 ;;
    esac
    if [[ ! ${B} =~ ^[0-9]+$ ]] || (( B < 2 || B > 9 )); then
        echo "[FATAL] B_ENV must be an integer in [2,9] (1 sign bit + B-1 extra), got '${B}'" >&2
        return 1
    fi
    if (( MEMDIM % 64 != 0 )); then
        echo "[FATAL] MEMDIM_ENV must be a multiple of 64, got ${MEMDIM}" >&2
        return 1
    fi
}

# Vectors are float32 <stem>.fbin, or uint8 <stem>.u8bin or <stem>.bvecs;
# exactly one must exist in <dir>. Prints its path.
resolve_vectors() {
    local dir=$1 stem=$2 ext
    local found=()
    for ext in fbin u8bin bvecs; do
        [[ -e "${dir}/${stem}.${ext}" ]] && found+=("${dir}/${stem}.${ext}")
    done
    if (( ${#found[@]} == 1 )); then
        echo "${found[0]}"
        return 0
    fi
    if (( ${#found[@]} == 0 )); then
        echo "[FATAL] found none of ${stem}.fbin, ${stem}.u8bin, ${stem}.bvecs in ${dir}" >&2
    else
        echo "[FATAL] ${dir} holds more than one of ${stem}.fbin, ${stem}.u8bin, ${stem}.bvecs; keep only one" >&2
    fi
    return 1
}
