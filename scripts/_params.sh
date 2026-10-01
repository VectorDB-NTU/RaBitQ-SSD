# _params.sh — derive the index parameters from the dataset itself.
# Sourced by the three stage runners; not executable on its own.
#
#   derive_params <train.fbin>   sets N D C MEMDIM METRIC B ORDER
#
# There is no configuration file. A .fbin begins with [int32 n][int32 d], so
# the two numbers every parameter depends on are already in the data:
#
#   C       clusters              default ceil(N / 1024), i.e. ~1024 points per
#                                 cluster                       override: C_ENV
#   MEMDIM  in-RAM prefix dims    default the full vector width, rounded up to
#                                 a multiple of 64. Lower it to trade memory
#                                 for SSD reads               override: MEMDIM_ENV
#   METRIC  l2 | ip               default l2                  override: METRIC_ENV
#   B       total bits per dim    default 9 = 1 sign bit + 8 extra-precision
#                                 bits. Fewer bits shrink the index and lower
#                                 recall at a given nprobe        override: B_ENV
#   ORDER   ordered | unordered   default ordered: points within a cluster are
#                                 stored in centroid-distance order
#                                                              override: ORDER_ENV
#
# All five appear in the index directory and file names, which is why they are
# derived here rather than in each driver: the build and the search then read
# the same values from the same rule.
#
# Every stage derives them the same way from the same file, so the clustering,
# the index and the query sweep agree without anything being written down. An
# override must therefore be repeated across stages -- and if it is not, the
# mismatch shows up immediately as a missing-file error naming the path that
# was expected, because all five appear in the index filenames.

derive_params() {
    local train=$1
    if [[ ! -s "${train}" ]]; then
        echo "[FATAL] cannot read the base vectors at ${train}" >&2
        echo "        Point DATA_ROOT at the directory holding your dataset folders," >&2
        echo "        or DATA_ENV straight at a train.fbin." >&2
        return 1
    fi
    read -r N D < <(od -An -tu4 -N8 "${train}" | awk '{print $1, $2}')
    if (( N <= 0 || D <= 0 )); then
        echo "[FATAL] ${train} does not start with a plausible [int32 n][int32 d] header" >&2
        return 1
    fi
    C=${C_ENV:-$(( (N + 1023) / 1024 ))}
    MEMDIM=${MEMDIM_ENV:-$(( (D + 63) / 64 * 64 ))}
    METRIC=${METRIC_ENV:-l2}
    B=${B_ENV:-9}
    ORDER=${ORDER_ENV:-ordered}
    case ${METRIC} in
        l2|ip) ;;
        *) echo "[FATAL] METRIC_ENV must be l2 or ip, got '${METRIC}'" >&2; return 1 ;;
    esac
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
