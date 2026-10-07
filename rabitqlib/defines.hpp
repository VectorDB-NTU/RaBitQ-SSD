#pragma once

#include <cstdint>
#include <limits>

#include "rabitqlib/third/Eigen/Dense"

#define BIT_ID(x) (__builtin_popcount((x) - 1))
#define LOWBIT(x) ((x) & (-(x)))

namespace rabitqlib {

// Integer ids by role. A point id indexes the base vectors; it is 32-bit unless
// the build is configured with RABITQ_PID64=ON, which widens it to 64 bits for
// datasets with more than 2^32 - 1 points. Cluster ids and per-query page-slot
// ids are 32-bit in every build, so the coarse-quantizer files are the same in
// both builds.
#if defined(RABITQ_PID64)
using PID = uint64_t;
#else
using PID = uint32_t;
#endif
using CID = uint32_t;
using SlotID = uint32_t;

constexpr PID kPidMax = std::numeric_limits<PID>::max();
constexpr SlotID kSlotMax = std::numeric_limits<SlotID>::max();

template <typename T>
using RowMajorMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using RowMajorMatrixMap = Eigen::Map<RowMajorMatrix<T>>;

template <typename T>
using ConstRowMajorMatrixMap = Eigen::Map<const RowMajorMatrix<T>>;

template <typename T>
using RowMajorArray = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using RowMajorArrayMap = Eigen::Map<RowMajorArray<T>>;

template <typename T>
using ConstRowMajorArrayMap = Eigen::Map<const RowMajorArray<T>>;

template <typename T>
using VectorMap = Eigen::Map<Vector<T>>;

template <typename T>
using ConstVectorMap = Eigen::Map<const Vector<T>>;

template <typename T, typename TP = PID>
struct AnnCandidate {
    TP id = 0;
    T distance = std::numeric_limits<T>::max();

    AnnCandidate() = default;
    explicit AnnCandidate(TP vec_id, T dis) : id(vec_id), distance(dis) {}

    friend bool operator<(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance < second.distance;
    }
    friend bool operator>(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance > second.distance;
    }
    friend bool operator>=(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance >= second.distance;
    }
    friend bool operator<=(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance <= second.distance;
    }
};

enum MetricType : std::uint8_t { METRIC_L2, METRIC_IP };
enum ScalarQuantizerType : std::uint8_t { RECONSTRUCTION, UNBIASED_ESTIMATION, PLAIN };
}  // namespace rabitqlib