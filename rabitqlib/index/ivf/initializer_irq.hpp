#pragma once

// IvfRabitqInitializer: an Initializer (rabitqlib::ivf::Initializer subtype)
// whose centroids_distances() runs an embedded IVF-RaBitQ search over the
// first-level centroids. The embedded index uses second-level ("inner")
// centroids produced offline by clustering/ivf_centroids.py and 9-bit RaBitQ
// quantization (1 bit batch + 8 bits ex). At query time it visits inner
// clusters nearest-first — all of them, or a budgeted prefix, depending on
// SearchMode below — and returns the top-nprobe first-level centroids
// ordered by ascending RaBitQ-estimated distance.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/io_auto.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::ivf_irq {

// Total bits for the embedded IVF-RaBitQ on first-level centroids.
// Fixed at 9 (= 1-bit binary + 8-bit ex code).
constexpr size_t kInnerTotalBits = 9;
constexpr size_t kInnerExBits = kInnerTotalBits - 1;

// .irq sidecar file-format magic + version. Sidecars without a header begin
// directly with a size_t pdim field; load() peeks the first 32 bits and
// treats anything != kIrqMagic as such a headerless file (mean scheme, no
// scheme tag).
constexpr uint32_t kIrqMagic   = 0xC1F8E0CAU;
// v3 appends (after ex_data_): [u32 ip_normalize][float*num_cluster norms]
// when ip_normalize is set. v2 and earlier are read with ip_normalize=false.
constexpr uint32_t kIrqVersion = 3U;

// Inner-centroid scheme tag persisted in the .irq sidecar (v2+). The mean
// scheme is the only valid value: each inner centroid is the kmeans MEAN of
// its cluster's first-level centroids — a synthetic point — and RaBitQ codes
// encode every first-level centroid relative to its assigned mean. Any other
// tag value is rejected at load.
constexpr uint32_t kSchemeMean = 0U;

// Inner-IVF traversal mode at query time. Both modes order inner clusters
// by ascending query-to-inner-centroid distance (nearest first); they
// differ only in how many of those clusters get scanned per query.
//
//   Full     : scan every inner cluster (= every first-level centroid gets
//              its 9-bit RaBitQ distance estimated). Result is ranked over
//              the entire centroid set.
//
//   KFixedMult: scan k = min(num_inner, ceil(outer_multiplier * nprobe))
//              inner clusters in nearest-first order. Independent of how
//              many first-level centroids those clusters actually contain.
//              A cluster budget proportional to nprobe caps coarse FastScan
//              work predictably without a per-dataset points budget, which
//              matters on ~100M datasets where the coarse step dominates.
//
// Refinement method used inside scan_one_batch when a centroid passes the
// 1-bit lower-bound prune (low_dist < distk). The 1-bit fastscan stage is
// always done first either way; this only controls the per-survivor refine.
//
//   Exbits        : default. split_distance_boosting on the 8-bit ex_code
//                   (1024 B per centroid, mixed-type SIMD ip16_fxu8_avx).
//                   Returns RaBitQ's estimate of ||q - p||^2.
//   FullPrecision : euclidean_sqr(rotated_query, rotated_centroids_[id])
//                   (4096 B per centroid float read, pure SIMD FMA).
//                   Returns exact ||q - p||^2 in the rotated padded_dim
//                   space (rotation is orthogonal so == original space).
//                   Useful for ablating whether ex-bits is the bottleneck.
enum class RefineMode : uint8_t {
    Exbits        = 0,
    FullPrecision = 1,
};

inline const char* refine_mode_name(RefineMode m) {
    return (m == RefineMode::FullPrecision) ? "full_precision" : "exbits";
}

// Env vars (read once per IvfRabitqInitializer instance):
//   RABITQ_IRQ_MODE              = "full" | "kfixedmult"               (default: kfixedmult)
//   RABITQ_IRQ_OUTER_MULT        = float > 0                          (kfixedmult only, default: 0.5)
//   RABITQ_IRQ_REFINE            = "exbits" | "full_precision"        (default: exbits)
enum class SearchMode : uint8_t {
    Full = 0,
    KFixedMult = 1,
};

inline const char* search_mode_name(SearchMode m) {
    if (m == SearchMode::KFixedMult) return "kfixedmult";
    return "full";
}

struct SearchParams {
    // Defaults: kfixedmult with outer_multiplier 0.5 and exbits refine.
    // The environment variables above override them.
    SearchMode mode = SearchMode::KFixedMult;
    // KFixedMult only: k_to_scan = min(num_inner, ceil(outer_multiplier * nprobe)).
    float outer_multiplier = 0.5F;
    RefineMode refine_mode = RefineMode::Exbits;
};

inline SearchParams load_search_params_from_env() {
    SearchParams p;
    if (const char* m = std::getenv("RABITQ_IRQ_MODE")) {
        const std::string s(m);
        if (s == "kfixedmult" || s == "kfm" || s == "fixedmult") {
            p.mode = SearchMode::KFixedMult;
        } else if (s == "full" || s == "scanall" || s == "all") {
            p.mode = SearchMode::Full;
        } else {
            std::cerr << "IvfRabitqInitializer: unknown RABITQ_IRQ_MODE='"
                      << s << "', falling back to 'kfixedmult'\n";
        }
    }
    if (const char* v = std::getenv("RABITQ_IRQ_REFINE")) {
        const std::string s(v);
        if (s == "full_precision" || s == "fullprec" || s == "full" || s == "fp") {
            p.refine_mode = RefineMode::FullPrecision;
        } else if (s == "exbits" || s == "ex" || s == "rabitq") {
            p.refine_mode = RefineMode::Exbits;
        } else {
            std::cerr << "IvfRabitqInitializer: unknown RABITQ_IRQ_REFINE='"
                      << s << "', falling back to 'exbits'\n";
        }
    }
    if (const char* v = std::getenv("RABITQ_IRQ_OUTER_MULT")) {
        try {
            const float f = std::stof(v);
            if (f > 0.0F) {
                p.outer_multiplier = f;
            } else {
                std::cerr << "IvfRabitqInitializer: ignoring "
                             "RABITQ_IRQ_OUTER_MULT='" << v
                          << "' (must be > 0); using "
                          << p.outer_multiplier << '\n';
            }
        } catch (...) {
            std::cerr << "IvfRabitqInitializer: bad RABITQ_IRQ_OUTER_MULT='"
                      << v << "'; using " << p.outer_multiplier << '\n';
        }
    }
    return p;
}

// Bounded max-heap top-K for centroids_distances. A heap keeps the coarse
// step free of the O(nprobe) per-insert memmove a sorted candidate array
// costs, which otherwise dominates at large nprobe:
//   insert         : O(log nprobe) worst-case
//   top_dist       : O(1)
//   sorted_results : O(nprobe log nprobe) once at the end
//
// The structure keeps the K smallest distances seen. Ties are broken loosely:
// a candidate equal to the current top distance is skipped rather than
// replacing it, which leaves top-K membership (and hence recall) unaffected.
class MaxHeapTopK {
   public:
    explicit MaxHeapTopK(size_t capacity) : capacity_(capacity) {
        if (capacity_ > 0) {
            data_.reserve(capacity_);
        }
    }

    [[nodiscard]] float top_dist() const {
        return (data_.size() == capacity_)
                   ? data_.front().distance
                   : std::numeric_limits<float>::max();
    }

    void insert(PID id, float dist) {
        if (data_.size() < capacity_) {
            data_.push_back(AnnCandidate<float>(id, dist));
            std::push_heap(data_.begin(), data_.end(), Less{});
        } else if (capacity_ > 0 && dist < data_.front().distance) {
            std::pop_heap(data_.begin(), data_.end(), Less{});
            data_.back().id = id;
            data_.back().distance = dist;
            std::push_heap(data_.begin(), data_.end(), Less{});
        }
    }

    // Sort heap into ascending-distance order in-place and return view.
    // After this call the structure is no longer a heap; do not call insert
    // again.
    [[nodiscard]] const std::vector<AnnCandidate<float>>& sorted_results() {
        std::sort_heap(data_.begin(), data_.end(), Less{});
        return data_;
    }

   private:
    struct Less {
        bool operator()(const AnnCandidate<float>& a,
                        const AnnCandidate<float>& b) const {
            return a.distance < b.distance;  // max-heap on distance
        }
    };

    std::vector<AnnCandidate<float>> data_;
    size_t capacity_;
};

class IvfRabitqInitializer : public rabitqlib::ivf::Initializer {
   public:
    // Indexing-time ctor. IVFSSD_Index supplies the outer rotator — which
    // maps the original-space second-level centroids loaded from disk into
    // the same padded_dim space the rotated first-level centroids live in —
    // and the paths to the python-produced inner-centroids / inner-cluster-
    // ids. All input paths accept either .fvecs/.ivecs or .fbin/.bin
    // (auto-detected).
    IvfRabitqInitializer(size_t padded_dim,
                         size_t num_cluster,
                         size_t outer_dim,
                         const Rotator<float>* outer_rotator,
                         std::string inner_centroids_path,
                         std::string inner_cids_path,
                         bool ip_normalize = false)
        : Initializer(padded_dim, num_cluster),
          outer_dim_(outer_dim),
          outer_rotator_(outer_rotator),
          inner_centroids_path_(std::move(inner_centroids_path)),
          inner_cids_path_(std::move(inner_cids_path)),
          search_params_(load_search_params_from_env()),
          ip_normalize_(ip_normalize) {
        ip_func_ = select_excode_ipfunc(kInnerExBits);
        announce_search_params();
        std::cout << "IvfRabitqInitializer: ip_normalize="
                  << (ip_normalize_ ? "true" : "false") << "\n";
    }

    // Loading-time ctor. Inner index payload is read from the .irq sidecar
    // in load(); no outer rotator / paths needed.
    IvfRabitqInitializer(size_t padded_dim, size_t num_cluster)
        : Initializer(padded_dim, num_cluster),
          search_params_(load_search_params_from_env()) {
        ip_func_ = select_excode_ipfunc(kInnerExBits);
        announce_search_params();
    }

    ~IvfRabitqInitializer() override {
        std::free(batch_data_);
        std::free(ex_data_);
    }

    // Returns the (rotated, padded_dim) first-level centroid by id. In
    // ip_normalize_ mode this is the *normalized* (unit-norm) centroid, which
    // the outer SSD loop does not read for METRIC_IP: it reconstructs <q,c>
    // from centroid_norm(id) and dist_IRQ instead (see common.tpp).
    [[nodiscard]] const float* centroid(PID id) const override {
        return rotated_centroids_.data() + (static_cast<size_t>(id) * dim_);
    }

    // Raw (un-normalized) L2 norm ||c|| of first-level rotated centroid `id`.
    // Valid only in ip_normalize_ mode; used by the outer SSD loop to turn the
    // normalized-space coarse distance dist_IRQ into <q,c> and ||q-c||.
    [[nodiscard]] bool ip_normalize() const { return ip_normalize_; }
    [[nodiscard]] float centroid_norm(PID id) const {
        return centroid_norms_[static_cast<size_t>(id)];
    }

    // Build the embedded IVF-RaBitQ. Called by IVFSSD_Index::construct after
    // it has rotated the first-level centroids; the rotated centroids are
    // passed in here. Loads the python-generated second-level centroids from
    // disk, rotates them with the outer rotator, then quantizes the
    // first-level centroids relative to their assigned second-level centroid.
    void add_vectors(const float* rotated_centroids) override {
        if (outer_rotator_ == nullptr) {
            throw std::runtime_error(
                "IvfRabitqInitializer::add_vectors called without an outer rotator");
        }

        // Cache rotated first-level centroids. For METRIC_L2 these stay raw.
        // Under ip_normalize_ (METRIC_IP) each centroid's raw norm ||c|| is
        // recorded and rotated_centroids_ is overwritten with the L2-
        // normalized vectors, so the whole embedded IVF-RaBitQ operates in
        // unit-norm space (cosine selection). The python-generated inner
        // centroids must likewise have been computed on normalized
        // first-level centroids.
        rotated_centroids_.assign(rotated_centroids,
                                  rotated_centroids + (num_cluster_ * dim_));
        if (ip_normalize_) {
            centroid_norms_.assign(num_cluster_, 0.0F);
            for (size_t i = 0; i < num_cluster_; ++i) {
                float* row = rotated_centroids_.data() + (i * dim_);
                const float nrm = std::sqrt(
                    dot_product(row, row, dim_));
                centroid_norms_[i] = nrm;
                if (nrm > 0.0F) {
                    const float inv = 1.0F / nrm;
                    for (size_t d = 0; d < dim_; ++d) {
                        row[d] *= inv;
                    }
                }
            }
        }

        // ----- Load python-generated inner centroids and assignments -----
        rabitqlib::RowMajorArray<float> inner_orig;
        rabitqlib::RowMajorArray<uint32_t> inner_cids_arr;
        rabitqlib::load_matrix_auto<float, rabitqlib::RowMajorArray<float>>(
            inner_centroids_path_.c_str(), inner_orig);
        rabitqlib::load_matrix_auto<CID, rabitqlib::RowMajorArray<uint32_t>>(
            inner_cids_path_.c_str(), inner_cids_arr);

        if (static_cast<size_t>(inner_orig.cols()) != outer_dim_) {
            std::cerr << "IvfRabitqInitializer: inner centroids dim ("
                      << inner_orig.cols() << ") != outer_dim_ ("
                      << outer_dim_ << ")\n";
            std::exit(1);
        }
        if (static_cast<size_t>(inner_cids_arr.rows()) != num_cluster_) {
            std::cerr << "IvfRabitqInitializer: inner cids rows ("
                      << inner_cids_arr.rows() << ") != num_cluster_ ("
                      << num_cluster_ << ")\n";
            std::exit(1);
        }

        num_inner_cluster_ = static_cast<size_t>(inner_orig.rows());

        // Rotate inner centroids into the same padded_dim space.
        inner_centroids_rot_.assign(num_inner_cluster_ * dim_, 0.0F);
        for (size_t i = 0; i < num_inner_cluster_; ++i) {
            outer_rotator_->rotate(
                inner_orig.data() + (i * outer_dim_),
                inner_centroids_rot_.data() + (i * dim_));
        }

        // ----- Group first-level centroids by inner cluster -----
        std::vector<std::vector<CID>> id_lists(num_inner_cluster_);
        for (size_t i = 0; i < num_cluster_; ++i) {
            const CID cid = static_cast<CID>(inner_cids_arr.data()[i]);
            if (cid >= num_inner_cluster_) {
                std::cerr << "IvfRabitqInitializer: bad inner cluster id "
                          << cid << " (num_inner=" << num_inner_cluster_ << ")\n";
                std::exit(1);
            }
            id_lists[cid].push_back(static_cast<CID>(i));
        }

        // ----- Layout (sizes, batch/ex offsets) -----
        // num_rabitq_coded: total first-level centroids that get RaBitQ codes
        // (== num_cluster_: every centroid is a member of exactly one inner
        // cluster).
        size_t num_rabitq_coded = 0;
        for (const auto& l : id_lists) {
            num_rabitq_coded += l.size();
        }

        cluster_sizes_.assign(num_inner_cluster_, 0);
        cluster_batch_offsets_.assign(num_inner_cluster_ + 1, 0);
        cluster_ex_offsets_.assign(num_inner_cluster_ + 1, 0);
        cluster_id_offsets_.assign(num_inner_cluster_ + 1, 0);

        const size_t batch_block_bytes = BatchDataMap<float>::data_bytes(dim_);
        const size_t ex_one_bytes = ExDataMap<float>::data_bytes(dim_, kInnerExBits);

        for (size_t c = 0; c < num_inner_cluster_; ++c) {
            const size_t n = id_lists[c].size();
            cluster_sizes_[c] = n;
            const size_t blocks = div_round_up(n, fastscan::kBatchSize);
            cluster_batch_offsets_[c + 1] =
                cluster_batch_offsets_[c] + blocks * batch_block_bytes;
            cluster_ex_offsets_[c + 1] =
                cluster_ex_offsets_[c] + n * ex_one_bytes;
            cluster_id_offsets_[c + 1] = cluster_id_offsets_[c] + n;
        }

        const size_t batch_total_bytes = cluster_batch_offsets_[num_inner_cluster_];
        const size_t ex_total_bytes = cluster_ex_offsets_[num_inner_cluster_];

        std::free(batch_data_);
        std::free(ex_data_);
        batch_data_ = memory::align_allocate<64, char, true>(batch_total_bytes);
        ex_data_ = memory::align_allocate<64, char, true>(ex_total_bytes);
        ids_.assign(num_rabitq_coded, 0);

        // ----- Quantize each inner cluster -----
        quant::RabitqConfig config;  // default quantization configuration
        for (size_t c = 0; c < num_inner_cluster_; ++c) {
            const auto& ids = id_lists[c];
            const size_t n = ids.size();
            if (n == 0) {
                continue;
            }

            // Gather rotated first-level centroids for this inner cluster.
            // The source is rotated_centroids_ (the cached member), which in
            // ip_normalize_ mode holds the normalized vectors, so the RaBitQ
            // codes are built in the same normalized space the inner
            // centroids and the normalized query live in. For METRIC_L2
            // rotated_centroids_ holds the raw input.
            std::vector<float> gathered(n * dim_);
            for (size_t i = 0; i < n; ++i) {
                std::memcpy(gathered.data() + (i * dim_),
                            rotated_centroids_.data() +
                                (static_cast<size_t>(ids[i]) * dim_),
                            sizeof(float) * dim_);
            }

            const float* inner_c = inner_centroids_rot_.data() + (c * dim_);
            char* batch_ptr = batch_data_ + cluster_batch_offsets_[c];
            char* ex_ptr = ex_data_ + cluster_ex_offsets_[c];

            for (size_t i = 0; i < n; i += fastscan::kBatchSize) {
                const size_t bn = std::min(fastscan::kBatchSize, n - i);
                // A short last batch leaves factor slots unwritten; zero the
                // block first so the .irq file does not depend on what memory held.
                if (bn < fastscan::kBatchSize) {
                    std::memset(batch_ptr, 0, batch_block_bytes);
                }
                quant::quantize_split_batch(
                    gathered.data() + (i * dim_), inner_c, bn, dim_,
                    kInnerExBits, batch_ptr, ex_ptr,
                    rabitqlib::METRIC_L2, config);
                batch_ptr += batch_block_bytes;
                ex_ptr += bn * ex_one_bytes;
            }

            // Copy ids into the cluster-ordered ids_ array.
            std::memcpy(ids_.data() + cluster_id_offsets_[c], ids.data(),
                        sizeof(CID) * n);
        }

        std::cout << "IvfRabitqInitializer: built inner IVF-RaBitQ over "
                  << num_cluster_ << " centroids in " << num_inner_cluster_
                  << " inner clusters (" << kInnerTotalBits << "-bit, padded_dim="
                  << dim_ << ", rabitq_coded=" << num_rabitq_coded << ")\n";
    }

    // The interface the outer SSD search loop uses: given the rotated
    // (outer padded_dim) query, return the top-nprobe first-level centroids
    // sorted by ascending RaBitQ-estimated distance.
    void centroids_distances(
        const float* query,
        size_t nprobe,
        std::vector<AnnCandidate<float>>& candidates) const override {
        if (nprobe == 0) {
            return;
        }
        nprobe = std::min(nprobe, num_cluster_);

        // ip_normalize_: the embedded index lives in unit-norm space, so the
        // query must be L2-normalized before scanning. The normalization runs
        // on a local copy (qbuf) that `query` is rebound to, leaving the
        // caller's rotated query untouched — the outer SSD loop still needs
        // the raw rotated query for its own RaBitQ LUT and reconstructs ||q||
        // separately. The returned candidate .distance is then
        // dist_IRQ = ||q_hat - c_hat||.
        std::vector<float> qbuf;
        if (ip_normalize_) {
            qbuf.assign(query, query + dim_);
            const float qn = std::sqrt(dot_product(qbuf.data(), qbuf.data(),
                                                    dim_));
            if (qn > 0.0F) {
                const float inv = 1.0F / qn;
                for (float& x : qbuf) {
                    x *= inv;
                }
            }
            query = qbuf.data();
        }

        // 1) Distance from query to every inner centroid (Flat scan).
        //    Cheap: O(num_inner * dim), num_inner ~ num_cluster / inner_size.
        std::vector<float> inner_dist(num_inner_cluster_);
        for (size_t c = 0; c < num_inner_cluster_; ++c) {
            inner_dist[c] = std::sqrt(euclidean_sqr(
                query, inner_centroids_rot_.data() + (c * dim_), dim_));
        }

        // 2) Order inner clusters by ascending query-to-inner-centroid
        //    distance (nearest first). This matters even in Full mode: the
        //    top-K bound distk shrinks as candidates accumulate, so visiting
        //    near clusters first makes far clusters' survivors face a tighter
        //    distk and be pruned before the ex-bits rerank (fewer
        //    split_distance_boosting calls). It is also what makes the
        //    KFixedMult "stop after k clusters" cap meaningful.
        std::vector<uint32_t> order(num_inner_cluster_);
        std::iota(order.begin(), order.end(), 0U);
        std::sort(order.begin(), order.end(),
                  [&](uint32_t a, uint32_t b) {
                      return inner_dist[a] < inner_dist[b];
                  });

        // 3) Decide how many inner clusters to actually scan.
        //    Full      -> num_inner_cluster_
        //    KFixedMult-> k_to_scan = min(num_inner_cluster_,
        //                                 max(1, ceil(outer_multiplier * nprobe))).
        //                 Pure cluster-count cap; ignores per-cluster sizes.
        size_t k_to_scan = num_inner_cluster_;
        if (search_params_.mode == SearchMode::KFixedMult
            && num_inner_cluster_ > 0) {
            const size_t k_mult = std::max<size_t>(
                1U,
                static_cast<size_t>(std::ceil(
                    static_cast<double>(search_params_.outer_multiplier)
                    * static_cast<double>(nprobe))));
            k_to_scan = std::min(num_inner_cluster_, k_mult);
        }

        // 4) Scan k_to_scan clusters in nearest-first order with FastScan +
        //    ex-bits boosting, inserting first-level centroids into knns.
        //
        // knns is a bounded max-heap (MaxHeapTopK above). Its O(log nprobe)
        // insert and O(1) top_dist keep candidate bookkeeping from dominating
        // the coarse step at large nprobe. The heap is sorted exactly once,
        // at the end, before the results are copied out.
        MaxHeapTopK knns(nprobe);
        SplitBatchQuery<float> q_obj(
            query, dim_, kInnerExBits, rabitqlib::METRIC_L2, /*use_hacc=*/true);

        const size_t batch_block_bytes = BatchDataMap<float>::data_bytes(dim_);
        const size_t ex_one_bytes = ExDataMap<float>::data_bytes(dim_, kInnerExBits);

        // Roofline counters: split into 3 buckets so the work model can
        // charge each kind of coarse-search work at its correct flop-eq
        // cost (full L2 vs 1-bit fastscan vs ex-bits refine). See
        // rabitqlib::ivf::last_full_l2_dists / _fastscan_attempts /
        // _exbits_refines in initializer.hpp for the shared thread_locals
        // that all Initializer subtypes write.
        //
        // full_l2_dists for IRQ accumulates the num_inner_cluster_ Flat
        // distances from the loop at the top of this function (query ->
        // each inner centroid). These are real full L2 evaluations and
        // feed both the cluster ordering and q_obj.set_g_add() below.
        uint64_t full_l2_dists = num_inner_cluster_;
        uint64_t fastscan_attempts = 0;
        uint64_t exbits_refines = 0;

        // Scan + per-survivor refine, cluster by cluster in
        // nearest-first order.
        for (size_t idx = 0; idx < k_to_scan; ++idx) {
            const size_t c = order[idx];
            const size_t n = cluster_sizes_[c];
            if (n == 0) {
                continue;
            }
            q_obj.set_g_add(inner_dist[c]);  // L2: stores g_add = norm^2

            const char* batch_ptr = batch_data_ + cluster_batch_offsets_[c];
            const char* ex_ptr = ex_data_ + cluster_ex_offsets_[c];
            const CID* ids = ids_.data() + cluster_id_offsets_[c];

            const size_t iter = n / fastscan::kBatchSize;
            const size_t remain = n - (iter * fastscan::kBatchSize);

            for (size_t b = 0; b < iter; ++b) {
                scan_one_batch(batch_ptr, ex_ptr, ids, q_obj,
                               fastscan::kBatchSize, knns,
                               fastscan_attempts, exbits_refines);
                batch_ptr += batch_block_bytes;
                ex_ptr += fastscan::kBatchSize * ex_one_bytes;
                ids += fastscan::kBatchSize;
            }
            if (remain > 0) {
                scan_one_batch(batch_ptr, ex_ptr, ids, q_obj, remain, knns,
                               fastscan_attempts, exbits_refines);
            }
        }

        // Roofline: report this thread's centroid coarse-search work.
        rabitqlib::ivf::last_full_l2_dists = full_l2_dists;
        rabitqlib::ivf::last_fastscan_attempts = fastscan_attempts;
        rabitqlib::ivf::last_exbits_refines = exbits_refines;
        // 5) Sort the heap and copy out top-nprobe in ascending-distance
        //    order. knns stores ex_dist = RaBitQ's estimate of *squared* L2
        //    distance (sqrt is monotonic so ranking is preserved). The
        //    downstream boundary_qd loop expects the same convention as
        //    Flat/HNSW initializers — distance = sqrt(squared_dist), i.e.
        //    an L2 *norm* — because it later calls set_g_add(dist), which
        //    itself squares the input. Returning a squared distance here
        //    propagates as dist^4 inside g_add and silently breaks cluster
        //    pruning, costing substantial recall.
        //    Clamp to 0 to absorb tiny numerical negatives that 9-bit
        //    RaBitQ estimation can produce.
        const auto& buf = knns.sorted_results();
        const size_t n_out = std::min(nprobe, buf.size());
        for (size_t i = 0; i < n_out; ++i) {
            candidates[i].id = buf[i].id;
            candidates[i].distance = std::sqrt(std::max(0.0f, buf[i].distance));
        }
        // Defensive: in KFixedMult mode, if k_to_scan did not yield nprobe
        // centroids the tail is filled with sentinel
        // max-distance entries so callers iterating over `nprobe` slots see a
        // stable shape. (For Full mode with nprobe <= num_cluster_ this never
        // fires.)
        for (size_t i = n_out; i < nprobe; ++i) {
            candidates[i].id = std::numeric_limits<PID>::max();
            candidates[i].distance = std::numeric_limits<float>::max();
        }
    }

    // ---------------- save / load to .irq sidecar -----------------------
    // The payload lives in a sidecar file, as for HNSW: the mem-index
    // ofstream argument is ignored and a separate file at filename + ".irq"
    // is opened instead.
    //
    // File layout (version 2+):
    //
    //   [u32]  magic = kIrqMagic
    //   [u32]  version = kIrqVersion
    //   [u32]  scheme tag; only 0 (= kSchemeMean) is valid, any other value
    //          is rejected at load
    //   [u32]  num_rabitq_coded   (== num_cluster; size_t fits <= 4G)
    //   [size_t] pdim
    //   [size_t] num_cluster
    //   [size_t] num_inner_cluster
    //   [size_t] ex_bits
    //   [float * num_cluster * pdim]      rotated_centroids_
    //   [float * num_inner * pdim]        inner_centroids_rot_
    //   [size_t * num_inner]              cluster_sizes_
    //   [u32    * num_rabitq_coded]       ids_ (first-level centroid ids)
    //   [bytes]                            batch_data_
    //   [bytes]                            ex_data_
    //
    // The headerless (pre-magic) layout starts directly with the size_t pdim
    // field and carries no scheme tag. load() peeks 4 bytes and, if the magic
    // does not match, rewinds and parses that layout instead.
    void save(std::ofstream& /*output*/, const char* filename) const override {
        const std::string path = std::string(filename) + ".irq";
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            std::cerr << "IvfRabitqInitializer::save: cannot open " << path << '\n';
            std::exit(1);
        }
        const uint32_t magic = kIrqMagic;
        const uint32_t version = kIrqVersion;
        const uint32_t scheme_u32 = kSchemeMean;
        const uint32_t rabitq_coded_u32 = static_cast<uint32_t>(ids_.size());
        out.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&scheme_u32), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&rabitq_coded_u32), sizeof(uint32_t));

        const size_t pdim = dim_;
        const size_t nc = num_cluster_;
        const size_t nic = num_inner_cluster_;
        const size_t exb = kInnerExBits;
        out.write(reinterpret_cast<const char*>(&pdim), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(&nc), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(&nic), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(&exb), sizeof(size_t));

        out.write(reinterpret_cast<const char*>(rotated_centroids_.data()),
                  static_cast<std::streamsize>(sizeof(float) * nc * pdim));
        out.write(reinterpret_cast<const char*>(inner_centroids_rot_.data()),
                  static_cast<std::streamsize>(sizeof(float) * nic * pdim));
        out.write(reinterpret_cast<const char*>(cluster_sizes_.data()),
                  static_cast<std::streamsize>(sizeof(size_t) * nic));
        out.write(reinterpret_cast<const char*>(ids_.data()),
                  static_cast<std::streamsize>(sizeof(CID) * ids_.size()));

        const size_t batch_total_bytes = cluster_batch_offsets_[nic];
        const size_t ex_total_bytes = cluster_ex_offsets_[nic];
        out.write(batch_data_, static_cast<std::streamsize>(batch_total_bytes));
        out.write(ex_data_, static_cast<std::streamsize>(ex_total_bytes));

        // v3 tail: ip_normalize flag, and (if set) the per-centroid raw norms.
        // Appended after ex_data_ so v2 readers' fixed layout is unaffected.
        const uint32_t ip_norm_u32 = ip_normalize_ ? 1U : 0U;
        out.write(reinterpret_cast<const char*>(&ip_norm_u32), sizeof(uint32_t));
        if (ip_normalize_) {
            out.write(reinterpret_cast<const char*>(centroid_norms_.data()),
                      static_cast<std::streamsize>(sizeof(float) * nc));
        }
        out.close();
    }

    void load(std::ifstream& /*input*/, const char* filename) override {
        const std::string path = std::string(filename) + ".irq";
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::cerr << "IvfRabitqInitializer::load: cannot open " << path << '\n';
            std::exit(1);
        }

        // Peek 4 bytes to detect the headerless (pre-magic) format, which
        // starts directly with size_t pdim; on little-endian x86_64 that
        // word essentially never collides with kIrqMagic.
        uint32_t maybe_magic = 0;
        in.read(reinterpret_cast<char*>(&maybe_magic), sizeof(uint32_t));
        bool has_header = (maybe_magic == kIrqMagic);
        if (!has_header) {
            in.seekg(0, std::ios::beg);
        }

        uint32_t version = 1U;  // assumed when no header is present
        uint32_t scheme_u32 = kSchemeMean;
        uint32_t rabitq_coded_u32 = 0U;
        if (has_header) {
            in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
            in.read(reinterpret_cast<char*>(&scheme_u32), sizeof(uint32_t));
            in.read(reinterpret_cast<char*>(&rabitq_coded_u32), sizeof(uint32_t));
            if (version > kIrqVersion) {
                std::cerr << "IvfRabitqInitializer::load: file version "
                          << version << " is newer than this binary supports ("
                          << kIrqVersion << ")\n";
                std::exit(1);
            }
        }
        if (scheme_u32 != kSchemeMean) {
            std::cerr << "IvfRabitqInitializer::load: " << path
                      << " carries non-mean scheme tag " << scheme_u32
                      << (scheme_u32 == 1U
                              ? " (the removed snapped scheme)"
                              : " (unknown/corrupt)")
                      << "; rebuild the coarse quantizer with build_coarse\n";
            std::exit(1);
        }

        size_t pdim = 0, nc = 0, nic = 0, exb = 0;
        in.read(reinterpret_cast<char*>(&pdim), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&nc), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&nic), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&exb), sizeof(size_t));
        if (pdim != dim_ || nc != num_cluster_ || exb != kInnerExBits) {
            std::cerr << "IvfRabitqInitializer::load: header mismatch (file padded_dim="
                      << pdim << " num_cluster=" << nc << " ex_bits=" << exb
                      << " | expected padded_dim=" << dim_ << " num_cluster="
                      << num_cluster_ << " ex_bits=" << kInnerExBits << ")\n";
            std::exit(1);
        }
        num_inner_cluster_ = nic;

        // rabitq_coded == num_cluster (every first-level centroid has codes);
        // headerless files carry no explicit count.
        const size_t rabitq_coded =
            has_header ? static_cast<size_t>(rabitq_coded_u32) : nc;

        rotated_centroids_.assign(nc * pdim, 0.0F);
        inner_centroids_rot_.assign(nic * pdim, 0.0F);
        cluster_sizes_.assign(nic, 0);
        ids_.assign(rabitq_coded, 0);

        in.read(reinterpret_cast<char*>(rotated_centroids_.data()),
                static_cast<std::streamsize>(sizeof(float) * nc * pdim));
        in.read(reinterpret_cast<char*>(inner_centroids_rot_.data()),
                static_cast<std::streamsize>(sizeof(float) * nic * pdim));
        in.read(reinterpret_cast<char*>(cluster_sizes_.data()),
                static_cast<std::streamsize>(sizeof(size_t) * nic));
        in.read(reinterpret_cast<char*>(ids_.data()),
                static_cast<std::streamsize>(sizeof(CID) * rabitq_coded));

        // Rebuild offset presums and allocate batch/ex blobs.
        cluster_batch_offsets_.assign(nic + 1, 0);
        cluster_ex_offsets_.assign(nic + 1, 0);
        cluster_id_offsets_.assign(nic + 1, 0);
        const size_t batch_block_bytes = BatchDataMap<float>::data_bytes(pdim);
        const size_t ex_one_bytes = ExDataMap<float>::data_bytes(pdim, kInnerExBits);
        for (size_t c = 0; c < nic; ++c) {
            const size_t n = cluster_sizes_[c];
            const size_t blocks = div_round_up(n, fastscan::kBatchSize);
            cluster_batch_offsets_[c + 1] =
                cluster_batch_offsets_[c] + blocks * batch_block_bytes;
            cluster_ex_offsets_[c + 1] = cluster_ex_offsets_[c] + n * ex_one_bytes;
            cluster_id_offsets_[c + 1] = cluster_id_offsets_[c] + n;
        }
        const size_t batch_total_bytes = cluster_batch_offsets_[nic];
        const size_t ex_total_bytes = cluster_ex_offsets_[nic];

        std::free(batch_data_);
        std::free(ex_data_);
        batch_data_ = memory::align_allocate<64, char, true>(batch_total_bytes);
        ex_data_ = memory::align_allocate<64, char, true>(ex_total_bytes);

        in.read(batch_data_, static_cast<std::streamsize>(batch_total_bytes));
        in.read(ex_data_, static_cast<std::streamsize>(ex_total_bytes));
        if (!in) {
            std::cerr << "IvfRabitqInitializer::load: short read on " << path << '\n';
            std::exit(1);
        }

        // v3 tail: ip_normalize flag + (if set) per-centroid raw norms.
        // Absent in v1/v2 files -> ip_normalize_ stays false.
        ip_normalize_ = false;
        centroid_norms_.clear();
        if (version >= 3U) {
            uint32_t ip_norm_u32 = 0U;
            in.read(reinterpret_cast<char*>(&ip_norm_u32), sizeof(uint32_t));
            ip_normalize_ = (ip_norm_u32 != 0U);
            if (ip_normalize_) {
                centroid_norms_.assign(nc, 0.0F);
                in.read(reinterpret_cast<char*>(centroid_norms_.data()),
                        static_cast<std::streamsize>(sizeof(float) * nc));
            }
            if (!in) {
                std::cerr << "IvfRabitqInitializer::load: short read on v3 tail of "
                          << path << '\n';
                std::exit(1);
            }
        }
        in.close();

        std::cout << "IvfRabitqInitializer::load: num_cluster=" << nc
                  << " num_inner=" << nic
                  << " rabitq_coded=" << rabitq_coded
                  << " ip_normalize=" << (ip_normalize_ ? "true" : "false")
                  << " (file_version=" << version << ")\n";
    }

   private:
    void scan_one_batch(const char* batch_data,
                        const char* ex_data,
                        const CID* ids,
                        const SplitBatchQuery<float>& q_obj,
                        size_t num_points,
                        MaxHeapTopK& knns,
                        uint64_t& fastscan_attempts,
                        uint64_t& exbits_refines) const {
        std::array<float, fastscan::kBatchSize> est_distance;
        std::array<float, fastscan::kBatchSize> low_distance;
        std::array<float, fastscan::kBatchSize> ip_x0_qr;

        split_batch_estdist(batch_data, q_obj, dim_, est_distance.data(),
                            low_distance.data(), ip_x0_qr.data(),
                            /*use_hacc=*/true);

        float distk = knns.top_dist();
        // Hoist the refine mode out of the inner loop to keep the branch
        // predictable.
        const bool use_full_prec =
            (search_params_.refine_mode == RefineMode::FullPrecision);
        const float* rotated_query = q_obj.rotated_query();
        const float* rotated_centroids_ptr = rotated_centroids_.data();

        for (size_t i = 0; i < num_points; ++i) {
            ++fastscan_attempts;
            const float lower_dist = low_distance[i];
            if (lower_dist < distk) {
                float ex_dist;
                if (use_full_prec) {
                    // Pure SIMD-FMA exact squared L2 in padded_dim space.
                    // rotated_centroids_ holds the OUTER-rotated centroids
                    // (same orthogonal space as q_obj.rotated_query()), so
                    // this is the true ||q - p||^2. The exbits_refines
                    // counter tallies refine-step attempts regardless of
                    // which kernel ran.
                    ex_dist = euclidean_sqr(
                        rotated_query,
                        rotated_centroids_ptr +
                            (static_cast<size_t>(ids[i]) * dim_),
                        dim_);
                } else {
                    ex_dist = split_distance_boosting(
                        ex_data, ip_func_, q_obj, dim_, kInnerExBits,
                        ip_x0_qr[i]);
                }
                ++exbits_refines;
                knns.insert(ids[i], ex_dist);
                distk = knns.top_dist();
            }
            ex_data += ExDataMap<float>::data_bytes(dim_, kInnerExBits);
        }
    }

    void announce_search_params() const {
        std::cout << "IvfRabitqInitializer: search mode='"
                  << search_mode_name(search_params_.mode) << "'";
        if (search_params_.mode == SearchMode::KFixedMult) {
            std::cout << " outer_multiplier="
                      << search_params_.outer_multiplier;
        }
        std::cout << " refine='"
                  << refine_mode_name(search_params_.refine_mode) << "'";
        std::cout << " (override via RABITQ_IRQ_MODE / "
                     "RABITQ_IRQ_OUTER_MULT / RABITQ_IRQ_REFINE)\n";
    }

    size_t outer_dim_ = 0;
    const Rotator<float>* outer_rotator_ = nullptr;
    std::string inner_centroids_path_;
    std::string inner_cids_path_;
    SearchParams search_params_;

    // METRIC_IP (normalized-coarse) mode. When true, the embedded IVF-RaBitQ
    // runs entirely in *normalized* (unit-norm) space: rotated_centroids_ and
    // inner_centroids_rot_ hold L2-normalized vectors, and centroids_distances
    // normalizes the incoming query before scanning. The per-centroid raw
    // norms are kept in centroid_norms_ so the outer SSD loop can reconstruct
    // <q,c> and ||q-c|| from the normalized-space distance dist_IRQ:
    //   <q,c> = (2 - dist_IRQ^2) * ||q|| * ||c|| / 2
    //   ||q-c|| = sqrt(||q||^2 + ||c||^2 - 2<q,c>)
    // For METRIC_L2 this stays false and the index is the plain L2 variant.
    bool ip_normalize_ = false;
    // Raw (un-normalized) L2 norm of each first-level rotated centroid, indexed
    // by first-level centroid id. Populated only when ip_normalize_ is true.
    // Resident at query time; consumed via centroid_norm(id).
    std::vector<float> centroid_norms_;

    size_t num_inner_cluster_ = 0;
    std::vector<float> rotated_centroids_;
    std::vector<float> inner_centroids_rot_;
    std::vector<size_t> cluster_sizes_;
    std::vector<size_t> cluster_batch_offsets_;
    std::vector<size_t> cluster_ex_offsets_;
    std::vector<size_t> cluster_id_offsets_;
    std::vector<CID> ids_;
    char* batch_data_ = nullptr;
    char* ex_data_ = nullptr;

    float (*ip_func_)(const float*, const uint8_t*, size_t) = nullptr;
};

}  // namespace rabitqlib::ivf_irq
