#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/third/hnswlib/hnswlib.h"

namespace rabitqlib::ivf {

// Roofline counters: per-bucket thread-local counts of centroid coarse-search
// work performed inside the most recent Initializer::centroids_distances call
// on THIS thread. Each Initializer subtype writes whichever buckets apply to
// its algorithm; the roofline work model maps each bucket to a different
// flop-eq cost.
//
//   last_full_l2_dists      : full L2 distance evaluations over padded_dim.
//                             Flat: == num_cluster_; HNSW: == hnswlib
//                             distfunc count; IRQ: inner-IVF Flat scan over
//                             num_inner_cluster_ centroids.
//   last_fastscan_attempts  : 1-bit fastscan + factor-correction attempts
//                             over padded_dim (i.e. naive-scalar work of
//                             2*padded_dim + 8 flop-eq each). Used by IRQ
//                             for its embedded RaBitQ scan; 0 for Flat/HNSW.
//   last_exbits_refines     : ex-bits boosting evaluations (2*ex_bits*
//                             padded_dim + 6 flop-eq each). IRQ uses
//                             ex_bits=8 (= kInnerExBits in initializer_irq);
//                             0 for Flat/HNSW.
//
// Multiple OMP threads share one Initializer instance (centroids_distances
// is const), so these counters must be thread_local rather than per-instance.
inline thread_local uint64_t last_full_l2_dists = 0;
inline thread_local uint64_t last_fastscan_attempts = 0;
inline thread_local uint64_t last_exbits_refines = 0;

enum class InitializerType : uint8_t {
    Auto = 0,
    Flat = 1,
    HNSW = 2,
};

inline const char* initializer_type_name(InitializerType type) {
    switch (type) {
        case InitializerType::Auto:
            return "auto";
        case InitializerType::Flat:
            return "flat";
        case InitializerType::HNSW:
            return "hnsw";
    }
    return "unknown";
}

inline InitializerType resolve_initializer_type(
    InitializerType requested, size_t num_cluster
) {
    if (requested != InitializerType::Auto) {
        return requested;
    }
    return (num_cluster < 20000UL) ? InitializerType::Flat
                                   : InitializerType::HNSW;
}

template <class Function>
inline void parallel_for(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr last_exception = nullptr;
        std::mutex last_except_mutex;

        threads.reserve(numThreads);
        for (size_t thread_id = 0; thread_id < numThreads; ++thread_id) {
            threads.push_back(std::thread([&, thread_id] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, thread_id);
                    } catch (...) {
                        std::unique_lock<std::mutex> last_except_lock(last_except_mutex);
                        last_exception = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        if (last_exception) {
            std::rethrow_exception(last_exception);
        }
    }
}

/**
 * @brief For ivf centroids, we need an intializer to get the candidate clusters.
 */
class Initializer {
   protected:
    size_t dim_;
    size_t num_cluster_;

   public:
    explicit Initializer(size_t d, size_t k) : dim_(d), num_cluster_(k) {}
    virtual ~Initializer() = 0;
    [[nodiscard]] virtual const float* centroid(PID) const = 0;
    virtual void add_vectors(const float*) = 0;
    virtual void centroids_distances(
        const float*, size_t, std::vector<AnnCandidate<float>>&
    ) const = 0;
    virtual void load(std::ifstream&, const char*) = 0;
    virtual void save(std::ofstream&, const char*) const = 0;
};
inline Initializer::~Initializer() {}

class FlatInitializer : public Initializer {
   private:
    std::vector<float> centroids_;

   public:
    explicit FlatInitializer(size_t d, size_t k)
        : Initializer(d, k), centroids_(num_cluster_ * dim_) {}

    ~FlatInitializer() override = default;

    [[nodiscard]] const float* centroid(PID id) const override {
        return &centroids_[id * dim_];
    }

    void add_vectors(const float* cent) override {
        std::memcpy(centroids_.data(), cent, sizeof(float) * num_cluster_ * dim_);
    }

    void centroids_distances(
        const float* query, size_t nprobe, std::vector<AnnCandidate<float>>& candidates
    ) const override {
        std::vector<AnnCandidate<float>> centroid_dist(this->num_cluster_);
        for (PID i = 0; i < num_cluster_; ++i) {
            centroid_dist[i].id = i;
            centroid_dist[i].distance = std::sqrt(euclidean_sqr(query, centroid(i), dim_));
        }
        // Roofline counters: Flat unconditionally evaluates a full L2
        // distance for every centroid; no RaBitQ/ex-bits work.
        last_full_l2_dists = num_cluster_;
        last_fastscan_attempts = 0;
        last_exbits_refines = 0;
        std::partial_sort(
            centroid_dist.begin(),
            centroid_dist.begin() + static_cast<long>(nprobe),
            centroid_dist.end()
        );

        std::memcpy(
            candidates.data(), centroid_dist.data(), sizeof(AnnCandidate<float>) * nprobe
        );
    }

    // for flat initer, we save & load into the ifstream
    void save(std::ofstream& output, const char*) const override {
        output.write(
            reinterpret_cast<const char*>(centroids_.data()),
            static_cast<long>(sizeof(float) * dim_ * num_cluster_)
        );
    }

    void load(std::ifstream& input, const char*) override {
        input.read(
            reinterpret_cast<char*>(centroids_.data()),
            static_cast<long>(sizeof(float) * dim_ * num_cluster_)
        );
    }
};

class HNSWInitializer : public Initializer {
   private:
    int M_ = 16;
    int ef_construction_ = 400;
    hnswlib::HierarchicalNSW<float>* alg_hnsw_ = nullptr;
    hnswlib::L2Space space_;

   public:
    explicit HNSWInitializer(size_t d, size_t k) : Initializer(d, k), space_(d) {
        alg_hnsw_ = new hnswlib::HierarchicalNSW<float>(
            &space_, num_cluster_, M_, ef_construction_
        );
    }

    void add_vectors(const float* cent) override {
        std::cout << "Inserting vectors into hnsw...\n";
        size_t start = 0;
        size_t rows = num_cluster_;
        size_t num_threads = 0;
        parallel_for(start, rows, num_threads, [&](size_t row, size_t /*thread_id*/) {
            alg_hnsw_->addPoint(cent + (row * dim_), row);
        });
        std::cout << "Inserted vectors into hnsw...\n" << std::flush;
    }

    [[nodiscard]] const float* centroid(PID id) const override {
        return reinterpret_cast<const float*>(alg_hnsw_->getDataByInternalId(id));
    }

    void centroids_distances(
        const float* query, size_t nprobe, std::vector<AnnCandidate<float>>& candidates
    ) const override {
        // efSearch scales with the request: 2*nprobe gives the beam the usual
        // 2x headroom over k without imposing a fixed cost at small nprobe.
        alg_hnsw_->setEf(2 * nprobe);
        // Roofline counters: zero hnswlib's thread-local distance counter
        // before the search; after searchKnn it holds the exact number of
        // distance evaluations performed during this call (entry distances +
        // upper-layer descent + base-layer beam search). The callsites that
        // bump it live in hnswalg.h. Each is a full L2 over padded_dim; HNSW
        // does no fastscan/ex-bits work.
        hnswlib::roofline_search_distfunc_calls = 0;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            alg_hnsw_->searchKnn(query, nprobe);
        last_full_l2_dists = hnswlib::roofline_search_distfunc_calls;
        last_fastscan_attempts = 0;
        last_exbits_refines = 0;

        // hnswlib's searchKnn returns a max-heap by distance, so result.top()
        // is the farthest of the k-nearest; popping yields farthest->nearest.
        // The SSD search loop expects candidates sorted nearest-first (matching
        // FlatInitializer), so write in reverse to flip the order.
        const size_t got = result.size();
        for (size_t i = 0; i < got; ++i) {
            const size_t idx = got - 1 - i;
            candidates[idx].distance = std::sqrt(result.top().first);
            candidates[idx].id = result.top().second;
            result.pop();
        }
    }

    // for hnsw initer, we save & load into a separate file by hnswlib
    void save(std::ofstream&, const char* filename) const override {
        std::string hnsw(filename);
        hnsw += ".hnsw";
        alg_hnsw_->saveIndex(hnsw);
    }

    void load(std::ifstream&, const char* filename) override {
        std::string hnsw(filename);
        hnsw += ".hnsw";
        alg_hnsw_->loadIndex(hnsw, &space_, num_cluster_);
    }

    ~HNSWInitializer() override { delete alg_hnsw_; }
};

inline Initializer* create_initializer(
    InitializerType type, size_t dim, size_t num_cluster
) {
    switch (type) {
        case InitializerType::Flat:
            return new FlatInitializer(dim, num_cluster);
        case InitializerType::HNSW:
            return new HNSWInitializer(dim, num_cluster);
        case InitializerType::Auto:
            break;
    }
    return create_initializer(resolve_initializer_type(type, num_cluster), dim, num_cluster);
}
}  // namespace rabitqlib::ivf
