#pragma once

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/ivf/cluster.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/ivf/initializer_irq.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/ssd_utils/concurrent_queue.hpp"
#include "rabitqlib/ssd_utils/linux_aligned_file_reader.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/io_auto.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::ivf_ssd
{
    // Selects the coarse quantizer / initializer the index uses.
    //   Flat / HNSW -> built through the ivf::Initializer factory.
    //   IRQ         -> the embedded 9-bit IVF-RaBitQ initializer
    //                  (ivf_irq::IvfRabitqInitializer).
    enum class CoarseKind : uint8_t
    {
        Flat = 0,
        HNSW = 1,
        IRQ = 2,
    };

    // What one SSD record holds. Chosen when build_invlist runs and saved in
    // <base>, so querying reads it from the index rather than a flag.
    //   ExBits -> the 1-bit code of the dimensions past MEMDIM plus the
    //             extra-precision code (B-1 bits per dimension). Re-ranking
    //             estimates the distance from the codes. The default, and the
    //             configuration of the paper's main evaluation.
    //   Raw    -> the vector exactly as build_invlist read it, float32 or
    //             uint8, zero-padded to padded_dim. Re-ranking computes the
    //             exact distance to the unrotated query. The in-memory prefix
    //             and its bounds are the same as for ExBits, so the scan,
    //             pruning and I/O pipeline are shared unchanged.
    enum class SsdStore : uint8_t
    {
        ExBits = 0,
        Raw = 1,
    };

    // Cluster scan is batched: for each 32-point batch the scan refreshes
    // alpha_distk, scans the batch, then immediately polls/re-ranks/submits.
    // alpha_distk (see pruning_threshold) is the threshold used both for
    // mem-level pruning and for pending-page pruning.

    // Alpha control (drain-only).
    //
    // During the cluster-search loop (probe_idx in [0, nprobe)) alpha is fixed
    // at 1.0 -- the plain threshold, i.e. NO extra pruning -- so every probed
    // cluster contributes its full candidate set. Only the DRAIN phase
    // (probe_idx >= nprobe), where the remaining pending pages are far from the
    // query and unlikely to hold genuine topk, prunes: it uses a single
    // user-specified drain_alpha (set_drain_alpha). drain_alpha == 1.0 disables
    // drain pruning as well (plain threshold everywhere); a smaller drain_alpha
    // prunes the drain tail harder. The drain threshold is
    // pruning_threshold(knns, topk, drain_alpha), recomputed from the current
    // KNN buffer at each drain submit.

    constexpr size_t kInitialPageSlots = kMaxReadsInFlight + 1;
    constexpr size_t kEmptySsdGroupSize = 8;

    // ---------------------------------------------------------------------
    // Split-storage on-disk format.
    //
    // The storage separates the coarse-quantizer-independent inverted list
    // from the swappable coarse quantizer, so that switching coarse quantizer
    // (Flat / HNSW / IRQ) with everything else unchanged does NOT rebuild or
    // re-store the (large) inverted list. Three artifacts:
    //
    //   <base>      : magic+version, metadata, cluster sizes, rotator, the
    //                 rotated first-level centroids, mem_batch_data and ids.
    //                 Built once by build_invlist. Holds NO coarse quantizer.
    //   <base>.ssd  : the SSD inverted list (page-packed layout).
    //   <coarse>    : magic+version, a header identifying the base it pairs
    //                 with, then the initializer payload. Built per coarse
    //                 quantizer by build_coarse; tiny (operates on the C
    //                 centroids only). IRQ/HNSW also write a <coarse>.irq /
    //                 <coarse>.hnsw sidecar via the initializer's own save().
    //
    // Querying loads one <base> (+ <base>.ssd) plus one <coarse>.
    //
    // <base> versions. Version 1 is the ExBits layout, and every ExBits index
    // is still written as version 1, so its bytes never change. Version 2
    // inserts two uint32 fields after the metric -- the SsdStore and the
    // VecElemType of raw records -- and is written only for the Raw store.
    // Version 3 has the version-2 header and 64-bit point ids; only the
    // RABITQ_PID64 build writes it, and each build reads only its own ids.
    constexpr uint32_t kBaseMagic = 0x52424253U;   // 'RBBS' (rabitq base split)
    constexpr uint32_t kBaseVersion = 1U;
    constexpr uint32_t kBaseVersionWithStore = 2U;
    constexpr uint32_t kBaseVersionPid64 = 3U;
    constexpr uint32_t kCoarseMagic = 0x52424351U; // 'RBCQ' (rabitq coarse q)
    constexpr uint32_t kCoarseVersion = 1U;

    // 64-bit FNV-1a over a rotated-centroids block. Stored in <coarse> at
    // build time and re-checked at query time, so a coarse quantizer can only
    // be paired with the exact base (same rotator + same centroids) it was
    // built from -- a mismatched pairing would silently produce wrong results.
    inline uint64_t centroids_fingerprint(const std::vector<float> &rc)
    {
        uint64_t h = 1469598103934665603ULL;
        const auto *bytes = reinterpret_cast<const unsigned char *>(rc.data());
        const size_t n = rc.size() * sizeof(float);
        for (size_t i = 0; i < n; ++i)
        {
            h ^= static_cast<uint64_t>(bytes[i]);
            h *= 1099511628211ULL;
        }
        return h;
    }

    // Per-thread cap on in-flight SSD reads. Every thread runs with at most
    // this many outstanding reads, independent of the thread count, so thread
    // count and queue depth are independent knobs: the aggregate in-flight
    // depth is qd * nthreads.
    //
    // Overridable at run time through the RABITQ_QD_PER_THREAD environment
    // variable (parsed once, then cached); when unset the compiled default
    // below applies. The best depth is device- and thread-count-specific.
    constexpr size_t kDefaultQdPerThread = 16;

    // The requested depth, UNCLAMPED. init_buffers compares it against the
    // effective cap below and reports when a request has been clamped.
    inline size_t requested_qd_per_thread()
    {
        static const size_t v = []() -> size_t {
            if (const char *e = std::getenv("RABITQ_QD_PER_THREAD"))
            {
                const long parsed = std::atol(e);
                if (parsed > 0) { return static_cast<size_t>(parsed); }
            }
            return kDefaultQdPerThread;
        }();
        return v;
    }

    // The effective per-thread in-flight cap: the request clamped to the
    // per-thread request-slot / registered-buffer count (a deeper queue would
    // have nowhere to put its reads).
    inline size_t qd_per_thread()
    {
        const size_t qd = requested_qd_per_thread();
        return (qd > kMaxReadsInFlight) ? kMaxReadsInFlight : qd;
    }

    // ---------------------------------------------------------------------
    // The pruning threshold (alpha_distk).
    //
    // The plain threshold is knns.top_dist(), the distance of the
    // topk-th (worst) neighbour currently held; it decides (a) which scanned
    // points become page candidates and (b) which pending pages survive
    // submit-time pruning. With a small memdim the mem-level lower bound is
    // loose, so that threshold admits many marginal candidates -- a longer IO
    // queue and more re-ranks, both of which add latency.
    //
    // alpha_distk instead thresholds on the alpha-quantile of the KNN buffer:
    // with topk=100 and alpha=0.9, the 90th-nearest distance. Since alpha < 1
    // this is <= top_dist(), so pruning is tighter and fewer candidates / IOs
    // are produced (trading a little recall for latency). The KNN buffer still
    // holds the true topk and the final results are still the true topk;
    // alpha_distk is only a derived pruning threshold. The re-rank stage
    // (rerank_page) deliberately keeps using the true top_dist(), so no genuine
    // topk result is dropped once its distance has been computed.
    //
    // WHERE ALPHA IS APPLIED: the cluster-search loop always calls this with
    // alpha=1.0 (so it returns top_dist() = the plain threshold, no extra
    // pruning). Only the DRAIN phase calls it with the user's drain_alpha_
    // (< 1), pruning pending pages whose lower bound exceeds the
    // drain_alpha-quantile distance -- cutting tail-end SSD IOs / reranks that
    // would mostly miss the topk.
    // See the "Alpha control (drain-only)" note above.
    //
    // Until the KNN buffer is full the alpha-quantile element does not exist,
    // so this returns top_dist() (== +inf pre-full) and no pruning occurs.
    // alpha >= 1 likewise degenerates to the plain threshold.
    inline float pruning_threshold(const buffer::SearchBuffer<float> &knns,
                                   size_t topk, float alpha)
    {
        if (alpha >= 1.0F || !knns.is_full())
        {
            return knns.top_dist();
        }
        // 1-based rank of the alpha-quantile element, rounded to nearest and
        // clamped to [1, topk]; e.g. topk=100, alpha=0.9 -> rank 90.
        long rank = std::lround(alpha * static_cast<double>(topk));
        if (rank < 1)
        {
            rank = 1;
        }
        if (rank > static_cast<long>(topk))
        {
            rank = static_cast<long>(topk);
        }
        // knns.data() is sorted ascending by distance; convert to 0-based.
        return knns.data()[static_cast<size_t>(rank) - 1].distance;
    }

    struct SearchStats
    {
        // Total wall time of search_one for this query (us).
        double total_us = 0.0;
        // SSD read accounting, all counted at the single submit site:
        //   issued_read_reqs  : # read REQUESTS submitted
        //   issued_read_pages : 4 KiB pages covered (req.len / SECTOR_LEN)
        //   issued_read_bytes : bytes requested (exact)
        // pages != requests when one request spans multiple contiguous pages.
        uint64_t issued_read_reqs = 0;
        uint64_t issued_read_pages = 0;
        uint64_t issued_read_bytes = 0;
        uint64_t num_points_probed = 0;
        // Scan-phase bookkeeping-analysis counters (per query):
        //   scan_candidates_inserted : points that passed the mem-prune and
        //                              got a Candidate written.
        //   scan_pages_acquired      : page-candidate slots acquired in scan.
        //   peak_page_slots          : page_slots pool high-water mark.
        //
        // Main-pipeline rerank funnel (one completed page IO expands into many
        // record evaluations, only some of which pay the boost kernel):
        //   main_pages_reranked      : # calls to rerank_page,
        //                              i.e. # SSD page-IOs that completed and
        //                              were handed to the main-pipeline rerank
        //                              stage. A per-PAGE count (drained pages
        //                              included) -- NOT the number of distance
        //                              evaluations.
        //   main_records_examined    : sum of page_cand.candidate_num over
        //                              those pages = # records the rerank loop
        //                              iterated over.
        //   main_boost_dists         : # actual split_distance_boosting calls
        //                              = records that survived the cheap
        //                              mem_low_dist vs threshold early-exit.
        //                              THIS is the real rerank compute (each
        //                              costs 2*ex_bits*padded_dim + 6 flop-eq).
        //                              0 on the ex_bits_==0 && ssd_dim_==0 path
        //                              (no boosting: mem_est_dist used
        //                              directly). With the Raw store it counts
        //                              exact distance evaluations instead.
        //   main_knn_inserts         : # knns.insert calls (records that also
        //                              passed the final
        //                              full_dist <= threshold).
        // Useful ratios: boost_dists/records_examined = mem lower-bound prune
        // pass rate; boost_dists/pages_reranked = boosts per page IO.
        uint64_t scan_candidates_inserted = 0;
        uint64_t scan_pages_acquired = 0;
        uint64_t peak_page_slots = 0;
        uint64_t main_pages_reranked = 0;
        uint64_t main_records_examined = 0;
        uint64_t main_boost_dists = 0;
        uint64_t main_knn_inserts = 0;
        // Roofline buckets: centroid coarse-search work performed during
        // Initializer::centroids_distances. Each bucket maps to a different
        // flop-eq cost in the roofline work model (see work_model.hpp).
        //   full_l2_dists      : full L2 (2*padded_dim flop-eq each).
        //                        Flat / HNSW write here; IRQ writes its
        //                        inner Flat scan here.
        //   fastscan_attempts  : 1-bit RaBitQ fastscan + factor correction
        //                        (2*padded_dim + 8 flop-eq each). IRQ-only.
        //   exbits_refines     : ex-bits boosting (2*ex_bits*padded_dim + 6
        //                        flop-eq each). IRQ-only; ex_bits = 8.
        uint64_t centroid_full_l2_dists = 0;
        uint64_t centroid_fastscan_attempts = 0;
        uint64_t centroid_exbits_refines = 0;
        // Wall time spent inside the coarse-quantizer step
        // (Initializer::centroids_distances) for this query. Used to estimate
        // QPS-without-coarse-quantizer in offline analysis.
        double coarse_us = 0.0;
        // Wall time spent in the post-nprobe-loop drain phase for this query:
        // the tail where the main loop has stopped issuing cluster scans and
        // the pipeline only polls/refines/submits until io_queue is empty.
        // Because the main loop's poll is a non-blocking peek, every
        // SSD read that compute could NOT hide funnels into this phase. So a
        // large drain_us (relative to the SSD-pipeline wall time) means the
        // operating point is IO-bound; a near-zero drain_us means it is
        // compute-bound (IO fully overlapped behind cluster-scan compute).
        double drain_us = 0.0;

        void reset()
        {
            total_us = 0.0;
            issued_read_reqs = 0;
            issued_read_pages = 0;
            issued_read_bytes = 0;
            num_points_probed = 0;
            scan_candidates_inserted = 0;
            scan_pages_acquired = 0;
            peak_page_slots = 0;
            main_pages_reranked = 0;
            main_records_examined = 0;
            main_boost_dists = 0;
            main_knn_inserts = 0;
            centroid_full_l2_dists = 0;
            centroid_fastscan_attempts = 0;
            centroid_exbits_refines = 0;
            coarse_us = 0.0;
            drain_us = 0.0;
        }
    };

    // Global 4 KiB page index into <base>.ssd. int covers 2^31 - 1 pages
    // (8 TiB), and check_ssd_pages refuses a larger layout at build and load;
    // the 64-bit point-id build, meant for larger indexes, widens it.
#if defined(RABITQ_PID64)
    using PageIdx = int64_t;
#else
    using PageIdx = int;
#endif

    struct Candidate
    {
        PID pid;
        int in_page_idx;
        float mem_low_dist;
        float mem_est_dist;
        float mem_ip_x0_qr;
    };

    struct PageCandidate
    {
        float page_lower_bound = 0.0F;
        uint32_t probe_idx = static_cast<uint32_t>(-1);
        PageIdx page_id = -1; // Global 4 KiB page index.
        int req_slot = -1;

        uint16_t candidate_num = 0;
    };

    struct QueryBuffer
    {
        float *rotated_query = nullptr;
        // Raw store only (nullptr otherwise): the query as given, zero-padded
        // to padded_dim. Raw records are unrotated, so they are compared with
        // the unrotated query; the rotation is orthonormal, so the in-memory
        // bounds computed in rotated space still bound these distances.
        float *raw_query = nullptr;
        char *read_buffers = nullptr;

        std::vector<float> est_distance;
        std::vector<float> low_distance;
        std::vector<float> ip_x0_qr;
        std::vector<int32_t> accu_arr;
        uint64_t read_bytes = 0;
        uint16_t page_candidate_capacity = 0;
        std::vector<AnnCandidate<float>> centroid_dist;

        std::array<IORequest, kMaxReadsInFlight> requests{};
        std::array<SlotID, kMaxReadsInFlight> req_slot_owner{};
        std::vector<SlotID> completed_slots;

        std::vector<PageCandidate> page_slots;
        std::vector<Candidate> candidate_storage;
        std::vector<SlotID> free_page_slots;
        std::vector<int> free_req_slots;

        void init_page_candidate_storage(size_t candidate_capacity,
                                         size_t initial_slots);
        void ensure_page_slot_capacity(size_t target_slots);
        void reset();
        [[nodiscard]] char *read_buffer(int req_slot) const;
        [[nodiscard]] Candidate *slot_candidates(SlotID slot_id);
        [[nodiscard]] const Candidate *slot_candidates(SlotID slot_id) const;
    };

    class PendingPages
    {
    public:
        struct Entry
        {
            SlotID slot_id = kSlotMax;
            float key = 0.0F;
        };

        void reset();
        [[nodiscard]] bool empty() const;
        [[nodiscard]] size_t size() const;
        void insert(SlotID slot_id, float key);
        [[nodiscard]] SlotID front_slot() const;
        void pop_front();
        void prune_greater(float threshold, std::vector<SlotID> &pruned_slots);

    private:
        void compact_if_needed();

        std::vector<Entry> entries_;
        size_t head_ = 0;
    };


    class PageReadQueue
    {
    public:
        // Bind the query's buffer, reader, IO context, empty-SSD flag and stats until the next reset().
        void reset(QueryBuffer *query_buf, LinuxAlignedFileReader *file_reader,
                   void *ctx, bool empty_ssd, SearchStats *stats);
        [[nodiscard]] size_t inflight_count() const { return inflight_count_; }
        [[nodiscard]] bool reads_done() const
        {
            return pending_.empty() && inflight_count_ == 0 && ready_completed_.empty();
        }
        SlotID acquire_page_slot();

        void enqueue_page(SlotID slot_id);

        int submit_reads(size_t budget, float threshold);

        void poll_reads();

        [[nodiscard]] const char *read_data(SlotID slot_id) const;
        void release_read(SlotID slot_id);

    private:
        static constexpr int kInvalidReqSlot = -1;
        static constexpr uint8_t kStateFree = 0;
        static constexpr uint8_t kStateReserved = 1;
        static constexpr uint8_t kStatePending = 2;
        static constexpr uint8_t kStateInflight = 3;
        static constexpr uint8_t kStateCompleted = 4;

        void ensure_slot_capacity(size_t target_slots);
        void release_page_slot(SlotID slot_id);
        void prune_pending_above(float threshold);
        void prepare_read(SlotID slot_id, int req_slot);

        QueryBuffer *query_buf_ = nullptr;
        LinuxAlignedFileReader *file_reader_ = nullptr;
        void *ctx_ = nullptr;
        bool empty_ssd_ = false;
        SearchStats *stats_ = nullptr;
        PendingPages pending_;
        std::vector<uint8_t> slot_state_;
        std::vector<int> slot_req_slot_;
        std::vector<SlotID> ready_completed_;
        std::vector<SlotID> pruned_slots_;
        size_t inflight_count_ = 0;
    };

    class IvfSsdIndex
    {
    private:
        ivf::Initializer *initer_ = nullptr;
        char *mem_batch_data_ = nullptr;
        PID *ids_ = nullptr;
        size_t num_ = 0;
        size_t dim_ = 0;
        size_t padded_dim_ = 0;
        size_t mem_dim_ = 0;
        size_t requested_mem_dim_ = 0;
        size_t num_cluster_ = 0;
        size_t ex_bits_ = 0;
        // Coarse-quantizer selection. coarse_kind_ decides which initializer
        // is built by build_initializer(); initializer_type_ is only
        // meaningful for Flat / HNSW (fed to ivf::create_initializer).
        CoarseKind coarse_kind_ = CoarseKind::Flat;
        ivf::InitializerType initializer_type_ = ivf::InitializerType::Auto;
        // IRQ-only inputs, set by build_coarse_quantizer and consumed inside
        // ivf_irq::IvfRabitqInitializer. Unused for Flat / HNSW.
        std::string inner_centroids_path_;
        std::string inner_cids_path_;
        RotatorType type_;
        Rotator<float> *rotator_ = nullptr;
        std::vector<ivf::Cluster> cluster_lst_;
        // Rotated (padded_dim) first-level centroids -- the shared block the
        // coarse quantizer is built from. Populated by construct_invlist (for
        // save_base) and read back by load_base / load_base_meta. Keeping it
        // in <base> means build_coarse needs no separate centroids file and
        // cannot pick up a centroid set inconsistent with the inverted list.
        std::vector<float> rotated_centroids_;
        MetricType metric_type_ = rabitqlib::METRIC_L2;
        float (*ip_func_)(const float *, const uint8_t *, size_t) = nullptr;

        // What the SSD records hold (see SsdStore), and for the Raw store the
        // element type of a record. Set by set_ssd_store at build time and
        // restored from <base> by load_base / load_base_meta.
        SsdStore ssd_store_ = SsdStore::ExBits;
        VecElemType raw_elem_ = VecElemType::F32;

        bool empty_ssd_ = false;

        // Drain-phase alpha (see the "Alpha control (drain-only)" note above).
        // Cluster search always runs at alpha=1.0; only the drain phase prunes,
        // using this single value. Default 1.0 = no drain pruning (plain
        // threshold everywhere); set < 1.0 to prune the drain tail harder.
        float drain_alpha_ = 1.0F;

        [[nodiscard]] size_t ids_bytes() const { return sizeof(PID) * num_; }
        [[nodiscard]] size_t mem_batch_data_bytes(const std::vector<size_t> &cluster_sizes) const
        {
            assert(cluster_sizes.size() == num_cluster_);
            size_t total_blocks = 0;
            for (size_t i = 0; i < num_cluster_; ++i)
            {
                total_blocks += div_round_up(cluster_sizes[i], fastscan::kBatchSize);
            }
            return total_blocks * batch_bytes_;
        }

        // Split-storage build steps. allocate_invlist_memory allocates the
        // coarse-quantizer-independent inverted-list buffers (mem_batch_data_,
        // ids_) plus ip_func_; build_initializer constructs initer_ for the
        // current coarse_kind_ (Flat / HNSW / IRQ), consuming the IRQ-only
        // path members when coarse_kind_ is IRQ.
        void allocate_invlist_memory(const std::vector<size_t> &cluster_sizes);
        void build_initializer();
        void free_memory()
        {
            delete initer_;
            initer_ = nullptr;
            std::free(mem_batch_data_);
            mem_batch_data_ = nullptr;
            std::free(ids_);
            ids_ = nullptr;
        }

        std::shared_ptr<LinuxAlignedFileReader> file_reader_;
        std::string ssd_index_file_;

        size_t ssd_dim_ = 0;
        size_t record_bytes_ = 0;
        size_t records_per_read_ = 0;
        size_t pages_per_read_ = 0;
        size_t read_bytes_ = 0;
        size_t batch_bytes_ = 0;

        // First 4 KiB page of each cluster in <base>.ssd.
        std::vector<size_t> cluster_first_page_;

        void init_layout_metadata();
        // One log line with the SSD store and the page layout it implies.
        void log_ssd_layout() const;
        // Write one Raw record: the dim_ input values in raw_elem_, followed by
        // zeros up to padded_dim (dst points into a zeroed page buffer).
        void write_raw_record(char *dst, const float *src) const;
        // Validate the <base> version and, for version 2, read the SSD store
        // fields that follow the metric. Version 1 means ExBits.
        void read_base_store_fields(std::istream &input, uint32_t version,
                                    const std::string &base_index_file);
        // Gives <base>.ssd its full size before the build writes it, unless
        // its file system lacks room for it plus reserve_bytes.
        void preallocate_ssd_file(uint64_t reserve_bytes);
        // Exits when the clusters need more 4 KiB SSD pages than PageIdx can
        // address (2^31 - 1 in the default build), naming the 64-bit build.
        void check_ssd_pages(const std::vector<size_t> &cluster_sizes) const;
        // Exits when the SSD records would hold only the 1-bit code beyond
        // memdim, with no extra-precision code: a layout search cannot re-rank.
        void check_ssd_layout_searchable() const;
        void init_clusters(const std::vector<size_t> &cluster_sizes);
        void quantize_and_write_clusterssd(ivf::Cluster &cp,
                                           const std::vector<PID> &ids,
                                           const float *data,
                                           const float *cur_centroid,
                                           float *rotated_centroid,
                                           const quant::RabitqConfig &config,
                                           size_t cluster_offset);
        // Shared quantize+write core for one cluster. Decouples "the global pid
        // stored into cp.ids()" (pids) from "where the i-th point's dim_-length
        // source vector lives" (fetch_row(i)): the full-DRAM build fetches
        // data + pids[i]*dim_, the streaming build fetches a gathered group /
        // solo buffer by local row. batch_parallel turns on intra-cluster
        // OpenMP (rotate / quantize / page-write) for the big-cluster path
        // (one cluster processed alone, must use all cores); it MUST stay false
        // when running inside an outer parallel-for over clusters.
        template <class FetchRow>
        void quantize_and_write_cluster_core(
            ivf::Cluster &cp, const PID *pids, size_t num_points,
            FetchRow fetch_row, const float *cur_centroid,
            float *rotated_centroid, const quant::RabitqConfig &config,
            size_t cluster_offset, bool batch_parallel);

        // Stable order of a cluster's points by centroid distance (ascending),
        // tie-broken by pid. The full-DRAM build fetches from the in-RAM
        // dataset; the streaming build fetches from its reorder scratch.
        // Both build paths use this ordering before quantization.
        template <class FetchRow>
        std::vector<size_t> compute_centroid_order(
            const PID *pids, size_t num_points, FetchRow fetch_row,
            const float *cur_centroid) const;

        ConcurrentQueue<QueryBuffer *> thread_query_buffer_pool_;
        std::vector<QueryBuffer *> thread_data_bufs_;
        ConcurrentQueue<PageReadQueue *> thread_ioqueue_buffer_pool_;
        std::vector<PageReadQueue *> thread_ioqueue_bufs_;

        void init_query_buf(QueryBuffer &buf);
        QueryBuffer *pop_query_buf();
        void push_query_buf(QueryBuffer *buf);
        PageReadQueue *pop_ioqueue_buf();
        void push_ioqueue_buf(PageReadQueue *buf);

        // Per-cluster scan state shared across batches of a single cluster.
        // Holds the page-tracking cursor (a page may span several 32-point
        // batches and stays "open" here until it closes) plus the per-cluster
        // stat accumulators. It has to outlive a single batch because the
        // search interleaves poll / re-rank / submit between scan_batch
        // calls, and the open page must survive that.
        struct ClusterScanState
        {
            // Page-id tracking cursor (spans the whole cluster).
            PageIdx last_page = -1;
            SlotID cur_page_slot_id = kSlotMax;
            int cur_inner_pid = -1;
            PageIdx cur_page = 0;
            int in_page_idx = 0;
            int remaining_in_page = 0;
            char *mem_data = nullptr;
            const PID *ids = nullptr;
            // Per-cluster stat accumulators, flushed into SearchStats by
            // finish_cluster.
            size_t scan_cand_inserted_local = 0;
            size_t scan_pages_acq_local = 0;
        };

        // Initialise state at the start of a cluster scan.
        void init_cluster_scan_state(const ivf::Cluster &cur_cluster, CID cid,
                                     ClusterScanState &state);

        // Scan one 32-point batch (index `iter`) of the current cluster.
        // Page-tracking state in `state` carries over between batches; a page
        // that spans batches stays open and is only enqueued when it closes.
        void scan_batch(
            const ivf::Cluster &cur_cluster,
            const SplitBatchQuery_nprobe<float> &q_obj,
            size_t probe_idx,
            float threshold,
            QueryBuffer *query_buf,
            PageReadQueue *io_queue,
            bool use_hacc,
            ClusterScanState &state,
            size_t iter);

        // Enqueue the final open page (if any) and flush per-cluster stat
        // accumulators into SearchStats.
        void finish_cluster(
            PageReadQueue *io_queue,
            SearchStats *stats,
            ClusterScanState &state);

        // Per-query worker used by batch_search.
        // Search steps for one query:
        //   for each of nprobe clusters in centroid-distance order:
        //     A. scan the cluster (populates io_queue.pending), pruning
        //        mem-level points whose lower bound exceeds alpha_distk
        //     B. poll any completed IOs and re-rank them (re-rank still uses
        //        the true threshold = knns.top_dist(), refreshing alpha_distk)
        //     C. submit IOs from io_queue.pending up to qd_cap in-flight
        //        (submit_reads internally prunes pending whose page_lower_bound
        //        >= alpha_distk). qd_cap comes from qd_per_thread().
        //   drain: while io_queue is non-empty, repeat (poll, re-rank, submit)
        //          with the same qd_cap, until in-flight = 0 and pending is
        //          exhausted (either consumed or pruned).
        void search_one(const float *query, size_t topk, size_t nprobe,
                        PID *__restrict__ results, bool use_hacc,
                        SearchStats *stats);

        void estimate_batch(
            const char *mem_batch_data,
            const SplitBatchQuery_nprobe<float> &q_obj,
            const size_t probe_idx,
            float *est_distance,
            float *low_distance,
            float *ip_x0_qr,
            QueryBuffer *query_buf,
            bool use_hacc);

        void rerank_page(
            SlotID slot_id, QueryBuffer *query_buf,
            PageReadQueue *io_queue,
            const SplitBatchQuery_nprobe<float> &q_obj,
            buffer::SearchBuffer<float> &knns,
            SearchStats *stats = nullptr);

        void compute_probe_terms(
            SplitBatchQuery_nprobe<float> &q_obj, size_t probe_idx, CID cid,
            float dist, float qnorm, const QueryBuffer *query_buf);
        void harvest_reads(
            QueryBuffer *query_buf, PageReadQueue *io_queue,
            const SplitBatchQuery_nprobe<float> &q_obj,
            buffer::SearchBuffer<float> &knns, SearchStats *stats);

    public:
        explicit IvfSsdIndex() = default;
        // num: number of indexed points, from 0 through the PID range.
        // dim: positive input width supported by the selected rotator.
        // num_cluster: positive number of clusters within the CID range.
        // total_bits: quantization bits per dimension in [1, 9].
        // ssd_index_file: SSD file path, opened for writing and created if
        // missing.
        // metric_type: METRIC_L2 or METRIC_IP, default METRIC_L2.
        // type: MatrixRotator or FhtKacRotator, default FhtKacRotator.
        // mem_dim: 0 for padded width, or a multiple of 64 in
        // [64, padded width].
        explicit IvfSsdIndex(size_t num, size_t dim, size_t num_cluster,
                             size_t total_bits, std::string ssd_index_file,
                             MetricType metric_type = rabitqlib::METRIC_L2,
                             RotatorType type = RotatorType::FhtKacRotator,
                             size_t mem_dim = 0);
        ~IvfSsdIndex();

        [[nodiscard]] size_t num_clusters() const { return num_cluster_; }
        [[nodiscard]] size_t num_points() const { return num_; }
        [[nodiscard]] size_t dim() const { return dim_; }

        // ---- Split-storage build / load API --------------------------------
        //
        // Build (first time): construct_invlist + save_base once, then
        // build_coarse_quantizer + save_coarse once per coarse quantizer.
        // Query: load_base then load_coarse.

        // Inverted-list (base) build. Orders each cluster by centroid distance,
        // quantizes the data, writes <base>.ssd, and caches the rotated
        // centroids (rotated_centroids_) for save_base(). Builds no coarse
        // quantizer; the centroids are only stored, not indexed.
        void construct_invlist(const float *data, const float *centroids,
                               const CID *assignments, bool faster);
        // Memory-budgeted streaming variant of construct_invlist. Produces the
        // same base index + SSD file as construct_invlist (byte-for-byte when
        // RABITQ_ROTATOR_SEED pins the shared rotation), but never
        // holds the whole dataset in DRAM: it streams the data from
        // <data_file> (.fbin/.fvecs float32, .u8bin/.bvecs uint8) through a
        // cluster-partitioned reorder pass into a scratch subfolder (next to
        // <data_file> unless RABITQ_BUILD_SCRATCH_DIR says otherwise), then
        // quantizes group-by-group within mem_budget_bytes (total RSS ceiling).
        // centroids (C*dim) and assignments (N) are still passed in DRAM (small
        // relative to the dataset). See the .tpp definition for the P0-P3
        // pipeline and the budget math. A nonzero prefix_rows indexes only the
        // first prefix_rows rows of <data_file>.
        void construct_invlist_streaming(const std::string &data_file,
                                         const float *centroids,
                                         const CID *assignments, bool faster,
                                         size_t mem_budget_bytes,
                                         size_t prefix_rows = 0);
        // Write the coarse-quantizer-independent base index to <base>.
        void save_base(const std::string &base_index_file);

        // Read just the header of <base> (metadata + rotator + rotated
        // centroids). Used by build_coarse, which needs neither mem_batch_data
        // nor the SSD file. Leaves the index ready for build_coarse_quantizer.
        void load_base_meta(const std::string &base_index_file);
        // Build a coarse quantizer over the loaded rotated centroids. Call
        // after load_base_meta(). For Flat/HNSW the IRQ-only args are ignored;
        // for IRQ initializer_type is ignored.
        void build_coarse_quantizer(
            CoarseKind coarse_kind,
            ivf::InitializerType initializer_type = ivf::InitializerType::Auto,
            const std::string &inner_centroids_path = std::string(),
            const std::string &inner_cids_path = std::string());
        // Write the coarse quantizer to <coarse> (small file; IRQ/HNSW also
        // emit a <coarse>.irq / <coarse>.hnsw sidecar).
        void save_coarse(const std::string &coarse_index_file);

        // Full base load for querying: metadata + rotator + rotated centroids
        // + mem_batch_data + ids, and opens <base>.ssd. No coarse quantizer.
        void load_base(const std::string &base_index_file,
                       const std::string &ssd_index_file,
                       size_t mem_dim = 0);
        // Load a coarse quantizer built by build_coarse. Must be called AFTER
        // load_base(); validates (dims + centroid fingerprint) that the
        // coarse file pairs with this base.
        void load_coarse(const std::string &coarse_index_file);

        void init_buffers(uint64_t nthreads);
        void destroy_buffers();

        // Drain-phase alpha. Call once after load(), before batch_search.
        // Clamped to [0,1]; 1.0 = no drain pruning (plain threshold
        // everywhere), smaller prunes the drain tail harder. Cluster search
        // is always 1.0.
        void set_drain_alpha(float drain_alpha)
        {
            if (drain_alpha < 0.0F) drain_alpha = 0.0F;
            if (drain_alpha > 1.0F) drain_alpha = 1.0F;
            drain_alpha_ = drain_alpha;
        }

        // Select what the SSD records hold (see SsdStore). Call at build time,
        // after construction and before construct_invlist*. For the Raw store
        // raw_elem is the element type of the input vectors, which the records
        // keep; Raw has no extra-precision code, so it also sets ex_bits to 0.
        // Querying needs no call: load_base restores the store from <base>.
        void set_ssd_store(SsdStore store, VecElemType raw_elem);
        [[nodiscard]] SsdStore ssd_store() const { return ssd_store_; }
        [[nodiscard]] VecElemType raw_elem() const { return raw_elem_; }
        // "exbits" or "raw", the spelling build_invlist's store= option takes.
        [[nodiscard]] const char *ssd_store_name() const
        {
            return ssd_store_ == SsdStore::Raw ? "raw" : "exbits";
        }
        // One line for logs, e.g. "exbits (B=9)" or "raw (uint8 records)".
        [[nodiscard]] std::string ssd_store_description() const
        {
            if (ssd_store_ == SsdStore::Raw)
            {
                return std::string("raw (") + elem_type_name(raw_elem_) + " records)";
            }
            return "exbits (B=" + std::to_string(ex_bits_ + 1) + ")";
        }

        void batch_search(const float *queries, size_t nqueries, size_t topk,
                          size_t nprobe, PID *__restrict__ results,
                          bool use_hacc = false,
                          SearchStats *stats = nullptr);
    };
} // namespace rabitqlib::ivf_ssd

#include "rabitqlib/index/ivf/ivf_ssd.tpp"
