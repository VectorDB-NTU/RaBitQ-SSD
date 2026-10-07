#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <unordered_map>

#include "rabitqlib/index/ivf/ivf_ssd_boundary_qd_ms_common.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/streaming_reader.hpp"

namespace rabitqlib::ivf_ssd_boundary_qd_ms_detail
{
    inline void QueryBuffer::init_page_candidate_storage(
        size_t candidate_capacity, size_t initial_slots)
    {
        assert(candidate_capacity > 0);
        page_candidate_capacity = static_cast<uint16_t>(candidate_capacity);
        page_slots.clear();
        candidate_storage.clear();
        ensure_page_slot_capacity(initial_slots);
    }

    inline void QueryBuffer::ensure_page_slot_capacity(size_t target_slots)
    {
        if (page_slots.size() >= target_slots)
        {
            return;
        }
        assert(page_candidate_capacity > 0);
        const size_t old_size = page_slots.size();
        page_slots.resize(target_slots);
        candidate_storage.resize(target_slots *
                                 static_cast<size_t>(page_candidate_capacity));
        free_page_slots.reserve(target_slots);
        for (size_t i = target_slots; i > old_size; --i)
        {
            free_page_slots.push_back(static_cast<SlotID>(i - 1));
        }
    }

    inline void QueryBuffer::reset()
    {
        free_page_slots.clear();
        free_page_slots.reserve(page_slots.size());
        for (size_t i = page_slots.size(); i > 0; --i)
        {
            const SlotID slot_id = static_cast<SlotID>(i - 1);
            PageCandidates &slot = page_slots[slot_id];
            slot.page_lower_bound = 0.0F;
            slot.probe_idx = static_cast<CID>(-1);
            slot.page_id = -1;
            slot.req_slot = -1;
            slot.candidate_num = 0;
            free_page_slots.push_back(slot_id);
        }

        free_req_slots.clear();
        free_req_slots.reserve(N_REQ_BUF);
        completed_slots.clear();
        for (int req_slot = static_cast<int>(N_REQ_BUF); req_slot > 0; --req_slot)
        {
            const int slot = req_slot - 1;
            reqs[slot] = IORequest();
            req_slot_owner[slot] = kSlotMax;
            free_req_slots.push_back(slot);
        }
    }

    inline char *QueryBuffer::request_buffer(int req_slot) const
    {
        return sector_scratch + static_cast<size_t>(req_slot) * request_slot_bytes;
    }

    inline SingleCandidate *QueryBuffer::slot_candidates(SlotID slot_id)
    {
        assert(slot_id < page_slots.size());
        return candidate_storage.data() +
               static_cast<size_t>(slot_id) *
                   static_cast<size_t>(page_candidate_capacity);
    }

    inline const SingleCandidate *QueryBuffer::slot_candidates(SlotID slot_id) const
    {
        assert(slot_id < page_slots.size());
        return candidate_storage.data() +
               static_cast<size_t>(slot_id) *
                   static_cast<size_t>(page_candidate_capacity);
    }

    inline void ArrayPendingSet::reset()
    {
        entries_.clear();
        head_ = 0;
    }

    inline void ArrayPendingSet::on_slot_grow(size_t)
    {
    }

    inline bool ArrayPendingSet::empty() const
    {
        return size() == 0;
    }

    inline size_t ArrayPendingSet::size() const
    {
        return entries_.size() - head_;
    }

    inline void ArrayPendingSet::insert(SlotID slot_id, float key)
    {
        compact_if_needed();
        const auto begin_it = entries_.begin() + static_cast<ptrdiff_t>(head_);
        const auto it = std::lower_bound(
            begin_it, entries_.end(), key,
            [](const Entry &entry, float value)
            { return entry.key < value; });
        entries_.insert(it, Entry{slot_id, key});
    }

    inline SlotID ArrayPendingSet::front_slot() const
    {
        assert(!empty());
        return entries_[head_].slot_id;
    }

    inline void ArrayPendingSet::pop_front()
    {
        assert(!empty());
        ++head_;
        compact_if_needed();
    }

    inline void ArrayPendingSet::prune_greater(float distk,
                                               std::vector<SlotID> &pruned_slots)
    {
        const auto begin_it = entries_.begin() + static_cast<ptrdiff_t>(head_);
        const auto it = std::upper_bound(
            begin_it, entries_.end(), distk,
            [](float value, const Entry &entry)
            { return value < entry.key; });
        for (auto cur = it; cur != entries_.end(); ++cur)
        {
            pruned_slots.push_back(cur->slot_id);
        }
        entries_.erase(it, entries_.end());
        if (head_ == entries_.size())
        {
            entries_.clear();
            head_ = 0;
            return;
        }
        compact_if_needed();
    }

    inline void ArrayPendingSet::compact_if_needed()
    {
        if (head_ == 0)
        {
            return;
        }
        if (head_ >= 256 && head_ * 2 >= entries_.size())
        {
            entries_.erase(entries_.begin(),
                           entries_.begin() + static_cast<ptrdiff_t>(head_));
            head_ = 0;
        }
    }


    template <class PendingSet>
    inline void PageCandidate_IO_Queue<PendingSet>::ensure_slot_capacity(
        size_t target_slots)
    {
        if (slot_state_.size() >= target_slots)
        {
            return;
        }
        slot_state_.resize(target_slots, kStateFree);
        slot_req_slot_.resize(target_slots, kInvalidReqSlot);
        pending_.on_slot_grow(target_slots);
    }

    template <class PendingSet>
    inline void PageCandidate_IO_Queue<PendingSet>::reset(QueryBuffer *query_buf)
    {
        pending_.reset();
        ready_completed_.clear();
        inflight_count_ = 0;
        ensure_slot_capacity(query_buf->page_slots.size());
        std::fill(slot_state_.begin(), slot_state_.end(), kStateFree);
        std::fill(slot_req_slot_.begin(), slot_req_slot_.end(), kInvalidReqSlot);
    }

    template <class PendingSet>
    inline SlotID PageCandidate_IO_Queue<PendingSet>::acquire_page_candidate_slot(
        QueryBuffer *query_buf)
    {
        if (query_buf->free_page_slots.empty())
        {
            const size_t old_size = query_buf->page_slots.size();
            const size_t grow_by =
                std::max(old_size, static_cast<size_t>(INITIAL_PAGE_SLOT_CAPACITY));
            query_buf->ensure_page_slot_capacity(old_size + grow_by);
            ensure_slot_capacity(query_buf->page_slots.size());
        }

        const SlotID slot_id = query_buf->free_page_slots.back();
        query_buf->free_page_slots.pop_back();
        PageCandidates &slot = query_buf->page_slots[slot_id];
        slot.page_lower_bound = 0.0F;
        slot.probe_idx = static_cast<CID>(-1);
        slot.page_id = -1;
        slot.req_slot = -1;
        slot.candidate_num = 0;
        slot_state_[slot_id] = kStateReserved;
        return slot_id;
    }

    template <class PendingSet>
    inline void PageCandidate_IO_Queue<PendingSet>::release_page_slot(
        QueryBuffer *query_buf, SlotID slot_id)
    {
        assert(slot_id < query_buf->page_slots.size());
        PageCandidates &slot = query_buf->page_slots[slot_id];
        slot.page_lower_bound = 0.0F;
        slot.probe_idx = static_cast<CID>(-1);
        slot.page_id = -1;
        slot.req_slot = -1;
        slot.candidate_num = 0;
        slot_state_[slot_id] = kStateFree;
        slot_req_slot_[slot_id] = kInvalidReqSlot;
        query_buf->free_page_slots.push_back(slot_id);
    }

    template <class PendingSet>
    inline void PageCandidate_IO_Queue<PendingSet>::prune_pending_above(
        QueryBuffer *query_buf, float distk)
    {
        if (pending_.empty())
        {
            return;
        }
        std::vector<SlotID> pruned_slots;
        pruned_slots.reserve(pending_.size());
        pending_.prune_greater(distk, pruned_slots);
        for (SlotID slot_id : pruned_slots)
        {
            assert(slot_state_[slot_id] == kStatePending);
            release_page_slot(query_buf, slot_id);
        }
    }

    template <class PendingSet>
    inline int PageCandidate_IO_Queue<PendingSet>::push_into_submit_candidate_pool(
        SlotID slot_id, float page_lower_bound, SearchStats *)
    {
        assert(slot_id < slot_state_.size());
        assert(slot_state_[slot_id] == kStateReserved ||
               slot_state_[slot_id] == kStateFree);
        slot_state_[slot_id] = kStatePending;
        pending_.insert(slot_id, page_lower_bound);
        return -1;
    }

    template <class PendingSet>
    inline void PageCandidate_IO_Queue<PendingSet>::prepare_request(
        QueryBuffer *query_buf, SlotID slot_id, int req_slot)
    {
        PageCandidates &cand = query_buf->page_slots[slot_id];
        char *buf = query_buf->request_buffer(req_slot);
        query_buf->reqs[req_slot] = IORequest(
            static_cast<uint64_t>(cand.page_id) * SECTOR_LEN,
            query_buf->request_slot_bytes, buf);
        query_buf->req_slot_owner[req_slot] = slot_id;
        cand.req_slot = req_slot;
        slot_req_slot_[slot_id] = req_slot;
    }

    template <class PendingSet>
    template <bool LimitByCandidateCount>
    inline int PageCandidate_IO_Queue<PendingSet>::submit_impl(
        size_t budget, QueryBuffer *query_buf,
        LinuxAlignedFileReader *file_reader, void *ctx, float distk,
        bool empty_ssd, SearchStats *stats)
    {
        if (budget == 0 || pending_.empty())
        {
            return 0;
        }

        prune_pending_above(query_buf, distk);
        if (budget == 0 || pending_.empty())
        {
            return 0;
        }

        if (empty_ssd)
        {
            while (budget > 0 && !pending_.empty())
            {
                const SlotID slot_id = pending_.front_slot();
                pending_.pop_front();
                slot_state_[slot_id] = kStateCompleted;
                ready_completed_.push_back(slot_id);
                if constexpr (LimitByCandidateCount)
                {
                    const size_t n_cands =
                        query_buf->page_slots[slot_id].candidate_num;
                    budget = (budget > n_cands) ? (budget - n_cands) : 0;
                }
                else
                {
                    --budget;
                }
            }
            return 0;
        }

        rabitqlib::io_backend::SubmitBatch &io_batch =
            rabitqlib::io_backend::submit_batch();
        io_batch.begin(ctx);
        int submitted = 0;
        while (budget > 0 && !pending_.empty() &&
               !query_buf->free_req_slots.empty())
        {
            const SlotID slot_id = pending_.front_slot();
            pending_.pop_front();

            const int req_slot = query_buf->free_req_slots.back();
            query_buf->free_req_slots.pop_back();
            prepare_request(query_buf, slot_id, req_slot);

            IORequest &req = query_buf->reqs[req_slot];
            const uint64_t end_offset = req.offset + req.len;
            if (req.offset > file_reader->get_file_size() ||
                end_offset > file_reader->get_file_size())
            {
                LOG(ERROR) << "Read out of range: offset=" << req.offset
                           << ", len=" << req.len
                           << ", file_sz=" << file_reader->get_file_size();
                crash();
            }

            // Stage one read into the IO backend; libaio reads straight into
            // req.buf.
            if (!io_batch.stage(
                    file_reader->get_file_desc(), req.buf,
                    static_cast<unsigned>(req.len), req.offset, &req))
            {
                LOG(ERROR) << "io_backend stage failed (submission queue full)";
                crash();
            }

            slot_state_[slot_id] = kStateInflight;
            inflight_count_++;
            submitted++;
            if (stats != nullptr)
            {
                // Single submit chokepoint: count the request itself, the
                // exact bytes, and the page-equivalent separately (a multi-
                // page contiguous read is 1 req, N pages, len bytes).
                stats->issued_read_reqs += 1;
                stats->issued_read_pages += req.len / SECTOR_LEN;
                stats->issued_read_bytes += req.len;
            }

            if constexpr (LimitByCandidateCount)
            {
                const size_t n_cands = query_buf->page_slots[slot_id].candidate_num;
                budget = (budget > n_cands) ? (budget - n_cands) : 0;
            }
            else
            {
                --budget;
            }
        }

        if (submitted == 0)
        {
            return 0;
        }

        const int ret = io_batch.submit();
        if (ret < 0)
        {
            LOG(ERROR) << "io_backend submit failed: " << strerror(-ret);
            crash();
        }
        return submitted;
    }

    template <class PendingSet>
    inline int PageCandidate_IO_Queue<PendingSet>::submit_top_io_pagecandidates(
        size_t max_n_page_submit, QueryBuffer *query_buf,
        LinuxAlignedFileReader *file_reader, void *ctx, float distk,
        bool empty_ssd, SearchStats *stats)
    {
        return submit_impl<false>(max_n_page_submit, query_buf, file_reader, ctx,
                                  distk, empty_ssd, stats);
    }

    template <class PendingSet>
    inline void
    PageCandidate_IO_Queue<PendingSet>::poll_all_completed_out_of_order(
        LinuxAlignedFileReader *, void *ctx, QueryBuffer *query_buf,
        int &n_completed, bool empty_ssd, SearchStats *)
    {
        if (n_completed >= static_cast<int>(N_REQ_BUF))
        {
            return;
        }
        query_buf->completed_slots.clear();

        if (empty_ssd)
        {
            for (SlotID slot_id : ready_completed_)
            {
                query_buf->completed_slots.push_back(slot_id);
                ++n_completed;
            }
            ready_completed_.clear();
            return;
        }

        if (inflight_count_ == 0)
        {
            return;
        }

        static thread_local void *io_ud[MAX_EVENTS];
        static thread_local int io_res[MAX_EVENTS];
        const int ret =
            rabitqlib::io_backend::reap_batch(ctx, io_ud, io_res, MAX_EVENTS);
        if (ret <= 0)
        {
            return;
        }

        for (int i = 0; i < ret; ++i)
        {
            // A failed or short read leaves stale bytes in the request buffer,
            // and re-ranking them would silently return wrong neighbours, so
            // both are fatal. The SSD index is written in whole pages, so a
            // good read always returns exactly req->len bytes.
            if (io_res[i] < 0)
            {
                LOG(ERROR) << "SSD read failed: " << strerror(-io_res[i]);
                crash();
            }

            IORequest *req = reinterpret_cast<IORequest *>(io_ud[i]);
            if (req == nullptr)
            {
                continue;
            }
            if (static_cast<uint64_t>(io_res[i]) != req->len)
            {
                LOG(ERROR) << "short SSD read at offset " << req->offset
                           << ": got " << io_res[i] << " of " << req->len
                           << " bytes (was the SSD index file truncated?)";
                crash();
            }

            const ptrdiff_t diff = req - query_buf->reqs.data();
            if (diff < 0 || diff >= static_cast<ptrdiff_t>(N_REQ_BUF))
            {
                continue;
            }

            const int req_slot = static_cast<int>(diff);
            const SlotID slot_id = query_buf->req_slot_owner[req_slot];
            if (slot_id == kSlotMax || slot_id >= slot_state_.size() ||
                slot_state_[slot_id] != kStateInflight)
            {
                continue;
            }

            slot_state_[slot_id] = kStateCompleted;
            inflight_count_--;
            query_buf->completed_slots.push_back(slot_id);
            ++n_completed;
        }

        // reap_batch already consumed the completions (io_getevents drains
        // them as they are returned).
    }

    template <class PendingSet>
    inline const char *PageCandidate_IO_Queue<PendingSet>::completed_data(
        const QueryBuffer *query_buf, SlotID slot_id) const
    {
        if (slot_id >= slot_req_slot_.size())
        {
            return nullptr;
        }
        const int req_slot = slot_req_slot_[slot_id];
        if (req_slot == kInvalidReqSlot)
        {
            return nullptr;
        }
        return static_cast<const char *>(query_buf->reqs[req_slot].buf);
    }

    template <class PendingSet>
    inline void PageCandidate_IO_Queue<PendingSet>::release_completed_slot(
        QueryBuffer *query_buf, SlotID slot_id)
    {
        assert(slot_id < slot_state_.size());
        assert(slot_state_[slot_id] == kStateCompleted);

        const int req_slot = slot_req_slot_[slot_id];
        if (req_slot != kInvalidReqSlot)
        {
            assert(query_buf->req_slot_owner[req_slot] == slot_id);
            query_buf->reqs[req_slot] = IORequest();
            query_buf->req_slot_owner[req_slot] = kSlotMax;
            query_buf->free_req_slots.push_back(req_slot);
        }

        release_page_slot(query_buf, slot_id);
    }

    template <class PendingSet>
    inline IVFSSD_Index<PendingSet>::IVFSSD_Index(
        size_t num, size_t dim, size_t num_cluster, size_t total_bits,
        std::string ssd_index_file, MetricType metric_type, RotatorType type,
        size_t mem_dim, CoarseKind coarse_kind,
        ivf::InitializerType initializer_type,
        std::string inner_centroids_path, std::string inner_cids_path)
        : num_(num),
          dim_(dim),
          requested_mem_dim_(mem_dim),
          num_cluster_(num_cluster),
          ex_bits_(total_bits - 1),
          coarse_kind_(coarse_kind),
          initializer_type_(
              ivf::resolve_initializer_type(initializer_type, num_cluster)),
          inner_centroids_path_(std::move(inner_centroids_path)),
          inner_cids_path_(std::move(inner_cids_path)),
          type_(type),
          metric_type_(metric_type),
          ssd_index_file_(std::move(ssd_index_file))
    {
        if (total_bits < 1 || ex_bits_ > 8)
        {
            std::cerr << "Invalid number of bits for quantization in IVFSSD_Index\n";
            std::cerr << "Expected: 1 to 9  Input:" << total_bits << '\n';
            std::cerr.flush();
            std::exit(1);
        }

        rotator_ =
            choose_rotator<float>(dim, type, round_up_to_multiple(dim_, 64));
        padded_dim_ = rotator_->size();

        assert(padded_dim_ % 64 == 0);
        assert(padded_dim_ >= dim_);

        init_layout_metadata();
    }

    template <class PendingSet>
    inline IVFSSD_Index<PendingSet>::~IVFSSD_Index()
    {
        free_memory();
        delete rotator_;
        rotator_ = nullptr;
        if (file_reader_)
        {
            file_reader_->close();
        }
        destroy_buffers();
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::set_ssd_store(SsdStore store,
                                                        VecElemType raw_elem)
    {
        ssd_store_ = store;
        raw_elem_ = raw_elem;
        if (store == SsdStore::Raw)
        {
            // A raw record replaces both the SSD part of the 1-bit code and
            // the extra-precision code, so no ex-code is computed or stored.
            ex_bits_ = 0;
        }
        init_layout_metadata();
        log_ssd_layout();
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::write_raw_record(char *dst,
                                                           const float *src) const
    {
        if (raw_elem_ == VecElemType::F32)
        {
            std::memcpy(dst, src, dim_ * sizeof(float));
            return;
        }
        // uint8 input reached the build widened to float, which is exact, so
        // narrowing it back is lossless. That is checked rather than assumed:
        // a value that is not an integer in [0, 255] means the input was not
        // uint8 after all, and storing a rounded value would give wrong
        // distances without any error.
        auto *out = reinterpret_cast<uint8_t *>(dst);
        for (size_t j = 0; j < dim_; ++j)
        {
            const float v = src[j];
            if (!(v >= 0.0F && v <= 255.0F) || v != std::floor(v))
            {
                LOG(ERROR) << "raw uint8 store: input value " << v
                           << " in dimension " << j
                           << " is not an integer in [0, 255]";
                std::exit(1);
            }
            out[j] = static_cast<uint8_t>(v);
        }
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::read_base_store_fields(
        std::istream &input, uint32_t version, const std::string &base_index_file)
    {
#if defined(RABITQ_PID64)
        if (version == kBaseVersion || version == kBaseVersionWithStore)
        {
            LOG(ERROR) << base_index_file << " was built with 32-bit point ids; "
                          "use the default build (bin/) for it";
            std::exit(1);
        }
        if (version != kBaseVersionPid64)
        {
            LOG(ERROR) << "Unsupported base index version " << version
                       << " (expected " << kBaseVersionPid64
                       << "): " << base_index_file;
            std::exit(1);
        }
#else
        if (version == kBaseVersion)
        {
            ssd_store_ = SsdStore::ExBits;
            raw_elem_ = VecElemType::F32;
            return;
        }
        if (version == kBaseVersionPid64)
        {
            LOG(ERROR) << base_index_file << " was built with 64-bit point ids; "
                          "use the RABITQ_PID64 build (bin64/) for it";
            std::exit(1);
        }
        if (version != kBaseVersionWithStore)
        {
            LOG(ERROR) << "Unsupported base index version " << version
                       << " (expected " << kBaseVersion << " or "
                       << kBaseVersionWithStore << "): " << base_index_file;
            std::exit(1);
        }
#endif
        uint32_t store_u = 0;
        uint32_t elem_u = 0;
        input.read(reinterpret_cast<char *>(&store_u), sizeof(uint32_t));
        input.read(reinterpret_cast<char *>(&elem_u), sizeof(uint32_t));
        if (!input || store_u > static_cast<uint32_t>(SsdStore::Raw) ||
            elem_u > static_cast<uint32_t>(VecElemType::U8))
        {
            LOG(ERROR) << "Base index has an unknown SSD store (" << store_u
                       << ", " << elem_u << "): " << base_index_file;
            std::exit(1);
        }
        ssd_store_ = static_cast<SsdStore>(store_u);
        raw_elem_ = static_cast<VecElemType>(elem_u);
        if (ssd_store_ == SsdStore::Raw && ex_bits_ != 0)
        {
            LOG(ERROR) << "Base index claims raw SSD records but ex_bits="
                       << ex_bits_ << ": " << base_index_file;
            std::exit(1);
        }
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::check_ssd_pages(
        const std::vector<size_t> &cluster_sizes) const
    {
        if (data_num_in_one_page_ == 0)
        {
            return;  // nothing is stored on SSD
        }
        size_t pages = 0;
        for (const size_t n : cluster_sizes)
        {
            pages += div_round_up(n, data_num_in_one_page_) * page_num_per_io_;
        }
        const auto limit = static_cast<size_t>(std::numeric_limits<PageIdx>::max());
        if (pages > limit)
        {
            LOG(ERROR) << "the SSD index needs " << pages << " pages, more than the "
                       << limit << " this build can address; use the build with "
                          "64-bit point ids (-DRABITQ_PID64=ON, bin64/)";
            std::exit(1);
        }
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::check_ssd_layout_searchable() const
    {
        if (ssd_store_ == SsdStore::ExBits && ex_bits_ == 0 && ssd_dim_ != 0)
        {
            LOG(ERROR) << "total_bits=1 with memdim " << mem_dim_ << " leaves "
                       << ssd_dim_ << " dimensions of the 1-bit code on SSD with "
                          "no extra-precision code, which search cannot re-rank; "
                          "use total_bits of 2 or more, memdim "
                       << padded_dim_ << " (the whole code in RAM), or store=raw";
            std::exit(1);
        }
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::preallocate_ssd_file(
        uint64_t reserve_bytes)
    {
        if (one_data_ssd_bytes_ == 0 || num_cluster_ == 0)
        {
            return;
        }
        const size_t last = num_cluster_ - 1;
        const uint64_t total_pages =
            cluster_page_presums_[last] +
            (div_round_up(cluster_lst_[last].num(), data_num_in_one_page_) *
             page_num_per_io_);
        const uint64_t total_bytes = total_pages * SECTOR_LEN;
        const int fd = file_reader_->get_file_desc();
        auto gb = [](uint64_t b) { return static_cast<double>(b) / 1e9; };

        // Blocks the file already holds (a rebuild in place) need no new space.
        struct stat st {};
        const uint64_t held =
            (::fstat(fd, &st) == 0) ? static_cast<uint64_t>(st.st_blocks) * 512
                                    : 0;
        const uint64_t need = (total_bytes > held) ? total_bytes - held : 0;
        struct statvfs vfs {};
        if (::fstatvfs(fd, &vfs) == 0)
        {
            const uint64_t avail =
                static_cast<uint64_t>(vfs.f_bavail) * vfs.f_frsize;
            const uint64_t want = need + reserve_bytes;
            if (avail < want + (want / 100))
            {
                LOG(INFO) << "not preallocating " << ssd_index_file_ << ": it needs "
                          << gb(need) << " GB and the build writes "
                          << gb(reserve_bytes)
                          << " GB of scratch on the same file system, which has "
                          << gb(avail) << " GB free";
                return;
            }
        }
        if (::fallocate(fd, 0, 0, static_cast<off_t>(total_bytes)) == 0)
        {
            LOG(INFO) << "preallocated " << gb(total_bytes) << " GB for "
                      << ssd_index_file_;
            return;
        }
        LOG(INFO) << "not preallocating " << ssd_index_file_ << ": fallocate: "
                  << std::strerror(errno);
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::init_layout_metadata()
    {
        LOG(INFO) << "Initializing layout metadata...";

        const size_t target_mem_dim =
            (requested_mem_dim_ == 0) ? padded_dim_ : requested_mem_dim_;
        if (target_mem_dim == 0 || target_mem_dim > padded_dim_ ||
            (target_mem_dim % 64) != 0)
        {
            LOG(ERROR) << "Invalid MEMDIM=" << target_mem_dim
                       << " (padded_dim_=" << padded_dim_
                       << ", MEMDIM must be in [64, padded_dim_] and multiple of 64)";
            std::exit(1);
        }

        mem_dim_ = target_mem_dim;
        // The 1-bit code always covers all padded_dim dimensions, and the
        // in-memory bound's factors are computed over all of them; ssd_dim_ is
        // the part of that code that is not memory-resident. The Raw store
        // simply does not write it: its record is the vector itself.
        ssd_dim_ = padded_dim_ - mem_dim_;
        if (ssd_store_ == SsdStore::Raw)
        {
            empty_ssd_ = false;
        }
        else
        {
            empty_ssd_ = (ssd_dim_ == 0 && ex_bits_ == 0);
        }

        LOG(INFO) << "mem_dim_ = " << mem_dim_ << "    ssd_dim_ = " << ssd_dim_;
        LOG(INFO) << "empty_ssd_=" << empty_ssd_;

        if (ssd_store_ == SsdStore::Raw)
        {
            // The input vector, zero-padded to padded_dim.
            one_data_ssd_bytes_ = padded_dim_ * elem_type_bytes(raw_elem_);
        }
        else
        {
            one_data_ssd_bytes_ =
                ssd_dim_ / 8 + ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
        }

        if (one_data_ssd_bytes_ == 0)
        {
            data_num_in_one_page_ =
                std::min(fastscan::kBatchSize, DEFAULT_EMPTY_SSD_GROUP_SIZE);
            page_num_per_io_ = 1;
        }
        else
        {
            data_num_in_one_page_ =
                std::max(static_cast<size_t>(1),
                         SECTOR_LEN / one_data_ssd_bytes_);
            page_num_per_io_ = div_round_up(one_data_ssd_bytes_, SECTOR_LEN);
        }
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::log_ssd_layout() const
    {
        LOG(INFO) << "SSD store: " << ssd_store_description() << ", "
                  << one_data_ssd_bytes_ << " B per record, "
                  << data_num_in_one_page_ << " per page, " << page_num_per_io_
                  << " sector(s) per read";
    }

    // Allocate the coarse-quantizer-independent inverted-list buffers
    // (mem_batch_data_, ids_, ip_func_) required by build_invlist and
    // load_base. Does not touch initer_.
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::allocate_invlist_memory(
        const std::vector<size_t> &cluster_sizes)
    {
        LOG(INFO) << "Allocating memory for inverted-list components...";
        mem_batch_data_ = memory::align_allocate<64, char, true>(
            mem_batch_data_bytes(cluster_sizes));
        ids_ = memory::align_allocate<64, PID, true>(ids_bytes());
        ip_func_ = select_excode_ipfunc(ex_bits_);
    }

    // Construct initer_ for the current coarse_kind_. Called by
    // build_coarse_quantizer and by load_coarse. On the build path the IRQ
    // constructor consumes the inner-IVF inputs set by
    // build_coarse_quantizer; on the load path those path strings are empty
    // and IvfRabitqInitializer::load reads the payload from the <coarse>.irq
    // sidecar instead.
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::build_initializer()
    {
        delete initer_;
        initer_ = nullptr;

        if (coarse_kind_ == CoarseKind::IRQ)
        {
            // ip_normalize is keyed off the index's metric. For METRIC_IP the
            // embedded IVF-RaBitQ runs in normalized (unit-norm) space, so its
            // L2 cluster selection becomes cosine selection and the outer SSD
            // loop reconstructs <q,c> and ||q-c|| from the stored centroid
            // norms (see the METRIC_IP branch of query_one_boundary_qd). For
            // METRIC_L2 it stays false. Precondition on the build path: the
            // inner centroids must be computed on *normalized* first-level
            // centroids. On the load path the flag and the centroid norms are
            // restored from the .irq sidecar.
            initer_ = new ivf_irq::IvfRabitqInitializer(
                padded_dim_, num_cluster_, dim_, rotator_,
                inner_centroids_path_, inner_cids_path_,
                /*ip_normalize=*/metric_type_ == METRIC_IP);
        }
        else
        {
            initer_ = ivf::create_initializer(initializer_type_, padded_dim_,
                                              num_cluster_);
        }
    }

    template <class PendingSet>
    inline void
    IVFSSD_Index<PendingSet>::init_clusters(const std::vector<size_t> &cluster_sizes)
    {
        LOG(INFO) << "Initializing clusters...";
        size_t added_vectors = 0;
        size_t added_batches = 0;
        size_t added_pages = 0;

        cluster_lst_.clear();
        cluster_page_presums_.clear();

        for (size_t i = 0; i < num_cluster_; ++i)
        {
            const size_t num = cluster_sizes[i];
            const size_t num_batches = div_round_up(num, fastscan::kBatchSize);

            char *current_batch_data =
                mem_batch_data_ +
                (BatchDataMap<float>::data_bytes(mem_dim_) * added_batches);
            PID *ids = ids_ + added_vectors;

            ivf::Cluster cur_cluster(num, current_batch_data, nullptr, ids);
            cluster_lst_.push_back(std::move(cur_cluster));

            cluster_page_presums_.emplace_back(added_pages);

            added_pages += data_num_in_one_page_
                               ? div_round_up(num, data_num_in_one_page_) *
                                     page_num_per_io_
                               : 0;
            added_vectors += num;
            added_batches += num_batches;
        }
    }

    template <class PendingSet>
    template <class FetchRow>
    inline void IVFSSD_Index<PendingSet>::quantize_and_write_cluster_core(
        ivf::Cluster &cp, const PID *pids, size_t num_points,
        FetchRow fetch_row, const float *cur_centroid, float *rotated_centroid,
        const quant::RabitqConfig &config, size_t cluster_offset,
        bool batch_parallel)
    {
        if (cp.num() != num_points)
        {
            std::cerr << "Size of cluster and IDs are inequivalent\n";
            std::cerr << "Cluster: " << cp.num() << " IDs: " << num_points
                      << '\n';
            std::exit(1);
        }

        std::copy(pids, pids + num_points, cp.ids());
        rotator_->rotate(cur_centroid, rotated_centroid);

        std::vector<float> rotated_data(padded_dim_ * num_points);
#pragma omp parallel for schedule(static) if (batch_parallel)
        for (size_t i = 0; i < num_points; ++i)
        {
            rotator_->rotate(fetch_row(i),
                             rotated_data.data() + (i * padded_dim_));
        }

        char *mem_batch_data = cp.batch_data();

        const size_t ssd_bytes_per_point = ssd_dim_ / 8;
        const size_t ex_bytes_per_point =
            ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
        const size_t ssd_alloc_bytes =
            std::max<size_t>(1, num_points * ssd_bytes_per_point);
        const size_t ex_alloc_bytes =
            std::max<size_t>(1, num_points * ex_bytes_per_point);
        char *ssd_bin_data_tmp =
            memory::align_allocate<64, char, true>(ssd_alloc_bytes);
        char *ex_data_tmp =
            memory::align_allocate<64, char, true>(ex_alloc_bytes);

        // batch_parallel toggles intra-cluster OpenMP. It is off for the full
        // and small-group paths, whose outer per-cluster loop is already
        // parallel, and on for the big-cluster path, where a single cluster is
        // processed alone and must use all cores. Batches write disjoint
        // mem_batch / ssd / ex slices, so either setting is safe.
        const size_t num_batches =
            div_round_up(num_points, fastscan::kBatchSize);
#pragma omp parallel for schedule(static) if (batch_parallel)
        for (size_t b = 0; b < num_batches; ++b)
        {
            const size_t i = b * fastscan::kBatchSize;
            const size_t n = std::min(fastscan::kBatchSize, num_points - i);

            char *mem_batch_data_ptr =
                mem_batch_data + b * BatchDataMap<float>::data_bytes(mem_dim_);
            char *ssd_bin_data_ptr = ssd_bin_data_tmp + i * ssd_bytes_per_point;
            char *ex_data_ptr = ex_data_tmp + i * ex_bytes_per_point;
            // A short last batch leaves factor slots unwritten; zero the block
            // first so the base file does not depend on what memory held.
            if (n < fastscan::kBatchSize)
            {
                std::memset(mem_batch_data_ptr, 0,
                            BatchDataMap<float>::data_bytes(mem_dim_));
            }

            quant::quantize_split_batch_ssd_split_single_uint64(
                rotated_data.data() + (i * padded_dim_), rotated_centroid, n,
                mem_dim_, ssd_dim_, ex_bits_, mem_batch_data_ptr,
                ssd_bin_data_ptr, ex_data_ptr, metric_type_, config);
        }

        if (one_data_ssd_bytes_ == 0)
        {
            std::free(ssd_bin_data_tmp);
            std::free(ex_data_tmp);
            return;
        }

        // Write the cluster's SSD pages with pwrite(). Each cluster owns a
        // fixed, contiguous, non-overlapping byte range; page p lands at
        // cluster_offset + p*write_len, an absolute offset independent of the
        // shared fd's position. So concurrent pwrite() from parallel cluster
        // workers (or, here, parallel chunks) is safe. Pages are written in
        // chunks of kSsdWriteChunkPages; the bytes do not depend on the chunk
        // size.
        const int ssd_fd = file_reader_->get_file_desc();
        const size_t write_len = SECTOR_LEN * page_num_per_io_;
        const size_t num_pages = div_round_up(num_points, data_num_in_one_page_);
        const size_t pages_per_chunk =
            std::max<size_t>(1, kSsdWriteChunkPages / page_num_per_io_);
        const size_t num_chunks = div_round_up(num_pages, pages_per_chunk);

#pragma omp parallel for schedule(static) if (batch_parallel)
        for (size_t ch = 0; ch < num_chunks; ++ch)
        {
            const size_t first_page = ch * pages_per_chunk;
            const size_t pages_here =
                std::min(pages_per_chunk, num_pages - first_page);
            const size_t chunk_len = pages_here * write_len;
            const size_t page_offset = cluster_offset + first_page * write_len;

            char *page_buf =
                memory::align_allocate<SECTOR_LEN, char, true>(chunk_len);
            // Zero the chunk so the trailing padding (the bytes after the last
            // point in this cluster's final page, plus any intra-sector slack)
            // is deterministic; align_allocate does not zero. The padding is
            // never read at query time, but zeroing keeps the on-disk SSD file
            // reproducible and makes the streaming and full-DRAM builds
            // bit-identical.
            std::memset(page_buf, 0, chunk_len);

            for (size_t q = 0; q < pages_here; ++q)
            {
                const size_t pid = (first_page + q) * data_num_in_one_page_;
                const size_t points_in_page =
                    std::min(data_num_in_one_page_, num_points - pid);
                char *buf_ptr = page_buf + q * write_len;
                for (size_t i = 0; i < points_in_page; ++i)
                {
                    if (ssd_store_ == SsdStore::Raw)
                    {
                        // fetch_row(k) is the k-th point in stored order, the
                        // same order as cp.ids(), before any rotation.
                        write_raw_record(buf_ptr, fetch_row(pid + i));
                        buf_ptr += one_data_ssd_bytes_;
                        continue;
                    }
                    std::memcpy(buf_ptr,
                                ssd_bin_data_tmp + (pid + i) * (ssd_dim_ / 8),
                                ssd_dim_ / 8);
                    buf_ptr += ssd_dim_ / 8;

                    const size_t ex_bytes =
                        ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
                    std::memcpy(buf_ptr,
                                ex_data_tmp + (pid + i) * ex_bytes, ex_bytes);
                    buf_ptr += ex_bytes;
                }
            }

            // Retry loop guards against short writes. The file is O_DIRECT,
            // so any partial write still lands on a SECTOR_LEN boundary,
            // keeping buffer / offset / length aligned for the next pwrite.
            size_t written = 0;
            while (written < chunk_len)
            {
                const ssize_t n = ::pwrite(
                    ssd_fd, page_buf + written, chunk_len - written,
                    static_cast<off_t>(page_offset + written));
                if (n <= 0)
                {
                    LOG(ERROR) << "pwrite to ssd index failed at offset "
                               << (page_offset + written) << ": "
                               << std::strerror(errno);
                    crash();
                }
                written += static_cast<size_t>(n);
            }

            std::free(page_buf);
        }

        std::free(ssd_bin_data_tmp);
        std::free(ex_data_tmp);
    }

    // Full-DRAM path: fetch the i-th point from the in-memory dataset by global
    // pid and store the same global pids into cp.ids(). Forwards to the shared
    // core with batch_parallel off, since construct_invlist already
    // parallelises over clusters.
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::quantize_and_write_clusterssd(
        ivf::Cluster &cp, const std::vector<PID> &ids, const float *data,
        const float *cur_centroid, float *rotated_centroid,
        const quant::RabitqConfig &config, size_t cluster_offset)
    {
        const PID *ids_ptr = ids.data();
        quantize_and_write_cluster_core(
            cp, ids_ptr, ids.size(),
            [&](size_t i) { return data + (static_cast<size_t>(ids_ptr[i]) * dim_); },
            cur_centroid, rotated_centroid, config, cluster_offset,
            /*batch_parallel=*/false);
    }

    template <class PendingSet>
    template <class FetchRow>
    inline std::vector<size_t>
    IVFSSD_Index<PendingSet>::compute_centroid_order(
        const PID *pids, size_t num_points, FetchRow fetch_row,
        const float *cur_centroid) const
    {
        std::vector<double> centroid_distance(num_points, 0.0);
        for (size_t i = 0; i < num_points; ++i)
        {
            const float *x = fetch_row(i);
            double acc = 0.0;
            if (metric_type_ == METRIC_IP)
            {
                for (size_t d = 0; d < dim_; ++d)
                {
                    acc -= static_cast<double>(x[d]) *
                           static_cast<double>(cur_centroid[d]);
                }
            }
            else
            {
                for (size_t d = 0; d < dim_; ++d)
                {
                    const double diff = static_cast<double>(x[d]) -
                                        static_cast<double>(cur_centroid[d]);
                    acc += diff * diff;
                }
            }
            centroid_distance[i] = acc;
        }

        std::vector<size_t> order(num_points);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(
            order.begin(), order.end(),
            [&](size_t a, size_t b)
            {
                const double da = centroid_distance[a];
                const double db = centroid_distance[b];
                const bool a_nan = std::isnan(da);
                const bool b_nan = std::isnan(db);
                if (a_nan != b_nan)
                {
                    return !a_nan;
                }
                if (da < db)
                {
                    return true;
                }
                if (da > db)
                {
                    return false;
                }
                return pids[a] < pids[b];
            });
        return order;
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::quantize_and_write_ordered_clusterssd(
        ivf::Cluster &cp, const std::vector<PID> &ids, const float *data,
        const float *cur_centroid, float *rotated_centroid,
        const quant::RabitqConfig &config, size_t cluster_offset)
    {
        if (ids.empty())
        {
            quantize_and_write_clusterssd(cp, ids, data, cur_centroid,
                                          rotated_centroid, config,
                                          cluster_offset);
            return;
        }

        // The same permutation the streaming build computes, from the same
        // helper -- the two paths are contractually bit-identical, so the rule
        // must live in exactly one place.
        const std::vector<size_t> order = compute_centroid_order(
            ids.data(), ids.size(),
            [&](size_t i) { return data + (static_cast<size_t>(ids[i]) * dim_); },
            cur_centroid);

        std::vector<PID> ordered_ids;
        ordered_ids.reserve(ids.size());
        for (size_t idx : order)
        {
            ordered_ids.push_back(ids[idx]);
        }

        quantize_and_write_clusterssd(cp, ordered_ids, data, cur_centroid,
                                      rotated_centroid, config, cluster_offset);
    }

    // Build the coarse-quantizer-independent inverted list: quantize every
    // point, fill the resident mem_batch_data_ / ids_ structures and write the
    // SSD pages. The coarse quantizer is a separate artifact built by
    // build_coarse_quantizer, so no vectors are fed to initer_ here; the
    // rotated centroids are cached in rotated_centroids_ so that save_base can
    // persist them for the coarse build to reuse.
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::construct_invlist(
        const float *data, const float *centroids, const CID *assignments,
        bool faster, ClusterOrderMode order_mode)
    {
        LOG(INFO) << "Constructing inverted list (split-storage base)...";
        check_ssd_layout_searchable();

        LOG(INFO) << "Loading clustering information...";
        std::vector<size_t> counts(num_cluster_, 0);
        std::vector<std::vector<PID>> id_lists(num_cluster_);
        for (size_t i = 0; i < num_; ++i)
        {
            const CID cid = assignments[i];
            if (cid >= num_cluster_)
            {
                LOG(ERROR) << "Bad cluster id";
                std::exit(1);
            }
            id_lists[cid].push_back(static_cast<PID>(i));
            counts[cid] += 1;
        }

        check_ssd_pages(counts);
        allocate_invlist_memory(counts);
        init_clusters(counts);

        file_reader_ = std::make_shared<LinuxAlignedFileReader>();
        file_reader_->open(ssd_index_file_, true, true);
#if defined(RABITQ_PID64)
        preallocate_ssd_file(0);
#endif

        // Held as a member rather than a local so save_base can persist it;
        // the coarse quantizer is built from this block later.
        rotated_centroids_.assign(num_cluster_ * padded_dim_, 0.0F);

        quant::RabitqConfig config;
        if (faster)
        {
            config = quant::faster_config(padded_dim_, ex_bits_ + 1);
        }

        // Parallelise over clusters: every cluster writes to its own,
        // non-overlapping slices of mem_batch_data_ / ids_ / rotated_centroids_
        // and its own [cluster_offset, ...) byte range of the SSD file (via
        // pwrite, see quantize_and_write_clusterssd), so the num_cluster_
        // iterations are fully independent. schedule(dynamic, 1) load-balances
        // the uneven cluster sizes.
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < num_cluster_; ++i)
        {
            const float *cur_centroid = centroids + (i * dim_);
            float *cur_rotated_c = &rotated_centroids_[i * padded_dim_];
            ivf::Cluster &cp = cluster_lst_[i];
            const size_t cluster_offset = cluster_page_presums_[i] * SECTOR_LEN;

            if (order_mode == ClusterOrderMode::OrderedByCentroidDistance)
            {
                quantize_and_write_ordered_clusterssd(
                    cp, id_lists[i], data, cur_centroid, cur_rotated_c,
                    config, cluster_offset);
            }
            else
            {
                quantize_and_write_clusterssd(cp, id_lists[i], data,
                                              cur_centroid, cur_rotated_c,
                                              config, cluster_offset);
            }
        }
        // The coarse quantizer is not populated here: it is built separately
        // from rotated_centroids_ by build_coarse_quantizer.
    }

    // -----------------------------------------------------------------------
    // construct_invlist_streaming: memory-budgeted build. Same output as
    // construct_invlist -- byte-for-byte, once RABITQ_ROTATOR_SEED pins the
    // rotation both share -- but never holds the whole fp32 dataset in DRAM.
    //   P0  count clusters, allocate resident structures, pre-rotate centroids,
    //       derive theta/P_g from the budget, classify small/big clusters,
    //       FFD-pack small clusters into group files + give big clusters solo
    //       files.
    //   P1  one streaming pass over <data_file>: pwrite each row -- in
    //       increasing-pid order per file (the bit-identity invariant) -- to
    //       its destination scratch file (.vec + .meta sidecar).
    //   P2  quantize file-by-file (one resident buffer at a time): small groups
    //       OMP-parallel across clusters, big clusters OMP-parallel across
    //       batches. Fills mem_batch_/ids_ + SSD pages at global offsets.
    //   P3  remove the scratch dir (caller then calls save_base).
    // -----------------------------------------------------------------------
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::construct_invlist_streaming(
        const std::string &data_file, const float *centroids,
        const CID *assignments, bool faster, ClusterOrderMode order_mode,
        size_t mem_budget_bytes, size_t prefix_rows)
    {
        namespace fs = std::filesystem;
        check_ssd_layout_searchable();
        LOG(INFO) << "Constructing inverted list (streaming, mem_budget="
                  << (mem_budget_bytes >> 30) << " GiB)...";

        constexpr uint32_t kNoFile = std::numeric_limits<uint32_t>::max();
        const size_t T =
            static_cast<size_t>(std::max(1, omp_get_max_threads()));

        // Reorder-scratch metadata, one record per row: the point id and its
        // cluster id.
        struct ReorderMeta
        {
            PID pid;
            CID cid;
        };

        // Per-phase wall-clock timers (P0 setup / P1 reorder / P2 quantize)
        // plus a total for construct_invlist_streaming itself; all are
        // reported through LOG(INFO).
        StopW total_sw;
        StopW phase_sw;

        auto pwrite_full = [](int fd, const void *buf, size_t nbytes,
                              off_t offset) {
            const char *p = static_cast<const char *>(buf);
            size_t done = 0;
            while (done < nbytes)
            {
                const ssize_t n = ::pwrite(fd, p + done, nbytes - done,
                                           offset + static_cast<off_t>(done));
                if (n < 0)
                {
                    if (errno == EINTR) continue;
                    LOG(ERROR) << "streaming build: pwrite failed: "
                               << std::strerror(errno);
                    crash();
                }
                done += static_cast<size_t>(n);
            }
        };
        auto pread_full = [](int fd, void *buf, size_t nbytes, off_t offset) {
            char *p = static_cast<char *>(buf);
            size_t done = 0;
            while (done < nbytes)
            {
                const ssize_t n = ::pread(fd, p + done, nbytes - done,
                                          offset + static_cast<off_t>(done));
                if (n < 0)
                {
                    if (errno == EINTR) continue;
                    LOG(ERROR) << "streaming build: pread failed: "
                               << std::strerror(errno);
                    crash();
                }
                if (n == 0)
                {
                    LOG(ERROR) << "streaming build: unexpected EOF on scratch";
                    crash();
                }
                done += static_cast<size_t>(n);
            }
        };

        // ---- P0a: cluster counts + resident allocation --------------------
        std::vector<size_t> counts(num_cluster_, 0);
        for (size_t i = 0; i < num_; ++i)
        {
            const CID cid = assignments[i];
            if (cid >= num_cluster_)
            {
                LOG(ERROR) << "Bad cluster id " << cid << " at point " << i;
                std::exit(1);
            }
            counts[cid] += 1;
        }

        check_ssd_pages(counts);
        allocate_invlist_memory(counts);
        init_clusters(counts);

        file_reader_ = std::make_shared<LinuxAlignedFileReader>();
        file_reader_->open(ssd_index_file_, true, true);

        quant::RabitqConfig config;
        if (faster)
        {
            config = quant::faster_config(padded_dim_, ex_bits_ + 1);
        }

        // Pre-rotate ALL first-level centroids. The full path rotates each
        // centroid inside quantize_and_write_clusterssd (even for empty
        // clusters); doing it up-front makes rotated_centroids_ bit-identical
        // for every cluster, including 0-point/big ones whose per-cluster call
        // is skipped or differs.
        rotated_centroids_.assign(num_cluster_ * padded_dim_, 0.0F);
#pragma omp parallel for schedule(static)
        for (size_t c = 0; c < num_cluster_; ++c)
        {
            rotator_->rotate(centroids + c * dim_,
                             rotated_centroids_.data() + c * padded_dim_);
        }

        // ---- P0b: budget math ---------------------------------------------
        const size_t ex_bytes_per_point =
            ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
        const size_t ssd_bytes_per_point = ssd_dim_ / 8;

        const size_t r_res =
            mem_batch_data_bytes(counts) + ids_bytes() +
            (num_cluster_ * padded_dim_ * sizeof(float)) +  // rotated centroids
            (num_cluster_ * dim_ * sizeof(float)) +         // raw centroids
            (num_ * sizeof(CID));                           // assignments
        const double b_total = 0.85 * static_cast<double>(mem_budget_bytes);
        if (b_total <= static_cast<double>(r_res))
        {
            LOG(ERROR) << "streaming build: mem_budget too small. Resident "
                          "structures need ~"
                       << (r_res >> 30) << " GiB but 0.85*budget = "
                       << (static_cast<size_t>(b_total) >> 30)
                       << " GiB. Lower MEMDIM or raise mem_budget_gb.";
            std::exit(1);
        }
        const size_t b_trans = static_cast<size_t>(b_total) - r_res;

        // theta: small-cluster threshold. Up to T small clusters rotate
        // concurrently; cap their rotation temps at 25% of the transient.
        const size_t rot_bytes_per_pt = padded_dim_ * sizeof(float);
        size_t theta = static_cast<size_t>(
            (0.25 * static_cast<double>(b_trans)) /
            static_cast<double>(T * rot_bytes_per_pt));
        if (theta < fastscan::kBatchSize) theta = fastscan::kBatchSize;

        // P_g: group .vec buffer cap (dim*4 / pt) at 75% of the transient.
        size_t p_g = static_cast<size_t>(
            (0.75 * static_cast<double>(b_trans)) /
            static_cast<double>(dim_ * sizeof(float)));
        if (p_g < 1)
        {
            LOG(ERROR) << "streaming build: mem_budget too small for even one "
                          "group buffer. Raise mem_budget_gb.";
            std::exit(1);
        }
        if (theta > p_g) theta = p_g;  // a small cluster must fit one group

        // Largest single cluster that can be loaded and quantized whole in its
        // own solo file. Clusters above this size are rejected by the check in
        // P0c below.
        const size_t big_pt_bytes = (dim_ * sizeof(float)) + rot_bytes_per_pt +
                                    ssd_bytes_per_point + ex_bytes_per_point;
        const size_t max_big_cz =
            big_pt_bytes ? (b_trans / big_pt_bytes) : num_;

        // ---- P0c: classify + FFD-pack into files --------------------------
        std::vector<std::vector<uint32_t>> file_clusters;
        std::vector<char> file_is_big;
        std::vector<size_t> file_rows;
        std::vector<size_t> group_load;
        std::vector<uint32_t> cluster_to_file(num_cluster_, kNoFile);

        std::vector<uint32_t> small;
        small.reserve(num_cluster_);
        size_t n_big = 0;
        for (uint32_t c = 0; c < num_cluster_; ++c)
        {
            if (counts[c] == 0) continue;  // 0-point cluster: no file
            if (counts[c] > theta)
            {
                if (counts[c] > max_big_cz)
                {
                    LOG(ERROR)
                        << "streaming build: cluster " << c << " has "
                        << counts[c] << " points, exceeding the single-cluster "
                        << "budget of " << max_big_cz
                        << " points. Raise mem_budget_gb: clusters larger "
                           "than the transient budget are not supported.";
                    std::exit(1);
                }
                ++n_big;
            }
            else
            {
                small.push_back(c);
            }
        }
        // FFD: descending size, first-fit into group files capped at p_g.
        std::sort(small.begin(), small.end(),
                  [&](uint32_t a, uint32_t b) { return counts[a] > counts[b]; });
        for (uint32_t c : small)
        {
            size_t g = 0;
            for (; g < file_clusters.size(); ++g)
            {
                if (group_load[g] + counts[c] <= p_g) break;
            }
            if (g == file_clusters.size())
            {
                file_clusters.emplace_back();
                file_is_big.push_back(0);
                file_rows.push_back(0);
                group_load.push_back(0);
            }
            file_clusters[g].push_back(c);
            file_rows[g] += counts[c];
            group_load[g] += counts[c];
            cluster_to_file[c] = static_cast<uint32_t>(g);
        }
        const size_t n_small_groups = file_clusters.size();
        // big clusters: one solo file each, appended after the small groups.
        for (uint32_t c = 0; c < num_cluster_; ++c)
        {
            if (counts[c] == 0 || counts[c] <= theta) continue;
            const uint32_t f = static_cast<uint32_t>(file_clusters.size());
            file_clusters.push_back({c});
            file_is_big.push_back(1);
            file_rows.push_back(counts[c]);
            group_load.push_back(counts[c]);
            cluster_to_file[c] = f;
        }
        const size_t num_files = file_clusters.size();
        if (num_files > 4096)
        {
            LOG(ERROR) << "streaming build: " << num_files
                       << " scratch files (would exhaust fds). Raise "
                          "mem_budget_gb to pack more clusters per file.";
            std::exit(1);
        }

        LOG(INFO) << "streaming build: N=" << num_ << " dim=" << dim_
                  << " C=" << num_cluster_ << " T=" << T << " | R_res="
                  << (r_res >> 30) << "GiB B_trans=" << (b_trans >> 30)
                  << "GiB theta=" << theta << " P_g=" << p_g
                  << " | small_groups=" << n_small_groups << " big=" << n_big
                  << " files=" << num_files;

        // The data file decides the scratch row width. The reorder pass only
        // permutes rows, so it keeps the stored values -- one byte each for
        // uint8 input, four for float32 -- and P2 widens a group to float32 as
        // it loads it.
        StreamingRowReader reader(data_file);
        if (prefix_rows != 0)
        {
            if (prefix_rows > reader.rows())
            {
                LOG(ERROR) << "streaming build: asked for the first "
                           << prefix_rows << " rows of " << data_file
                           << ", which holds only " << reader.rows();
                std::exit(1);
            }
            reader.limit_rows(prefix_rows);
            LOG(INFO) << "streaming build: indexing the first " << prefix_rows
                      << " of " << reader.file_rows() << " rows of " << data_file;
        }
        if (reader.rows() != num_ || reader.cols() != dim_)
        {
            LOG(ERROR) << "streaming build: data_file shape (" << reader.rows()
                       << "x" << reader.cols() << ") != index (" << num_ << "x"
                       << dim_ << ")";
            std::exit(1);
        }
        const size_t scratch_row_bytes = dim_ * reader.elem_bytes();
        LOG(INFO) << "streaming build: input values are "
                  << elem_type_name(reader.elem_type()) << ", "
                  << scratch_row_bytes << " B per row in the reorder scratch";

        // ---- P0d: scratch dir --------------------------------------------
        // The reorder pass writes every vector once, so this directory grows to
        // the size of the dataset before P2 consumes and P3 removes it. It sits
        // next to the data file by default; RABITQ_BUILD_SCRATCH_DIR moves it,
        // which is what you want when the data lives on a volume without room
        // for a second copy.
        const fs::path data_path(data_file);
        const char *scratch_env = std::getenv("RABITQ_BUILD_SCRATCH_DIR");
        const fs::path scratch_parent =
            (scratch_env != nullptr && scratch_env[0] != '\0')
                ? fs::path(scratch_env)
                : data_path.parent_path();
        const fs::path scratch_dir =
            scratch_parent / (data_path.filename().string() + ".reorder_tmp");
        std::error_code ec;

        // Fail now rather than partway through a multi-hour reorder pass.
        {
            const size_t need = num_ * scratch_row_bytes + num_ * sizeof(ReorderMeta);
            fs::create_directories(scratch_parent, ec);
            const fs::space_info si = fs::space(scratch_parent, ec);
            if (!ec && si.available < need)
            {
                LOG(ERROR) << "streaming build: the reorder scratch needs "
                           << (need >> 30) << " GiB but " << scratch_parent
                           << " has only " << (si.available >> 30)
                           << " GiB free. Free space there, or set "
                              "RABITQ_BUILD_SCRATCH_DIR to a volume with room.";
                std::exit(1);
            }
            ec.clear();
        }
        fs::remove_all(scratch_dir, ec);  // clear any stale leftovers
        fs::create_directories(scratch_dir, ec);
        if (ec)
        {
            LOG(ERROR) << "streaming build: cannot create scratch dir "
                       << scratch_dir << ": " << ec.message();
            std::exit(1);
        }
#if defined(RABITQ_PID64)
        {
            // The scratch fills before the SSD file does, so preallocation
            // must leave room for it when both share a file system.
            struct stat scratch_st {};
            struct stat ssd_st {};
            const bool same_fs =
                ::stat(scratch_dir.c_str(), &scratch_st) == 0 &&
                ::fstat(file_reader_->get_file_desc(), &ssd_st) == 0 &&
                scratch_st.st_dev == ssd_st.st_dev;
            preallocate_ssd_file(
                same_fs ? num_ * (scratch_row_bytes + sizeof(ReorderMeta)) : 0);
        }
#endif
        auto vec_path = [&](size_t f) {
            return (scratch_dir / ("g" + std::to_string(f) + ".vec")).string();
        };
        auto meta_path = [&](size_t f) {
            return (scratch_dir / ("g" + std::to_string(f) + ".meta")).string();
        };

        LOG(INFO) << "[timing] P0 setup (counts/alloc/pre-rotate/classify/FFD): "
                  << phase_sw.get_elapsed_sec() << " s";
        phase_sw.reset();

        // ---- P1: streaming reorder pass -----------------------------------
        std::vector<int> vec_fd(num_files, -1), meta_fd(num_files, -1);
        for (size_t f = 0; f < num_files; ++f)
        {
            vec_fd[f] = ::open(vec_path(f).c_str(),
                               O_WRONLY | O_CREAT | O_TRUNC, 0644);
            meta_fd[f] = ::open(meta_path(f).c_str(),
                                O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (vec_fd[f] < 0 || meta_fd[f] < 0)
            {
                LOG(ERROR) << "streaming build: cannot create scratch file for "
                              "group "
                           << f << ": " << std::strerror(errno);
                std::exit(1);
            }
            if (::ftruncate(vec_fd[f], static_cast<off_t>(
                                           file_rows[f] * scratch_row_bytes)) != 0 ||
                ::ftruncate(meta_fd[f], static_cast<off_t>(
                                            file_rows[f] *
                                            sizeof(ReorderMeta))) != 0)
            {
                LOG(ERROR) << "streaming build: ftruncate failed: "
                           << std::strerror(errno);
                std::exit(1);
            }
        }

        // Block size: block_buf + stage_vec (2x) within a slice of b_trans,
        // capped at 8 GiB/block to bound a single read.
        size_t block_bytes = b_trans / 8;
        const size_t cap = static_cast<size_t>(8) << 30;
        if (block_bytes > cap) block_bytes = cap;
        size_t BN = block_bytes / scratch_row_bytes;
        if (BN < 1) BN = 1;
        if (BN > num_) BN = num_;

        char *block_buf =
            memory::align_allocate<64, char, true>(BN * scratch_row_bytes);
        char *stage_vec =
            memory::align_allocate<64, char, true>(BN * scratch_row_bytes);
        std::vector<ReorderMeta> stage_meta(BN);
        std::vector<size_t> pos(BN);
        std::vector<size_t> file_cursor(num_files, 0);

        for (size_t i0 = 0; i0 < num_; i0 += BN)
        {
            const size_t bn = std::min(BN, num_ - i0);
            reader.read_rows_raw(i0, bn, block_buf);

            std::vector<size_t> bcnt(num_files, 0);
            for (size_t r = 0; r < bn; ++r)
            {
                const uint32_t f = cluster_to_file[assignments[i0 + r]];
                if (f != kNoFile) bcnt[f] += 1;
            }
            std::vector<size_t> stage_off(num_files, 0);
            std::vector<size_t> base(num_files, 0);
            size_t acc = 0;
            for (size_t f = 0; f < num_files; ++f)
            {
                stage_off[f] = acc;
                acc += bcnt[f];
                base[f] = file_cursor[f];
                file_cursor[f] += bcnt[f];
            }
            // Slots are assigned in scan order, so each file receives its rows
            // in increasing-pid order: the bit-identity invariant.
            std::vector<size_t> fill(num_files, 0);
            for (size_t r = 0; r < bn; ++r)
            {
                const uint32_t c = assignments[i0 + r];
                const uint32_t f = cluster_to_file[c];
                if (f == kNoFile)
                {
                    pos[r] = std::numeric_limits<size_t>::max();
                    continue;
                }
                const size_t local = stage_off[f] + fill[f];
                fill[f] += 1;
                pos[r] = local;
                stage_meta[local].pid = static_cast<PID>(i0 + r);
                stage_meta[local].cid = c;
            }
#pragma omp parallel for schedule(static)
            for (size_t r = 0; r < bn; ++r)
            {
                if (pos[r] == std::numeric_limits<size_t>::max()) continue;
                std::memcpy(stage_vec + pos[r] * scratch_row_bytes,
                            block_buf + r * scratch_row_bytes, scratch_row_bytes);
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t f = 0; f < num_files; ++f)
            {
                if (bcnt[f] == 0) continue;
                pwrite_full(vec_fd[f], stage_vec + stage_off[f] * scratch_row_bytes,
                            bcnt[f] * scratch_row_bytes,
                            static_cast<off_t>(base[f] * scratch_row_bytes));
                pwrite_full(meta_fd[f], stage_meta.data() + stage_off[f],
                            bcnt[f] * sizeof(ReorderMeta),
                            static_cast<off_t>(base[f] * sizeof(ReorderMeta)));
            }
        }
        for (size_t f = 0; f < num_files; ++f)
        {
            ::close(vec_fd[f]);
            ::close(meta_fd[f]);
        }
        std::free(block_buf);
        std::free(stage_vec);
        std::vector<ReorderMeta>().swap(stage_meta);
        std::vector<size_t>().swap(pos);
        LOG(INFO) << "streaming build: reorder pass complete (" << num_files
                  << " files)";
        LOG(INFO) << "[timing] P1 reorder pass: " << phase_sw.get_elapsed_sec()
                  << " s";
        phase_sw.reset();

        // ---- P2: quantize file-by-file ------------------------------------
        for (size_t f = 0; f < num_files; ++f)
        {
            const size_t n = file_rows[f];
            if (n == 0) continue;

            const int vfd = ::open(vec_path(f).c_str(), O_RDONLY);
            const int mfd = ::open(meta_path(f).c_str(), O_RDONLY);
            if (vfd < 0 || mfd < 0)
            {
                LOG(ERROR) << "streaming build: cannot reopen scratch file " << f
                           << ": " << std::strerror(errno);
                std::exit(1);
            }
            float *gbuf = memory::align_allocate<64, float, true>(
                n * dim_ * sizeof(float));
            std::vector<ReorderMeta> meta(n);
            if (scratch_row_bytes == dim_ * sizeof(float))
            {
                pread_full(vfd, gbuf, n * dim_ * sizeof(float), 0);
            }
            else
            {
                // uint8 scratch: widen to float32 while reading, a bounded
                // chunk at a time, so the group needs no second full-size
                // buffer. The widening is exact.
                const size_t total = n * dim_;
                const size_t chunk = std::min(total, static_cast<size_t>(1) << 26);
                std::vector<uint8_t> u8(chunk);
                for (size_t off = 0; off < total; off += chunk)
                {
                    const size_t len = std::min(chunk, total - off);
                    pread_full(vfd, u8.data(), len, static_cast<off_t>(off));
                    for (size_t t = 0; t < len; ++t)
                    {
                        gbuf[off + t] = static_cast<float>(u8[t]);
                    }
                }
            }
            pread_full(mfd, meta.data(), n * sizeof(ReorderMeta), 0);
            ::close(vfd);
            ::close(mfd);

            if (!file_is_big[f])
            {
                // Small group: build per-cluster local row/pid lists in slot
                // (that is, increasing-pid) order, then quantize the clusters
                // in parallel.
                const auto &clusters = file_clusters[f];
                std::unordered_map<uint32_t, uint32_t> cpos;
                cpos.reserve(clusters.size() * 2);
                for (uint32_t k = 0; k < clusters.size(); ++k)
                {
                    cpos[clusters[k]] = k;
                }
                std::vector<std::vector<uint32_t>> loc_rows(clusters.size());
                std::vector<std::vector<PID>> loc_pids(clusters.size());
                for (size_t r = 0; r < n; ++r)
                {
                    const PID pid = meta[r].pid;
                    const CID c = meta[r].cid;
                    const uint32_t k = cpos[c];
                    loc_rows[k].push_back(static_cast<uint32_t>(r));
                    loc_pids[k].push_back(pid);
                }
#pragma omp parallel for schedule(dynamic, 1)
                for (size_t k = 0; k < clusters.size(); ++k)
                {
                    const uint32_t c = clusters[k];
                    ivf::Cluster &cp = cluster_lst_[c];
                    const float *cur_centroid = centroids + c * dim_;
                    float *rc = &rotated_centroids_[c * padded_dim_];
                    const size_t coff = cluster_page_presums_[c] * SECTOR_LEN;
                    const auto &rows = loc_rows[k];
                    const size_t nc = rows.size();
                    if (order_mode ==
                        ClusterOrderMode::OrderedByCentroidDistance)
                    {
                        auto order = compute_centroid_order(
                            loc_pids[k].data(), nc,
                            [&](size_t i) { return gbuf + rows[i] * dim_; },
                            cur_centroid);
                        std::vector<PID> opids(nc);
                        std::vector<uint32_t> orows(nc);
                        for (size_t j = 0; j < nc; ++j)
                        {
                            opids[j] = loc_pids[k][order[j]];
                            orows[j] = rows[order[j]];
                        }
                        quantize_and_write_cluster_core(
                            cp, opids.data(), nc,
                            [&](size_t i) { return gbuf + orows[i] * dim_; },
                            cur_centroid, rc, config, coff,
                            /*batch_parallel=*/false);
                    }
                    else
                    {
                        quantize_and_write_cluster_core(
                            cp, loc_pids[k].data(), nc,
                            [&](size_t i) { return gbuf + rows[i] * dim_; },
                            cur_centroid, rc, config, coff,
                            /*batch_parallel=*/false);
                    }
                }
            }
            else
            {
                // Big cluster: the solo file is already in increasing-pid
                // order and its rows are 0..n-1. Quantize with intra-cluster
                // (batch) parallelism, one cluster at a time, so that only one
                // large buffer is resident.
                const uint32_t c = file_clusters[f][0];
                ivf::Cluster &cp = cluster_lst_[c];
                const float *cur_centroid = centroids + c * dim_;
                float *rc = &rotated_centroids_[c * padded_dim_];
                const size_t coff = cluster_page_presums_[c] * SECTOR_LEN;
                std::vector<PID> pids(n);
                for (size_t r = 0; r < n; ++r) pids[r] = meta[r].pid;

                if (order_mode == ClusterOrderMode::OrderedByCentroidDistance)
                {
                    auto order = compute_centroid_order(
                        pids.data(), n,
                        [&](size_t i) { return gbuf + i * dim_; }, cur_centroid);
                    std::vector<PID> opids(n);
                    std::vector<size_t> orows(n);
                    for (size_t j = 0; j < n; ++j)
                    {
                        opids[j] = pids[order[j]];
                        orows[j] = order[j];
                    }
                    quantize_and_write_cluster_core(
                        cp, opids.data(), n,
                        [&](size_t i) { return gbuf + orows[i] * dim_; },
                        cur_centroid, rc, config, coff,
                        /*batch_parallel=*/true);
                }
                else
                {
                    quantize_and_write_cluster_core(
                        cp, pids.data(), n,
                        [&](size_t i) { return gbuf + i * dim_; }, cur_centroid,
                        rc, config, coff, /*batch_parallel=*/true);
                }
            }

            std::free(gbuf);
            fs::remove(vec_path(f), ec);
            fs::remove(meta_path(f), ec);
        }

        LOG(INFO) << "[timing] P2 quantize (" << num_files
                  << " files): " << phase_sw.get_elapsed_sec() << " s";

        // ---- P3: cleanup (caller then calls save_base) --------------------
        fs::remove_all(scratch_dir, ec);
        LOG(INFO) << "[timing] construct_invlist_streaming total: "
                  << total_sw.get_elapsed_sec() << " s";
        LOG(INFO) << "streaming build: done.";
    }

    // Persist the coarse-quantizer-independent base index. Layout (see
    // kBaseMagic comment): magic+version, metadata, cluster sizes, rotator,
    // rotated centroids, mem_batch_data, ids. No initializer payload.
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::save_base(
        const std::string &base_index_file)
    {
        LOG(INFO) << "Saving base index to " << base_index_file;

        if (rotated_centroids_.size() != num_cluster_ * padded_dim_)
        {
            LOG(ERROR) << "save_base: rotated_centroids_ not populated; "
                          "call construct_invlist first";
            std::exit(1);
        }

        std::ofstream output(base_index_file, std::ios::binary);
        if (!output.is_open())
        {
            LOG(ERROR) << "Cannot open file " << base_index_file
                       << " for writing base index";
            std::exit(1);
        }

        // An ExBits index stays at version 1, byte for byte; only the Raw
        // store needs the version-2 fields. The 64-bit point-id build always
        // writes version 3.
        const uint32_t magic = kBaseMagic;
#if defined(RABITQ_PID64)
        const uint32_t version = kBaseVersionPid64;
#else
        const uint32_t version = (ssd_store_ == SsdStore::ExBits)
                                     ? kBaseVersion
                                     : kBaseVersionWithStore;
#endif
        output.write(reinterpret_cast<const char *>(&magic), sizeof(uint32_t));
        output.write(reinterpret_cast<const char *>(&version), sizeof(uint32_t));

        output.write(reinterpret_cast<const char *>(&num_), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(&dim_), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(&num_cluster_), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(&ex_bits_), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(&type_), sizeof(type_));
        output.write(reinterpret_cast<const char *>(&metric_type_),
                     sizeof(metric_type_));
        if (version != kBaseVersion)
        {
            const auto store_u = static_cast<uint32_t>(ssd_store_);
            const auto elem_u = static_cast<uint32_t>(raw_elem_);
            output.write(reinterpret_cast<const char *>(&store_u), sizeof(uint32_t));
            output.write(reinterpret_cast<const char *>(&elem_u), sizeof(uint32_t));
        }

        std::vector<size_t> cluster_sizes;
        cluster_sizes.reserve(num_cluster_);
        for (const auto &cur_cluster : cluster_lst_)
        {
            cluster_sizes.push_back(cur_cluster.num());
        }
        output.write(reinterpret_cast<const char *>(cluster_sizes.data()),
                     static_cast<long>(sizeof(size_t) * num_cluster_));

        rotator_->save(output);

        // Shared rotated-centroids block -- build_coarse reads this back.
        output.write(
            reinterpret_cast<const char *>(rotated_centroids_.data()),
            static_cast<long>(sizeof(float) * num_cluster_ * padded_dim_));

        output.write(mem_batch_data_,
                     static_cast<long>(mem_batch_data_bytes(cluster_sizes)));
        output.write(reinterpret_cast<const char *>(ids_),
                     static_cast<long>(ids_bytes()));
        output.close();
    }

    // Read just the <base> header (metadata + rotator + rotated centroids),
    // skipping the cluster-sizes block and stopping before mem_batch_data.
    // build_coarse uses this -- it needs neither the inverted list nor the
    // SSD file.
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::load_base_meta(
        const std::string &base_index_file)
    {
        LOG(INFO) << "Loading base index header from " << base_index_file;

        std::ifstream input(base_index_file, std::ios::binary);
        if (!input.is_open())
        {
            LOG(ERROR) << "Failed to open base index file: "
                       << base_index_file;
            std::exit(1);
        }

        uint32_t magic = 0;
        uint32_t version = 0;
        input.read(reinterpret_cast<char *>(&magic), sizeof(uint32_t));
        input.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));
        if (magic != kBaseMagic)
        {
            LOG(ERROR) << "Not a split-storage base index (bad magic): "
                       << base_index_file;
            std::exit(1);
        }

        input.read(reinterpret_cast<char *>(&num_), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&dim_), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&num_cluster_), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&ex_bits_), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&type_), sizeof(type_));
        input.read(reinterpret_cast<char *>(&metric_type_),
                   sizeof(metric_type_));
        // build_coarse ignores the SSD store, but its fields sit before the
        // rotator, so they are read to stay aligned.
        read_base_store_fields(input, version, base_index_file);

        rotator_ =
            choose_rotator<float>(dim_, type_, round_up_to_multiple(dim_, 64));
        padded_dim_ = rotator_->size();

        // Skip the cluster-sizes block (build_coarse does not need it).
        input.seekg(
            static_cast<std::streamoff>(sizeof(size_t) * num_cluster_),
            std::ios::cur);

        rotator_->load(input);

        rotated_centroids_.assign(num_cluster_ * padded_dim_, 0.0F);
        input.read(reinterpret_cast<char *>(rotated_centroids_.data()),
                   static_cast<std::streamsize>(sizeof(float) * num_cluster_ *
                                                padded_dim_));
        if (!input)
        {
            LOG(ERROR) << "Failed to read base index header from "
                       << base_index_file;
            std::exit(1);
        }
        input.close();
        std::cout << "Base index header loaded\n";
    }

    // Build a coarse quantizer over the rotated centroids loaded from <base>:
    // set the coarse parameters, construct initer_, and feed it the centroids.
    // Precondition: load_base_meta() (or load_base()) has already run.
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::build_coarse_quantizer(
        CoarseKind coarse_kind, ivf::InitializerType initializer_type,
        const std::string &inner_centroids_path,
        const std::string &inner_cids_path)
    {
        if (rotated_centroids_.size() != num_cluster_ * padded_dim_)
        {
            LOG(ERROR) << "build_coarse_quantizer: base not loaded; call "
                          "load_base_meta first";
            std::exit(1);
        }

        coarse_kind_ = coarse_kind;
        inner_centroids_path_ = inner_centroids_path;
        inner_cids_path_ = inner_cids_path;

        // For Flat/HNSW resolve Auto against the cluster count, then pin
        // coarse_kind_ to the resolved family so save_coarse records it.
        if (coarse_kind_ != CoarseKind::IRQ)
        {
            initializer_type_ =
                ivf::resolve_initializer_type(initializer_type, num_cluster_);
            coarse_kind_ = (initializer_type_ == ivf::InitializerType::HNSW)
                               ? CoarseKind::HNSW
                               : CoarseKind::Flat;
        }

        build_initializer();
        initer_->add_vectors(rotated_centroids_.data());
    }

    // Persist the coarse quantizer. Layout (see kCoarseMagic comment):
    // magic+version+kind, the base-pairing header, then the initializer
    // payload (Flat inline; HNSW/IRQ via a <coarse>.hnsw / <coarse>.irq
    // sidecar written by initer_->save).
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::save_coarse(
        const std::string &coarse_index_file)
    {
        LOG(INFO) << "Saving coarse quantizer to " << coarse_index_file;

        if (initer_ == nullptr)
        {
            LOG(ERROR) << "save_coarse: no coarse quantizer built; call "
                          "build_coarse_quantizer first";
            std::exit(1);
        }

        // Drop stale sidecars from a previous build so the IRQ/HNSW loader
        // can never pick up a leftover written for a different coarse kind.
        std::remove((coarse_index_file + ".hnsw").c_str());
        std::remove((coarse_index_file + ".irq").c_str());

        std::ofstream output(coarse_index_file, std::ios::binary);
        if (!output.is_open())
        {
            LOG(ERROR) << "Cannot open file " << coarse_index_file
                       << " for writing coarse quantizer";
            std::exit(1);
        }

        const uint32_t magic = kCoarseMagic;
        const uint32_t version = kCoarseVersion;
        const uint32_t kind = static_cast<uint32_t>(coarse_kind_);
        output.write(reinterpret_cast<const char *>(&magic), sizeof(uint32_t));
        output.write(reinterpret_cast<const char *>(&version), sizeof(uint32_t));
        output.write(reinterpret_cast<const char *>(&kind), sizeof(uint32_t));
        output.write(reinterpret_cast<const char *>(&num_cluster_), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(&dim_), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(&padded_dim_), sizeof(size_t));
        output.write(reinterpret_cast<const char *>(&metric_type_),
                     sizeof(metric_type_));

        const uint64_t fp = centroids_fingerprint(rotated_centroids_);
        output.write(reinterpret_cast<const char *>(&fp), sizeof(uint64_t));

        initer_->save(output, coarse_index_file.c_str());
        output.close();
    }

    // Full base load for querying: metadata + rotator + rotated centroids +
    // mem_batch_data + ids, and opens <base>.ssd. The coarse quantizer is not
    // read here; load_coarse() loads it.
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::load_base(
        const std::string &base_index_file, const std::string &ssd_index_file,
        size_t mem_dim)
    {
        LOG(INFO) << "Loading base index from " << base_index_file
                  << " and ssd index file: " << ssd_index_file;

        ssd_index_file_ = ssd_index_file;
        file_reader_ = std::make_shared<LinuxAlignedFileReader>();
        file_reader_->open(ssd_index_file_, false, false);

        std::ifstream input(base_index_file, std::ios::binary);
        if (!input.is_open())
        {
            LOG(ERROR) << "Failed to open base index file: "
                       << base_index_file;
            std::exit(1);
        }

        uint32_t magic = 0;
        uint32_t version = 0;
        input.read(reinterpret_cast<char *>(&magic), sizeof(uint32_t));
        input.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));
        if (magic != kBaseMagic)
        {
            LOG(ERROR) << "Not a split-storage base index (bad magic): "
                       << base_index_file;
            std::exit(1);
        }

        input.read(reinterpret_cast<char *>(&num_), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&dim_), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&num_cluster_), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&ex_bits_), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&type_), sizeof(type_));
        input.read(reinterpret_cast<char *>(&metric_type_), sizeof(metric_type_));
        // The SSD store must be known before init_layout_metadata below: it
        // decides the record size and hence the page layout.
        read_base_store_fields(input, version, base_index_file);

        requested_mem_dim_ = mem_dim;

        rotator_ =
            choose_rotator<float>(dim_, type_, round_up_to_multiple(dim_, 64));
        padded_dim_ = rotator_->size();
        init_layout_metadata();
        log_ssd_layout();

        std::vector<size_t> cluster_sizes(num_cluster_, 0);
        input.read(reinterpret_cast<char *>(cluster_sizes.data()),
                   static_cast<std::streamsize>(sizeof(size_t) *
                                                cluster_sizes.size()));

        const size_t total_count = std::accumulate(
            cluster_sizes.begin(), cluster_sizes.end(), static_cast<size_t>(0));
        if (total_count != num_)
        {
            LOG(ERROR) << "The sum of cluster sizes != total number of points";
            std::exit(1);
        }
        check_ssd_pages(cluster_sizes);

        rotator_->load(input);

        rotated_centroids_.assign(num_cluster_ * padded_dim_, 0.0F);
        input.read(reinterpret_cast<char *>(rotated_centroids_.data()),
                   static_cast<std::streamsize>(sizeof(float) * num_cluster_ *
                                                padded_dim_));

        free_memory();
        allocate_invlist_memory(cluster_sizes);
        input.read(mem_batch_data_, static_cast<std::streamsize>(
                                        mem_batch_data_bytes(cluster_sizes)));
        input.read(reinterpret_cast<char *>(ids_),
                   static_cast<std::streamsize>(ids_bytes()));
        if (!input)
        {
            LOG(ERROR) << "Failed to read base index payload (MEMDIM="
                       << ((mem_dim == 0) ? padded_dim_ : mem_dim)
                       << "). Check that querying uses the same MEMDIM as "
                          "build_invlist.";
            std::exit(1);
        }

        char extra = 0;
        if (input.read(&extra, 1))
        {
            LOG(ERROR) << "Base index file has unexpected extra bytes "
                       << "(MEMDIM mismatch?). MEMDIM="
                       << ((mem_dim == 0) ? padded_dim_ : mem_dim);
            std::exit(1);
        }

        init_clusters(cluster_sizes);
        input.close();
        std::cout << "Base index loaded\n";
    }

    // Load a coarse quantizer built by build_coarse. Must run AFTER
    // load_base(): it validates that the coarse file pairs with this base
    // (matching num_cluster / dim / padded_dim / metric, and an FNV-1a
    // fingerprint of the rotated centroids) before constructing initer_.
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::load_coarse(
        const std::string &coarse_index_file)
    {
        LOG(INFO) << "Loading coarse quantizer from " << coarse_index_file;

        if (rotated_centroids_.size() != num_cluster_ * padded_dim_)
        {
            LOG(ERROR) << "load_coarse: base not loaded; call load_base first";
            std::exit(1);
        }

        std::ifstream input(coarse_index_file, std::ios::binary);
        if (!input.is_open())
        {
            LOG(ERROR) << "Failed to open coarse quantizer file: "
                       << coarse_index_file;
            std::exit(1);
        }

        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t kind = 0;
        input.read(reinterpret_cast<char *>(&magic), sizeof(uint32_t));
        input.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));
        input.read(reinterpret_cast<char *>(&kind), sizeof(uint32_t));
        if (magic != kCoarseMagic)
        {
            LOG(ERROR) << "Not a split-storage coarse quantizer (bad magic): "
                       << coarse_index_file;
            std::exit(1);
        }
        if (version != kCoarseVersion)
        {
            LOG(ERROR) << "Unsupported coarse quantizer version " << version
                       << " (expected " << kCoarseVersion << ")";
            std::exit(1);
        }
        if (kind > static_cast<uint32_t>(CoarseKind::IRQ))
        {
            LOG(ERROR) << "Coarse quantizer file has bad coarse kind " << kind;
            std::exit(1);
        }

        size_t file_num_cluster = 0;
        size_t file_dim = 0;
        size_t file_padded_dim = 0;
        MetricType file_metric = rabitqlib::METRIC_L2;
        uint64_t file_fp = 0;
        input.read(reinterpret_cast<char *>(&file_num_cluster), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&file_dim), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&file_padded_dim), sizeof(size_t));
        input.read(reinterpret_cast<char *>(&file_metric), sizeof(file_metric));
        input.read(reinterpret_cast<char *>(&file_fp), sizeof(uint64_t));

        if (file_num_cluster != num_cluster_ || file_dim != dim_ ||
            file_padded_dim != padded_dim_ || file_metric != metric_type_)
        {
            LOG(ERROR) << "Coarse quantizer does not match the loaded base "
                       << "(coarse: num_cluster=" << file_num_cluster
                       << " dim=" << file_dim
                       << " padded_dim=" << file_padded_dim
                       << " | base: num_cluster=" << num_cluster_
                       << " dim=" << dim_ << " padded_dim=" << padded_dim_
                       << ")";
            std::exit(1);
        }
        if (file_fp != centroids_fingerprint(rotated_centroids_))
        {
            LOG(ERROR) << "Coarse quantizer was built for a different base "
                          "(centroid fingerprint mismatch). Rebuild it "
                          "against this base with build_coarse.";
            std::exit(1);
        }

        coarse_kind_ = static_cast<CoarseKind>(kind);
        initializer_type_ = (coarse_kind_ == CoarseKind::HNSW)
                                ? ivf::InitializerType::HNSW
                                : ivf::InitializerType::Flat;
        LOG(INFO) << "Coarse quantizer kind: "
                  << (coarse_kind_ == CoarseKind::IRQ
                          ? "irq"
                          : (coarse_kind_ == CoarseKind::HNSW ? "hnsw"
                                                              : "flat"));

        build_initializer();
        initer_->load(input, coarse_index_file.c_str());
        input.close();
        std::cout << "Coarse quantizer loaded\n";
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::init_query_buf(QueryBuffer &buf,
                                                         size_t)
    {
        buf.rotated_query =
            memory::align_allocate<64, float, true>(padded_dim_ * sizeof(float));
        buf.raw_query =
            (ssd_store_ == SsdStore::Raw)
                ? memory::align_allocate<64, float, true>(padded_dim_ * sizeof(float))
                : nullptr;
        buf.request_slot_bytes = page_num_per_io_ * SECTOR_LEN;
        buf.sector_scratch = memory::align_allocate<SECTOR_LEN, char, true>(
            N_REQ_BUF * buf.request_slot_bytes);
        buf.est_distance.resize(fastscan::kBatchSize);
        buf.low_distance.resize(fastscan::kBatchSize);
        buf.ip_x0_qr.resize(fastscan::kBatchSize);
        buf.accu_arr.resize(fastscan::kBatchSize);
        buf.init_page_candidate_storage(data_num_in_one_page_,
                                        INITIAL_PAGE_SLOT_CAPACITY);
        buf.reset();
    }

    template <class PendingSet>
    inline QueryBuffer *IVFSSD_Index<PendingSet>::pop_query_buf()
    {
        QueryBuffer *data = thread_query_buffer_pool_.pop();
        while (data == thread_query_buffer_pool_.null_T)
        {
            thread_query_buffer_pool_.wait_for_push_notify();
            data = thread_query_buffer_pool_.pop();
        }
        return data;
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::push_query_buf(QueryBuffer *buf)
    {
        thread_query_buffer_pool_.push(buf);
        thread_query_buffer_pool_.push_notify_all();
    }

    template <class PendingSet>
    inline PageCandidate_IO_Queue<PendingSet> *
    IVFSSD_Index<PendingSet>::pop_ioqueue_buf()
    {
        auto *data = thread_ioqueue_buffer_pool_.pop();
        while (data == thread_ioqueue_buffer_pool_.null_T)
        {
            thread_ioqueue_buffer_pool_.wait_for_push_notify();
            data = thread_ioqueue_buffer_pool_.pop();
        }
        return data;
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::push_ioqueue_buf(
        PageCandidate_IO_Queue<PendingSet> *buf)
    {
        thread_ioqueue_buffer_pool_.push(buf);
        thread_ioqueue_buffer_pool_.push_notify_all();
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::init_buffers(uint64_t nthreads,
                                                       size_t topk)
    {
        const uint64_t n_buffers = nthreads;
        // Report the active per-thread queue depth once, on stderr. A
        // requested depth above N_REQ_BUF is clamped; the clamp is reported
        // explicitly rather than applied silently.
        {
            const size_t qd_req = qd_per_thread_requested();
            const size_t qd_cap = boundary_qd_per_thread();
            std::cerr << "[qd] per_thread_req=" << qd_req
                      << " nthreads=" << nthreads << " qd_cap=" << qd_cap
                      << " aggregate=" << qd_cap * nthreads;
            if (qd_req != qd_cap)
            {
                std::cerr << "  [WARNING] requested qd=" << qd_req
                          << " CLAMPED to N_REQ_BUF=" << N_REQ_BUF
                          << " (raise IO_QUEUE_DEPTH to go deeper)";
            }
            std::cerr << std::endl;
        }
        thread_query_buffer_pool_.null_T = nullptr;
        thread_ioqueue_buffer_pool_.null_T = nullptr;

        for (uint64_t i = 0; i < n_buffers; ++i)
        {
            auto *buf = new QueryBuffer();
            init_query_buf(*buf, topk);
            thread_data_bufs_.push_back(buf);
            thread_query_buffer_pool_.push(buf);

            auto *io_queue = new PageCandidate_IO_Queue<PendingSet>();
            thread_ioqueue_buffer_pool_.push(io_queue);
            thread_ioqueue_bufs_.push_back(io_queue);
        }

        LOG(INFO) << "Initialized " << n_buffers
                  << " query buffers for IVFSSD_Index.";
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::destroy_buffers()
    {
        while (!thread_data_bufs_.empty())
        {
            QueryBuffer *buf = thread_data_bufs_.back();
            std::free(buf->rotated_query);
            std::free(buf->raw_query);
            std::free(buf->sector_scratch);
            thread_data_bufs_.pop_back();
            delete buf;
        }
        while (!thread_ioqueue_bufs_.empty())
        {
            auto *io_queue = thread_ioqueue_bufs_.back();
            thread_ioqueue_bufs_.pop_back();
            delete io_queue;
        }
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::init_cluster_scan_state(
        const ivf::Cluster &cur_cluster, CID cid, ClusterScanState &state)
    {
        // Page-id tracking advances for every point in the cluster, regardless
        // of whether that point survives the mem-level prune. Only candidate
        // insertion is conditional on low_distance <= distk.
        state.last_page = -1;
        state.cur_page_slot_id = kSlotMax;
        state.cur_inner_pid = -1;
        state.cur_page = static_cast<PageIdx>(cluster_page_presums_[cid]);
        state.in_page_idx = 0;
        state.remaining_in_page = static_cast<int>(data_num_in_one_page_);
        state.mem_data = cur_cluster.batch_data();
        state.ids = cur_cluster.ids();
        state.scan_cand_inserted_local = 0;
        state.scan_pages_acq_local = 0;
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::scan_one_batch(
        const ivf::Cluster &cur_cluster,
        const SplitBatchQuery_nprobe<float> &q_obj, size_t probe_idx,
        float distk, QueryBuffer *query_buf,
        PageCandidate_IO_Queue<PendingSet> *io_queue, bool use_hacc,
        SearchStats *stats, ClusterScanState &state, size_t iter)
    {
        const size_t n_iter =
            div_round_up(cur_cluster.num(), fastscan::kBatchSize);

        const size_t batch_data_bytes = BatchDataMap<float>::data_bytes(mem_dim_);
        const int data_num_in_one_page = static_cast<int>(data_num_in_one_page_);
        const int page_num_per_io = static_cast<int>(page_num_per_io_);

        const size_t batch_size = std::min(
            fastscan::kBatchSize,
            cur_cluster.num() - iter * fastscan::kBatchSize);

        mem_batch_binary_estdist(state.mem_data, q_obj, probe_idx,
                                 query_buf->est_distance.data(),
                                 query_buf->low_distance.data(),
                                 query_buf->ip_x0_qr.data(), query_buf,
                                 use_hacc);

        for (size_t idx = 0; idx < batch_size; ++idx)
        {
            ++state.cur_inner_pid;
            if (state.remaining_in_page == 0)
            {
                state.cur_page += page_num_per_io;
                state.remaining_in_page = data_num_in_one_page;
                state.in_page_idx = 0;
            }
            const PageIdx cur_page_id = state.cur_page;
            const int cur_in_page_idx = state.in_page_idx;
            --state.remaining_in_page;
            ++state.in_page_idx;

            if (query_buf->low_distance[idx] > distk)
            {
                continue;
            }

            const size_t prefetch_pid =
                static_cast<size_t>(state.cur_inner_pid + 32);
            if (prefetch_pid < cur_cluster.num())
            {
                __builtin_prefetch(state.ids + prefetch_pid, 0, 1);
            }

            const PID true_data_id = state.ids[state.cur_inner_pid];

            if (cur_page_id != state.last_page)
            {
                if (state.last_page != -1)
                {
                    push_pagecand_into_candidate_pool(
                        query_buf, io_queue, state.cur_page_slot_id, stats);
                }
                state.last_page = cur_page_id;
                state.cur_page_slot_id =
                    io_queue->acquire_page_candidate_slot(query_buf);
                ++state.scan_pages_acq_local;

                PageCandidates &page_cand =
                    query_buf->page_slots[state.cur_page_slot_id];
                page_cand.probe_idx = static_cast<CID>(probe_idx);
                page_cand.page_id = cur_page_id;
                page_cand.candidate_num = 0;
                page_cand.page_lower_bound = query_buf->low_distance[idx];
            }

            PageCandidates &page_cand =
                query_buf->page_slots[state.cur_page_slot_id];
            SingleCandidate *slot_candidates =
                query_buf->slot_candidates(state.cur_page_slot_id);
            SingleCandidate &cand =
                slot_candidates[page_cand.candidate_num++];
            ++state.scan_cand_inserted_local;
            cand.true_data_id = true_data_id;
            cand.in_page_idx = cur_in_page_idx;
            cand.mem_low_dist_ = query_buf->low_distance[idx];
            cand.mem_est_dist_ = query_buf->est_distance[idx];
            cand.mem_ip_x0_qr_ = query_buf->ip_x0_qr[idx];
            page_cand.page_lower_bound = std::min(
                page_cand.page_lower_bound, query_buf->low_distance[idx]);
        }

        if (iter + 1 < n_iter)
        {
            __builtin_prefetch(state.mem_data + batch_data_bytes, 0, 1);
        }
        state.mem_data += batch_data_bytes;
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::finalize_cluster_scan(
        QueryBuffer *query_buf, PageCandidate_IO_Queue<PendingSet> *io_queue,
        SearchStats *stats, ClusterScanState &state)
    {
        if (state.last_page != -1)
        {
            push_pagecand_into_candidate_pool(query_buf, io_queue,
                                              state.cur_page_slot_id, stats);
        }

        if (stats != nullptr)
        {
            stats->scan_candidates_inserted += state.scan_cand_inserted_local;
            stats->scan_pages_acquired += state.scan_pages_acq_local;
        }
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::query_one_boundary_qd(
        const float *query, size_t topk, size_t nprobe,
        PID *__restrict__ results, bool use_hacc,
        SearchStats *stats)
    {
        // Whole-query timer. Together with the coarse and drain timers below
        // these are the only three on the query path, and none is inside a loop.
        StopW total_stopw;

        QueryBuffer *query_buf = pop_query_buf();
        query_buf->reset();
        auto *io_queue = pop_ioqueue_buf();
        io_queue->reset(query_buf);

        // Per-thread in-flight cap (see boundary_qd_per_thread() in the
        // header): identical for every thread, cached after the first call.
        const size_t qd_cap = boundary_qd_per_thread();

        void *ctx = file_reader_->get_ctx();

        rotator_->rotate(query, query_buf->rotated_query);
        if (ssd_store_ == SsdStore::Raw)
        {
            // Raw records are compared with the query as given.
            std::memcpy(query_buf->raw_query, query, dim_ * sizeof(float));
            std::fill(query_buf->raw_query + dim_,
                      query_buf->raw_query + padded_dim_, 0.0F);
        }

        auto &centroid_dist = query_buf->centroid_dist;
        centroid_dist.resize(nprobe);
        StopW coarse_stopw;
        initer_->centroids_distances(query_buf->rotated_query, nprobe,
                                     centroid_dist);
        const double coarse_us = coarse_stopw.get_elapsed_micro();
        if (stats != nullptr)
        {
            // Roofline buckets: each thread_local is written by the
            // Initializer subtype (Flat/HNSW: only full_l2; IRQ: all three).
            stats->centroid_full_l2_dists += ivf::last_full_l2_dists;
            stats->centroid_fastscan_attempts += ivf::last_fastscan_attempts;
            stats->centroid_exbits_refines += ivf::last_exbits_refines;
            stats->coarse_us += coarse_us;
        }

        buffer::SearchBuffer<float> knns(topk);
        SplitBatchQuery_nprobe<float> q_obj(query_buf->rotated_query,
                                            padded_dim_, mem_dim_, ex_bits_,
                                            nprobe, metric_type_, use_hacc);

        // ||q|| of the rotated query. Needed only on the METRIC_IP + IRQ
        // coarse path, where the coarse step returns the normalized-space
        // distance dist_IRQ and <q,c> / ||q-c|| are reconstructed from it plus
        // the cached centroid norms. Constant over nprobe, so computed once
        // per query.
        const float qnorm =
            (metric_type_ == METRIC_IP && coarse_kind_ == CoarseKind::IRQ)
                ? std::sqrt(dot_product<float>(query_buf->rotated_query,
                                               query_buf->rotated_query,
                                               padded_dim_))
                : 0.0F;

        for (size_t i = 0; i < nprobe; ++i)
        {
            const CID cid = static_cast<CID>(centroid_dist[i].id);
            const float dist = centroid_dist[i].distance;
            const ivf::Cluster &cur_cluster = cluster_lst_[cid];
            if (stats != nullptr)
            {
                stats->num_points_probed += cur_cluster.num();
            }

            if (metric_type_ == METRIC_L2)
            {
                q_obj.set_g_add(i, dist);
            }
            else if (metric_type_ == METRIC_IP)
            {
                // set_g_add(IP) expects g_error = ||q-c|| and g_add = -<q,c>
                // (the negation happens inside set_g_add). How those two are
                // obtained depends on the coarse quantizer.
                float g_add_ip;
                float g_error;
                if (coarse_kind_ == CoarseKind::IRQ)
                {
                    // IRQ coarse runs in normalized space, so `dist` is
                    // dist_IRQ = ||q_hat - c_hat||. Reconstruct from norms:
                    //   <q,c>   = (2 - dist_IRQ^2) * ||q|| * ||c|| / 2
                    //   ||q-c|| = sqrt(||q||^2 + ||c||^2 - 2<q,c>)
                    const auto *irq = static_cast<
                        const ivf_irq::IvfRabitqInitializer *>(initer_);
                    const float cnorm = irq->centroid_norm(cid);
                    g_add_ip = 0.5F * (2.0F - dist * dist) * qnorm * cnorm;
                    g_error = std::sqrt(std::max(
                        0.0F,
                        (qnorm * qnorm) + (cnorm * cnorm) -
                            (2.0F * g_add_ip)));
                }
                else
                {
                    // Flat / HNSW coarse: `dist` is already the raw ||q-c||
                    // and the raw centroid is available, so take the exact
                    // dot product directly.
                    g_add_ip = dot_product<float>(query_buf->rotated_query,
                                                  initer_->centroid(cid),
                                                  padded_dim_);
                    g_error = dist;
                }
                q_obj.set_g_add(i, g_error, g_add_ip);
            }
            else
            {
                LOG(ERROR) << "Unsupported metric type";
                crash();
            }

            // Cluster search runs at alpha=1.0 (unpruned, no extra
            // pruning) for every probed cluster; only the drain phase prunes,
            // via drain_alpha_. alpha_distk is still recomputed from the
            // current knns at each alpha_quantile_dist call, so a re-rank
            // within a cluster can still tighten this (alpha=1.0) threshold to
            // the live topk-th distance.
            const float cluster_alpha = 1.0F;

            // For each 32-point batch refresh alpha_distk, scan just that
            // batch (Phase A), then immediately poll/re-rank (Phase B) +
            // submit (Phase C). The mem-prune threshold is refreshed every
            // batch, so a tighter alpha_distk produced by an earlier batch's
            // re-rank can prune later batches harder.
            {
                const size_t n_iter =
                    div_round_up(cur_cluster.num(), fastscan::kBatchSize);
                ClusterScanState state;
                init_cluster_scan_state(cur_cluster, cid, state);

                for (size_t iter = 0; iter < n_iter; ++iter)
                {
                    // Phase A (this batch): scan with the freshest
                    // alpha_distk. cluster_alpha is fixed per cluster (per
                    // probe_idx), but alpha_quantile_dist re-reads knns on
                    // every call, so the per-batch threshold tracks the
                    // current KNN buffer.
                    const float distk =
                        alpha_quantile_dist(knns, topk, cluster_alpha);
                    scan_one_batch(cur_cluster, q_obj, i, distk, query_buf,
                                   io_queue, use_hacc, stats, state, iter);

                    int n_completed = 0;
                    io_queue->poll_all_completed_out_of_order(
                        file_reader_.get(), ctx, query_buf, n_completed,
                        empty_ssd_);
                    for (int j = 0; j < n_completed; ++j)
                    {
                        const SlotID slot_id = query_buf->completed_slots[j];
                        complete_distance_for_page_candidate(
                            slot_id, query_buf, io_queue, q_obj, knns, stats);
                    }
                    const size_t in_flight = io_queue->in_flight_size();
                    if (in_flight < qd_cap)
                    {
                        io_queue->submit_top_io_pagecandidates(
                            qd_cap - in_flight, query_buf, file_reader_.get(),
                            ctx,
                            alpha_quantile_dist(knns, topk, cluster_alpha),
                            empty_ssd_, stats);
                    }
                }
                // finalize: push the last open page.
                finalize_cluster_scan(query_buf, io_queue, stats, state);
            }
        }

        // Drain phase: poll/submit until in-flight = 0 AND pending is empty
        // (pending is drained either by submission or by alpha_distk-driven
        // pruning inside submit_top_io_pagecandidates).
        //
        // The wall time of this phase is the "exposed IO" signal: the main
        // loop above never blocks on IO (poll is a non-blocking peek), so any
        // SSD read that cluster-scan compute failed to hide ends up draining
        // here. drain_us / SSD-pipeline-wall therefore classifies the
        // operating point as IO-bound (large) vs compute-bound (~0).
        //
        // Drain runs PAST the last cluster and is the ONLY phase that prunes:
        // submission here uses drain_alpha_ (cluster search ran at alpha=1.0).
        // Pending pages whose lower bound exceeds the drain_alpha-quantile
        // distance get dropped instead of read, cutting tail-end IOs that
        // would mostly miss the topk anyway. The rerank stage still uses
        // the true distk = knns.top_dist() so genuinely good candidates
        // whose IO already completed are never dropped after-the-fact.
        StopW drain_stopw;
        while (!io_queue->empty())
        {
            int n_completed = 0;
            io_queue->poll_all_completed_out_of_order(
                file_reader_.get(), ctx, query_buf, n_completed, empty_ssd_);
            for (int j = 0; j < n_completed; ++j)
            {
                const SlotID slot_id = query_buf->completed_slots[j];
                complete_distance_for_page_candidate(slot_id, query_buf,
                                                     io_queue, q_obj, knns,
                                                     stats);
            }
            const size_t in_flight = io_queue->in_flight_size();
            if (in_flight < qd_cap)
            {
                io_queue->submit_top_io_pagecandidates(
                    qd_cap - in_flight, query_buf,
                    file_reader_.get(), ctx,
                    alpha_quantile_dist(knns, topk, drain_alpha_),
                    empty_ssd_, stats);
            }
        }

        const double drain_us = drain_stopw.get_elapsed_micro();

        knns.copy_results(results);

        if (stats != nullptr)
        {
            stats->drain_us += drain_us;
            stats->total_us += total_stopw.get_elapsed_micro();
            // Page-candidate pool high-water mark (per-thread QueryBuffer,
            // grows monotonically): proxy for candidate-pool memory footprint.
            stats->peak_page_slots = query_buf->page_slots.size();
        }

        push_query_buf(query_buf);
        push_ioqueue_buf(io_queue);
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::batch_search(
        const float *queries, size_t nqueries, size_t topk, size_t nprobe,
        PID *__restrict__ results, bool use_hacc, SearchStats *stats)
    {
        nprobe = std::min(nprobe, num_cluster_);

#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < nqueries; ++i)
        {
            query_one_boundary_qd(queries + i * dim_, topk, nprobe,
                                  results + i * topk, use_hacc,
                                  stats ? &stats[i] : nullptr);
        }
    }
    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::mem_batch_binary_estdist(
        const char *mem_batch_data, const SplitBatchQuery_nprobe<float> &q_obj,
        const size_t probe_idx, float *est_distance, float *low_distance,
        float *ip_x0_qr, QueryBuffer *query_buf, bool use_hacc)
    {
        int32_t *accu_arr = query_buf->accu_arr.data();
        std::memset(accu_arr, 0, sizeof(int32_t) * fastscan::kBatchSize);

        ConstBatchDataMap<float> cur_batch(mem_batch_data, mem_dim_);
        ConstRowMajorArrayMap<float> f_add_arr(cur_batch.f_add(), 1,
                                               fastscan::kBatchSize);
        ConstRowMajorArrayMap<float> f_rescale_arr(cur_batch.f_rescale(), 1,
                                                   fastscan::kBatchSize);
        ConstRowMajorArrayMap<float> f_error_arr(cur_batch.f_error(), 1,
                                                 fastscan::kBatchSize);
        RowMajorArrayMap<float> est_dist_arr(est_distance, 1,
                                             fastscan::kBatchSize);
        RowMajorArrayMap<float> ip_x0_qr_arr(ip_x0_qr, 1,
                                             fastscan::kBatchSize);
        RowMajorArrayMap<float> low_dist_arr(low_distance, 1,
                                             fastscan::kBatchSize);

        if (use_hacc)
        {
            std::array<int32_t, fastscan::kBatchSize> accu_res;
            fastscan::accumulate_hacc(cur_batch.bin_code(), q_obj.lut(),
                                      accu_res.data(), mem_dim_);
            for (size_t i = 0; i < fastscan::kBatchSize; ++i)
            {
                accu_arr[i] = accu_res[i];
            }
        }
        else
        {
            std::array<uint16_t, fastscan::kBatchSize> accu_res;
            const uint8_t *codes_ptr =
                reinterpret_cast<const uint8_t *>(cur_batch.bin_code());
            const uint8_t *lut_ptr =
                reinterpret_cast<const uint8_t *>(q_obj.lut());
            // A segment of D dims sums D/4 uint8 LUT entries (each <= 255) into
            // a uint16 lane that wraps on overflow, so (D/4) * 255 <= 65535.
            constexpr int kNoHaccSegmentDim = 1024;
            static_assert((kNoHaccSegmentDim / 4) * 255 <= 65535,
                          "segment too large for the uint16 fastscan accumulator");
            static_assert(kNoHaccSegmentDim % 64 == 0,
                          "segment must be a multiple of 64 for the SIMD kernel");
            int delta_dim = static_cast<int>(mem_dim_);
            while (delta_dim > 0)
            {
                const size_t scan_dim = std::min(kNoHaccSegmentDim, delta_dim);
                fastscan::accumulate_segment_no_repeat(codes_ptr, lut_ptr,
                                                       accu_res.data(), scan_dim);
                for (size_t i = 0; i < fastscan::kBatchSize; ++i)
                {
                    accu_arr[i] += accu_res[i];
                }
                delta_dim -= kNoHaccSegmentDim;
            }
        }

        RowMajorArrayMap<int32_t> batch_accu_arr(accu_arr, 1,
                                                 fastscan::kBatchSize);
        ip_x0_qr_arr = q_obj.delta() *
                           (batch_accu_arr.template cast<float>()) +
                       q_obj.sum_vl_lut();
        est_dist_arr = f_add_arr + q_obj.g_add(probe_idx) +
                       f_rescale_arr * (ip_x0_qr_arr + q_obj.k1xsumq());
        low_dist_arr = est_dist_arr - f_error_arr * q_obj.g_error(probe_idx);
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::push_pagecand_into_candidate_pool(
        QueryBuffer *query_buf, PageCandidate_IO_Queue<PendingSet> *io_queue,
        SlotID slot_id, SearchStats *stats)
    {
        PageCandidates &page_cand = query_buf->page_slots[slot_id];
        io_queue->push_into_submit_candidate_pool(slot_id,
                                                  page_cand.page_lower_bound,
                                                  stats);
    }

    template <class PendingSet>
    inline void IVFSSD_Index<PendingSet>::complete_distance_for_page_candidate(
        SlotID slot_id, QueryBuffer *query_buf,
        PageCandidate_IO_Queue<PendingSet> *io_queue,
        const SplitBatchQuery_nprobe<float> &q_obj,
        buffer::SearchBuffer<float> &knns, SearchStats *stats)
    {
        // One call here == one completed page IO handed to the main-pipeline
        // rerank stage. See SearchStats::main_pages_reranked doc: that counter
        // is per-PAGE. The per-RECORD work below is tallied into locals and
        // folded into stats once at the end, so the hot loop does no pointer
        // chasing and the timing phases stay undisturbed.
        if (stats != nullptr)
        {
            ++stats->main_pages_reranked;
        }
        PageCandidates &page_cand = query_buf->page_slots[slot_id];
        const char *data = io_queue->completed_data(query_buf, slot_id);

        uint64_t boost_dists = 0;  // # split_distance_boosting calls (real rerank compute)
        uint64_t knn_inserts = 0;  // # knns.insert calls

        // The re-rank stage uses the true distk = knns.top_dist(), never
        // alpha_distk. The IO for this page has already completed, so pruning
        // here with the tighter alpha_distk would save no work and could drop
        // a genuine topk result after its distance had been computed.
        // alpha_distk only throttles candidate generation (scan) and IO
        // submission upstream.
        float distk = knns.top_dist();

        if (ssd_store_ == SsdStore::Raw)
        {
            // Raw records: the exact distance to the unrotated query, on the
            // scale the in-memory estimator uses (squared L2, or 1 - <q, o>
            // for inner product). The rotation is orthonormal, so the
            // in-memory lower bound still bounds this distance and the cheap
            // early exit is kept. Records and query are both zero-padded to
            // padded_dim, which leaves the distance unchanged.
            const float *rq = query_buf->raw_query;
            const bool l2 = (metric_type_ == METRIC_L2);
            const bool u8 = (raw_elem_ == VecElemType::U8);
            const SingleCandidate *slot_candidates =
                query_buf->slot_candidates(slot_id);
            for (size_t cidx = 0; cidx < page_cand.candidate_num; ++cidx)
            {
                const SingleCandidate &cand = slot_candidates[cidx];
                if (cand.mem_low_dist_ > distk)
                {
                    continue;
                }
                const char *rec = data + cand.in_page_idx * one_data_ssd_bytes_;
                float full_dist = 0.0F;
                if (u8)
                {
                    const auto *v = reinterpret_cast<const uint8_t *>(rec);
                    full_dist = l2 ? l2sqr_f32_u8(rq, v, padded_dim_)
                                   : 1.0F - ip_f32_u8(rq, v, padded_dim_);
                }
                else
                {
                    const auto *v = reinterpret_cast<const float *>(rec);
                    full_dist = l2 ? euclidean_sqr<float>(rq, v, padded_dim_)
                                   : dot_product_dis<float>(rq, v, padded_dim_);
                }
                ++boost_dists;

                if (full_dist > distk)
                {
                    continue;
                }
                knns.insert(cand.true_data_id, full_dist);
                ++knn_inserts;
                distk = knns.top_dist();
            }
        }
        else if (ex_bits_ == 0 && ssd_dim_ == 0)
        {
            const SingleCandidate *slot_candidates =
                query_buf->slot_candidates(slot_id);
            for (size_t cidx = 0; cidx < page_cand.candidate_num; ++cidx)
            {
                const SingleCandidate &cand = slot_candidates[cidx];
                if (cand.mem_low_dist_ > distk)
                {
                    continue;
                }
                const float est_dist = cand.mem_est_dist_;
                if (est_dist > distk)
                {
                    continue;
                }
                knns.insert(cand.true_data_id, est_dist);
                ++knn_inserts;
                distk = knns.top_dist();
            }
            // No boosting on this path: mem_est_dist_ is used directly.
        }
        else if (ex_bits_ == 0)
        {
            LOG(ERROR) << "Not implemented: ex_bits_==0 && ssd_dim_!=0 case";
            std::exit(1);
        }
        else
        {
            const size_t probe_idx = static_cast<size_t>(page_cand.probe_idx);
            const SingleCandidate *slot_candidates =
                query_buf->slot_candidates(slot_id);
            for (size_t cidx = 0; cidx < page_cand.candidate_num; ++cidx)
            {
                const SingleCandidate &cand = slot_candidates[cidx];
                if (cand.mem_low_dist_ > distk)
                {
                    continue;
                }

                const char *data_ptr = data + cand.in_page_idx * one_data_ssd_bytes_;
                const uint64_t *ssd_bin_ptr =
                    reinterpret_cast<const uint64_t *>(data_ptr);
                const float ssd_ip_x0_qr = mask_ip_x0_q(
                    q_obj.rotated_query() + mem_dim_, ssd_bin_ptr, ssd_dim_);
                const float ip_x0_qr = cand.mem_ip_x0_qr_ + ssd_ip_x0_qr;

                const char *ex_data_ptr = data_ptr + (ssd_dim_ / 8);
                const float full_dist = split_distance_boosting(
                    ex_data_ptr, ip_func_, q_obj, probe_idx, padded_dim_,
                    ex_bits_, ip_x0_qr);
                ++boost_dists;

                if (full_dist > distk)
                {
                    continue;
                }
                knns.insert(cand.true_data_id, full_dist);
                ++knn_inserts;
                distk = knns.top_dist();
            }
        }

        // Fold the per-record tallies in once. candidate_num is exactly the
        // number of records both loops iterate over.
        if (stats != nullptr)
        {
            stats->main_records_examined += page_cand.candidate_num;
            stats->main_boost_dists += boost_dists;
            stats->main_knn_inserts += knn_inserts;
        }

        io_queue->release_completed_slot(query_buf, slot_id);
    }
} // namespace rabitqlib::ivf_ssd_boundary_qd_ms_detail
