#pragma once
//
// SSD IO backend for the query path: Linux libaio (io_setup / io_submit /
// io_getevents). Reads go straight into req.buf with io_prep_pread; O_DIRECT
// with SECTOR_LEN-aligned offsets, lengths and buffers satisfies libaio's
// alignment requirements.
//
// The interface is a per-thread context lifecycle for the reader, and batched
// submit + non-blocking batched reap for the search hot path. All
// backend-specific code lives ONLY in this header; the algorithm code stays
// backend-agnostic and passes the context as void*.
//
// The build path (build_invlist / build_coarse) does not use this backend --
// it writes the index with ordinary pwrite()/ofstream -- so the backend never
// affects index construction or the on-disk format.

#include <cstdint>
#include <cstring>

#include "rabitqlib/ssd_utils/ssd_index_defs.hpp" // IO_QUEUE_DEPTH, crash()
#include "rabitqlib/ssd_utils/log.hpp"

#include <libaio.h>
#include <ctime>

namespace rabitqlib::io_backend
{

// Human-readable name of the backend (for logging and the result CSVs).
inline const char *name()
{
    return "libaio";
}

// ---- per-thread context lifecycle -----------------------------------------
// Create a per-thread IO context and return it as an opaque handle (void*).
// queue_depth maps to the libaio io_setup nr_events. Returns nullptr on
// failure.
inline void *ctx_create(unsigned queue_depth)
{
    io_context_t aio = nullptr;
    const int ret = io_setup(static_cast<int>(queue_depth), &aio);
    if (ret < 0)
    {
        LOG(ERROR) << "io_setup failed: " << strerror(-ret);
        return nullptr;
    }
    // io_context_t is itself a pointer, so it round-trips through void* cleanly.
    return reinterpret_cast<void *>(aio);
}

inline void ctx_destroy(void *ctx)
{
    if (ctx == nullptr)
    {
        return;
    }
    io_destroy(reinterpret_cast<io_context_t>(ctx));
}

// ---- submission batch ------------------------------------------------------
// Thread-local staging for one submit round. Usage:
//   SubmitBatch &b = submit_batch();
//   b.begin(ctx);
//   for each read: if (!b.stage(fd, buf, len, offset, ud)) crash();
//   int n = b.submit();   // # submitted (>=0), or <0 on a real error
struct SubmitBatch
{
    void *ctx_ = nullptr;
    int n_ = 0;
    struct iocb cbs_[IO_QUEUE_DEPTH];
    struct iocb *ptrs_[IO_QUEUE_DEPTH];

    inline void begin(void *ctx)
    {
        ctx_ = ctx;
        n_ = 0;
    }

    inline bool stage(int fd, void *buf, unsigned len, uint64_t offset,
                      void *user_data)
    {
        if (n_ >= IO_QUEUE_DEPTH)
        {
            return false;
        }
        struct iocb *cb = &cbs_[n_];
        io_prep_pread(cb, fd, buf, static_cast<size_t>(len),
                      static_cast<long long>(offset));
        cb->data = user_data;
        ptrs_[n_] = cb;
        ++n_;
        return true;
    }

    // Flush the staged reads. Returns the number submitted (>=0) or -errno.
    inline int submit()
    {
        if (n_ == 0)
        {
            return 0;
        }
        io_context_t aio = reinterpret_cast<io_context_t>(ctx_);
        int done = 0;
        while (done < n_)
        {
            const int r = io_submit(aio, n_ - done, ptrs_ + done);
            if (r < 0)
            {
                if (r == -EINTR || r == -EAGAIN)
                {
                    continue; // transient; queue is sized to max in-flight
                }
                return r;
            }
            if (r == 0)
            {
                break; // defensive: avoid spinning
            }
            done += r;
        }
        return done;
    }
};

inline SubmitBatch &submit_batch()
{
    static thread_local SubmitBatch b;
    return b;
}

// ---- completion reaping ----------------------------------------------------
// Non-blocking, batched. Fills user_data_out[] / res_out[] with up to `max`
// completed requests and CONSUMES them. res is bytes transferred (>=0) or a
// negative errno. Returns the number reaped (>=0); 0 means nothing ready.
inline int reap_batch(void *ctx, void **user_data_out, int *res_out, int max)
{
    if (max > IO_QUEUE_DEPTH)
    {
        max = IO_QUEUE_DEPTH;
    }
    static thread_local struct io_event evs[IO_QUEUE_DEPTH];
    struct timespec zero{0, 0}; // non-blocking poll
    const int ret =
        io_getevents(reinterpret_cast<io_context_t>(ctx), 0, max, evs, &zero);
    if (ret <= 0)
    {
        return 0;
    }
    for (int i = 0; i < ret; ++i)
    {
        user_data_out[i] = evs[i].data;
        // libaio res holds bytes transferred (>=0) or a negative errno; it is
        // an unsigned field, so round-trip through long to recover the sign.
        res_out[i] = static_cast<int>(static_cast<long>(evs[i].res));
    }
    return ret;
}

} // namespace rabitqlib::io_backend
