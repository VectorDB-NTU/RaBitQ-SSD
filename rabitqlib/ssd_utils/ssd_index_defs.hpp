#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

// Device sector size. O_DIRECT requires every read offset, length and buffer
// address to be a multiple of this value; the on-disk index is laid out in
// pages of exactly SECTOR_LEN bytes.
constexpr size_t SECTOR_LEN = 4096;

// Pages the index build fills into one buffer before issuing a single pwrite.
// The 64-bit point-id build writes multi-terabyte SSD files, so it batches
// 1024 pages (4 MiB) per write and preallocates the file; the default build
// writes one page at a time.
#if defined(RABITQ_PID64)
constexpr size_t kSsdWriteChunkPages = 1024;
#else
constexpr size_t kSsdWriteChunkPages = 1;
#endif

// Maximum number of reads a single thread may keep in flight. It sizes the
// reader's event limit (MAX_EVENTS), the libaio queue (io_setup nr_events) and
// the per-query request-slot count (N_REQ_BUF), which must all stay equal:
// staging more than IO_QUEUE_DEPTH requests in one submit round is a fatal
// error.
constexpr int IO_QUEUE_DEPTH = 200;

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

// Unrecoverable-error handler used in place of exceptions: it never returns
// and aborts the process at the point of failure. SIGABRT rather than a trap
// instruction, so it is not mistaken for a CPU lacking the compiled-in ISA.
inline void crash()
{
    std::abort();
}

// One SECTOR_LEN-aligned read. All three fields must satisfy the O_DIRECT
// alignment contract (see the assertions below), otherwise the read fails
// with EINVAL. The IO backend addresses a request by its pointer (passed as
// user_data), so the struct itself carries no completion state.
struct IORequest
{
    uint64_t offset; // where to read from (page-aligned byte offset)
    uint64_t len;    // how much to read (multiple of SECTOR_LEN)
    void *buf;       // where to read into (SECTOR_LEN-aligned)

    IORequest() : offset(0), len(0), buf(nullptr)
    {
    }

    IORequest(uint64_t offset, uint64_t len, void *buf)
        : offset(offset), len(len), buf(buf)
    {
        assert((uint64_t)buf % SECTOR_LEN == 0);
        assert(offset % SECTOR_LEN == 0);
        assert(len % SECTOR_LEN == 0);
    }
};
