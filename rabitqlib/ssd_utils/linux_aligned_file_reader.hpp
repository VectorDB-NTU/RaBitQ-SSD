// Derived from Microsoft DiskANN (https://github.com/microsoft/DiskANN),
// include/linux_aligned_file_reader.h.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "ssd_index_defs.hpp"
#include "rabitqlib/ssd_utils/io_backend.hpp"
#include <vector>
#include <cerrno>
#include "rabitqlib/ssd_utils/log.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>

#define MAX_EVENTS IO_QUEUE_DEPTH

// Owns the O_DIRECT file descriptor of the SSD index plus the per-thread IO
// contexts (created lazily via get_ctx). The actual submit/reap hot path
// lives in rabitqlib::io_backend (io_backend.hpp); this class only manages
// the fd and the per-thread context lifecycle.
class LinuxAlignedFileReader
{
private:
    uint64_t file_sz;
    int file_desc;

public:
    LinuxAlignedFileReader();
    ~LinuxAlignedFileReader();

    u_int64_t get_file_size() { return file_sz; }

    int get_file_desc() { return file_desc; }

    // IO context of the calling thread, created on first use. Never null.
    void *get_ctx();

    // Open & close ops. open() uses O_DIRECT, so all reads must be
    // SECTOR_LEN-aligned in offset, length and buffer address.
    void open(const std::string &fname, bool enable_writes, bool enable_create);
    void close();

    // Create the calling thread's IO context.
    void register_thread();

};
