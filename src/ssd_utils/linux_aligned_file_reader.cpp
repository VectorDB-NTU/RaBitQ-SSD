// Derived from Microsoft DiskANN (https://github.com/microsoft/DiskANN),
// src/linux_aligned_file_reader.cpp.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "rabitqlib/ssd_utils/linux_aligned_file_reader.hpp"
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>

LinuxAlignedFileReader::LinuxAlignedFileReader()
{
    this->file_desc = -1;
    this->file_sz = 0;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader()
{
    int64_t ret;
    // check to make sure file_desc is closed
    ret = ::fcntl(this->file_desc, F_GETFD);
    if (ret == -1)
    {
        if (errno != EBADF)
        {
            std::cerr << "close() not called" << std::endl;
            // close file desc
            ret = ::close(this->file_desc);
            // error checks
            if (ret == -1)
            {
                std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
            }
        }
    }
}

namespace ioctx
{
    // Per-thread libaio IO context, held as an opaque void*. Created and
    // destroyed through rabitqlib::io_backend, which owns all backend-specific
    // code.
    static thread_local void *ctx = nullptr;
};

void *LinuxAlignedFileReader::get_ctx()
{
    if (unlikely(ioctx::ctx == nullptr))
    {
        register_thread();
    }
    return ioctx::ctx;
}

void LinuxAlignedFileReader::register_thread()
{
    if (ioctx::ctx == nullptr)
    {
        ioctx::ctx = rabitqlib::io_backend::ctx_create(MAX_EVENTS);
        if (ioctx::ctx == nullptr)
        {
            LOG(ERROR) << "io_backend ctx_create (" << rabitqlib::io_backend::name()
                       << ") failed";
            exit(1);
        }
    }
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false)
{
    int flags = O_DIRECT | O_LARGEFILE | (enable_writes ? O_RDWR : O_RDONLY);
    if (enable_create)
    {
        flags |= O_CREAT;
    }
    this->file_desc = ::open(fname.c_str(), flags, 0644);
    if (this->file_desc == -1)
    {
        const int err = errno;
        LOG(ERROR) << "cannot open SSD index " << fname << ": " << strerror(err);
        if (err == EINVAL)
        {
            LOG(ERROR) << "the SSD index is opened with O_DIRECT, which this file "
                          "system does not support (e.g. tmpfs on older kernels); "
                          "put the index on a local disk";
        }
        exit(1);
    }
    struct stat st;
    if (::fstat(this->file_desc, &st) == 0)
    {
        this->file_sz = static_cast<uint64_t>(st.st_size);
    }
    else
    {
        LOG(ERROR) << "fstat failed: " << strerror(errno);
        this->file_sz = 0;
    }
}

void LinuxAlignedFileReader::close()
{
    ::fcntl(this->file_desc, F_GETFD);
    ::close(this->file_desc);
}
