#pragma once

// Streaming row reader for the memory-budgeted build.
//
// The full-DRAM build path loads the entire dataset with
// load_matrix_auto<float, RowMajorArray<float>>() before quantizing. For
// datasets that do not fit in DRAM the
// streaming build reads the dataset in bounded blocks instead. This reader
// exposes exactly that: open a float .fbin/.fvecs file, learn (rows, cols)
// from its header, and pread arbitrary row ranges into a caller buffer.
//
// CRITICAL invariant: read_rows() must yield byte-for-byte the same float
// values that load_bin<float>/load_vecs<float> (io.hpp) would have placed in
// row-major order, so that the streaming build produces an index identical
// to the full-DRAM build. Both on-disk layouts store raw float32 values:
//   * .fbin/.bin : [u32 rows][u32 cols] then rows*cols contiguous floats
//                  (row i data at byte 8 + i*cols*4, length cols*4)
//   * .fvecs     : per row [u32 dim][dim floats]
//                  (row i data at byte i*(4+dim*4) + 4, length dim*4)
// Only float payloads are supported, matching the full-DRAM path's
// load_matrix_auto<float, ...> call (it reads sizeof(float) per value).

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/utils/io_auto.hpp"  // VecFileFormat + detail::{format_from_extension, sniff_format}

namespace rabitqlib {

class StreamingRowReader {
   public:
    explicit StreamingRowReader(const std::string& path) : path_(path) {
        VecFileFormat fmt = VecFileFormat::Bin;
        if (!detail::format_from_extension(path, fmt)) {
            fmt = detail::sniff_format<float>(path.c_str());
        }
        vecs_ = (fmt == VecFileFormat::Vecs);

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("StreamingRowReader: cannot open " + path +
                                     ": " + std::strerror(errno));
        }

        if (vecs_) {
            // .fvecs: dim from the first record's u32 prefix; rows from size.
            uint32_t dim = 0;
            pread_full(&dim, sizeof(uint32_t), 0);
            cols_ = dim;
            const size_t fsz = filesize();
            row_stride_ = sizeof(uint32_t) + cols_ * sizeof(float);
            row_data_off_ = sizeof(uint32_t);
            base_off_ = 0;
            rows_ = (row_stride_ > 0) ? (fsz / row_stride_) : 0;
        } else {
            // .fbin: [u32 rows][u32 cols] then contiguous float data.
            uint32_t hdr[2] = {0, 0};
            pread_full(hdr, 2 * sizeof(uint32_t), 0);
            rows_ = hdr[0];
            cols_ = hdr[1];
            row_stride_ = cols_ * sizeof(float);
            row_data_off_ = 0;
            base_off_ = 2 * sizeof(uint32_t);
        }
        if (cols_ == 0) {
            throw std::runtime_error("StreamingRowReader: zero cols in " + path);
        }
    }

    ~StreamingRowReader() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    StreamingRowReader(const StreamingRowReader&) = delete;
    StreamingRowReader& operator=(const StreamingRowReader&) = delete;

    [[nodiscard]] size_t rows() const { return rows_; }
    [[nodiscard]] size_t cols() const { return cols_; }
    [[nodiscard]] bool is_vecs() const { return vecs_; }

    // Fill out[0 .. count*cols_) with the float values of rows
    // [start_row, start_row+count). out must hold count*cols_ floats.
    void read_rows(size_t start_row, size_t count, float* out) const {
        if (count == 0) {
            return;
        }
        if (start_row + count > rows_) {
            throw std::runtime_error("StreamingRowReader::read_rows out of range");
        }
        if (!vecs_) {
            // Contiguous: one pread straight into the caller buffer.
            pread_full(out, count * cols_ * sizeof(float),
                       base_off_ + start_row * row_stride_);
            return;
        }
        // .fvecs: rows are interleaved with 4-byte dim prefixes. Read the raw
        // span once, then strip the prefixes into the packed float output.
        const size_t span = count * row_stride_;
        std::vector<char> scratch(span);
        pread_full(scratch.data(), span, base_off_ + start_row * row_stride_);
        for (size_t r = 0; r < count; ++r) {
            std::memcpy(out + r * cols_,
                        scratch.data() + r * row_stride_ + row_data_off_,
                        cols_ * sizeof(float));
        }
    }

   private:
    [[nodiscard]] size_t filesize() const {
        struct stat st {};
        if (::fstat(fd_, &st) != 0) {
            throw std::runtime_error("StreamingRowReader: fstat failed on " +
                                     path_ + ": " + std::strerror(errno));
        }
        return static_cast<size_t>(st.st_size);
    }

    void pread_full(void* buf, size_t nbytes, off_t offset) const {
        auto* p = static_cast<char*>(buf);
        size_t done = 0;
        while (done < nbytes) {
            const ssize_t n =
                ::pread(fd_, p + done, nbytes - done,
                        offset + static_cast<off_t>(done));
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("StreamingRowReader: pread failed on " +
                                         path_ + ": " + std::strerror(errno));
            }
            if (n == 0) {
                throw std::runtime_error(
                    "StreamingRowReader: unexpected EOF on " + path_);
            }
            done += static_cast<size_t>(n);
        }
    }

    std::string path_;
    int fd_ = -1;
    size_t rows_ = 0;
    size_t cols_ = 0;
    bool vecs_ = false;
    size_t row_stride_ = 0;    // bytes between consecutive row starts
    size_t row_data_off_ = 0;  // byte offset of the float payload within a row
    size_t base_off_ = 0;      // byte offset of row 0's stride start
};

}  // namespace rabitqlib
