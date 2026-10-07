#pragma once

// Streaming row reader for the memory-budgeted build.
//
// The full-DRAM build path loads the entire dataset with
// load_matrix_as_float() before quantizing. For datasets that do not fit in
// DRAM the streaming build reads the dataset in bounded blocks instead. This
// reader exposes exactly that: open a vector file, learn (rows, cols) from its
// header, and pread arbitrary row ranges into a caller buffer.
//
// Four layouts are read, two row layouts times two value widths:
//   * .fbin  : [u32 rows][u32 cols] then rows*cols float32
//   * .u8bin : [u32 rows][u32 cols] then rows*cols uint8
//   * .fvecs : per row [u32 dim][dim float32]
//   * .bvecs : per row [u32 dim][dim uint8]
// The extension decides the value width (see io_auto.hpp); a file without one
// is read as float32. read_rows() always yields float32, widening uint8
// exactly. read_rows_raw() yields the stored values unchanged, which is what
// the reorder pass copies: it only permutes rows, so uint8 data stays at one
// byte per value in its scratch.
//
// limit_rows(n) restricts the reader to the first n rows, for indexing a
// prefix of a larger file.
//
// CRITICAL invariant: read_rows() must yield byte-for-byte the same float
// values that load_matrix_as_float() (io_auto.hpp) places in row-major order,
// so that the streaming build produces an index identical to the full-DRAM
// build.

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

#include "rabitqlib/utils/io_auto.hpp"  // VecFileFormat, VecElemType + detail helpers

namespace rabitqlib {

class StreamingRowReader {
   public:
    explicit StreamingRowReader(const std::string& path) : path_(path) {
        VecFileFormat fmt = VecFileFormat::Bin;
        if (!detail::format_from_extension(path, fmt)) {
            fmt = detail::sniff_format<float>(path.c_str());
        }
        vecs_ = (fmt == VecFileFormat::Vecs);
        elem_ = vector_elem_type(path);
        elem_bytes_ = elem_type_bytes(elem_);

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("StreamingRowReader: cannot open " + path +
                                     ": " + std::strerror(errno));
        }

        const size_t fsz = filesize();
        if (vecs_) {
            // .fvecs / .bvecs: dim from the first record's u32 prefix; rows
            // from the size, which must be a whole number of records.
            uint32_t dim = 0;
            pread_full(&dim, sizeof(uint32_t), 0);
            cols_ = dim;
            row_stride_ = sizeof(uint32_t) + cols_ * elem_bytes_;
            row_data_off_ = sizeof(uint32_t);
            base_off_ = 0;
            if (cols_ == 0 || fsz % row_stride_ != 0) {
                throw std::runtime_error(
                    "StreamingRowReader: " + path + " is not a whole number of " +
                    std::to_string(cols_) + "-dim " + elem_type_name(elem_) +
                    " rows");
            }
            rows_ = fsz / row_stride_;
        } else {
            // .fbin / .u8bin: [u32 rows][u32 cols] then contiguous values.
            uint32_t hdr[2] = {0, 0};
            pread_full(hdr, 2 * sizeof(uint32_t), 0);
            rows_ = hdr[0];
            cols_ = hdr[1];
            row_stride_ = cols_ * elem_bytes_;
            row_data_off_ = 0;
            base_off_ = 2 * sizeof(uint32_t);
            if (base_off_ + rows_ * row_stride_ > fsz) {
                throw std::runtime_error(
                    "StreamingRowReader: " + path + " declares " +
                    std::to_string(rows_) + " x " + std::to_string(cols_) + " " +
                    elem_type_name(elem_) + " values but holds only " +
                    std::to_string(fsz) + " bytes");
            }
        }
        if (cols_ == 0) {
            throw std::runtime_error("StreamingRowReader: zero cols in " + path);
        }
        file_rows_ = rows_;
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
    // Rows in the file, whatever limit_rows() set.
    [[nodiscard]] size_t file_rows() const { return file_rows_; }

    // Read only the first n rows of the file from now on.
    void limit_rows(size_t n) {
        if (n > file_rows_) {
            throw std::runtime_error(
                "StreamingRowReader::limit_rows: " + path_ + " holds " +
                std::to_string(file_rows_) + " rows, fewer than the " +
                std::to_string(n) + " requested");
        }
        rows_ = n;
    }
    [[nodiscard]] bool is_vecs() const { return vecs_; }
    [[nodiscard]] VecElemType elem_type() const { return elem_; }
    // Bytes per stored value: 4 for float32, 1 for uint8.
    [[nodiscard]] size_t elem_bytes() const { return elem_bytes_; }

    // Fill out[0 .. count*cols_) with the float values of rows
    // [start_row, start_row+count). out must hold count*cols_ floats.
    void read_rows(size_t start_row, size_t count, float* out) const {
        if (count == 0) {
            return;
        }
        if (elem_ == VecElemType::F32) {
            read_rows_raw(start_row, count, out);
            return;
        }
        std::vector<uint8_t> raw(count * cols_);
        read_rows_raw(start_row, count, raw.data());
        for (size_t i = 0; i < raw.size(); ++i) {
            out[i] = static_cast<float>(raw[i]);
        }
    }

    // Fill out with the stored values of rows [start_row, start_row+count),
    // packed row after row without any per-row header. out must hold
    // count*cols_*elem_bytes() bytes.
    void read_rows_raw(size_t start_row, size_t count, void* out) const {
        if (count == 0) {
            return;
        }
        if (start_row + count > rows_) {
            throw std::runtime_error("StreamingRowReader::read_rows out of range");
        }
        const size_t row_bytes = cols_ * elem_bytes_;
        if (!vecs_) {
            // Contiguous: one pread straight into the caller buffer.
            pread_full(out, count * row_bytes, base_off_ + start_row * row_stride_);
            return;
        }
        // .fvecs / .bvecs: rows are interleaved with 4-byte dim prefixes. Read
        // the raw span once, then strip the prefixes into the packed output.
        // The span buffer is kept between calls, so this is not thread-safe.
        const size_t span = count * row_stride_;
        if (scratch_.size() < span) {
            scratch_.resize(span);
        }
        pread_full(scratch_.data(), span, base_off_ + start_row * row_stride_);
        char* dst = static_cast<char*>(out);
        for (size_t r = 0; r < count; ++r) {
            std::memcpy(dst + r * row_bytes,
                        scratch_.data() + r * row_stride_ + row_data_off_, row_bytes);
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
    size_t file_rows_ = 0;
    size_t cols_ = 0;
    bool vecs_ = false;
    VecElemType elem_ = VecElemType::F32;
    size_t elem_bytes_ = sizeof(float);
    size_t row_stride_ = 0;    // bytes between consecutive row starts
    size_t row_data_off_ = 0;  // byte offset of the payload within a row
    size_t base_off_ = 0;      // byte offset of row 0's stride start
    mutable std::vector<char> scratch_;  // read_rows_raw span buffer (.fvecs/.bvecs)
};

}  // namespace rabitqlib
