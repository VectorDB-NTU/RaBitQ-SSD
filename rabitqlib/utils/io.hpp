#pragma once

#include <sys/stat.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <type_traits>

namespace rabitqlib {
// get num of bytes
inline size_t get_filesize(const char* filename) noexcept {
    try {
        return std::filesystem::file_size(filename);
    } catch (const std::filesystem::filesystem_error& e) {
        // Log the error and return -1 to maintain original behavior on error.
        std::cerr << "Error getting file size for '" << filename << "': " << e.what()
                  << '\n';
        return static_cast<size_t>(-1);
    }
}

inline bool file_exists(const char* filename) { return std::filesystem::exists(filename); }

// load .*vecs file to a matrix (e.g., RowMajorFloatMat)
template <typename T, class M>
void load_vecs(const char* filename, M& row_mat) {
    if (!file_exists(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        exit(1);
    }

    static_assert(std::is_same_v<T*, std::decay_t<decltype(row_mat.data())>>,
                  "the element type T must match the matrix element type");

    uint32_t tmp;
    size_t file_size = get_filesize(filename);
    std::ifstream input(filename, std::ios::binary);

    input.read(reinterpret_cast<char*>(&tmp), sizeof(uint32_t));

    size_t cols = tmp;
    // Every row is [u32 dim][dim values of T]. A file that is not a whole
    // number of such rows holds values of a different width -- a uint8 .bvecs
    // handed to a float reader, say -- and would otherwise be read as garbage.
    const size_t row_bytes = cols * sizeof(T) + sizeof(uint32_t);
    if (cols == 0 || file_size % row_bytes != 0) {
        std::cerr << "File " << filename << " is not a whole number of rows of " << cols
                  << " values of " << sizeof(T)
                  << " bytes each; it was written with a different value type\n";
        exit(1);
    }
    size_t rows = file_size / row_bytes;
    row_mat = M(rows, cols);

    input.seekg(0, std::ifstream::beg);

    for (size_t i = 0; i < rows; i++) {
        input.read(reinterpret_cast<char*>(&tmp), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&row_mat(i, 0)), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded\n";
    std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
    input.close();
}

// load .*bin file to a matrix (e.g., RowMajorFloatMat)
template <typename T, class M>
void load_bin(const char* filename, M& row_mat) {
    if (!file_exists(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        exit(1);
    }

    static_assert(std::is_same_v<T*, std::decay_t<decltype(row_mat.data())>>,
                  "the element type T must match the matrix element type");

    uint32_t rows;
    uint32_t cols;
    std::ifstream input(filename, std::ios::binary);

    input.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
    input.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

    // The header promises rows*cols values of T. A smaller file was written
    // with narrower values (a .u8bin read as float) or is truncated; reading it
    // anyway would fill the matrix with garbage. A larger file is allowed: a
    // DiskANN type-1 ground truth keeps a distance block after the ids.
    const size_t need = 2 * sizeof(uint32_t) + static_cast<size_t>(rows) * cols * sizeof(T);
    const size_t have = get_filesize(filename);
    if (have < need) {
        std::cerr << "File " << filename << " declares " << rows << " x " << cols
                  << " values of " << sizeof(T) << " bytes, which needs " << need
                  << " bytes, but holds " << have
                  << "; it is truncated or was written with a narrower value type\n";
        exit(1);
    }

    row_mat = M(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        input.read(reinterpret_cast<char*>(&row_mat(i, 0)), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded\n";
    std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
    input.close();
}
}  // namespace rabitqlib