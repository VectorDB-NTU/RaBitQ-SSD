#pragma once

// Format-detecting matrix loader.
//
// The shared loaders in rabitqlib/utils/io.hpp come in two flavors:
//   * load_vecs : per-row  [dim:u32][dim values]      (.fvecs / .ivecs / .bvecs)
//   * load_bin  : global   [rows:u32][cols:u32][data] (.fbin / .bin, big-ann style)
//
// Datasets ship one format or the other -- e.g. yfcc100m_sample_query1k only
// has .fbin files, while older datasets are .fvecs. load_matrix_auto() picks
// the matching loader so call sites stay format-agnostic: it trusts a
// recognized file extension, and falls back to sniffing the header for
// extensionless files (e.g. a plain "gt100" groundtruth file).
//
// Orthogonal to the row layout is the width of the values. Base and query
// vectors are float32 (.fbin, .fvecs) or uint8 (.u8bin, .bvecs); the
// extension says which. load_matrix_as_float() reads either into a float
// matrix, and load_matrix_auto() refuses a file whose extension names a value
// width other than the one it was asked to read.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/io.hpp"

namespace rabitqlib {

enum class VecFileFormat { Vecs, Bin };

// Width of the vector values in a file, independent of its row layout:
//   .fvecs -> (Vecs, F32)   .bvecs -> (Vecs, U8)
//   .fbin  -> (Bin,  F32)   .u8bin -> (Bin,  U8)
// The numeric values are stored in a raw-store <base>; do not renumber them.
enum class VecElemType : uint32_t { F32 = 0, U8 = 1 };

inline const char* elem_type_name(VecElemType t) {
    return t == VecElemType::U8 ? "uint8" : "float32";
}

inline size_t elem_type_bytes(VecElemType t) {
    return t == VecElemType::U8 ? sizeof(uint8_t) : sizeof(float);
}

namespace detail {

// Lowercased file extension (without the dot), or "" when the basename has
// no extension. Guards against dots that belong to a parent directory
// (e.g. "./data/gt100").
inline std::string lowercase_ext(const std::string& path) {
    const auto slash = path.find_last_of("/\\");
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos ||
        (slash != std::string::npos && dot < slash)) {
        return {};
    }
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

// Recognize the format from the file extension. Returns false when the
// extension is missing or unknown, in which case the caller sniffs content.
inline bool format_from_extension(const std::string& path, VecFileFormat& out) {
    const std::string ext = lowercase_ext(path);
    if (ext == "fvecs" || ext == "ivecs" || ext == "bvecs") {
        out = VecFileFormat::Vecs;
        return true;
    }
    if (ext == "fbin" || ext == "bin" || ext == "u8bin" || ext == "i8bin") {
        out = VecFileFormat::Bin;
        return true;
    }
    return false;
}

// The value width a vector-file extension names. Returns false for an
// extension that names none (.bin, .ivecs, or no extension); callers keep
// their historical default for those. .i8bin names signed int8, which nothing
// in this project reads, so it is rejected rather than misread.
inline bool elem_type_from_extension(const std::string& path, VecElemType& out) {
    const std::string ext = lowercase_ext(path);
    if (ext == "fvecs" || ext == "fbin") {
        out = VecElemType::F32;
        return true;
    }
    if (ext == "bvecs" || ext == "u8bin") {
        out = VecElemType::U8;
        return true;
    }
    if (ext == "i8bin") {
        std::cerr << path << ": int8 vectors (.i8bin) are not supported; "
                  << "use float32 (.fbin) or uint8 (.u8bin)\n";
        exit(1);
    }
    return false;
}

// Sniff the format from the first 8 bytes plus the total file size. Both
// layouts impose a tight size constraint, so in practice exactly one fits.
template <typename T>
VecFileFormat sniff_format(const char* filename) {
    const size_t file_size = get_filesize(filename);
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error(std::string("sniff_format: cannot open ") +
                                 filename);
    }
    uint32_t a = 0;
    uint32_t b = 0;
    input.read(reinterpret_cast<char*>(&a), sizeof(uint32_t));
    input.read(reinterpret_cast<char*>(&b), sizeof(uint32_t));
    input.close();

    // .bin layout: [rows=a][cols=b] then rows*cols values of type T.
    const bool bin_ok = a > 0 && b > 0 &&
                        file_size == 8 + (static_cast<size_t>(a) * b * sizeof(T));

    // .vecs layout: each row is [dim=a] then a values of type T.
    const size_t vecs_row = sizeof(uint32_t) + (static_cast<size_t>(a) * sizeof(T));
    const bool vecs_ok =
        a > 0 && file_size >= vecs_row && file_size % vecs_row == 0;

    if (bin_ok && !vecs_ok) {
        return VecFileFormat::Bin;
    }
    if (vecs_ok && !bin_ok) {
        return VecFileFormat::Vecs;
    }
    if (bin_ok && vecs_ok) {
        // Both layouts fit the file size (rare). Prefer .bin: an 8-byte global
        // header is more likely to also satisfy the .vecs row divisor than a
        // genuine .vecs file is to satisfy the exact .bin size equation.
        return VecFileFormat::Bin;
    }
    throw std::runtime_error(
        std::string("load_matrix_auto: cannot determine format of ") + filename +
        " (neither .vecs nor .bin layout matches the file size)");
}

}  // namespace detail

// Value width of a base or query vector file: what its extension names, and
// float32 for extensions that name none (.bin, no extension), which is how
// such files have always been read.
inline VecElemType vector_elem_type(const std::string& path) {
    VecElemType elem = VecElemType::F32;
    detail::elem_type_from_extension(path, elem);
    return elem;
}

// Load a row-major matrix, auto-detecting .fvecs/.ivecs vs .fbin/.bin.
// Dispatches by file extension when recognized, otherwise sniffs the header.
template <typename T, class M>
void load_matrix_auto(const char* filename, M& row_mat) {
    if (!file_exists(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        exit(1);
    }
    // An extension that names a value width must match the width asked for:
    // reading a uint8 file as float (or the reverse) yields garbage, not an
    // error, so it is stopped here.
    VecElemType named = VecElemType::F32;
    if (detail::elem_type_from_extension(filename, named) &&
        elem_type_bytes(named) != sizeof(T)) {
        std::cerr << "File " << filename << " holds " << elem_type_name(named)
                  << " values, but this reader expects " << sizeof(T)
                  << "-byte values\n";
        exit(1);
    }
    VecFileFormat fmt = VecFileFormat::Vecs;
    if (!detail::format_from_extension(filename, fmt)) {
        fmt = detail::sniff_format<T>(filename);
    }
    if (fmt == VecFileFormat::Bin) {
        load_bin<T, M>(filename, row_mat);
    } else {
        load_vecs<T, M>(filename, row_mat);
    }
}

// Load base or query vectors as float32, whatever their stored width. A uint8
// file (.u8bin, .bvecs) is read as uint8 and widened in memory; the widening
// is exact, so the original bytes stay recoverable. Everything downstream
// (the rotator, the fastscan LUT) works in float.
template <class M>
void load_matrix_as_float(const char* filename, M& row_mat) {
    if (vector_elem_type(filename) == VecElemType::F32) {
        load_matrix_auto<float, M>(filename, row_mat);
        return;
    }
    RowMajorArray<uint8_t> u8;
    load_matrix_auto<uint8_t, RowMajorArray<uint8_t>>(filename, u8);
    row_mat = M(u8.rows(), u8.cols());
    const size_t n = static_cast<size_t>(u8.rows()) * static_cast<size_t>(u8.cols());
    const uint8_t* src = u8.data();
    float* dst = row_mat.data();
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
    std::cout << "  (uint8 values widened to float32)\n" << std::flush;
}

}  // namespace rabitqlib
