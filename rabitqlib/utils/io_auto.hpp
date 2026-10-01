#pragma once

// Format-detecting matrix loader for the mode_switch_alpha_threshold variant.
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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "rabitqlib/utils/io.hpp"

namespace rabitqlib {

enum class VecFileFormat { Vecs, Bin };

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

// Load a row-major matrix, auto-detecting .fvecs/.ivecs vs .fbin/.bin.
// Dispatches by file extension when recognized, otherwise sniffs the header.
template <typename T, class M>
void load_matrix_auto(const char* filename, M& row_mat) {
    if (!file_exists(filename)) {
        std::cerr << "File " << filename << " not exists\n";
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

}  // namespace rabitqlib
