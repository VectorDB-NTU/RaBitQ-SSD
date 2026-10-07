#pragma once

// Readers for the id files whose counts are 64-bit, so they can describe more
// than 2^31 points (all fields little-endian):
//
//   .cid64  per-point cluster ids
//           [u64 magic][u64 version = 1][u64 n][u64 num_clusters]
//           then n x u32, the cluster of point i at position i
//   .gt64   ground-truth neighbor ids
//           [u64 magic][u64 version = 1][u64 nq][u64 k]
//           then nq x k x u64, row-major, nearest first
//
// scripts/dino_exp/gpu_assign.py writes .cid64; tools/dino_prepare.py writes
// .gt64 for the sizes whose ids do not fit in 32 bits.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/io.hpp"

namespace rabitqlib {

constexpr uint64_t kCid64Magic = 0x3436444943444E44ULL;  // "DNDCID64"
constexpr uint64_t kGt64Magic = 0x343654474F4E4944ULL;   // "DINOGT64"
constexpr uint64_t kIdFileVersion = 1;

namespace detail {
inline uint64_t read_u64_or_die(std::ifstream& in, const char* path) {
    uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) {
        std::cerr << path << ": truncated header\n";
        std::exit(1);
    }
    return v;
}

inline void open_id_file(const char* path, uint64_t magic, const char* kind,
                         std::ifstream& in, uint64_t& a, uint64_t& b) {
    if (!file_exists(path)) {
        std::cerr << "File " << path << " not exists\n";
        std::exit(1);
    }
    in.open(path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot open " << path << '\n';
        std::exit(1);
    }
    if (read_u64_or_die(in, path) != magic) {
        std::cerr << path << " is not a " << kind << " file (bad magic)\n";
        std::exit(1);
    }
    const uint64_t version = read_u64_or_die(in, path);
    if (version != kIdFileVersion) {
        std::cerr << path << ": unsupported " << kind << " version " << version << '\n';
        std::exit(1);
    }
    a = read_u64_or_die(in, path);
    b = read_u64_or_die(in, path);
}
}  // namespace detail

inline bool has_u64_magic(const char* path, uint64_t magic) {
    std::ifstream in(path, std::ios::binary);
    uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return static_cast<bool>(in) && v == magic;
}

// Loads a .cid64 into an n x 1 matrix of cluster ids and returns the header's
// cluster count in num_clusters.
template <class M>
void load_cid64(const char* path, M& ids, uint64_t& num_clusters) {
    static_assert(std::is_same_v<CID*, std::decay_t<decltype(ids.data())>>,
                  "load_cid64 fills a matrix of CID");
    std::ifstream in;
    uint64_t n = 0;
    detail::open_id_file(path, kCid64Magic, ".cid64", in, n, num_clusters);
    const size_t expect = (4 * sizeof(uint64_t)) + (n * sizeof(uint32_t));
    if (get_filesize(path) != expect) {
        std::cerr << path << ": header declares " << n << " ids (" << expect
                  << " bytes) but the file has " << get_filesize(path) << " bytes\n";
        std::exit(1);
    }
    ids = M(static_cast<Eigen::Index>(n), 1);
    in.read(reinterpret_cast<char*>(ids.data()),
            static_cast<std::streamsize>(n * sizeof(uint32_t)));
    if (!in) {
        std::cerr << path << ": short read of the id block\n";
        std::exit(1);
    }
    std::cout << "File " << path << " loaded\n"
              << "Rows " << n << " Cols 1 (.cid64, " << num_clusters
              << " clusters)\n"
              << std::flush;
}

// Loads a .gt64 into an nq x k matrix of point ids. A build with 32-bit point
// ids accepts it only when every id fits in 32 bits.
template <class M>
void load_gt64(const char* path, M& ids) {
    using T = std::remove_pointer_t<std::decay_t<decltype(ids.data())>>;
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>,
                  "load_gt64 fills a matrix of uint32_t or uint64_t");
    std::ifstream in;
    uint64_t nq = 0;
    uint64_t k = 0;
    detail::open_id_file(path, kGt64Magic, ".gt64", in, nq, k);
    const size_t expect = (4 * sizeof(uint64_t)) + (nq * k * sizeof(uint64_t));
    if (get_filesize(path) != expect) {
        std::cerr << path << ": header declares " << nq << " x " << k << " ids ("
                  << expect << " bytes) but the file has " << get_filesize(path)
                  << " bytes\n";
        std::exit(1);
    }
    ids = M(static_cast<Eigen::Index>(nq), static_cast<Eigen::Index>(k));
    const size_t count = nq * k;
    if constexpr (std::is_same_v<T, uint64_t>) {
        in.read(reinterpret_cast<char*>(ids.data()),
                static_cast<std::streamsize>(count * sizeof(uint64_t)));
    } else {
        constexpr size_t kBlock = static_cast<size_t>(1) << 20;
        std::vector<uint64_t> buf(std::min(count, kBlock));
        for (size_t off = 0; off < count && in; off += buf.size()) {
            const size_t len = std::min(buf.size(), count - off);
            in.read(reinterpret_cast<char*>(buf.data()),
                    static_cast<std::streamsize>(len * sizeof(uint64_t)));
            for (size_t i = 0; i < len; ++i) {
                if (buf[i] > std::numeric_limits<uint32_t>::max()) {
                    std::cerr << path << " holds point id " << buf[i]
                              << ", beyond 32 bits; use the RABITQ_PID64 build"
                                 " (bin64/)\n";
                    std::exit(1);
                }
                ids.data()[off + i] = static_cast<uint32_t>(buf[i]);
            }
        }
    }
    if (!in) {
        std::cerr << path << ": short read of the id block\n";
        std::exit(1);
    }
    std::cout << "File " << path << " loaded\n"
              << "Rows " << nq << " Cols " << k << " (.gt64)\n"
              << std::flush;
}

}  // namespace rabitqlib
