# Third-party notices

This project is released under the MIT License (see [LICENSE](LICENSE)). It
also ships, or derives from, the components below, each of which keeps its own
license.

| Component | Upstream | Where it lives here | License |
|---|---|---|---|
| RaBitQ-Library | https://github.com/VectorDB-NTU/RaBitQ-Library | `rabitqlib/` (vendored and extended) | Apache-2.0 |
| hnswlib | https://github.com/nmslib/hnswlib | `rabitqlib/third/hnswlib/` | Apache-2.0 |
| Eigen | https://eigen.tuxfamily.org | `rabitqlib/third/Eigen/` | MPL-2.0 |
| FFHT | https://github.com/FALCONN-LIB/FFHT | `rabitqlib/utils/fht_avx.hpp` | MIT |
| DiskANN | https://github.com/microsoft/DiskANN | `rabitqlib/ssd_utils/concurrent_queue.hpp`, `rabitqlib/ssd_utils/linux_aligned_file_reader.hpp`, `src/ssd_utils/linux_aligned_file_reader.cpp` | MIT |

License texts:

- `rabitqlib/LICENSE` (Apache-2.0, RaBitQ-Library)
- `rabitqlib/third/hnswlib/LICENSE`
- `rabitqlib/third/Eigen/LICENSE`

FFHT carries its MIT text inline at the top of `fht_avx.hpp`. The three
DiskANN-derived files carry Microsoft's MIT attribution in their headers; they
were adapted, not copied wholesale, and the derivation is noted per file.

Notes:

- Eigen's MPL-2.0 is file-level copyleft: modifying a vendored Eigen file
  obliges you to publish that file's source, but it does not affect the rest of
  this repository.
- hnswlib is only compiled in through the optional `hnsw` coarse quantizer.
