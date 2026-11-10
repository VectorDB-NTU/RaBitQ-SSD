#pragma once

#include <cassert>

#include "rabitqlib/defines.hpp"

namespace rabitqlib::ivf {

/** Location of one cluster: its point count, its in-memory batches and its point ids. */
class Cluster {
   private:
    size_t num_;                  // Num of vectors in this cluster
    char* batch_data_ = nullptr;  // RaBitQ code and factors
    PID* ids_ = nullptr;          // PID of vectors

   public:
    explicit Cluster(size_t, char*, PID*);
    Cluster(const Cluster&) = default;
    Cluster(Cluster&&) noexcept = default;
    Cluster& operator=(const Cluster&) = default;
    Cluster& operator=(Cluster&&) noexcept = default;
    ~Cluster() = default;

    [[nodiscard]] char* batch_data() const { return this->batch_data_; }


    [[nodiscard]] PID* ids() const { return this->ids_; }

    [[nodiscard]] size_t num() const { return num_; }
};

inline Cluster::Cluster(size_t num, char* batch_data, PID* ids)
    : num_(num), batch_data_(batch_data), ids_(ids) {}
}  // namespace rabitqlib::ivf
