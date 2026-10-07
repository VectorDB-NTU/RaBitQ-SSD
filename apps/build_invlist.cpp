// build_invlist: builds the coarse-quantizer-independent part of the index.
//
// It quantizes the data, writes the SSD inverted list (<ssd_index>), and
// saves the base index (<base_index>: metadata + rotator + rotated centroids
// + mem_batch_data + ids). It builds no coarse quantizer; that is a separate,
// cheap step (build_coarse), so one base index can be paired with flat / hnsw
// / irq coarse quantizers without rebuilding the inverted list.
#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/id_files.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/io_auto.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/streaming_reader.hpp"
#include "rabitqlib/utils/tools.hpp"
#include "rabitqlib/index/ivf/ivf_ssd_boundary_qd_ms.hpp"

using PID = rabitqlib::PID;
using CID = rabitqlib::CID;
using index_type = rabitqlib::ivf_ssd_boundary_qd_ms::IVFSSD_Index;
using SsdStore = rabitqlib::ivf_ssd_boundary_qd_ms::SsdStore;
using data_type = rabitqlib::RowMajorArray<float>;
using cid_type = rabitqlib::RowMajorArray<CID>;

static std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    return s;
}

static bool parse_bool(const std::string &s, bool &out)
{
    const std::string v = to_lower(s);
    if (v == "1" || v == "true" || v == "yes" || v == "y" || v == "on")
    {
        out = true;
        return true;
    }
    if (v == "0" || v == "false" || v == "no" || v == "n" || v == "off")
    {
        out = false;
        return true;
    }
    return false;
}

static bool parse_order_mode(const std::string &s, index_type::ClusterOrderMode &out)
{
    const std::string v = to_lower(s);
    if (v == "unordered")
    {
        out = index_type::ClusterOrderMode::Unordered;
        return true;
    }
    if (v == "ordered" || v == "orderedbycentroiddistance" || v == "ordered_by_centroid_distance")
    {
        out = index_type::ClusterOrderMode::OrderedByCentroidDistance;
        return true;
    }
    return false;
}

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <data> <centroids> <cluster_ids> <total_bits>"
                     " <base_index> <ssd_index> [options...]\n\n"
                  << "data        : vectors to index: float32 .fbin/.fvecs or uint8 .u8bin/.bvecs\n"
                  << "              (the extension decides)\n"
                  << "centroids   : centroids .fvecs from clustering/run_kmeans.sh\n"
                  << "cluster_ids : per-point cluster assignment from the same run: .ivecs, or\n"
                  << "              .cid64 (64-bit header) from the DINO GPU clustering\n"
                  << "total_bits  : bits per dimension for the quantized code (the scripts use 9);\n"
                  << "              ignored by store=raw\n"
                  << "base_index  : path for saving the in-RAM part of the index\n"
                  << "ssd_index   : path for saving the on-SSD part\n\n"
                  << "options, in any order:\n"
                  << "  l2 | ip                  metric, l2 by default\n"
                  << "  true | false             faster quantization, false by default\n"
                  << "  unordered | ordered      cluster order mode, unordered by default\n"
                  << "  <integer>                MEMDIM: memory-resident bincode dim, a multiple\n"
                  << "                           of 64; default is the padded vector width\n"
                  << "  mem_budget_gb=<GiB>      build-time DRAM ceiling. When set, the dataset is\n"
                  << "                           streamed within this budget instead of loaded\n"
                  << "                           whole; the index is the same either way.\n"
                  << "                           Keyed because the bare-integer slot is MEMDIM.\n"
                  << "  store=exbits | raw       what the SSD records hold, exbits by default:\n"
                  << "                           exbits = the rest of the 1-bit code plus\n"
                  << "                           (total_bits-1)-bit extra-precision codes;\n"
                  << "                           raw = the input vectors as given (float32 or\n"
                  << "                           uint8), re-ranked with exact distances\n"
                  << "  rows=<N>                 index only the first N rows of <data>; the\n"
                  << "                           cluster ids must cover exactly those N\n\n"
                  << "This builds ONLY the inverted list. Build a coarse quantizer separately\n"
                  << "with build_coarse <base_index> <coarse_index> <kind>.\n"
                  << "Full reference: docs/cli_reference.md\n";
        return 1;
    }

    const char *data_file = argv[1];
    const char *centroids_file = argv[2];
    const char *cids_file = argv[3];
    size_t total_bits = static_cast<size_t>(std::stoull(argv[4]));
    const char *base_index_file = argv[5];
    const char *ssd_index_file = argv[6];

    rabitqlib::MetricType metric_type = rabitqlib::METRIC_L2;
    bool faster_quant = false;
    index_type::ClusterOrderMode order_mode = index_type::ClusterOrderMode::Unordered;
    size_t mem_dim = 0;
    // Build-time DRAM budget (total RSS ceiling), in GiB. 0 = unset selects the
    // full-DRAM path (loads the whole dataset, then quantizes). When set, the
    // dataset is streamed within this budget (the whole fp32 dataset is never
    // held in DRAM); both paths produce the same index. Keyed
    // (mem_budget_gb=N) rather than positional because the bare-integer
    // optional slot is already taken by MEMDIM.
    size_t mem_budget_bytes = 0;
    // What the SSD records hold (store=exbits|raw). Saved in <base>, so the
    // query side needs no matching flag.
    SsdStore ssd_store = SsdStore::ExBits;
    // rows=<N>: index only the first N rows of <data> (0 = all of them).
    size_t prefix_rows = 0;

    for (int i = 7; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg.empty())
        {
            continue;
        }

        const std::string lower = to_lower(arg);
        if (lower == "ip")
        {
            metric_type = rabitqlib::METRIC_IP;
            continue;
        }
        if (lower == "l2")
        {
            metric_type = rabitqlib::METRIC_L2;
            continue;
        }

        // mem_budget_gb=<GiB> : enable the streaming build within this budget.
        {
            const std::string key = "mem_budget_gb=";
            if (lower.rfind(key, 0) == 0)
            {
                try
                {
                    const double gb = std::stod(arg.substr(key.size()));
                    if (gb <= 0.0)
                    {
                        throw std::invalid_argument("non-positive");
                    }
                    mem_budget_bytes = static_cast<size_t>(
                        gb * static_cast<double>(static_cast<size_t>(1) << 30));
                }
                catch (const std::exception &)
                {
                    std::cerr << "Invalid mem_budget_gb value: " << arg << '\n';
                    return 1;
                }
                continue;
            }
        }

        // store=exbits|raw : what the SSD records hold.
        {
            const std::string key = "store=";
            if (lower.rfind(key, 0) == 0)
            {
                const std::string v = lower.substr(key.size());
                if (v == "exbits")
                {
                    ssd_store = SsdStore::ExBits;
                }
                else if (v == "raw")
                {
                    ssd_store = SsdStore::Raw;
                }
                else
                {
                    std::cerr << "Invalid store value: " << arg
                              << " (expect store=exbits or store=raw)\n";
                    return 1;
                }
                continue;
            }
        }

        // rows=<N> : index a prefix of <data>.
        {
            const std::string key = "rows=";
            if (lower.rfind(key, 0) == 0)
            {
                try
                {
                    size_t pos = 0;
                    const std::string v = arg.substr(key.size());
                    prefix_rows = static_cast<size_t>(std::stoull(v, &pos));
                    if (pos != v.size() || prefix_rows == 0)
                    {
                        throw std::invalid_argument("bad rows");
                    }
                }
                catch (const std::exception &)
                {
                    std::cerr << "Invalid rows value: " << arg
                              << " (expect a positive integer)\n";
                    return 1;
                }
                continue;
            }
        }

        bool bool_value = false;
        if (parse_bool(lower, bool_value))
        {
            faster_quant = bool_value;
            continue;
        }

        index_type::ClusterOrderMode parsed_order_mode = index_type::ClusterOrderMode::Unordered;
        if (parse_order_mode(lower, parsed_order_mode))
        {
            order_mode = parsed_order_mode;
            continue;
        }

        try
        {
            size_t pos = 0;
            const size_t parsed_mem_dim = static_cast<size_t>(std::stoull(arg, &pos));
            if (pos != arg.size())
            {
                throw std::invalid_argument("trailing chars");
            }
            mem_dim = parsed_mem_dim;
            continue;
        }
        catch (const std::exception &)
        {
            std::cerr << "Invalid optional arg: " << arg << " (expect metric/ip|l2, faster true|false, order unordered|ordered, store=exbits|raw, mem_budget_gb=<GiB>, rows=<N>, or MEMDIM)\n";
            return 1;
        }
    }

    std::cout << "Metric Type: " << ((metric_type == rabitqlib::METRIC_IP) ? "IP" : "L2") << '\n';
    std::cout << "Faster quantization: " << (faster_quant ? "true" : "false") << '\n';
    std::cout << "Cluster order mode: " << ((order_mode == index_type::ClusterOrderMode::OrderedByCentroidDistance) ? "ordered" : "unordered") << '\n';
    // The input's value width comes from its extension; a raw store keeps it.
    const rabitqlib::VecElemType data_elem = rabitqlib::vector_elem_type(data_file);
    std::cout << "Input values: " << rabitqlib::elem_type_name(data_elem) << '\n';
    if (ssd_store == SsdStore::Raw)
    {
        std::cout << "SSD store: raw (" << rabitqlib::elem_type_name(data_elem)
                  << " records; total_bits is ignored)\n";
    }
    else
    {
        std::cout << "SSD store: exbits (total_bits=" << total_bits << ")\n";
    }

#if defined(RABITQ_PID64)
    std::cout << "Point ids: 64-bit\n";
#endif
    const bool streaming = (mem_budget_bytes > 0);
    std::cout << "Build mode: "
              << (streaming ? "streaming (memory-budgeted)" : "full-DRAM")
              << '\n';
    if (streaming)
    {
        std::cout << "Build memory budget: "
                  << (mem_budget_bytes >> 30) << " GiB\n";
    }

    data_type data;  // dataset is loaded only on the full-DRAM path
    data_type centroids;
    cid_type cids;

    // Centroids (C*dim) and cluster ids (N) are always loaded -- small relative
    // to the dataset. The dataset itself is only loaded for the full path; the
    // streaming path reads it block-by-block from <data_file> instead.
    rabitqlib::load_matrix_auto<float, data_type>(centroids_file, centroids);
    if (rabitqlib::has_u64_magic(cids_file, rabitqlib::kCid64Magic))
    {
        uint64_t cid64_clusters = 0;
        rabitqlib::load_cid64(cids_file, cids, cid64_clusters);
        if (cid64_clusters != static_cast<uint64_t>(centroids.rows()))
        {
            std::cerr << cids_file << " assigns to " << cid64_clusters
                      << " clusters, but " << centroids_file << " holds "
                      << centroids.rows() << " centroids\n";
            return 1;
        }
    }
    else
    {
        rabitqlib::load_matrix_auto<CID, cid_type>(cids_file, cids);
    }
    if (static_cast<size_t>(cids.rows()) > static_cast<size_t>(rabitqlib::kPidMax))
    {
        std::cerr << cids.rows() << " points do not fit in " << (8 * sizeof(PID))
                  << "-bit point ids; build with -DRABITQ_PID64=ON (bin64/)\n";
        return 1;
    }

    size_t num_points;
    size_t dim;
    size_t k = centroids.rows();

    if (prefix_rows != 0 && prefix_rows != static_cast<size_t>(cids.rows()))
    {
        std::cerr << "rows=" << prefix_rows << " but " << cids_file << " holds "
                  << cids.rows() << " cluster ids\n";
        return 1;
    }

    if (!streaming)
    {
        if (prefix_rows == 0)
        {
            rabitqlib::load_matrix_as_float<data_type>(data_file, data);
        }
        else
        {
            // Read just the prefix, a bounded block at a time.
            rabitqlib::StreamingRowReader reader(data_file);
            if (prefix_rows > reader.rows())
            {
                std::cerr << "rows=" << prefix_rows << " but " << data_file
                          << " holds only " << reader.rows() << " rows\n";
                return 1;
            }
            reader.limit_rows(prefix_rows);
            data = data_type(static_cast<Eigen::Index>(prefix_rows),
                             static_cast<Eigen::Index>(reader.cols()));
            const size_t block = static_cast<size_t>(1) << 20;
            for (size_t r = 0; r < prefix_rows; r += block)
            {
                const size_t n = std::min(block, prefix_rows - r);
                reader.read_rows(r, n, data.data() + (r * reader.cols()));
            }
            std::cout << "Read the first " << prefix_rows << " of "
                      << reader.file_rows() << " rows of " << data_file << '\n';
        }
        num_points = data.rows();
        dim = data.cols();
        if (static_cast<size_t>(cids.rows()) != num_points)
        {
            std::cerr << cids_file << " holds " << cids.rows()
                      << " cluster ids but " << data_file << " has "
                      << num_points << " rows\n";
            return 1;
        }
        // The streaming build checks the data against the centroid width;
        // this path takes its width from the data, so check the centroids.
        if (static_cast<size_t>(centroids.cols()) != dim)
        {
            std::cerr << centroids_file << " holds " << centroids.cols()
                      << "-dimensional centroids but " << data_file << " has "
                      << dim << " dimensions\n";
            return 1;
        }
    }
    else
    {
        // Shape from the small artifacts; the streaming build also revalidates
        // against the dataset header.
        num_points = cids.rows();
        dim = centroids.cols();
    }

    std::cout << (streaming ? "metadata loaded\n" : "data loaded\n");
    std::cout << "\tN: " << num_points << '\n';
    std::cout << "\tDIM: " << dim << '\n';
    std::cout << "\tK: " << k << '\n';

    rabitqlib::StopW stopw;
    // The coarse-quantizer ctor args are left at their defaults: build_invlist
    // builds no coarse quantizer, so coarse_kind / initializer_type are unused.
    // A raw store has no extra-precision code, so total_bits does not apply;
    // 1 keeps the constructor's range check satisfied whatever was passed.
    index_type ivf_ssd_index(num_points, dim, k,
                             (ssd_store == SsdStore::Raw) ? 1 : total_bits,
                             ssd_index_file, metric_type,
                             rabitqlib::RotatorType::FhtKacRotator, mem_dim);
    if (ssd_store == SsdStore::Raw)
    {
        ivf_ssd_index.set_ssd_store(SsdStore::Raw, data_elem);
    }
    if (!streaming)
    {
        ivf_ssd_index.construct_invlist(data.data(), centroids.data(),
                                        cids.data(), faster_quant, order_mode);
    }
    else
    {
        ivf_ssd_index.construct_invlist_streaming(
            data_file, centroids.data(), cids.data(), faster_quant, order_mode,
            mem_budget_bytes, prefix_rows);
    }

    const float construct_min = stopw.get_elapsed_mili() / 1000 / 60;
    std::cout << "inverted list constructed\n";

    rabitqlib::StopW save_sw;
    ivf_ssd_index.save_base(base_index_file);
    const float save_min = save_sw.get_elapsed_mili() / 1000 / 60;
    std::cout << "base index saved to " << base_index_file << '\n';

    std::cout << "[timing] inverted-list construct: " << construct_min
              << " min\n";
    std::cout << "[timing] save_base: " << save_min << " min\n";
    std::cout << "[timing] total (construct + save_base): "
              << (construct_min + save_min) << " min\n";
    return 0;
}
