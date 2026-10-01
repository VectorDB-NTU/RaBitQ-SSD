// build_coarse: builds ONE coarse quantizer over an existing base index's
// centroids.
//
// It reads only the header of <base_index> (metadata + rotator + rotated
// centroids -- neither the inverted list nor the SSD file is touched), builds
// the requested coarse quantizer (flat / hnsw / irq) over those centroids,
// and writes it to <coarse_index> (+ a <coarse_index>.irq / .hnsw sidecar for
// irq / hnsw). The cost is low: it operates on the C centroids only.
//
// One invocation produces one coarse quantizer; querying then pairs any one
// <coarse_index> with the shared <base_index> + <ssd_index>.
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/index/ivf/ivf_ssd_boundary_qd_ms.hpp"

using index_type = rabitqlib::ivf_ssd_boundary_qd_ms::IVFSSD_Index;
using CoarseKind = rabitqlib::ivf_ssd_boundary_qd_ms::CoarseKind;

static std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                   { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Parse the coarse-kind argument into (CoarseKind, InitializerType).
// auto|flat|hnsw map to the parent ivf::Initializer factory; irq selects the
// embedded IVF-RaBitQ coarse quantizer.
static bool parse_coarse_kind(const std::string &s, CoarseKind &kind,
                              rabitqlib::ivf::InitializerType &out)
{
    const std::string v = to_lower(s);
    if (v == "auto")
    {
        kind = CoarseKind::Flat; // resolved later by num_cluster
        out = rabitqlib::ivf::InitializerType::Auto;
        return true;
    }
    if (v == "flat")
    {
        kind = CoarseKind::Flat;
        out = rabitqlib::ivf::InitializerType::Flat;
        return true;
    }
    if (v == "hnsw")
    {
        kind = CoarseKind::HNSW;
        out = rabitqlib::ivf::InitializerType::HNSW;
        return true;
    }
    if (v == "irq")
    {
        kind = CoarseKind::IRQ;
        out = rabitqlib::ivf::InitializerType::Auto; // unused for IRQ
        return true;
    }
    return false;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <base_index> <coarse_index> <coarse_kind> [key=value...]\n"
                  << "base_index   : path for an existing base index built by build_invlist\n"
                  << "coarse_index : path for saving this coarse quantizer\n"
                  << "coarse_kind  : \"auto\", \"flat\", \"hnsw\", or \"irq\"\n"
                  << "key=value    : when coarse_kind=irq, also pass:\n"
                  << "                 inner_centroids=PATH  inner_clusterids=PATH\n"
                  << "               (a second-level k-means over the centroids, from\n"
                  << "                clustering/ivf_centroids.py)\n\n"
                  << "Full reference: docs/cli_reference.md\n";
        return 1;
    }

    const char *base_index_file = argv[1];
    const char *coarse_index_file = argv[2];

    CoarseKind coarse_kind = CoarseKind::Flat;
    rabitqlib::ivf::InitializerType initializer_type =
        rabitqlib::ivf::InitializerType::Auto;
    if (!parse_coarse_kind(argv[3], coarse_kind, initializer_type))
    {
        std::cerr << "Invalid coarse_kind: " << argv[3]
                  << " (expect auto|flat|hnsw|irq)\n";
        return 1;
    }

    // IRQ-only inputs (consumed only when coarse_kind=irq).
    std::string inner_centroids_path;
    std::string inner_cids_path;

    for (int i = 4; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg.empty())
        {
            continue;
        }
        const auto eq = arg.find('=');
        if (eq == std::string::npos)
        {
            std::cerr << "Unexpected arg (expect key=value): " << arg << '\n';
            return 1;
        }
        const std::string key = to_lower(arg.substr(0, eq));
        const std::string value = arg.substr(eq + 1);
        if (key == "inner_centroids" || key == "innercentroids")
        {
            inner_centroids_path = value;
        }
        else if (key == "inner_clusterids" || key == "innerclusterids" ||
                 key == "inner_cids")
        {
            inner_cids_path = value;
        }
        else
        {
            std::cerr << "Unknown key=value arg: " << arg << '\n';
            return 1;
        }
    }

    if (coarse_kind == CoarseKind::IRQ)
    {
        if (inner_centroids_path.empty() || inner_cids_path.empty())
        {
            std::cerr << "coarse_kind=irq requires inner_centroids=<path> and "
                         "inner_clusterids=<path>\n";
            return 1;
        }
    }

    if (coarse_kind == CoarseKind::IRQ)
    {
        std::cout << "Coarse quantizer: irq (9-bit embedded IVF-RaBitQ)\n";
        std::cout << "  inner_centroids : " << inner_centroids_path << '\n';
        std::cout << "  inner_clusterids: " << inner_cids_path << '\n';
    }
    else
    {
        std::cout << "Coarse quantizer: "
                  << rabitqlib::ivf::initializer_type_name(initializer_type)
                  << '\n';
    }

    rabitqlib::StopW stopw;

    index_type ivf_ssd_index;
    // Read just the base header (metadata + rotator + rotated centroids).
    ivf_ssd_index.load_base_meta(base_index_file);
    // Build the coarse quantizer over the base's centroids.
    ivf_ssd_index.build_coarse_quantizer(coarse_kind, initializer_type,
                                         inner_centroids_path, inner_cids_path);

    float seconds = stopw.get_elapsed_mili() / 1000;
    std::cout << "coarse quantizer constructed\n";

    ivf_ssd_index.save_coarse(coarse_index_file);
    std::cout << "coarse quantizer saved to " << coarse_index_file << '\n';

    std::cout << "Coarse-quantizer build time: " << seconds << " seconds.\n";
    return 0;
}
