#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/io_auto.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/tools.hpp"
#include "rabitqlib/index/ivf/ivf_ssd_boundary_qd_ms.hpp"
#include <iomanip>

using PID = rabitqlib::PID;
using index_type = rabitqlib::ivf_ssd_boundary_qd_ms::IVFSSD_Index;
using SearchStats = rabitqlib::ivf_ssd_boundary_qd_ms::SearchStats;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;
using gt_dist_type = rabitqlib::RowMajorArray<float>;

// ---- groundtruth loading with optional distances (DiskANN-style tie handling) ----
// Supports DiskANN "type-1": [int32 nq][int32 k][nq*k uint32 ids][nq*k float32 dists]
// (file size 8 + 2*nq*k*4), and falls back to ids-only (.bin / .ivecs). Distances let
// recall extend the GT set through all neighbors tied with the k-th, which is required
// for a meaningful recall on datasets containing duplicate base vectors.
// Returns true iff distances were loaded.
static bool load_gt_maybe_type1(const char *path, gt_type &ids, gt_dist_type &dist)
{
    const size_t fsz = rabitqlib::get_filesize(path);
    int32_t nq = 0, k = 0;
    {
        std::ifstream in(path, std::ios::binary);
        in.read(reinterpret_cast<char *>(&nq), sizeof(int32_t));
        in.read(reinterpret_cast<char *>(&k), sizeof(int32_t));
    }
    const size_t sz_type1 =
        8 + 2 * static_cast<size_t>(nq) * static_cast<size_t>(k) * sizeof(uint32_t);
    if (nq > 0 && k > 0 && fsz == sz_type1)
    {
        rabitqlib::load_bin<uint32_t, gt_type>(path, ids); // [nq][k]+ids (ignores dist block)
        dist = gt_dist_type(nq, k);
        std::ifstream in(path, std::ios::binary);
        in.seekg(8 + static_cast<std::streamoff>(static_cast<size_t>(nq) * k * sizeof(uint32_t)),
                 std::ios::beg);
        in.read(reinterpret_cast<char *>(dist.data()),
                static_cast<std::streamsize>(static_cast<size_t>(nq) * k * sizeof(float)));
        std::cout << "Groundtruth: type-1 (ids+distances) -> tie-aware recall enabled\n";
        return true;
    }
    rabitqlib::load_matrix_auto<uint32_t, gt_type>(path, ids); // ids-only
    std::cout << "Groundtruth: ids-only (no distances) -> plain id-match recall\n";
    return false;
}

// DiskANN-style recall count for one query: a result is correct if it matches any GT id
// within top-`topk` EXTENDED through all neighbors tied (equal distance) with the topk-th.
// Denominator stays `topk`. With gt_dist == nullptr this is plain id-intersection.
static inline size_t count_correct_with_ties(const PID *res_ids, const uint32_t *gt_ids,
                                             const float *gt_dist, size_t topk, size_t gt_stride)
{
    size_t tie_breaker = topk;
    if (gt_dist != nullptr)
    {
        tie_breaker = topk - 1;
        while (tie_breaker < gt_stride && gt_dist[tie_breaker] == gt_dist[topk - 1])
            ++tie_breaker;
    }
    size_t correct = 0;
    for (size_t j = 0; j < topk; ++j)
    {
        for (size_t k = 0; k < tie_breaker; ++k)
        {
            if (res_ids[j] == gt_ids[k])
            {
                ++correct;
                break;
            }
        }
    }
    return correct;
}

// ---- per-query latency / metrics CSV helpers ----
// Percentile by linear interpolation on an ASCENDING-sorted sample vector.
// p in [0,1]; p999 needs >= 1000 samples to be statistically meaningful
// (with fewer it degenerates toward max).
static double percentile_sorted(const std::vector<double> &sorted, double p)
{
    if (sorted.empty())
    {
        return 0.0;
    }
    const double idx = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static std::string path_basename(const std::string &p)
{
    const auto pos = p.find_last_of("/\\");
    return (pos == std::string::npos) ? p : p.substr(pos + 1);
}

static std::vector<size_t> get_nprobes(
    index_type &ivf,
    const std::vector<size_t> &all_nprobes,
    data_type &query,
    gt_type &gt,
    const gt_dist_type &gt_dist,
    bool gt_has_dist,
    size_t topk,
    bool use_hacc);

static size_t test_round = 1;

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

static bool is_integer_arg(const char *s)
{
    if (s == nullptr || *s == '\0')
    {
        return false;
    }
    char *end = nullptr;
    std::strtol(s, &end, 10);
    return end != s && *end == '\0';
}

// Build a compact, log-spaced list of nprobe candidates that adapts to the
// cluster count. The start is the smallest nprobe whose clusters are expected
// to collectively contain >= topk points (using the mean cluster size as a
// proxy); below that the search cannot even return topk candidates. Geometric
// spacing keeps the low-nprobe region dense without inflating the total number
// of points for large C; the recall-plateau early stop in get_nprobes prunes
// the tail.
static std::vector<size_t> build_nprobe_candidates(
    size_t num_cluster, size_t num_points, size_t topk)
{
    std::vector<size_t> nprobes;
    if (num_cluster == 0 || topk == 0)
    {
        return nprobes;
    }

    // nprobe_start = ceil(topk / avg_cluster_size)
    //              = ceil(topk * num_cluster / num_points)
    size_t start = 1;
    if (num_points > 0)
    {
        start = (topk * num_cluster + num_points - 1) / num_points;
    }
    start = std::max<size_t>(start, 1);
    start = std::min<size_t>(start, num_cluster);

    // Geometric ratio between consecutive candidates: 1.4 yields ~17 points
    // across ~2 orders of magnitude. Lower it for a denser candidate set.
    constexpr double kRatio = 1.4;

    double cur = static_cast<double>(start);
    while (true)
    {
        size_t np = static_cast<size_t>(std::round(cur));
        if (np >= num_cluster)
        {
            break;
        }
        if (nprobes.empty() || np > nprobes.back())
        {
            nprobes.push_back(np);
        }
        cur *= kRatio;
    }
    nprobes.push_back(num_cluster);
    return nprobes;
}

static double compute_avg_issued_read_pages(
    const std::vector<SearchStats> &stats)
{
    if (stats.empty())
    {
        return 0.0;
    }

    const double total_pages = std::accumulate(
        stats.begin(), stats.end(), 0.0,
        [](double sum, const SearchStats &stat)
        {
            return sum + static_cast<double>(stat.issued_read_pages);
        });
    return total_pages / static_cast<double>(stats.size());
}

static double compute_avg_num_points_probed(
    const std::vector<SearchStats> &stats)
{
    if (stats.empty())
    {
        return 0.0;
    }
    const double total = std::accumulate(
        stats.begin(), stats.end(), 0.0,
        [](double sum, const SearchStats &stat)
        {
            return sum + static_cast<double>(stat.num_points_probed);
        });
    return total / static_cast<double>(stats.size());
}

// Sum of per-query coarse-quantizer time (microseconds). Divide by nthreads
// to estimate the coarse-only wall-clock contribution under perfect dynamic
// scheduling. Used to derive QPS-without-coarse-quantizer.
static double compute_total_coarse_us(
    const std::vector<SearchStats> &stats)
{
    return std::accumulate(
        stats.begin(), stats.end(), 0.0,
        [](double sum, const SearchStats &stat)
        {
            return sum + stat.coarse_us;
        });
}

// Sum of per-query drain-phase time (microseconds). Divide by nthreads to
// estimate the drain-phase wall-clock contribution under perfect dynamic
// scheduling, mirroring compute_total_coarse_us. The drain phase is the
// post-nprobe-loop tail where the pipeline only polls/refines/submits; its
// wall time is the SSD IO that cluster-scan compute could not hide.
static double compute_total_drain_us(
    const std::vector<SearchStats> &stats)
{
    return std::accumulate(
        stats.begin(), stats.end(), 0.0,
        [](double sum, const SearchStats &stat)
        {
            return sum + stat.drain_us;
        });
}

// Mean of one SearchStats field over the queries of one operating point.
template <typename Getter>
static double avg_per_query(const std::vector<SearchStats> &stats, Getter get)
{
    if (stats.empty())
    {
        return 0.0;
    }
    double total = 0.0;
    for (const SearchStats &st : stats)
    {
        total += get(st);
    }
    return total / static_cast<double>(stats.size());
}

// Per-query average of `stats[i].centroid_exbits_refines`. For IRQ this
// counts how many first-level centroids the inner-IVF coarse step ran
// `split_distance_boosting` on per query — i.e. the survivors of the
// `lower_dist < distk` low-bound prune in scan_one_batch. Each such call
// costs O(padded_dim) for the ex-code SIMD inner product, so this counter
// tracks the coarse-step cost that grows with nprobe.
static double compute_avg_centroid_exbits_refines(
    const std::vector<SearchStats> &stats)
{
    if (stats.empty())
    {
        return 0.0;
    }
    const double total = std::accumulate(
        stats.begin(), stats.end(), 0.0,
        [](double sum, const SearchStats &stat)
        {
            return sum + static_cast<double>(stat.centroid_exbits_refines);
        });
    return total / static_cast<double>(stats.size());
}

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <base_index> <ssd_index> <coarse_index> <query> <gt> <threads> [key=value...]\n\n"
                  << "base_index  : path for the base index (inverted list) built by build_invlist\n"
                  << "ssd_index   : path for ssd index\n"
                  << "coarse_index: path for the coarse quantizer built by build_coarse\n"
                  << "query       : path for query file, format .fvecs or .fbin (auto-detected)\n"
                  << "gt          : path for groundtruth file, format .ivecs or .bin (auto-detected; must contain >= topk neighbors per row)\n"
                  << "threads     : number of threads for querying\n"
                  << "key=value   : any order of:\n"
                  << "              csv=<path>        write the result CSV, one row per nprobe (see OUTPUT_SCHEMA.md).\n"
                  << "                                Columns it cannot fill (cpu_util_pct,peak_rss_mb,resident_index_mb,\n"
                  << "                                index_ssd_bytes) are left BLANK for the driver to fill by run_id.\n"
                  << "              method_csv=<path> write method-specific diagnostics (scan funnel /\n"
                  << "                                roofline buckets); NOT part of the result schema.\n"
                  << "              run_id=<str>      run identifier (default: <dataset>_topk<K>_T<threads>)\n"
                  << "              system=<str>      system label for the CSV (default: RaBitQ-SSD)\n"
                  << "              metric_space=<l2|ip|cosine>  recorded in the CSV (default: l2)\n"
                  << "              dataset=<name>    dataset label recorded in the CSV (default: base-index file stem)\n"
                  << "              topk=<K>          number of neighbors to retrieve, default 100\n"
                  << "              memdim=<D>        or bare integer: memory-resident bincode dim (multiple of 64)\n"
                  << "              use_hacc=<bool>   or bare true/false: default true\n"
                  << "              drain_alpha=<a in [0,1]>   # alpha used ONLY in the drain phase, default 1.0\n"
                  << "                  cluster search always runs at alpha=1.0 (unpruned, no extra\n"
                  << "                  pruning); the drain phase (probe_idx >= nprobe) prunes pending pages\n"
                  << "                  against the drain_alpha-quantile distance. drain_alpha=1.0 disables\n"
                  << "                  drain pruning (unpruned everywhere); smaller prunes the drain tail harder.\n"
                  << "              nprobes=v1,v2,...  explicit operating points; SKIPS the recall-map/adaptive\n"
                  << "                  selection and times exactly these nprobe values (empty = adaptive path).\n"
                  << "                  Use to reuse a completed run's op_param list on a slower thread count.\n\n"
                  << "Full reference, including the environment variables: docs/cli_reference.md\n\n";
        return 1;
    }

    char *base_index_file = argv[1];
    char *ssd_index_file = argv[2];
    char *coarse_index_file = argv[3];
    char *query_file = argv[4];
    char *gt_file = argv[5];
    const int nthreads_arg_idx = 6;
    const int optional_arg_start = 7;

    if (!is_integer_arg(argv[nthreads_arg_idx]))
    {
        std::cerr << "Invalid threads argument.\n";
        return 1;
    }

    int nthreads = std::atoi(argv[nthreads_arg_idx]);
    omp_set_num_threads(nthreads);

    size_t mem_dim = 0;
    bool use_hacc = true;
    size_t topk = 100;
    // csv=<path> writes the result CSV defined by OUTPUT_SCHEMA.md (the only comparison
    // output). method_csv=<path> writes the method-specific diagnostics
    // (scan funnel / roofline buckets) -- NOT part of the
    // result schema. dataset= overrides the auto-derived dataset label.
    std::string csv_file;
    std::string method_csv_file;
    std::string dataset_name;
    // Config columns supplied by the caller; constant within a run.
    std::string run_id;
    std::string system_name = "RaBitQ-SSD";
    std::string metric_space = "l2";
    // Drain-phase alpha. Cluster search always runs at alpha=1.0 (no extra
    // pruning); only the drain phase prunes, using this single value. Default
    // 1.0 = no drain pruning (unpruned everywhere).
    float drain_alpha = 1.0F;
    // Optional explicit operating-point list (nprobes=v1,v2,...): when non-empty,
    // the recall-map + adaptive selection (get_nprobes) is SKIPPED and these exact
    // nprobe values are timed. Since recall is essentially thread-count
    // invariant, a run may reuse the operating points selected by another run at
    // a different thread count. Empty value (nprobes=) = normal adaptive path.
    std::vector<size_t> explicit_nprobes;
    for (int i = optional_arg_start; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg.empty())
        {
            continue;
        }

        const auto eq_pos = arg.find('=');
        if (eq_pos != std::string::npos)
        {
            const std::string key = to_lower(arg.substr(0, eq_pos));
            const std::string value = arg.substr(eq_pos + 1);
            if (key == "topk")
            {
                topk = static_cast<size_t>(std::stoull(value));
                continue;
            }
            if (key == "memdim")
            {
                mem_dim = static_cast<size_t>(std::stoull(value));
                continue;
            }
            if (key == "use_hacc" || key == "usehacc")
            {
                bool bool_value = false;
                if (!parse_bool(value, bool_value))
                {
                    std::cerr << "Invalid use_hacc value: " << value << '\n';
                    return 1;
                }
                use_hacc = bool_value;
                continue;
            }
            if (key == "csv" || key == "csv_file" || key == "csvfile")
            {
                csv_file = value;
                continue;
            }
            if (key == "method_csv" || key == "methodcsv")
            {
                method_csv_file = value;
                continue;
            }
            if (key == "dataset")
            {
                dataset_name = value;
                continue;
            }
            if (key == "run_id" || key == "runid")
            {
                run_id = value;
                continue;
            }
            if (key == "system")
            {
                system_name = value;
                continue;
            }
            if (key == "metric_space" || key == "metricspace")
            {
                metric_space = value;
                continue;
            }
            if (key == "drain_alpha" || key == "drainalpha")
            {
                try
                {
                    size_t pos = 0;
                    drain_alpha = std::stof(value, &pos);
                    if (pos != value.size())
                    {
                        throw std::invalid_argument("trailing chars");
                    }
                }
                catch (const std::exception &)
                {
                    std::cerr << "Invalid drain_alpha value: " << value
                              << " (expect a float in [0,1])\n";
                    return 1;
                }
                if (drain_alpha < 0.0F || drain_alpha > 1.0F)
                {
                    std::cerr << "drain_alpha out of range: " << drain_alpha
                              << " (expect a float in [0,1])\n";
                    return 1;
                }
                continue;
            }
            if (key == "nprobes" || key == "nprobe_list" || key == "nprobelist")
            {
                explicit_nprobes.clear();
                std::stringstream ss(value);
                std::string tok;
                while (std::getline(ss, tok, ','))
                {
                    if (tok.empty())
                    {
                        continue;
                    }
                    try
                    {
                        size_t pos = 0;
                        const size_t v = static_cast<size_t>(std::stoull(tok, &pos));
                        if (pos != tok.size())
                        {
                            throw std::invalid_argument("trailing chars");
                        }
                        if (v > 0)
                        {
                            explicit_nprobes.push_back(v);
                        }
                    }
                    catch (const std::exception &)
                    {
                        std::cerr << "Invalid nprobes entry: " << tok << '\n';
                        return 1;
                    }
                }
                // Empty value (nprobes=) leaves the list empty => normal adaptive
                // recall-map path; only a non-empty-but-unparseable entry errors.
                continue;
            }
            std::cerr << "Unknown key=value arg: " << arg << '\n';
            return 1;
        }

        bool bool_value = false;
        if (parse_bool(arg, bool_value))
        {
            use_hacc = bool_value;
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
            std::cerr << "Invalid optional arg: " << arg << " (expect topk=K, memdim=D, use_hacc=bool, bare integer=MEMDIM, or bare true|false=use_hacc)\n";
            return 1;
        }
    }

    std::cout << "io_backend: " << rabitqlib::io_backend::name()
              << " (compile-time)\n";
    std::cout << "topk: " << topk << '\n';
    std::cout << "use_hacc: " << (use_hacc ? "true" : "false") << '\n';
    std::cout << "scan_mode: batch\n";
    std::cout << "drain_alpha: " << drain_alpha
              << "  (cluster search alpha=1.0 for all probe_idx; drain -> alpha="
              << drain_alpha << ")\n";

    data_type query;
    gt_type gt;
    gt_dist_type gt_dist;
    rabitqlib::load_matrix_auto<float, data_type>(query_file, query);
    bool gt_has_dist = load_gt_maybe_type1(gt_file, gt, gt_dist);
    size_t nq = query.rows();
    if (static_cast<size_t>(gt.cols()) < topk)
    {
        std::cerr << "Groundtruth file has " << gt.cols()
                  << " neighbors per row, which is fewer than topk=" << topk << '\n';
        return 1;
    }
    size_t total_count = nq * topk;
    size_t gt_stride = static_cast<size_t>(gt.cols());

    index_type ivf_ssd;
    // Split storage: load the shared base (inverted list) first, then the
    // chosen coarse quantizer (load_coarse validates the pairing).
    ivf_ssd.load_base(base_index_file, ssd_index_file, mem_dim);
    ivf_ssd.load_coarse(coarse_index_file);
    ivf_ssd.init_buffers(nthreads, topk);
    // Apply the drain-phase alpha before any search runs, so the get_nprobes
    // recall probing below already reflects it.
    ivf_ssd.set_drain_alpha(drain_alpha);

    // A ground truth computed over a different base than the one just loaded
    // scores every query as a miss, and nothing else about the run looks wrong:
    // it completes, writes a full CSV and exits 0, with recall 0 on every row.
    // Ids outside [0, N) prove the mismatch, so refuse now instead. (Ids inside
    // the range cannot be checked this cheaply -- validate_gt_type1.py is the
    // thorough check.)
    {
        const size_t n_points = ivf_ssd.num_points();
        const uint32_t *gt_ids = gt.data();
        const size_t n_gt = static_cast<size_t>(gt.rows()) * gt_stride;
        size_t bad = 0;
        uint32_t worst = 0;
        for (size_t i = 0; i < n_gt; ++i)
        {
            if (static_cast<size_t>(gt_ids[i]) >= n_points)
            {
                ++bad;
                worst = std::max(worst, gt_ids[i]);
            }
        }
        if (bad != 0)
        {
            std::cerr << "Groundtruth does not match this index: " << bad << " of "
                      << n_gt << " ids fall outside the base's " << n_points
                      << " points; the largest is " << worst << ".\n"
                      << "The ground truth was computed over a different base file.\n";
            return 1;
        }
    }

    rabitqlib::StopW stopw;

    std::vector<size_t> nprobes;
    if (!explicit_nprobes.empty())
    {
        // REUSE path: use the caller-supplied operating points and SKIP the
        // recall-map/adaptive selection entirely. Clamp each to [1, num_clusters].
        const size_t nc = ivf_ssd.num_clusters();
        for (size_t np : explicit_nprobes)
        {
            nprobes.push_back(np > nc ? nc : np);
        }
        std::cout << "nprobes: REUSED " << nprobes.size()
                  << " explicit operating points (recall-map SKIPPED):";
        for (size_t np : nprobes)
        {
            std::cout << ' ' << np;
        }
        std::cout << '\n';
    }
    else
    {
        std::vector<size_t> all_nprobes = build_nprobe_candidates(
            ivf_ssd.num_clusters(), ivf_ssd.num_points(), topk);
        std::cout << "nprobe candidates (" << all_nprobes.size() << "):";
        for (size_t np : all_nprobes)
        {
            std::cout << ' ' << np;
        }
        std::cout << '\n';
        nprobes = get_nprobes(ivf_ssd, all_nprobes, query, gt, gt_dist,
                              gt_has_dist, topk, use_hacc);
    }
    size_t length = nprobes.size();

    // ---- CSV output setup (single per-nprobe summary, enabled by csv=...) ----
    // The summary CSV is the ONLY file output; it is written after aggregation
    // below. Every data row repeats the run-config columns so rows from
    // different configurations can never be confused if CSVs are concatenated.
    // All detailed print-outs stay on stdout for the run logfile to capture.
    // scan_mode is a method_ext CSV column with the constant value "batch".
    const char *scan_mode_name = "batch";
    if (dataset_name.empty())
    {
        // derive from the base-index filename stem, e.g.
        // ss_<dataset>_C<C>_cpu_B9_memdim<D>_l2_ordered.base
        std::string stem = path_basename(base_index_file);
        const auto dot = stem.find_last_of('.');
        if (dot != std::string::npos)
        {
            stem = stem.substr(0, dot);
        }
        dataset_name = stem;
    }
    const std::string coarse_tag = path_basename(coarse_index_file);
    if (run_id.empty())
    {
        run_id = dataset_name + "_topk" + std::to_string(topk) + "_T" +
                 std::to_string(nthreads);
    }

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_qps_no_coarse(test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(length));
    std::vector<std::vector<double>> all_io_pages(test_round, std::vector<double>(length));
    // SSD read accounting, split: requests vs pages vs exact bytes.
    std::vector<std::vector<double>> all_io_reqs(test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_io_bytes(test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_num_points(test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_coarse_us(test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_drain_us(test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_drain_fraction(
        test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_exbits_refines(
        test_round, std::vector<double>(length));
    // Scan-phase bookkeeping-analysis counters (avg per query).
    std::vector<std::vector<double>> all_cand_inserted(
        test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_pages_acquired(
        test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_peak_page_slots(
        test_round, std::vector<double>(length));
    // Main-pipeline rerank funnel (per query). pages_reranked is PER-PAGE
    // (# completed page IOs handed to the SSD-side rerank stage); the other
    // three are PER-RECORD: records the rerank loop walked, split_distance_
    // boosting calls actually paid (the real rerank compute), and knn inserts.
    std::vector<std::vector<double>> all_main_pages_reranked(
        test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_main_records_examined(
        test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_main_boost_dists(
        test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_main_knn_inserts(
        test_round, std::vector<double>(length));
    // Pooled per-query total_us samples per nprobe (across all rounds), for
    // tail-latency percentiles (mean/p50/p90/p95/p99/p999/max).
    std::vector<std::vector<double>> pooled_lat(length);
    for (auto &v : pooled_lat)
    {
        v.reserve(test_round * nq);
    }
    // Pooled per-query SSD-read pages/bytes per nprobe, for io p99/max.
    std::vector<std::vector<double>> pooled_io_pages(length), pooled_io_bytes(length);
    for (size_t i = 0; i < length; ++i)
    {
        pooled_io_pages[i].reserve(test_round * nq);
        pooled_io_bytes[i].reserve(test_round * nq);
    }
    // Effective bandwidth/IOPS (internal counters / timed wall) per nprobe.
    std::vector<std::vector<double>> all_eff_bw(test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_eff_iops(test_round, std::vector<double>(length));
    // Roofline coarse-search work buckets (method_ext only): full-L2 distances
    // and 1-bit fastscan attempts (IRQ-only).
    std::vector<std::vector<double>> all_full_l2(test_round, std::vector<double>(length));
    std::vector<std::vector<double>> all_fastscan_att(test_round, std::vector<double>(length));

    for (size_t r = 0; r < test_round; r++)
    {
        for (size_t l = 0; l < length; ++l)
        {
            size_t nprobe = nprobes[l];
            if (nprobe > ivf_ssd.num_clusters())
            {
                std::cout << "nprobe " << nprobe << " is larger than number of clusters, ";
                std::cout << "will use nprobe = num_cluster (" << ivf_ssd.num_clusters() << ").\n";
            }

            std::cout << "Round " << r + 1 << ", nprobe=" << nprobe << std::endl;

            size_t total_correct = 0;
            double total_time_us = 0.0;

            std::vector<PID> results(nq * topk);
            std::vector<SearchStats> stats(nq);

            stopw.reset();
            ivf_ssd.batch_search(query.data(), nq, topk, nprobe, results.data(), use_hacc, stats.data());
            total_time_us = stopw.get_elapsed_micro();

            std::vector<uint32_t> q_correct(nq);
            for (size_t i = 0; i < nq; ++i)
            {
                const uint32_t *gt_ids = gt.data() + i * gt_stride;
                const float *gt_d = gt_has_dist ? (gt_dist.data() + i * gt_stride) : nullptr;
                const PID *res_ids = results.data() + i * topk;
                q_correct[i] = static_cast<uint32_t>(
                    count_correct_with_ties(res_ids, gt_ids, gt_d, topk, gt_stride));
                total_correct += q_correct[i];
            }

            // Pool per-query latency + SSD-read pages/bytes for percentiles.
            for (size_t i = 0; i < nq; ++i)
            {
                pooled_lat[l].push_back(stats[i].total_us);
                pooled_io_pages[l].push_back(static_cast<double>(stats[i].issued_read_pages));
                pooled_io_bytes[l].push_back(static_cast<double>(stats[i].issued_read_bytes));
            }

            all_recall[r][l] = static_cast<float>(total_correct) / total_count;
            all_qps[r][l] = (double)nq / (total_time_us / 1e6);
            all_io_pages[r][l] = compute_avg_issued_read_pages(stats);
            all_io_reqs[r][l] = avg_per_query(
                stats, [](const SearchStats &s)
                { return static_cast<double>(s.issued_read_reqs); });
            all_io_bytes[r][l] = avg_per_query(
                stats, [](const SearchStats &s)
                { return static_cast<double>(s.issued_read_bytes); });
            all_num_points[r][l] = compute_avg_num_points_probed(stats);
            all_exbits_refines[r][l] = compute_avg_centroid_exbits_refines(stats);
            all_full_l2[r][l] = avg_per_query(
                stats, [](const SearchStats &s)
                { return static_cast<double>(s.centroid_full_l2_dists); });
            all_fastscan_att[r][l] = avg_per_query(
                stats, [](const SearchStats &s)
                { return static_cast<double>(s.centroid_fastscan_attempts); });
            // Effective device read bandwidth / IOPS from internal counters over
            // this nprobe's timed wall: bytes/us == MB/s; reqs / wall_seconds.
            all_eff_bw[r][l] = (all_io_bytes[r][l] * static_cast<double>(nq)) /
                               std::max(total_time_us, 1e-6);
            all_eff_iops[r][l] = (all_io_reqs[r][l] * static_cast<double>(nq)) /
                                 (std::max(total_time_us, 1e-6) / 1e6);
            all_cand_inserted[r][l] = avg_per_query(
                stats, [](const SearchStats &s)
                { return static_cast<double>(s.scan_candidates_inserted); });
            all_pages_acquired[r][l] = avg_per_query(
                stats, [](const SearchStats &s)
                { return static_cast<double>(s.scan_pages_acquired); });
            all_peak_page_slots[r][l] = avg_per_query(
                stats, [](const SearchStats &s)
                { return static_cast<double>(s.peak_page_slots); });
            all_main_pages_reranked[r][l] = avg_per_query(
                stats, [](const SearchStats &s)
                { return static_cast<double>(s.main_pages_reranked); });
            all_main_records_examined[r][l] = avg_per_query(
                stats, [](const SearchStats &s)
                { return static_cast<double>(s.main_records_examined); });
            all_main_boost_dists[r][l] = avg_per_query(
                stats, [](const SearchStats &s)
                { return static_cast<double>(s.main_boost_dists); });
            all_main_knn_inserts[r][l] = avg_per_query(
                stats, [](const SearchStats &s)
                { return static_cast<double>(s.main_knn_inserts); });

            // Coarse-only wall-clock contribution (us): per-query coarse times
            // are independent across queries, so under good dynamic-schedule
            // load balancing each thread sees about sum(coarse_us) / nthreads.
            const double total_coarse_us = compute_total_coarse_us(stats);
            const double coarse_wall_us =
                total_coarse_us / static_cast<double>(std::max(nthreads, 1));
            all_coarse_us[r][l] = coarse_wall_us;
            const double no_coarse_wall_us =
                std::max(total_time_us - coarse_wall_us, 1e-6);
            all_qps_no_coarse[r][l] =
                static_cast<float>(static_cast<double>(nq) / (no_coarse_wall_us / 1e6));

            // Drain-phase wall-clock contribution (us): same per-thread
            // load-balancing argument as coarse_us. The SSD-pipeline wall is
            // the measured wall minus the coarse-quantizer wall; the drain
            // fraction of it is the IO-bound vs compute-bound indicator.
            const double total_drain_us = compute_total_drain_us(stats);
            const double drain_wall_us =
                total_drain_us / static_cast<double>(std::max(nthreads, 1));
            all_drain_us[r][l] = drain_wall_us;
            all_drain_fraction[r][l] = drain_wall_us / no_coarse_wall_us;

        }
    }

    auto avg_qps = rabitqlib::horizontal_avg(all_qps);
    auto avg_qps_no_coarse = rabitqlib::horizontal_avg(all_qps_no_coarse);
    auto avg_recall = rabitqlib::horizontal_avg(all_recall);
    auto avg_io_pages = rabitqlib::horizontal_avg(all_io_pages);
    auto avg_io_reqs = rabitqlib::horizontal_avg(all_io_reqs);
    auto avg_io_bytes = rabitqlib::horizontal_avg(all_io_bytes);
    auto avg_num_points = rabitqlib::horizontal_avg(all_num_points);
    auto avg_coarse_us = rabitqlib::horizontal_avg(all_coarse_us);
    auto avg_drain_us = rabitqlib::horizontal_avg(all_drain_us);
    auto avg_drain_fraction = rabitqlib::horizontal_avg(all_drain_fraction);
    auto avg_exbits_refines = rabitqlib::horizontal_avg(all_exbits_refines);
    auto avg_cand_inserted = rabitqlib::horizontal_avg(all_cand_inserted);
    auto avg_pages_acquired = rabitqlib::horizontal_avg(all_pages_acquired);
    auto avg_peak_page_slots = rabitqlib::horizontal_avg(all_peak_page_slots);
    auto avg_main_pages_reranked = rabitqlib::horizontal_avg(all_main_pages_reranked);
    auto avg_main_records_examined = rabitqlib::horizontal_avg(all_main_records_examined);
    auto avg_main_boost_dists = rabitqlib::horizontal_avg(all_main_boost_dists);
    auto avg_main_knn_inserts = rabitqlib::horizontal_avg(all_main_knn_inserts);
    auto avg_eff_bw = rabitqlib::horizontal_avg(all_eff_bw);
    auto avg_eff_iops = rabitqlib::horizontal_avg(all_eff_iops);
    auto avg_full_l2 = rabitqlib::horizontal_avg(all_full_l2);
    auto avg_fastscan_att = rabitqlib::horizontal_avg(all_fastscan_att);

    // ---- per-nprobe latency percentiles from the pooled per-query samples ----
    const size_t n_samples = test_round * nq;
    std::vector<double> lat_mean(length), lat_p50(length), lat_p90(length),
        lat_p95(length), lat_p99(length), lat_p999(length), lat_max(length);
    for (size_t i = 0; i < length; ++i)
    {
        auto &v = pooled_lat[i];
        std::sort(v.begin(), v.end());
        lat_mean[i] = v.empty() ? 0.0
                                : std::accumulate(v.begin(), v.end(), 0.0) /
                                      static_cast<double>(v.size());
        lat_p50[i] = percentile_sorted(v, 0.50);
        lat_p90[i] = percentile_sorted(v, 0.90);
        lat_p95[i] = percentile_sorted(v, 0.95);
        lat_p99[i] = percentile_sorted(v, 0.99);
        lat_p999[i] = percentile_sorted(v, 0.999);
        lat_max[i] = v.empty() ? 0.0 : v.back();
    }
    // ---- per-nprobe SSD-read pages/bytes p99 + max from pooled samples ----
    std::vector<double> iopg_p99(length), iopg_max(length), iob_p99(length),
        iob_max(length);
    for (size_t i = 0; i < length; ++i)
    {
        auto &vp = pooled_io_pages[i];
        std::sort(vp.begin(), vp.end());
        iopg_p99[i] = percentile_sorted(vp, 0.99);
        iopg_max[i] = vp.empty() ? 0.0 : vp.back();
        auto &vb = pooled_io_bytes[i];
        std::sort(vb.begin(), vb.end());
        iob_p99[i] = percentile_sorted(vb, 0.99);
        iob_max[i] = vb.empty() ? 0.0 : vb.back();
    }

    std::cout << std::setw(6) << "nprobe" << std::setw(12) << "QPS"
              << std::setw(14) << "QPSnoCoarse" << std::setw(12) << "Recall"
              << std::setw(14) << "AvgIOReqs"
              << std::setw(16) << "AvgIOPages"
              << std::setw(16) << "AvgIOBytes"
              << std::setw(18) << "AvgPointsProbed"
              << std::setw(16) << "CoarseUsWall"
              << std::setw(16) << "DrainUsWall"
              << std::setw(12) << "DrainFrac"
              << std::setw(12) << "Bound"
              << std::setw(18) << "AvgExbitsRefines" << '\n';

    for (size_t i = 0; i < length; ++i)
    {
        size_t nprobe = nprobes[i];
        float qps = avg_qps[i];
        float recall = avg_recall[i];
        // DrainFrac is the drain-phase share of the SSD-pipeline wall time.
        // The Bound column labels it heuristically: > 0.5 -> IO-bound,
        // < 0.2 -> compute-bound, else mixed. DrainFrac itself is the
        // quantitative signal; the cutoffs are only a reading aid.
        const double df = avg_drain_fraction[i];
        const char *bound = (df > 0.5) ? "io" : (df < 0.2 ? "compute" : "mixed");

        std::cout << std::setw(6) << nprobe << std::setw(12) << qps
                  << std::setw(14) << avg_qps_no_coarse[i]
                  << std::setw(12) << recall
                  << std::setw(14) << avg_io_reqs[i]
                  << std::setw(16) << avg_io_pages[i]
                  << std::setw(16) << avg_io_bytes[i]
                  << std::setw(18)
                  << avg_num_points[i] << std::setw(16)
                  << avg_coarse_us[i] << std::setw(16)
                  << avg_drain_us[i] << std::setw(12)
                  << df << std::setw(12)
                  << bound << std::setw(18)
                  << avg_exbits_refines[i] << '\n';
    }

    // ---- Per-query latency percentiles (us) per operating point ----
    // Pooled over all rounds (test_round x nq samples per nprobe). With
    // threads=1 this is the single-thread latency-recall view; under
    // multithread it is latency-under-load (queueing included). p999 needs
    // >= 1000 samples to be meaningful (otherwise it degenerates to ~max).
    std::cout << "\n=== Per-query latency percentiles (us, pooled over "
              << test_round << " round(s) x " << nq << " queries, threads="
              << nthreads << ") ===\n";
    std::cout << std::setw(8) << "nprobe" << std::setw(12) << "mean"
              << std::setw(12) << "p50" << std::setw(12) << "p90"
              << std::setw(12) << "p95" << std::setw(12) << "p99"
              << std::setw(12) << "p999" << std::setw(12) << "max"
              << std::setw(12) << "recall" << '\n';
    for (size_t i = 0; i < length; ++i)
    {
        std::cout << std::setw(8) << nprobes[i] << std::setw(12) << lat_mean[i]
                  << std::setw(12) << lat_p50[i] << std::setw(12) << lat_p90[i]
                  << std::setw(12) << lat_p95[i] << std::setw(12) << lat_p99[i]
                  << std::setw(12) << lat_p999[i] << std::setw(12) << lat_max[i]
                  << std::setw(12) << avg_recall[i] << '\n';
    }
    if (n_samples < 1000)
    {
        std::cout << "[warn] only " << n_samples << " latency samples per "
                  "nprobe; p999 (and p99 below 100) degenerate toward max.\n";
    }

    // ---- Result CSV (schema in OUTPUT_SCHEMA.md; one row per nprobe) ----
    // Fixed column names/order shared across all methods. Columns this binary
    // cannot fill are left BLANK and filled by the driver via run_id:
    //   cpu_util_pct, peak_rss_mb, resident_index_mb, index_ssd_bytes
    //     -> /usr/bin/time -v + stat (driver merge)
    // Every other column, eff_bw_mbps and eff_iops included, is filled here
    // from internal counters over the timed wall.
    if (!csv_file.empty())
    {
        std::ofstream csv(csv_file);
        if (!csv)
        {
            std::cerr << "[csv] failed to open output file: " << csv_file << '\n';
        }
        else
        {
            csv << "run_id,system,dataset,metric_space,topk,threads,io_backend,"
                   "cache_state,rounds,nq,op_param_name,op_param,"
                   "recall_at_topk,qps,"
                   "lat_mean_us,lat_p50_us,lat_p95_us,lat_p99_us,lat_p999_us,"
                   "io_reqs_mean,io_pages_mean,io_bytes_mean,"
                   "io_pages_p99,io_pages_max,io_bytes_p99,io_bytes_max,"
                   "eff_bw_mbps,eff_iops,"
                   "cpu_util_pct,peak_rss_mb,resident_index_mb,index_ssd_bytes\n";
            for (size_t i = 0; i < length; ++i)
            {
                csv << run_id << ',' << system_name << ',' << dataset_name << ','
                    << metric_space << ',' << topk << ',' << nthreads << ','
                    << rabitqlib::io_backend::name() << ',' << "odirect" << ','
                    << test_round << ',' << nq << ',' << "nprobe" << ','
                    << nprobes[i] << ','
                    << avg_recall[i] << ',' << avg_qps[i] << ','
                    << lat_mean[i] << ',' << lat_p50[i] << ',' << lat_p95[i] << ','
                    << lat_p99[i] << ',' << lat_p999[i] << ','
                    << avg_io_reqs[i] << ',' << avg_io_pages[i] << ',' << avg_io_bytes[i] << ','
                    << iopg_p99[i] << ',' << iopg_max[i] << ',' << iob_p99[i] << ',' << iob_max[i] << ','
                    << avg_eff_bw[i] << ',' << avg_eff_iops[i];
                // The trailing ",,,," are the 4 driver-filled columns
                // (cpu_util_pct,peak_rss_mb,resident_index_mb,index_ssd_bytes).
                csv << ",,,,\n";
            }
            csv.close();
            std::cout << "[csv] wrote result summary (" << length
                      << " rows) to " << csv_file << '\n';
        }
    }
    else
    {
        std::cout << "[csv] no csv= path given; result CSV skipped "
                     "(stdout tables above are the full record)\n";
    }

    // ---- method_ext CSV (method-specific diagnostics; NOT part of the schema) ----
    // Joins to the result CSV on run_id + op_param. Holds the scan funnel and
    // the roofline coarse-search buckets: counters the engine maintains anyway,
    // pooled after timing ends.
    if (!method_csv_file.empty())
    {
        std::ofstream mcsv(method_csv_file);
        if (!mcsv)
        {
            std::cerr << "[method_csv] failed to open: " << method_csv_file << '\n';
        }
        else
        {
            mcsv << "run_id,system,dataset,topk,threads,mem_dim,scan_mode,drain_alpha,"
                    "op_param_name,op_param,"
                    "points_probed,cand_inserted,pages_acquired,peak_page_slots,"
                    "main_pages_reranked,main_records_examined,main_boost_dists,"
                    "main_knn_inserts,exbits_refines,full_l2_dists,fastscan_attempts,"
                    "coarse_us_wall,drain_us_wall,drain_fraction,bound\n";
            for (size_t i = 0; i < length; ++i)
            {
                const double df = avg_drain_fraction[i];
                const char *bound =
                    (df > 0.5) ? "io" : (df < 0.2 ? "compute" : "mixed");
                mcsv << run_id << ',' << system_name << ',' << dataset_name << ','
                     << topk << ',' << nthreads << ',' << mem_dim << ','
                     << scan_mode_name << ',' << drain_alpha << ','
                     << "nprobe" << ',' << nprobes[i] << ','
                     << avg_num_points[i] << ',' << avg_cand_inserted[i] << ','
                     << avg_pages_acquired[i] << ',' << avg_peak_page_slots[i] << ','
                     << avg_main_pages_reranked[i] << ',' << avg_main_records_examined[i] << ','
                     << avg_main_boost_dists[i] << ',' << avg_main_knn_inserts[i] << ','
                     << avg_exbits_refines[i] << ','
                     << avg_full_l2[i] << ',' << avg_fastscan_att[i] << ','
                     << avg_coarse_us[i] << ',' << avg_drain_us[i] << ',' << df << ','
                     << bound << '\n';
            }
            mcsv.close();
            std::cout << "[method_csv] wrote diagnostics (" << length
                      << " rows) to " << method_csv_file << '\n';
        }
    }

    // ---- Scan-phase bookkeeping analysis (avg per query) ----
    // Bookkeeping cost grows as the cluster count shrinks: a large cluster
    // shares a single per-cluster distk, so the in-memory prune is weak and
    // more points reach the candidate pool. Column meanings:
    // cand_inserted/probed = in-memory prune pass rate;
    // cand_inserted/iopages = inserted candidates discarded before SSD re-rank
    // (wasted insertion work); peak_slots = per-thread candidate-pool
    // high-water mark, a proxy for memory footprint.
    std::cout << "\n=== Scan candidate-insertion stats (avg per query) ===\n";
    std::cout << std::setw(8) << "nprobe"
              << std::setw(14) << "probed"
              << std::setw(14) << "cand_ins"
              << std::setw(12) << "pages_acq"
              << std::setw(12) << "peak_slot"
              << std::setw(12) << "iopages"
              << std::setw(12) << "pages_rr"
              << std::setw(12) << "recs_exam"
              << std::setw(12) << "boosts"
              << std::setw(12) << "boost/rec"
              << std::setw(12) << "boost/page"
              << std::setw(12) << "ins/probe"
              << std::setw(12) << "ins/iopag"
              << std::setw(12) << "recall" << '\n';
    for (size_t i = 0; i < length; ++i)
    {
        const double probed = avg_num_points[i];
        const double ins = avg_cand_inserted[i];
        const double iop = avg_io_pages[i];
        const double recs = avg_main_records_examined[i];
        const double boosts = avg_main_boost_dists[i];
        const double pages_rr = avg_main_pages_reranked[i];
        std::cout << std::setw(8) << nprobes[i]
                  << std::setw(14) << probed
                  << std::setw(14) << ins
                  << std::setw(12) << avg_pages_acquired[i]
                  << std::setw(12) << avg_peak_page_slots[i]
                  << std::setw(12) << iop
                  << std::setw(12) << pages_rr
                  << std::setw(12) << recs
                  << std::setw(12) << boosts
                  << std::setw(12) << (recs > 0.0 ? boosts / recs : 0.0)
                  << std::setw(12) << (pages_rr > 0.0 ? boosts / pages_rr : 0.0)
                  << std::setw(12) << (probed > 0.0 ? ins / probed : 0.0)
                  << std::setw(12) << (iop > 0.0 ? ins / iop : 0.0)
                  << std::setw(12) << avg_recall[i] << '\n';
    }
}

// Read a numeric environment variable, falling back to `def` when it is unset,
// empty, or unparseable. Used to expose the converged-region thinning knobs.
static double env_double(const char *name, double def)
{
    const char *v = std::getenv(name);
    if (v == nullptr || *v == '\0')
    {
        return def;
    }
    try
    {
        return std::stod(v);
    }
    catch (...)
    {
        return def;
    }
}

static size_t env_size(const char *name, size_t def)
{
    const char *v = std::getenv(name);
    if (v == nullptr || *v == '\0')
    {
        return def;
    }
    try
    {
        return static_cast<size_t>(std::stoull(v));
    }
    catch (...)
    {
        return def;
    }
}

// Adaptive, RECALL-TARGETED operating-point selection for a smooth QPS-recall
// curve. Log-spaced nprobe samples recall unevenly (steep at low nprobe, flat
// near saturation), so the selection targets a uniform RECALL grid instead:
//   1) MAP: coarse log-nprobe sweep measuring recall, stopping once
//      RECALL_HIGH_KEEP points lie above RECALL_PLATEAU (so the costly
//      near-full-cluster tail is never run) or recall stalls.
//   2) TARGET: on the (recall -> log nprobe) map, interpolate the nprobe that
//      hits each recall in [RECALL_TARGET_LO, RECALL_TARGET_HI] step
//      RECALL_TARGET_STEP -> dense, uniform-in-recall mid band.
//   3) Keep the lowest measured point + any map points with recall < LO (a few
//      low-end points), and RECALL_HIGH_KEEP points above PLATEAU.
// The main benchmark loop re-measures recall/qps/io at each returned nprobe;
// this routine only CHOOSES operating points. All knobs env-overridable.
// Defaults: LO=0.80 HI=0.98 STEP=0.01 PLATEAU=0.99 HIGH_KEEP=2.

// Measure recall at one nprobe (no timing / no stats).
static float probe_recall(index_type &ivf, size_t nprobe, data_type &query,
                          gt_type &gt, const gt_dist_type &gt_dist,
                          bool gt_has_dist, size_t topk, bool use_hacc)
{
    const size_t nq = query.rows();
    const size_t gt_stride = static_cast<size_t>(gt.cols());
    std::vector<PID> results(nq * topk);
    ivf.batch_search(query.data(), nq, topk, nprobe, results.data(), use_hacc,
                     nullptr);
    size_t correct = 0;
    for (size_t i = 0; i < nq; ++i)
    {
        const uint32_t *gt_ids = gt.data() + i * gt_stride;
        const float *gt_d = gt_has_dist ? (gt_dist.data() + i * gt_stride) : nullptr;
        const PID *res_ids = results.data() + i * topk;
        correct += count_correct_with_ties(res_ids, gt_ids, gt_d, topk, gt_stride);
    }
    return static_cast<float>(correct) / static_cast<float>(topk * nq);
}

static std::vector<size_t> get_nprobes(
    index_type &ivf,
    const std::vector<size_t> &all_nprobes,
    data_type &query,
    gt_type &gt,
    const gt_dist_type &gt_dist,
    bool gt_has_dist,
    size_t topk,
    bool use_hacc)
{
    const double rt_lo = env_double("RECALL_TARGET_LO", 0.80);
    const double rt_hi = env_double("RECALL_TARGET_HI", 0.98);
    const double rt_step = env_double("RECALL_TARGET_STEP", 0.01);
    const double plateau = env_double("RECALL_PLATEAU", 0.99);
    const size_t high_keep = env_size("RECALL_HIGH_KEEP", 2);
    const size_t low_keep = env_size("RECALL_LOW_KEEP", 4);  // # points below LO
    const size_t num_clusters = ivf.num_clusters();

    // ---- Phase 1: coarse MAP nprobe -> recall ----
    std::vector<std::pair<size_t, double>> map;
    size_t high_seen = 0;
    double prev_r = -1.0;
    int stall = 0;
    for (size_t cand : all_nprobes)
    {
        const size_t np = (cand > num_clusters) ? num_clusters : cand;
        const float r = probe_recall(ivf, np, query, gt, gt_dist, gt_has_dist,
                                     topk, use_hacc);
        map.emplace_back(np, r);
        std::cout << "[map] nprobe=" << np << " recall=" << r << std::endl;
        if (r > plateau && ++high_seen >= high_keep)
            break;
        if (np >= num_clusters)
            break;
        if (prev_r >= 0.0 && r - prev_r < 1e-4)
        {
            if (++stall >= 2)
            {
                std::cout << "[get_nprobes] recall stalled below plateau; stop map\n";
                break;
            }
        }
        else
        {
            stall = 0;
        }
        prev_r = r;
    }
    if (map.empty())
        return {};

    // Monotonize recall (running max) so every in-range target is bracketed.
    std::vector<double> mono(map.size());
    double run = -1.0;
    for (size_t i = 0; i < map.size(); ++i)
    {
        run = std::max(run, map[i].second);
        mono[i] = run;
    }
    auto interp_np = [&](double t) -> long long {
        if (t < mono.front() || t > mono.back())
            return -1;
        for (size_t i = 0; i + 1 < map.size(); ++i)
        {
            if (t >= mono[i] && t <= mono[i + 1])
            {
                const double n0 = std::log(static_cast<double>(map[i].first));
                const double n1 = std::log(static_cast<double>(map[i + 1].first));
                const double f = (mono[i + 1] > mono[i])
                                     ? (t - mono[i]) / (mono[i + 1] - mono[i])
                                     : 0.0;
                long long np = std::llround(std::exp(n0 + f * (n1 - n0)));
                if (np < 1) np = 1;
                if (np > static_cast<long long>(num_clusters))
                    np = static_cast<long long>(num_clusters);
                return np;
            }
        }
        return -1;
    };

    // ---- Phase 2: assemble operating points ----
    std::vector<size_t> keep;
    keep.push_back(map.front().first);                       // low-end anchor
    // A few points BELOW rt_lo: thin the map's <LO points to ~low_keep, evenly
    // spread (by index, ends included) so a slow-climbing curve doesn't pile up.
    std::vector<size_t> low;
    for (const auto &pr : map)
        if (pr.second < rt_lo)
            low.push_back(pr.first);
    if (low.size() <= low_keep || low_keep <= 1)
    {
        for (size_t v : low) keep.push_back(v);
    }
    else
    {
        for (size_t k = 0; k < low_keep; ++k)
            keep.push_back(low[(low.size() - 1) * k / (low_keep - 1)]);
    }
    for (double t = rt_lo; t <= rt_hi + 1e-9; t += rt_step)   // uniform-recall mid band
    {
        const long long np = interp_np(t);
        if (np > 0)
            keep.push_back(static_cast<size_t>(np));
    }
    size_t hk = 0;                                           // points above plateau
    for (const auto &pr : map)
        if (pr.second > plateau && hk < high_keep)
        {
            keep.push_back(pr.first);
            ++hk;
        }

    std::sort(keep.begin(), keep.end());
    keep.erase(std::unique(keep.begin(), keep.end()), keep.end());
    std::cout << "[get_nprobes] adaptive: " << keep.size()
              << " operating points | target recall [" << rt_lo << "," << rt_hi
              << "] step " << rt_step << " | >" << plateau << " keep " << high_keep
              << " | map probes=" << map.size() << '\n';
    return keep;
}
