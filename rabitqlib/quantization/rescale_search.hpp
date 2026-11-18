#pragma once

// Search for the rescale factor of the extra-precision (RaBitQ+) code.
//
// Adapted from RaBitQ-Library 11c0ee9: the event sweep and code emission of
// include/rabitqlib/core/detail/rabitq_impl.hpp, and the SIMD interval search
// of src/simd/rescale_search.hpp and src/simd/quantization_avx{2,512}.cpp. The
// instruction set is chosen at compile time, like the rest of this library.
//
// For a normalized magnitude vector a (a_i >= 0, ||a|| = 1) and a scale t, the
// code is c_i = min(floor(t * a_i), max_code) and q_i = c_i + 0.5. The search
// maximizes cos(a, q) = N / sqrt(S), with N = sum(a_i * q_i), S = sum(q_i^2),
// over t in [t_start, t_end), and returns the first scale of the best state.

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace rabitqlib::quant::rabitq_impl::ex_bits
{
    // Lower end of the scale search as a fraction of its upper end, by ex_bits.
    constexpr std::array<float, 9> kTightStart = {
        0,
        0.15,
        0.20,
        0.52,
        0.59,
        0.71,
        0.75,
        0.77,
        0.81,
    };

    // Code of a nonnegative magnitude at scale t, at most max_code. Code k
    // starts exactly at t = k / magnitude: the boundary is decided by division,
    // as the search places its events, because the product alone can round to
    // the wrong side of an event.
    inline int quantized_level_at_scale(double magnitude, double t, int max_code)
    {
        if (magnitude == 0)
        {
            return 0;
        }
        int code = static_cast<int>(std::min(t * magnitude, static_cast<double>(max_code)));
        if (code < max_code && (code + 1.0) / magnitude <= t)
        {
            ++code;
        }
        else if (code > 0 && static_cast<double>(code) / magnitude > t)
        {
            --code;
        }
        return code;
    }

    namespace rescale_detail
    {
        // (scale at which a coordinate's code next increases, coordinate)
        using Event = std::pair<double, size_t>;

        // Per-thread buffers, reused across calls; a call overwrites only the
        // prefix it uses.
        struct Scratch
        {
            std::vector<Event> heap;
            std::vector<int> codes;
            std::vector<double> magnitudes;
            std::vector<double> reciprocals;
        };

        inline Scratch &thread_scratch()
        {
            thread_local Scratch scratch;
            return scratch;
        }

        // Exact search: visit, in order, every scale in [t_start, t_end) at
        // which a code changes. All coordinates crossing the same scale change
        // together, so every evaluated state is one some t produces.
        template <typename T>
        inline double event_sweep(const T *o_abs, size_t dim, int max_code, double t_start,
                                  double t_end)
        {
            Scratch &scratch = thread_scratch();
            std::vector<Event> &next_t = scratch.heap;
            std::vector<int> &cur_o_bar = scratch.codes;
            next_t.clear();
            next_t.reserve(dim);
            cur_o_bar.resize(dim);

            double sqr_denominator = static_cast<double>(dim) * 0.25;
            double numerator = 0;
            for (size_t i = 0; i < dim; ++i)
            {
                const double magnitude = static_cast<double>(o_abs[i]);
                const int cur = quantized_level_at_scale(magnitude, t_start, max_code);
                cur_o_bar[i] = cur;
                sqr_denominator += (cur * cur) + cur;
                numerator += (cur + 0.5) * magnitude;
                // A saturated or zero coordinate never changes again.
                if (magnitude > 0 && cur < max_code)
                {
                    const double next = static_cast<double>(cur + 1) / magnitude;
                    if (next < t_end)
                    {
                        next_t.emplace_back(next, i);
                    }
                }
            }
            std::make_heap(next_t.begin(), next_t.end(), std::greater<Event>{});

            // The state at t_start is a candidate too.
            double max_ip = numerator / std::sqrt(sqr_denominator);
            double best_t = t_start;
            while (!next_t.empty())
            {
                const double cur_t = next_t.front().first;
                do
                {
                    const size_t i = next_t.front().second;
                    ++cur_o_bar[i];
                    // q_i goes from k - 0.5 to k + 0.5 for the new code k.
                    sqr_denominator += 2.0 * cur_o_bar[i];
                    const double magnitude = static_cast<double>(o_abs[i]);
                    numerator += magnitude;

                    Event next{t_end, i};
                    if (cur_o_bar[i] < max_code)
                    {
                        next.first = static_cast<double>(cur_o_bar[i] + 1) / magnitude;
                    }
                    if (next.first >= t_end)
                    {
                        // No further event for this coordinate: move the last
                        // entry into the root and restore the heap below.
                        next = next_t.back();
                        next_t.pop_back();
                    }
                    if (!next_t.empty())
                    {
                        // Replace the root with next. It cannot precede the old
                        // minimum, so one sift-down restores the heap.
                        size_t parent = 0;
                        size_t child = 1;
                        while (child < next_t.size())
                        {
                            if (child + 1 < next_t.size() && next_t[child + 1] < next_t[child])
                            {
                                ++child;
                            }
                            if (!(next_t[child] < next))
                            {
                                break;
                            }
                            next_t[parent] = next_t[child];
                            parent = child;
                            child = (2 * parent) + 1;
                        }
                        next_t[parent] = next;
                    }
                } while (!next_t.empty() && next_t.front().first == cur_t);

                const double cur_ip = numerator / std::sqrt(sqr_denominator);
                if (cur_ip > max_ip)
                {
                    max_ip = cur_ip;
                    best_t = cur_t;
                }
            }
            return best_t;
        }

#if defined(__AVX512F__) || defined(__AVX2__)
        // A code state and the scales [first, next) over which it holds:
        //   first = max(t_start, max over c_i > 0 of c_i / a_i)
        //   next  = min(t_end, min over a_i > 0, c_i < max_code of (c_i + 1) / a_i)
        struct SearchState
        {
            double numerator;     // N
            double squared_norm;  // S
            double first;
            double next;
        };

        // The adjacent doubles above and below a finite value. Bounds below are
        // rounded outward with these, so rounding can never make an interval
        // look worse than it is and cause a wrong prune.
        inline double next_double_up(double value)
        {
            static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8);
            if (value == 0)
            {
                return std::numeric_limits<double>::denorm_min();
            }
            uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            bits += value > 0 ? uint64_t{1} : static_cast<uint64_t>(-1);
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }

        inline double next_double_down(double value)
        {
            if (value == 0)
            {
                return -std::numeric_limits<double>::denorm_min();
            }
            uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            bits += value > 0 ? static_cast<uint64_t>(-1) : uint64_t{1};
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }

        // An evaluated state with bounds that enclose the true N despite SIMD
        // summation rounding; S is exact.
        struct BoundedState
        {
            SearchState state{};
            double numerator_lower = 0;
            double numerator_upper = 0;
            double score_lower = 0;
            double score_upper = 0;
        };

        // Upper bound on the squared cosine of every state strictly between two
        // evaluated states. An event at tau = k / a_i adds a_i to N and 2k to S,
        // so dS = 2 * tau * dN with tau in [l, u] = [left.next, right.first].
        // Hence S(x) >= max(S0 + 2l(x - N0), S1 - 2u(N1 - x)) for an
        // intermediate numerator x, and x^2 over that maximum, which is convex
        // on each linear piece, peaks at a piece endpoint.
        inline double interval_squared_cosine_upper_bound(const BoundedState &left,
                                                          const BoundedState &right)
        {
            const double endpoints = std::max(left.score_upper, right.score_upper);
            if (left.state.next == right.state.first ||
                left.numerator_upper >= right.numerator_upper)
            {
                return endpoints;
            }
            const double lower_t = next_double_down(left.state.next);
            const double upper_t = next_double_up(right.state.first);
            const double delta_n_lower = next_double_down(right.numerator_upper - left.numerator_upper);
            const double delta_n_upper = next_double_up(right.numerator_upper - left.numerator_upper);
            const double delta_d_lower =
                next_double_down(right.state.squared_norm - left.state.squared_norm);
            const double delta_d_upper =
                next_double_up(right.state.squared_norm - left.state.squared_norm);

            // Offset from N0 at which the two denominator lines intersect:
            //   p = (2u (N1 - N0) - (S1 - S0)) / (2 (u - l)), enclosed below.
            const double intersection_n_lower =
                next_double_down(next_double_down(2 * upper_t * delta_n_lower) - delta_d_upper);
            const double intersection_n_upper =
                next_double_up(next_double_up(2 * upper_t * delta_n_upper) - delta_d_lower);
            const double intersection_d_lower = 2 * next_double_down(upper_t - lower_t);
            const double intersection_d_upper = 2 * next_double_up(upper_t - lower_t);
            if (intersection_d_lower <= 0)
            {
                // The scale gap is too small to divide by: fall back to the
                // looser bound N <= N1, S >= S0.
                return next_double_up(next_double_up(right.numerator_upper * right.numerator_upper) /
                                      left.state.squared_norm);
            }
            const double offset_lower = std::max(
                0.0, std::min(delta_n_lower, next_double_down(std::max(0.0, intersection_n_lower) /
                                                              intersection_d_upper)));
            const double offset_upper = std::max(
                0.0, std::min(delta_n_upper, next_double_up(std::max(0.0, intersection_n_upper) /
                                                            intersection_d_lower)));
            const double numerator = next_double_up(left.numerator_upper + offset_upper);
            const double left_denominator = next_double_down(
                left.state.squared_norm + next_double_down(2 * lower_t * offset_lower));
            const double right_denominator = next_double_down(
                right.state.squared_norm -
                next_double_up(2 * upper_t * next_double_up(delta_n_upper - offset_lower)));
            const double denominator =
                std::max({left.state.squared_norm, left_denominator, right_denominator});
            return std::max(endpoints,
                            next_double_up(next_double_up(numerator * numerator) / denominator));
        }

        // The state at one scale. Magnitudes are float values promoted exactly
        // to double. Lanes near an integer boundary are recomputed with
        // quantized_level_at_scale, so an event at k / a_i yields code k here
        // exactly as in the sweep; the two boundary scales are recomputed by
        // division from the coordinates the reciprocal products selected.
#if defined(__AVX512F__)
        inline SearchState evaluate_scale(const double *magnitudes, const double *reciprocals,
                                          size_t dim, int max_code, double t_start, double t_end,
                                          double scale)
        {
            const __m512d z = _mm512_setzero_pd();
            const __m512d one = _mm512_set1_pd(1);
            const __m512d half = _mm512_set1_pd(.5);
            const __m512d vt = _mm512_set1_pd(scale);
            const __m512d vcap = _mm512_set1_pd(max_code);
            const __m512d inf = _mm512_set1_pd(std::numeric_limits<double>::infinity());
            const __m512d guard =
                _mm512_set1_pd(2 * std::numeric_limits<double>::epsilon() * (max_code + 1.0));
            __m512d vn = z;
            __m512d vd = z;
            __m512d vfirst = z;
            __m512d vnext = inf;
            __m512i positions = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i first_idx = _mm512_set1_epi64(-1);
            __m512i next_idx = first_idx;
            size_t i = 0;
            for (; i + 8 <= dim; i += 8)
            {
                const __m512d v = _mm512_loadu_pd(magnitudes + i);
                const __m512d vi = _mm512_loadu_pd(reciprocals + i);
                const __m512d product = _mm512_mul_pd(v, vt);
                __m512d c = _mm512_min_pd(_mm512_floor_pd(product), vcap);
                const __mmask8 interior =
                    _mm512_cmp_pd_mask(product, _mm512_add_pd(c, guard), _CMP_GT_OQ) &
                    (_mm512_cmp_pd_mask(c, vcap, _CMP_EQ_OQ) |
                     _mm512_cmp_pd_mask(product, _mm512_sub_pd(_mm512_add_pd(c, one), guard),
                                        _CMP_LT_OQ));
                const __mmask8 boundary = ~interior & _mm512_cmp_pd_mask(v, z, _CMP_GT_OQ);
                if (boundary)
                {
                    alignas(64) double corrected[8];
                    _mm512_store_pd(corrected, c);
                    for (unsigned mask = boundary; mask != 0; mask &= mask - 1)
                    {
                        const unsigned k = static_cast<unsigned>(__builtin_ctz(mask));
                        corrected[k] = quantized_level_at_scale(magnitudes[i + k], scale, max_code);
                    }
                    c = _mm512_load_pd(corrected);
                }
                const __m512d q = _mm512_add_pd(c, half);
                vn = _mm512_add_pd(vn, _mm512_mul_pd(v, q));
                vd = _mm512_add_pd(vd, _mm512_mul_pd(q, q));
                const __mmask8 active_first = _mm512_cmp_pd_mask(c, z, _CMP_GT_OQ);
                const __mmask8 active_next = _mm512_cmp_pd_mask(c, vcap, _CMP_LT_OQ) &
                                             _mm512_cmp_pd_mask(v, z, _CMP_GT_OQ);
                const __m512d f = _mm512_maskz_mul_pd(active_first, c, vi);
                const __m512d n = _mm512_mask_mul_pd(inf, active_next, _mm512_add_pd(c, one), vi);
                const __mmask8 fm = active_first & _mm512_cmp_pd_mask(f, vfirst, _CMP_GT_OQ);
                const __mmask8 nm = active_next & _mm512_cmp_pd_mask(n, vnext, _CMP_LT_OQ);
                first_idx = _mm512_mask_mov_epi64(first_idx, fm, positions);
                next_idx = _mm512_mask_mov_epi64(next_idx, nm, positions);
                vfirst = _mm512_max_pd(vfirst, f);
                vnext = _mm512_min_pd(vnext, n);
                positions = _mm512_add_epi64(positions, _mm512_set1_epi64(8));
            }
            SearchState s{_mm512_reduce_add_pd(vn), _mm512_reduce_add_pd(vd), t_start, t_end};
            alignas(64) size_t fi[8];
            alignas(64) size_t ni[8];
            _mm512_store_si512(fi, first_idx);
            _mm512_store_si512(ni, next_idx);
            const double f = _mm512_reduce_max_pd(vfirst);
            const double n = _mm512_reduce_min_pd(vnext);
            const unsigned fk = static_cast<unsigned>(__builtin_ctz(
                static_cast<unsigned>(_mm512_cmp_pd_mask(vfirst, _mm512_set1_pd(f), _CMP_EQ_OQ))));
            const unsigned nk = static_cast<unsigned>(__builtin_ctz(
                static_cast<unsigned>(_mm512_cmp_pd_mask(vnext, _mm512_set1_pd(n), _CMP_EQ_OQ))));
            if (fi[fk] != std::numeric_limits<size_t>::max())
            {
                s.first = std::max(s.first, static_cast<double>(quantized_level_at_scale(
                                                magnitudes[fi[fk]], scale, max_code)) /
                                                magnitudes[fi[fk]]);
            }
            if (ni[nk] != std::numeric_limits<size_t>::max())
            {
                s.next = std::min(
                    s.next, (quantized_level_at_scale(magnitudes[ni[nk]], scale, max_code) + 1.0) /
                                magnitudes[ni[nk]]);
            }
            for (; i < dim; ++i)
            {
                const int c = quantized_level_at_scale(magnitudes[i], scale, max_code);
                const double q = c + .5;
                s.numerator += magnitudes[i] * q;
                s.squared_norm += q * q;
                if (c > 0)
                {
                    s.first = std::max(s.first, static_cast<double>(c) / magnitudes[i]);
                }
                if (c < max_code && magnitudes[i] > 0)
                {
                    s.next = std::min(s.next, (c + 1.0) / magnitudes[i]);
                }
            }
            return s;
        }
#else
        inline SearchState evaluate_scale(const double *magnitudes, const double *reciprocals,
                                          size_t dim, int max_code, double t_start, double t_end,
                                          double scale)
        {
            const __m256d z = _mm256_setzero_pd();
            const __m256d one = _mm256_set1_pd(1);
            const __m256d half = _mm256_set1_pd(.5);
            const __m256d vt = _mm256_set1_pd(scale);
            const __m256d vcap = _mm256_set1_pd(max_code);
            const __m256d inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
            const __m256d guard =
                _mm256_set1_pd(2 * std::numeric_limits<double>::epsilon() * (max_code + 1.0));
            __m256d vn = z;
            __m256d vd = z;
            __m256d vfirst = z;
            __m256d vnext = inf;
            __m256d positions = _mm256_setr_pd(0, 1, 2, 3);
            __m256d first_idx = _mm256_set1_pd(-1);
            __m256d next_idx = first_idx;
            size_t i = 0;
            for (; i + 4 <= dim; i += 4)
            {
                const __m256d v = _mm256_loadu_pd(magnitudes + i);
                const __m256d vi = _mm256_loadu_pd(reciprocals + i);
                const __m256d product = _mm256_mul_pd(v, vt);
                __m256d c = _mm256_min_pd(_mm256_floor_pd(product), vcap);
                const __m256d interior = _mm256_and_pd(
                    _mm256_cmp_pd(product, _mm256_add_pd(c, guard), _CMP_GT_OQ),
                    _mm256_or_pd(_mm256_cmp_pd(c, vcap, _CMP_EQ_OQ),
                                 _mm256_cmp_pd(product,
                                               _mm256_sub_pd(_mm256_add_pd(c, one), guard),
                                               _CMP_LT_OQ)));
                const unsigned boundary =
                    (~static_cast<unsigned>(_mm256_movemask_pd(interior))) &
                    static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(v, z, _CMP_GT_OQ))) &
                    15U;
                if (boundary != 0)
                {
                    alignas(32) double corrected[4];
                    _mm256_store_pd(corrected, c);
                    for (unsigned mask = boundary; mask != 0; mask &= mask - 1)
                    {
                        const unsigned k = static_cast<unsigned>(__builtin_ctz(mask));
                        corrected[k] = quantized_level_at_scale(magnitudes[i + k], scale, max_code);
                    }
                    c = _mm256_load_pd(corrected);
                }
                const __m256d q = _mm256_add_pd(c, half);
                vn = _mm256_add_pd(vn, _mm256_mul_pd(v, q));
                vd = _mm256_add_pd(vd, _mm256_mul_pd(q, q));
                const __m256d af = _mm256_cmp_pd(c, z, _CMP_GT_OQ);
                const __m256d an =
                    _mm256_and_pd(_mm256_cmp_pd(c, vcap, _CMP_LT_OQ), _mm256_cmp_pd(v, z, _CMP_GT_OQ));
                const __m256d f = _mm256_blendv_pd(z, _mm256_mul_pd(c, vi), af);
                const __m256d n = _mm256_blendv_pd(inf, _mm256_mul_pd(_mm256_add_pd(c, one), vi), an);
                const __m256d fm = _mm256_and_pd(af, _mm256_cmp_pd(f, vfirst, _CMP_GT_OQ));
                const __m256d nm = _mm256_and_pd(an, _mm256_cmp_pd(n, vnext, _CMP_LT_OQ));
                first_idx = _mm256_blendv_pd(first_idx, positions, fm);
                next_idx = _mm256_blendv_pd(next_idx, positions, nm);
                vfirst = _mm256_max_pd(vfirst, f);
                vnext = _mm256_min_pd(vnext, n);
                positions = _mm256_add_pd(positions, _mm256_set1_pd(4));
            }
            alignas(32) double ns[4];
            alignas(32) double ds[4];
            alignas(32) double fs[4];
            alignas(32) double ts[4];
            _mm256_store_pd(ns, vn);
            _mm256_store_pd(ds, vd);
            _mm256_store_pd(fs, vfirst);
            _mm256_store_pd(ts, vnext);
            SearchState s{ns[0] + ns[1] + ns[2] + ns[3], ds[0] + ds[1] + ds[2] + ds[3], t_start,
                          t_end};
            alignas(32) double first_positions[4];
            alignas(32) double next_positions[4];
            _mm256_store_pd(first_positions, first_idx);
            _mm256_store_pd(next_positions, next_idx);
            size_t fi[4];
            size_t ni[4];
            for (size_t k = 0; k < 4; ++k)
            {
                fi[k] = first_positions[k] < 0 ? std::numeric_limits<size_t>::max()
                                               : static_cast<size_t>(first_positions[k]);
                ni[k] = next_positions[k] < 0 ? std::numeric_limits<size_t>::max()
                                              : static_cast<size_t>(next_positions[k]);
            }
            const size_t fk = std::max_element(fs, fs + 4) - fs;
            const size_t nk = std::min_element(ts, ts + 4) - ts;
            if (fi[fk] != std::numeric_limits<size_t>::max())
            {
                s.first = std::max(s.first, static_cast<double>(quantized_level_at_scale(
                                                magnitudes[fi[fk]], scale, max_code)) /
                                                magnitudes[fi[fk]]);
            }
            if (ni[nk] != std::numeric_limits<size_t>::max())
            {
                s.next = std::min(
                    s.next, (quantized_level_at_scale(magnitudes[ni[nk]], scale, max_code) + 1.0) /
                                magnitudes[ni[nk]]);
            }
            for (; i < dim; ++i)
            {
                const int c = quantized_level_at_scale(magnitudes[i], scale, max_code);
                const double q = c + .5;
                s.numerator += magnitudes[i] * q;
                s.squared_norm += q * q;
                if (c > 0)
                {
                    s.first = std::max(s.first, static_cast<double>(c) / magnitudes[i]);
                }
                if (c < max_code && magnitudes[i] > 0)
                {
                    s.next = std::min(s.next, (c + 1.0) / magnitudes[i]);
                }
            }
            return s;
        }
#endif

        // Branch and bound over the scale range: evaluate both ends, then
        // repeatedly bound the states between two evaluated ones, discard
        // intervals that cannot beat the best certified score, and split the
        // rest at a midpoint. Returns false, leaving the answer to the event
        // sweep, whenever it cannot certify a winner (for example when two
        // different states tie within rounding, or the work limits are
        // reached); it never returns an uncertified winner.
        inline bool interval_search(const float *input, size_t dim, int max_code, double t_start,
                                    double t_end, double &best_t)
        {
            if (dim == 0 || max_code == 0)
            {
                best_t = t_start;
                return true;
            }
            // S is a sum of quarter integers; this keeps every addition exact.
            const uint64_t largest_odd = static_cast<uint64_t>((2 * max_code) + 1);
            if (dim > (uint64_t{1} << 53) / (largest_odd * largest_odd))
            {
                return false;
            }

            Scratch &scratch = thread_scratch();
            if (scratch.magnitudes.size() < dim)
            {
                scratch.magnitudes.resize(dim);
                scratch.reciprocals.resize(dim);
            }
            double *magnitudes = scratch.magnitudes.data();
            double *reciprocals = scratch.reciprocals.data();
            for (size_t i = 0; i < dim; ++i)
            {
                magnitudes[i] = static_cast<double>(input[i]);
                reciprocals[i] = input[i] > 0 ? 1.0 / magnitudes[i] : 0;
            }

            // Relative error of the summed N, including the SIMD reduction.
            const double delta = static_cast<double>(dim + 8) * std::numeric_limits<double>::epsilon();
            const double lower_divisor = next_double_up(1 + delta);
            const double upper_divisor = next_double_down(1 - delta);
            constexpr size_t kEvaluationBudget = 512;
            constexpr size_t kMaxDepth = 64;
            size_t evaluations = 0;
            double best_lower = 0;  // largest certified score lower bound so far
            BoundedState best;
            bool have_best = false;

            const auto evaluate = [&](double t, BoundedState &result) {
                if (evaluations == kEvaluationBudget)
                {
                    return false;
                }
                ++evaluations;
                result.state =
                    evaluate_scale(magnitudes, reciprocals, dim, max_code, t_start, t_end, t);
                if (result.state.first > t || result.state.next <= t)
                {
                    return false;
                }
                result.numerator_lower = next_double_down(result.state.numerator / lower_divisor);
                result.numerator_upper = next_double_up(result.state.numerator / upper_divisor);
                result.score_lower = next_double_down(
                    next_double_down(result.numerator_lower * result.numerator_lower) /
                    result.state.squared_norm);
                result.score_upper = next_double_up(
                    next_double_up(result.numerator_upper * result.numerator_upper) /
                    result.state.squared_norm);
                best_lower = std::max(best_lower, result.score_lower);
                // Replace the winner only when the score enclosures are
                // disjoint; equal S means the same state, since codes only grow
                // with t. Overlapping scores of different states need the
                // sweep's exact arithmetic and order.
                if (!have_best || result.score_lower > best.score_upper)
                {
                    best = result;
                    have_best = true;
                }
                else if (result.score_upper >= best.score_lower &&
                         result.state.squared_norm != best.state.squared_norm)
                {
                    return false;
                }
                return true;
            };

            BoundedState first;
            BoundedState last;
            // t_end itself is excluded, as in the sweep.
            if (!evaluate(t_start, first) || !evaluate(next_double_down(t_end), last))
            {
                return false;
            }

            const auto search = [&](auto &&self, const BoundedState &left, const BoundedState &right,
                                    double upper_bound, size_t depth) -> bool {
                if (left.state.squared_norm == right.state.squared_norm ||
                    left.state.next == right.state.first || upper_bound < best_lower)
                {
                    return true;
                }
                if (left.state.next > right.state.first || depth == kMaxDepth)
                {
                    return false;
                }
                // Split only the gap between the two evaluated states.
                double midpoint = left.state.next + ((right.state.first - left.state.next) * 0.5);
                midpoint = std::max(left.state.next,
                                    std::min(midpoint, next_double_down(right.state.first)));
                if (midpoint >= right.state.first)
                {
                    return false;
                }
                BoundedState middle;
                if (!evaluate(midpoint, middle) ||
                    middle.state.squared_norm <= left.state.squared_norm ||
                    middle.state.squared_norm >= right.state.squared_norm)
                {
                    return false;
                }
                const double left_upper = interval_squared_cosine_upper_bound(left, middle);
                const double right_upper = interval_squared_cosine_upper_bound(middle, right);
                // The more promising half first: a better state found there
                // can prune the other half.
                if (left_upper > right_upper)
                {
                    return self(self, left, middle, left_upper, depth + 1) &&
                           self(self, middle, right, right_upper, depth + 1);
                }
                return self(self, middle, right, right_upper, depth + 1) &&
                       self(self, left, middle, left_upper, depth + 1);
            };
            if (!search(search, first, last, interval_squared_cosine_upper_bound(first, last), 0))
            {
                return false;
            }
            best_t = best.state.first;
            return true;
        }
#endif  // __AVX512F__ || __AVX2__
    }  // namespace rescale_detail

    // The rescale factor of the best code for o_abs (normalized magnitudes):
    // the first scale of the best state in [t_start, t_end). The SIMD interval
    // search answers almost every float input; the event sweep answers the
    // rest and the double inputs.
    template <typename T>
    inline double best_rescale_factor(const T *o_abs, size_t dim, size_t ex_bits)
    {
        if (dim == 0)
        {
            return 0;
        }
        const double max_o = static_cast<double>(*std::max_element(o_abs, o_abs + dim));
        if (max_o == 0)
        {
            return 0;
        }
        const int max_code = (1 << ex_bits) - 1;
        const double t_end = static_cast<double>(max_code + 10) / max_o;
        const double t_start = t_end * kTightStart[ex_bits];
#if defined(__AVX512F__) || defined(__AVX2__)
        if constexpr (std::is_same_v<T, float>)
        {
            double t = 0;
            if (rescale_detail::interval_search(o_abs, dim, max_code, t_start, t_end, t))
            {
                return t;
            }
        }
#endif
        return rescale_detail::event_sweep(o_abs, dim, max_code, t_start, t_end);
    }
}  // namespace rabitqlib::quant::rabitq_impl::ex_bits
