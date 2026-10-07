#pragma once

#include "rabitqlib/index/ivf/ivf_ssd_boundary_qd_ms_common.hpp"

namespace rabitqlib::ivf_ssd_boundary_qd_ms
{
    using CoarseKind = ivf_ssd_boundary_qd_ms_detail::CoarseKind;
    using SsdStore = ivf_ssd_boundary_qd_ms_detail::SsdStore;
    using SearchStats = ivf_ssd_boundary_qd_ms_detail::SearchStats;
    using SingleCandidate = ivf_ssd_boundary_qd_ms_detail::SingleCandidate;
    using PageCandidates = ivf_ssd_boundary_qd_ms_detail::PageCandidates;
    using QueryBuffer = ivf_ssd_boundary_qd_ms_detail::QueryBuffer;
    using PageCandidate_IO_Queue =
        ivf_ssd_boundary_qd_ms_detail::PageCandidate_IO_Queue<
            ivf_ssd_boundary_qd_ms_detail::ArrayPendingSet>;
    using IVFSSD_Index =
        ivf_ssd_boundary_qd_ms_detail::IVFSSD_Index<ivf_ssd_boundary_qd_ms_detail::ArrayPendingSet>;
} // namespace rabitqlib::ivf_ssd_boundary_qd_ms
