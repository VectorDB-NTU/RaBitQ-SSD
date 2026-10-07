cmake_minimum_required(VERSION 3.13)

# RaBitQ-SSD — self-contained build of the command-line drivers in apps/.
#
# Core design:
#   * switchable coarse quantizer {flat, hnsw, irq}; the main configuration
#     uses IRQ (an embedded 9-bit IVF-RaBitQ over the first-level centroids)
#   * split storage (in-memory prefix dims + on-SSD remainder, read
#     asynchronously during search)
#   * drain-phase pruning: cluster search runs at alpha=1.0 (baseline distk);
#     only the post-nprobe drain phase prunes pending SSD pages against the
#     drain_alpha-quantile distance (querying arg drain_alpha=, main config 0.4)
#   * L2 + IP metrics, memory-budgeted (streaming) build, float32 or uint8
#     input; the SSD records hold ex-codes or, with store=raw, the vectors
#     themselves for an exact re-rank.
#
# Three executables (binaries land in ./bin/):
#   build_invlist   builds the base index + SSD index
#   build_coarse    builds one coarse quantizer
#   querying        pairs one base + one coarse and runs queries
#
# Every rabitqlib header, plus the header-only third-party libraries (Eigen,
# hnswlib), is vendored under rabitqlib/. The only externally discovered
# dependencies are the system OpenMP runtime, libaio and (optionally) BLAS.
#
#   cmake -S . -B build
#   cmake --build build -j
#
# Point ids are 32-bit. For more than 2^32 - 1 points (DINO-10B), configure a
# second build with 64-bit point ids; its binaries land in ./bin64/ and each
# build reads only the indexes it wrote:
#
#   cmake -S . -B build64 -DRABITQ_PID64=ON
#   cmake --build build64 -j

project(RabitqSSD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optimization and ISA flags applied to every target.
set(CMAKE_CXX_FLAGS "-Wall -Wextra -march=native -fpic -fopenmp -ftree-vectorize -fexceptions")

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(BLAS)
if(BLAS_FOUND)
    message(STATUS "BLAS found ${BLAS_LIBRARIES}")
    if("${BLAS_LIBRARIES}" MATCHES "mkl")
        message(STATUS "BLAS is MKL, adding mkl_rt")
        set(BLAS_LIBRARIES mkl_rt)
    endif()
    link_libraries(${BLAS_LIBRARIES})
endif()

# --- SSD IO backend: Linux libaio (io_setup/io_submit/io_getevents) ----------
# Only querying reads through it; build_invlist/build_coarse write with plain
# pwrite(), so the on-disk index format does not depend on the backend.
add_library(rabitq_io INTERFACE)

find_path(RABITQ_LIBAIO_INCLUDE_DIR NAMES libaio.h)
find_library(RABITQ_LIBAIO_LIBRARY NAMES aio)
if(NOT RABITQ_LIBAIO_INCLUDE_DIR OR NOT RABITQ_LIBAIO_LIBRARY)
    message(FATAL_ERROR
        "libaio not found. Install libaio-dev (Debian/Ubuntu) or "
        "libaio-devel (RHEL/Fedora), or pass -DRABITQ_LIBAIO_INCLUDE_DIR / "
        "-DRABITQ_LIBAIO_LIBRARY.")
endif()

message(STATUS "SSD IO backend: libaio")
message(STATUS "Using libaio include dir: ${RABITQ_LIBAIO_INCLUDE_DIR}")
message(STATUS "Using libaio library:    ${RABITQ_LIBAIO_LIBRARY}")

target_include_directories(rabitq_io INTERFACE ${RABITQ_LIBAIO_INCLUDE_DIR})
target_link_libraries(rabitq_io INTERFACE ${RABITQ_LIBAIO_LIBRARY})

# rabitqlib/ sits directly in this directory, so the sources' "rabitqlib/..."
# includes resolve from here.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

option(RABITQ_PID64 "64-bit point ids, for more than 2^32 - 1 points" OFF)
if(RABITQ_PID64)
    message(STATUS "Point ids: 64-bit (binaries in bin64/)")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin64)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
endif()

# Shared translation units linked into every executable.
set(RABITQ_CORE_SRC      ${CMAKE_CURRENT_SOURCE_DIR}/src/ivf_ssd_boundary_qd_ms.cpp)
set(RABITQ_SSD_UTILS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/ssd_utils/linux_aligned_file_reader.cpp)

foreach(app build_invlist build_coarse querying)
    add_executable(${app}
        apps/${app}.cpp
        ${RABITQ_CORE_SRC}
        ${RABITQ_SSD_UTILS_SRC})
    target_link_libraries(${app} PRIVATE rabitq_io)
    if(RABITQ_PID64)
        target_compile_definitions(${app} PRIVATE RABITQ_PID64)
    endif()
endforeach()
