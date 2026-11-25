# HACKE: Hardware Aware Compute Kernel Engine

HACKE is a transparent, modular, and dependency-free high-performance dense linear algebra compute engine written from scratch in modern C++ (C++20).

The engine focuses on optimizing double-precision General Matrix Multiplication (`DGEMM`: `C = A * B`) on modern x86-64 architectures. Rather than relying on opaque, assembly-heavy vendor libraries like Intel MKL or OpenBLAS, HACKE systematically bridges the gap between a naive triple-loop baseline and theoretical peak hardware throughput through four readable, hardware-sympathetic vertical slices.

---

## Architectural Motivation: Defeating the Memory Wall

Modern x86-64 processors equipped with 256-bit AVX2 and Fused Multiply-Add (FMA) pipelines (such as the AMD Ryzen 7 PRO 8840HS, rated at **81.6 GFLOPS FP64 single-core peak** at 5.1 GHz with 16 DP FLOPs/cycle) have massive compute throughput. However, a naive $O(N^3)$ matrix multiplication implementation routinely achieves less than **5% of theoretical peak FLOPS**.

This bottleneck occurs because standard column-wise traversal of matrix `B` steps through memory with a stride of `N` doubles. For large matrices, every inner-loop iteration jumps across memory addresses separated by kilobytes, causing catastrophic **spatial locality failure**, **64-byte cache line waste** (fetching 8 doubles only to use 1), and **TLB thrashing**. The CPU functional units spend the vast majority of execution time stalled waiting on main DRAM fetches (~60-100ns latency) rather than computing in L1d cache (~1-2ns latency).

HACKE eliminates this bottleneck step-by-step using the GotoBLAS / BLIS analytical cache-hierarchy methodology.

---

## Target Hardware Reference (AMD Ryzen 7 PRO 8840HS)

Cache capacity boundaries used to evaluate working-set spillovers across matrix dimensions ($N \times N$ in FP64):

- **L1 Data Cache (32 KB):** Spillover threshold around `N ≈ 36`
- **L2 Cache (1 MB):** Spillover threshold around `N ≈ 208`
- **L3 Cache (16 MB):** Spillover threshold around `N ≈ 836`
- **Single-Core Theoretical Peak (FP64):** `81.6 GFLOPS`

---

## Optimization Roadmap (Vertical Slices)

### Phase 1: Baseline Suite & Automated Telemetry Pipeline
- **Flat 1D Memory Representation:** Replaces pointer-chasing 2D heap allocations with contiguous row-major `std::vector<double>` storage (`index = row * n + col`).
- **Deterministic Initialization:** Populates test matrices using a seeded Mersenne Twister (`std::mt19937`) for reproducible benchmarking and epsilon verification (`1e-9` for FP64).
- **Monotonic Telemetry & CSV Export:** Measures execution latency via `std::chrono::steady_clock`, computes throughput in GFLOPS ($\frac{2.0 \times N^3}{\text{time\_sec} \times 10^9}$), and emits structured CSV rows (`N,seconds,GFLOPS,pct_peak`).
- **Automated Sweep & PDF Reporting:** Python automation scripts sweep `N` from `16` to `2048` (powers of two plus `1.5x` midpoints) and generate a multi-page visual report (`naive_baseline_report.pdf`) tracking GFLOPS, % of theoretical peak, and effective memory bandwidth against L1d/L2/L3 boundaries.

### Phase 2: Multi-Level Cache Loop Tiling
- Partitions matrices into hierarchical sub-blocks (`Mc`, `Kc`, `Nc`) sized analytically to fit within the target processor's L1 Data, L2, and L3 cache capacities.
- Eliminates pointer aliasing ambiguity via `__restrict__` pointers and compile-time block constants (`constexpr`).

### Phase 3: Contiguous Memory Packing & Alignment
- Allocates 64-byte cache-line-aligned working buffers (`packA` and `packB`) managed via custom C++ RAII wrappers.
- Reorganizes strided matrix sub-views into sequential unit-stride memory panels prior to execution, ensuring 100% cache line utilization and hardware prefetcher synergy while eliminating TLB misses.

### Phase 4: Explicit AVX2 / FMA Register Micro-Kernel
- Implements a register-blocked inner micro-kernel (`Mr x Nr`, e.g., `6x8` or `8x6` in FP64) using explicit `<immintrin.h>` intrinsics (`__m256d`).
- Maximizes arithmetic intensity by pinning output sub-tiles inside `ymm0-ymm15` vector registers, streaming packed panels via `_mm256_load_pd` and `_mm256_set1_pd`, and executing parallel fused multiply-adds via `_mm256_fmadd_pd`.

---

## Repository Structure

```text
hacke/
├── CMakeLists.txt              # C++20 build configuration & hardware flags (-O3 -march=native)
├── README.md                   # Project documentation & profiling commands
├── include/
│   └── kernels.hpp             # Public kernel declarations & benchmark interfaces
├── src/
│   └── main.cpp                # CLI benchmark driver, Matrix container, and kernel implementations
├── scripts/
│   ├── run_benchmarks.py       # Automated sweep across N=[16..2048] -> naive_baseline.csv
│   └── plot_results.py         # Matplotlib PDF report generator -> naive_baseline_report.pdf
└── build/                      # Out-of-source CMake build directory (git-ignored)
```

---

## Toolchain & Build Instructions

HACKE targets Linux (or WSL2 Ubuntu) using GCC 13+ and CMake 3.16+.

### 1. Install Dependencies
```bash
sudo apt update && sudo apt install -y build-essential cmake gdb python3-pandas python3-matplotlib
```

### 2. Configure and Build
```bash
mkdir -p build && cd build
cmake ..
make
```

### 3. Run Single Benchmark or Full Automated Sweep
Run the compiled binary directly for a specific matrix dimension `N` (defaults to `256` if omitted):
```bash
./build/matrix_app 1024
```

Or run the full automated sweep across `N = 16` to `2048` to generate `naive_baseline.csv` and `naive_baseline_report.pdf`:
```bash
python3 scripts/run_benchmarks.py
```

---

## Hardware Cache Profiling (`perf`)

To empirically verify cache eviction reductions across Phases 1 through 4, run the binary under Linux `perf`:

```bash
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./build/matrix_app 1024
```

---

## Performance Log

| Phase | Kernel Variant | Matrix Size (`N`) | Execution Time (ms) | Throughput (GFLOPS) | % Peak (81.6 GFLOPS) | L1d Miss Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.1** | Naive `(i, j, k)` | 1024 | TBD | TBD | TBD | TBD |
| **1.2** | Loop Reordered `(i, k, j)` | 1024 | TBD | TBD | TBD | TBD |
| **2.0** | Cache Tiled | 1024 | TBD | TBD | TBD | TBD |
| **3.0** | GotoBLAS Packed | 1024 | TBD | TBD | TBD | TBD |
| **4.0** | AVX2 + FMA Micro-Kernel | 1024 | TBD | TBD | TBD | TBD |