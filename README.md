# HACKE: Hardware Aware Compute Kernel Engine

HACKE is a transparent, modular, and dependency-free high-performance dense linear algebra compute engine written from scratch in modern C++ (C++20).

The engine focuses on optimizing single-precision General Matrix Multiplication (`SGEMM`: `C = A * B`) on modern x86-64 architectures. Rather than relying on opaque, assembly-heavy vendor libraries like Intel MKL or OpenBLAS, HACKE systematically bridges the gap between a naive triple-loop baseline and theoretical peak hardware throughput through four readable, hardware-sympathetic vertical slices.

---

## Architectural Motivation: Defeating the Memory Wall

Modern x86-64 processors equipped with 256-bit AVX2 and Fused Multiply-Add (FMA) pipelines can retire up to 32 single-precision floating-point operations per clock cycle per core. However, a naive `O(N^3)` matrix multiplication implementation routinely achieves less than **5% of theoretical peak FLOPS**. 

This bottleneck occurs because standard column-wise traversal of matrix `B` steps through memory with a stride of `N` floats. For large matrices, every inner-loop iteration jumps across memory addresses separated by kilobytes, causing catastrophic **spatial locality failure**, **64-byte cache line waste** (fetching 16 floats only to use 1), and **TLB thrashing**. The CPU functional units spend the vast majority of execution time stalled waiting on main DRAM fetches (~60-100ns latency) rather than computing in L1d cache (~1-2ns latency).

HACKE eliminates this bottleneck step-by-step using the GotoBLAS / BLIS analytical cache-hierarchy methodology.

---

## Optimization Roadmap (Vertical Slices)

### Phase 1: Baseline Suite & Benchmarking Harness
* **Flat 1D Memory Representation:** Replaces pointer-chasing 2D heap allocations with contiguous row-major `std::vector<float>` storage (`index = i * N + j`).
* **Deterministic Verification Oracle:** Initializes test matrices using a seeded Mersenne Twister (`std::mt19937`) and validates optimized kernels against reference output within floating-point epsilon (`1e-4`).
* **Monotonic Telemetry:** Measures execution latency via `std::chrono::steady_clock` and normalizes throughput into GFLOPS (`(2.0 * N^3) / (time_sec * 10^9)`).

### Phase 2: Multi-Level Cache Loop Tiling
* Partitions matrices into hierarchical sub-blocks (`Mc`, `Kc`, `Nc`) sized analytically to fit within the target processor's L1 Data, L2, and L3 cache capacities.
* Eliminates pointer aliasing ambiguity via `__restrict__` pointers and compile-time block constants (`constexpr`).

### Phase 3: Contiguous Memory Packing & Alignment
* Allocates 64-byte cache-line-aligned working buffers (`packA` and `packB`) managed via custom C++ RAII wrappers.
* Reorganizes strided matrix sub-views into sequential unit-stride memory panels prior to execution, ensuring 100% cache line utilization and hardware prefetcher synergy while eliminating TLB misses.

### Phase 4: Explicit AVX2 / FMA Register Micro-Kernel
* Implements a register-blocked inner micro-kernel (`Mr x Nr`, e.g., `6x16` or `8x8`) using explicit `<immintrin.h>` intrinsics (`__m256`).
* Maximizes arithmetic intensity by pinning output sub-tiles inside `ymm0-ymm15` vector registers, streaming packed panels via `_mm256_load_ps` and `_mm256_set1_ps`, and executing parallel fused multiply-adds via `_mm256_fmadd_ps`.

---

## Repository Structure

```text
hacke/
├── CMakeLists.txt          # C++20 build configuration & hardware flags (-O3 -march=native)
├── README.md               # Project documentation & profiling commands
├── include/
│   └── kernels.hpp         # Public kernel declarations & benchmark interfaces
├── src/
│   └── main.cpp            # Benchmark driver, verification oracle, and kernel implementations
└── build/                  # Out-of-source CMake build directory (git-ignored)
```

---

## Toolchain & Build Instructions

HACKE targets Linux (or WSL2 Ubuntu) using GCC 13+ and CMake 3.16+.

### 1. Install Dependencies
```bash
sudo apt update && sudo apt install -y build-essential cmake gdb
```

### 2. Configure and Build
```bash
mkdir -p build && cd build
cmake ..
make
```

### 3. Run the Benchmark Suite
```bash
./hacke_bench
```

---

## Hardware Cache Profiling (`perf`)

To empirically verify cache eviction reductions across Phases 1 through 4, run the binary under Linux `perf` inside the `build/` directory:

```bash
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./hacke_bench
```

---

## Performance Log

| Phase | Kernel Variant | Matrix Size (`N`) | Execution Time (ms) | Throughput (GFLOPS) | % Peak (Single Core) | L1d Miss Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.1** | Naive `(i, j, k)` | 1024 | TBD | TBD | TBD | TBD |
| **1.2** | Loop Reordered `(i, k, j)` | 1024 | TBD | TBD | TBD | TBD |
| **2.0** | Cache Tiled | 1024 | TBD | TBD | TBD | TBD |
| **3.0** | GotoBLAS Packed | 1024 | TBD | TBD | TBD | TBD |
| **4.0** | AVX2 + FMA Micro-Kernel | 1024 | TBD | TBD | TBD | TBD |
