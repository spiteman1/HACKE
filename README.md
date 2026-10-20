# HACKE: Hardware Aware Compute Kernel Engine

An open-source, dependency-free dense linear algebra (GEMM) library in modern C++ that aims to close the gap between a CPU's practical and theoretical peak performance.

> **Status: work in progress (started October 2026).** The design below is the plan, not the result. Benchmarks and measured numbers will be added here as each component lands.

## The problem

Dense linear algebra sits at the heart of AI training and inference, yet naive implementations typically use under 5% of what the hardware can deliver. The cause is the memory wall: processor speed has grown much faster than DRAM speed, so the CPU spends most of its time idle, waiting for data. In a naive matrix multiply, useful data is evicted from cache and then fetched again by later operations.

This also matters for energy. Moving data from DRAM costs up to two orders of magnitude more energy per bit than reading from local cache, so memory stalls directly inflate electricity usage and heat.

Industry-standard libraries such as Intel oneMKL solve this, but they are closed source and large, which makes their internal scheduling hard to inspect or tailor. HACKE is meant to be a transparent, modular alternative where you can see and control memory layout and register allocation.

## The approach

HACKE attacks the memory wall with three optimisations, following the structure described in Goto's *Anatomy of High-Performance Matrix Multiplication* and the BLIS framework.

1. **Hierarchical loop tiling.** Matrices are split into blocks sized to fit the L2/L3 caches. Each block is brought into cache once and fully reused before being discarded, instead of being refetched from main memory.
2. **Contiguous memory packing.** Sub-blocks are copied into sequential, 64-byte cache-line-aligned buffers. This stops the CPU loading a whole cache line just to use one value out of 16 because of strided access.
3. **SIMD micro-kernel.** The innermost loop uses explicit SIMD intrinsics (AVX2 as the initial target) and register blocking. Multiple elements are processed per instruction, and intermediate results stay pinned in registers instead of bouncing through cache.

## Goals

These are targets, not measured results.

- Reach 75-85% of theoretical peak floating-point throughput, versus under 5% for a naive triple loop.
- Reduce the L1 data cache miss rate to under 3%, down from roughly 25% in naive code.
- Show an order of magnitude reduction in L2/L3 evictions.
- Keep the library dependency-free at runtime. Build and profiling tools (CMake, `perf`, PAPI) are development-only.

## Planned architecture

```
naive baseline -> loop tiling -> memory packing -> SIMD micro-kernel
        \______________ benchmark + correctness harness ______________/
```

Planned layout (subject to change):

```
hacke/
├── include/hacke/     # public headers
├── src/
│   ├── naive/         # triple-loop baseline
│   ├── tiling/        # cache blocking
│   ├── packing/       # aligned pack buffers
│   └── kernel/        # SIMD micro-kernel
├── bench/             # GFLOPS harness and perf/PAPI counters
├── tests/             # numerical correctness checks
└── CMakeLists.txt
```

## Methodology

Development uses agile sprints built on **vertical slices**: a minimal working version of every layer first, then iterative refinement once an end-to-end pipeline exists. Each sprint ends with an evaluation against the goals set at its start. The two metrics checked every sprint are mathematical correctness and peak GFLOPS.

## Evaluation plan

- **Throughput:** GFLOPS across square and non-square matrices, scaling from N=64 to N=4096.
- **Baselines:** the naive triple loop and Intel oneMKL, plotted against HACKE.
- **Hardware counters:** L1 data cache miss rates and L2/L3 evictions sampled with Linux `perf` and PAPI.
- **Correctness:** results compared against the naive implementation within a floating-point tolerance.
- **Method notes:** theoretical peak, clock speed and compiler flags will be documented alongside every result so the numbers can be reproduced.

## Roadmap

- [ ] Naive triple-loop baseline
- [ ] Benchmark harness (GFLOPS, timing, correctness check)
- [ ] Loop tiling engine
- [ ] Memory packing layer
- [ ] SIMD micro-kernel
- [ ] `perf` / PAPI cache-counter integration
- [ ] oneMKL comparison
- [ ] Results table and plots

## Building

Build instructions will be added once there is code to build. The plan is CMake with a C++17 (or newer) compiler.

## References

- Goto, K. (2008). *Anatomy of High-Performance Matrix Multiplication*. The University of Texas at Austin.
- Van Zee, F. G. and van de Geijn, R. A. (2015). *BLIS: A Framework for Rapidly Instantiating BLAS Functionality*. The University of Texas at Austin.
- Wolfe, M. (1989). *More Iteration Space Tiling*. Oregon Graduate Center.
- Wulf, W. and McKee, S. (1995). *Hitting the Memory Wall: Implications of the Obvious*.
- Horowitz, M. (2014). *Computing's Energy Problem (and what we can do about it)*.

## Licence

To be decided (MIT is a common choice for a project like this).
