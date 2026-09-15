# gsplat on AMD gfx1151 — kernel optimization port kit

Everything needed to move three 3D Gaussian Splatting kernel optimizations from
AMD RDNA 4 (`gfx1201`, Radeon AI PRO R9700) to RDNA 3.5 (`gfx1151`, Ryzen AI Max
"Strix Halo" / Radeon 8060S). This repository is self-contained: the kernels,
the build patches, the test harnesses, and the reference measurements are all
here, so a machine with a Strix Halo part and Docker needs nothing else.

**Start with [`MIGRATION.md`](MIGRATION.md).** It is written as an execution
plan with per-phase checks. [`BASELINE.md`](BASELINE.md) is the R9700 comparison
column.

## The three optimizations

Each targets a different stage of the training iteration, which is why their
speedups compose rather than overlap. On R9700 the stack takes a step from
70.4 ms to 24.2 ms.

**TriSSIM** — the SSIM loss ran through MIOpen grouped convolutions and took
30.6% of GPU time. Four Triton kernels replace it: a separable 11×11 Gaussian
blur with all five blur quantities fused into one pass, forward and backward
autotuned independently. 25.89 → 7.21 ms, 3.59×.

**TriRaster** — one `tile_size` governs intersection, sorting, forward and
backward, and their optima conflict. A `SPLIT` parameter decouples backward
execution granularity from the geometric tile, cutting atomics from 18 scalar to
4 vector per (Gaussian, tile) pair. 2.32× on the backward at tile 8.

**TriIsect** — output-parallel emit with one lane per pair, a depth pre-sort that
narrows the radix sort from 46-bit to 14-bit keys, `searchsorted` in place of the
boundary-scan kernel, and optional exact ellipse culling that drops 27% of pairs
without changing the render. 6.11× on the stage in isolation.

## Layout

```
Dockerfile.gfx1151     ROCm 7.2.1 + torch 2.9.1, all six build fixes applied
MIGRATION.md           the execution plan — read this first
BASELINE.md            R9700 reference measurements
patches/               build and correctness patches for the ROCm/gsplat fork
kernels/               the three Triton packages, installed by the Dockerfile
tests/                 correctness gates, benchmarks, and recalibration harnesses
shims/                 pure-torch fused_ssim, only used if TriSSIM is unavailable
docs/                  design notes carried over from the RDNA 4 work
```

## Two things that will bite you

**ROCm 7.2.0 is a hard floor on this hardware.** ROCm 7.0 builds and imports
fine, then null-dereferences inside `libhsa-runtime64.so` on the first kernel
dispatch — [ROCm #5853](https://github.com/ROCm/ROCm/issues/5853), a gfx1151
CWSR/VGPR-count mismatch. You also need Linux ≥ 6.18.4 for the kernel half of
the fix. None of this applies to gfx1201, so it is easy to trip over.

**A wrong wave32 reduction is silent.** rocPRIM's `check_virtual_wave_size`
guard turns a hardcoded `warp_reduce<…,64>` into a no-op instead of an error, so
training keeps running and still produces believable timings. Run the Phase 2
gates before looking at a single benchmark number.

## Provenance

The kernels come from the three thread branches on
[charyang-ai/gsplat-rocm-rdna4](https://github.com/charyang-ai/gsplat-rocm-rdna4)
(`trissim`, `triraster`, `triisect`), already merged here — see the last section
of `MIGRATION.md` for what was resolved. The gfx1151 platform patches come from
earlier RDNA 3.5 work that was validated on real silicon and proposed upstream as
[ROCm/gsplat#17](https://github.com/ROCm/gsplat/pull/17).

gsplat is Apache-2.0; the base image is AMD's `rocm/pytorch`.
