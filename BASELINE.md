# R9700 baseline

Every number here was measured on an **AMD Radeon AI PRO R9700** (RDNA 4,
`gfx1201`, 64 CU, 640 GB/s GDDR6) under ROCm 7.2.1 and PyTorch 2.9.1+rocm.
They are the comparison column for the gfx1151 port — not targets to hit.

Unless stated otherwise the workload is 500k Gaussians, 1920×1080, SH degree 3.

## Hardware

Fill in the right-hand column from `rocminfo` on the Strix Halo box before
using any of the derived numbers below.

| | R9700 (measured) | Strix Halo (fill in) |
|---|---|---|
| Architecture | RDNA 4 / gfx1201 | RDNA 3.5 / gfx1151 |
| Wavefront | 32 | 32 |
| Compute units | 64 | ? (spec sheets say 40) |
| Memory | GDDR6, 640 GB/s | LPDDR5X unified, ? (spec sheets say ~256 GB/s) |
| LDS per workgroup | 64 KB | ? |

## Cumulative training step, tile_size 8

Each row adds one substitution to the row above. Derived from 50-iteration
totals, expressed per step.

| Stack | ms/step | vs stock |
|---|---|---|
| stock ROCm gsplat | 70.36 | 1.00× |
| + TriSSIM | 49.80 | 1.41× |
| + TriRaster | 43.06 | 1.63× |
| + TriIsect (presort) | 25.92 | 2.71× |
| + exact ellipse culling | 24.16 | 2.91× |

At tile_size 16 the same stack lands at 27.62 ms stock and 23.04 ms with exact
culling.

## TriSSIM

Kernel level, one 1×3×1080×1920 pair, `padding="valid"`, fp32, 5 runs × 100
iterations, mean ± std in ms.

| Implementation | Forward | Fwd speedup | Fwd+bwd | Fwd+bwd speedup |
|---|---|---|---|---|
| Baseline (5× grouped conv2d → MIOpen) | 15.15 ± 0.16 | 1.00× | 25.89 ± 0.23 | 1.00× |
| Separable | 11.37 ± 0.05 | 1.33× | 19.75 ± 0.09 | 1.31× |
| torch.compile | 10.07 ± 0.11 | 1.50× | 16.68 ± 0.14 | 1.55× |
| Triton forward only | 2.72 ± 0.00 | 5.56× | 16.02 ± 0.04 | 1.62× |
| **Triton + bi-autotune** | **2.48 ± 0.00** | **6.11×** | **7.21 ± 0.01** | **3.59×** |

In the full training step, 5 paired profiler runs over 50 iterations:

| | Baseline SSIM | TriSSIM | Speedup |
|---|---|---|---|
| SSIM forward + backward | 1070.9 ± 3.8 ms (30.8%) | 87.3 ± 0.8 ms (3.52%) | 12.3× |
| Total self GPU time | 3475.3 ± 4.3 ms | 2479.1 ± 11.6 ms | 1.40× |

Numerics: gradient error vs reference about 1e-12; value error ≤ 4.8e-8.

Negative result worth not rediscovering: a single-kernel variant that recomputes
the K² window instead of splitting into row and column passes ran the forward in
26.94 ms — 9.7× slower than the two-kernel design. The bottleneck is issue and
ALU, not bandwidth.

## TriRaster

Cross-stage coupling of `tile_size`, 5-run mean over 30 iterations, HIP backward,
TriSSIM on (`tests/table1_results.json`):

| Stage | τ=8 | τ=16 | Ratio |
|---|---|---|---|
| Tile intersection | 333.25 ms | 51.95 ms | 6.41× |
| Radix sort | 276.63 ms | 85.61 ms | 3.23× |
| Rasterization forward | 15.24 ms | 17.76 ms | 0.86× |
| Rasterization backward | 364.17 ms | 295.20 ms | 1.23× |
| **Total GPU** | **1524.51 ms** | **967.29 ms** | **1.58×** |

Backward kernel replacement, 50 iterations:

| | Reference (HIP) | TriRaster | Speedup |
|---|---|---|---|
| τ=8 | 597.0 ms | 257.2 ms | 2.32× |
| τ=16 | 488.7 ms | 328.5 ms | 1.49× |

Autotune, measured 2026-08-04. The spread is the reason this must be re-run
rather than assumed:

| tile_size | Best | Winner | Worst | Spread |
|---|---|---|---|---|
| 8 | 5.634 ms | `SPLIT=1, num_warps=1` | 32.701 ms | 5.80× |
| 16 | 6.888 ms | `SPLIT=2, num_warps=1` | 11.717 ms | 1.70× |

Numerics: worst relative gradient error 8.3e-7 across all autotune candidates;
per-Gaussian signed bias within ±3e-9; densification-threshold bias 2.4e-8.

## TriIsect

End-to-end, 50 iterations, with TriSSIM and the Triton backward already active
(`results/synthetic/e2e.json`):

| Config | Step ms | Stage ms | Step speedup |
|---|---|---|---|
| tile 8, baseline | 43.00 | 20.95 | 1.00× |
| tile 8, emit only | 33.12 | 10.77 | 1.30× |
| tile 8, presort | 25.92 | 3.53 | 1.66× |
| tile 8, exact | 24.16 | 2.82 | 1.78× |
| tile 16, baseline | 27.62 | 5.09 | 1.00× |
| tile 16, presort | 24.22 | 1.40 | 1.14× |
| tile 16, exact | 23.04 | 1.14 | 1.20× |
| tile 32, baseline | 31.14 | 1.36 | 1.00× |
| tile 32, presort | 30.36 | 0.57 | 1.03× |

Stage in isolation, 1080p:

| Shape | tile | pairs/Gaussian | HIP | TriIsect | Speedup |
|---|---|---|---|---|---|
| 1920×1080 | 8 | 69.0 | 22.940 ms | 3.756 ms | 6.11× |
| 1920×1080 | 16 | 21.6 | 5.385 ms | 1.495 ms | 3.60× |
| 618×411 | 8 | 12.0 | 2.039 ms | 1.005 ms | 2.03× |
| 618×411 | 16 | 5.0 | 0.601 ms | 0.698 ms | 0.86× |

That last row is below the dispatch crossover and is why the fallback exists.

Phase breakdown at 1920×1080, tile 16, 10,763,024 pairs — the two rows that
matter most for a bandwidth-limited part are the sort pair and the offsets pair:

| Phase | ms |
|---|---|
| count kernel | 0.014 |
| emit kernel | 0.298 |
| depth pre-sort | 0.264 |
| ownership map (`repeat_interleave`) | 0.408 |
| sort, 8-byte key | 3.106 |
| **sort, 4-byte key** | **0.754** |
| offsets, boundary scan | 0.138 |
| **offsets, `searchsorted`** | **0.016** |

Peak memory and pair count at 1920×1080, tile 8:

| Mode | Peak bytes | Pairs |
|---|---|---|
| HIP | 839,516,160 | 34,462,420 |
| emit + presort | 571,749,888 | 34,462,420 |
| + exact culling | 539,120,128 | 25,094,892 |

Dispatch rule: `n_isects >= 1.5 * n_elements and n_isects >= 3_000_000`, fitted
over 80 real-scene operating points, reaching 99.5% of oracle performance.

## Validation floors

PSNR cannot resolve a kernel substitution at this sample size. Over 6 repeats
per arm on bicycle at `data_factor=4`, 30k steps:

| τ | Backward | Mean PSNR | Within-arm range |
|---|---|---|---|
| 8 | HIP | 24.952 dB | 0.631 dB |
| 8 | TriRaster | 25.047 dB | 0.441 dB |
| 16 | HIP | 24.739 dB | 0.507 dB |
| 16 | TriRaster | 24.834 dB | 0.413 dB |

Pooled σ = 0.200 dB against a between-arm difference of 0.094 dB. Two identical
HIP baselines at `data_factor=8` differed by 0.14 dB. Re-derive this on gfx1151;
do not carry the number over.

## Cross-architecture reference: RTX 4090 (Ada, sm_89)

The same harness on a different vendor, which is the best available evidence for
what travels:

| Optimization | On Ada |
|---|---|
| TriSSIM (convolution-based loss) | reproduces |
| tile_size preference | reproduces |
| TriIsect | 1.31× (1.78× on RDNA 4) |
| TriRaster | **0.77× — a regression** |

gfx1151 is far closer to gfx1201 than Ada is, so read this as a floor rather
than a forecast.
