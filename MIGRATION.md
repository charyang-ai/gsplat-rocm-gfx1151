# Porting three 3DGS kernel optimizations to gfx1151

This is an execution plan, not a description. Work through the phases in order;
each one ends with a check that decides whether to continue.

**Target:** AMD Ryzen AI Max "Strix Halo" / Radeon 8060S — RDNA 3.5, `gfx1151`.
**Source:** the same three kernels running on Radeon AI PRO R9700 — RDNA 4, `gfx1201`.

## What is already true, so you do not redo it

The three kernels are **pure Triton**. There is no `.hip` or `.cu` in
`kernels/`; Triton compiles for whatever target it finds at run time. Nothing in
them needs porting in the "rewrite for a new ISA" sense.

Both parts are **wave32**. This is the single most useful fact here. TriRaster's
central design argument — keep the reduction inside one wave so it rides DPP,
because crossing waves forces it through LDS and never pays — holds on gfx1151
unchanged. Do not re-derive it.

`triton.autotune` decorates every kernel and no winning config is hardcoded, so
the kernel-level search re-runs by itself on the new part. What does *not*
re-run by itself is anything fitted from measurements; see Phase 3.

The platform layer is done and was validated on real gfx1151 silicon: gradcheck
agreed to 2.0e-6 and a 7000-step `simple_trainer.py` reached PSNR 20.94 /
SSIM 0.60 at 2.66M Gaussians.

## Phase 0 — Confirm the box, and record what it actually is

**ROCm 7.2.0 or newer is mandatory, not a preference.** On ROCm 7.0 the first
kernel dispatch null-dereferences inside `libhsa-runtime64.so`
([ROCm #5853](https://github.com/ROCm/ROCm/issues/5853), a gfx1151 CWSR/VGPR-count
mismatch). The fix has a kernel half (CWSR/ctl_stack export, Linux ≥ 6.18.4) and
a userspace half ([rocm-systems#2200](https://github.com/ROCm/rocm-systems/pull/2200)),
and the userspace half first ships in stock ROCm 7.2.0. This failure does not
exist on gfx1201, so nothing upstream of this repo warns you about it.

```bash
uname -r                  # need >= 6.18.4
rocminfo | grep -i gfx    # expect gfx1151
rocminfo | grep -iE 'Compute Unit|LDS|Wavefront'
rocm-smi --showmeminfo vram
```

Write the CU count, memory bandwidth, and LDS size into `BASELINE.md` next to
the R9700 column. Two numbers in this plan are quoted from general spec sheets
rather than measured — roughly 256 GB/s LPDDR5X and 40 CU — and every
recalibration below depends on them being right.

**Check:** `gfx1151` reported, kernel ≥ 6.18.4. If ROCm is older than 7.2.0,
stop and upgrade; no amount of patching the kernels gets around it.

## Phase 1 — Build

```bash
docker build -f Dockerfile.gfx1151 -t gsplat-rocm:gfx1151 .
```

Six fixes are applied inside, in this order:

1. `setup.py.gfx1151.patch` — honour `PYTORCH_ROCM_ARCH` so the build does not
   silently fall back to gfx942, and put the vendored glm include first.
2. glm `.inl` top-up between the two build passes (hipify copies `.hpp` but not
   the sibling `.inl`, so pass 1 is *expected* to fail).
3. `wave_size.gfx1151.patch` — compile-time `WARP_SIZE`, 64 on CDNA/gfx9 and 32
   on RDNA. This is a correctness fix, not a build fix; see Phase 2.
4. `cub_allocator_torch210.patch` — torch ≥ 2.10 removed
   `c10::hip::HIPCachingAllocator::get()`; route CUB temp storage through
   `at::empty`.
5. `expose_sort.patch` — expose the fork's rocprim `DeviceRadixSort::SortPairs`
   to Python so triisect can sort an explicit bit range.
6. The three kernel packages install from `kernels/`.

> Patch 4 is deliberately WARP_SIZE-free. The version of it that lives in the
> gfx1201 tree carries its own copy of the wave32 fix, and applying that one
> alongside `wave_size.gfx1151.patch` conflicts — both edit the same region of
> `gsplat/cuda/include/Common.cuh`. The copy here has that hunk removed and its
> offsets recomputed against the post-wave32 file.

**Check:** the image builds, and `import gsplat, triraster, triisect, fused_ssim`
all succeed inside it.

## Phase 2 — Correctness gates, before any timing

Do not skip ahead to benchmarks. A broken wave32 reduction is **silent** on this
hardware: rocPRIM's `check_virtual_wave_size` guard turns a hardcoded
`warp_reduce<…,64>` into a no-op rather than an error, so training still runs,
still converges to something, and still produces plausible timings. That failure
mode is the entire subject of the gradient-gate work.

```bash
python tests/smoke_test.py rasterize          # imports + one GPU rasterize
python tests/correctness_test.py              # HIP fwd+bwd vs torch reference
python tests/rasbwd_correctness_test.py --verify-configs   # every autotune candidate
python tests/isect_correctness_test.py        # pairs/offsets, exact comparison
python tests/isect_correctness_test.py --render --exact    # ellipse culling vs oracle
```

**Check, in order of how much they tell you:**

- `isect_correctness_test.py` compares with `torch.equal`. It is exact or it is
  broken; there is no tolerance to argue about.
- `rasbwd_correctness_test.py --verify-configs` must pass for *every* autotune
  candidate, not just the winner — the winner changes on new hardware, so a
  candidate that is wrong but slow on R9700 can become the wrong-and-selected
  one here. On R9700 the worst relative gradient error across all candidates was
  8.3e-7 and the per-Gaussian signed bias stayed within ±3e-9.
- `correctness_test.py` is the wave32 gate proper. On R9700, TriSSIM gradients
  matched the torch reference to about 1e-12.

If a gate fails, stop. Everything after this point is meaningless until it
passes, and a plausible-looking number is worse than no number.

## Phase 3 — Recalibrate

This is the real work. Five values were fitted on R9700 and will be wrong here.
They do not raise errors when wrong; they quietly pick worse paths.

| Value | Location | R9700 value | Why it moves |
|---|---|---|---|
| `min_density` | `kernels/triisect/src/triisect/_core.py:627` | `1.5` | The HIP/Triton crossover. Lower bandwidth should let the Triton path win at *lower* densities, so this is likely too conservative here. |
| `min_pairs` | `kernels/triisect/src/triisect/_core.py:628` | `3_000_000` | Same crossover; fitted over 80 real-scene operating points on R9700. |
| `_PEAK_GBPS` | `tests/isect_sortbits.py:43` | `640.0` | Hardcoded R9700 peak. Only affects reported achieved-bandwidth, but it will make every gfx1151 figure wrong. |
| SPLIT rule | `kernels/triraster/src/triraster/_core.py` docstring | ≤4 px/lane → `SPLIT=1` at tile 8, `2` at tile 16 | The threshold is RDNA4's VGPR spill point. autotune re-searches, but the *rule* needs re-checking. |
| `tile_size` | `--tile-size` on the trainer | `16` | The cross-stage balance point, which depends on bandwidth and CU count. |

```bash
# kernel-level autotune re-runs by itself; capture what it picks
python tests/rasbwd_correctness_test.py --tune-report

# tile_size coupling — produces this box's own version of the table
python tests/bench_table1.py                   # writes table1_results.json

# refit the triisect dispatch rule from real scenes
python tests/train_realscene.py                # 30k steps, needs Mip-NeRF 360 data
python tests/isect_realscene.py --bench
python tests/isect_fit_dispatch.py             # compares candidate rules against the oracle

# sanity: where does the sort sit against this box's real peak?
python tests/isect_sortbits.py

# re-measure the SSIM speedup with the same 5-run statistics as the R9700 tables
python tests/ssim_bench_repeat.py      # kernel level, regenerates BASELINE.md's Table 1 shape
python tests/bench_table2.py           # whole training step, paired profiler runs
```

**Check:** you have a gfx1151 `table1_results.json`, a refitted
`(min_density, min_pairs)` pair, and a recorded autotune winner per tile size.
Compare each against the R9700 column in `BASELINE.md` and write down which
direction it moved and why that is physically plausible.

## Phase 4 — Measure, and be ready for a different answer

```bash
python tests/profile_trainer.py --ssim baseline --ras_bwd hip  --tile-size 8   # stock
python tests/profile_trainer.py --ssim trissim                 --tile-size 8
python tests/profile_trainer.py --ssim trissim --ras_bwd triton --tile-size 8
python tests/profile_trainer.py --ssim trissim --ras_bwd triton --isect triton --tile-size 8
python tests/profile_trainer.py --ssim trissim --ras_bwd triton --isect exact  --tile-size 16
```

There is already cross-architecture evidence for what will survive. Repeating
this harness on an RTX 4090 showed the convolution-based loss and the tile-size
preference reproduce, **the backward-rasterizer substitution becomes a 0.77×
regression**, and the intersection substitution still gives 1.31×. Ada is much
further from RDNA4 than gfx1151 is, so treat that as a pessimistic floor — but
treat TriRaster specifically as an open question, not a given.

Two predictions worth stating up front, so that confirming or breaking them is
informative either way:

- **TriIsect should do relatively better here than on R9700.** Its wins — the
  sort narrowed from 46-bit to 14-bit keys (3.106 → 0.754 ms on R9700) and
  exact ellipse culling dropping 27% of pairs — are pure memory-traffic
  reductions, and this part has roughly 40% of R9700's bandwidth.
- **TriRaster is the one that might not pay.** Its gains come from atomics
  reduction and register-pressure tuning, and the SPLIT thresholds were fitted
  to RDNA4's register budget with 64 CUs behind them.

If TriRaster regresses, that is a result, not a failure. It is the cleanest
possible support for the claim the journal draft already makes — that some of
these wins belong to `gsplat` and some belong to RDNA4 — and gfx1151 isolates
that better than Ada does, because bandwidth and CU count are the only things
that changed while vendor, wavefront width, and compiler stack stayed put.

**Do not reuse the R9700 PSNR noise floor.** The pooled standard deviation of
0.200 dB was measured on that box; re-derive it here with
`tests/run_psnr_matrix.sh` and `tests/collect_psnr_matrix.py` before using PSNR
to compare anything.

## If you are starting from the upstream branches instead of this kit

The three threads live as branches on
[charyang-ai/gsplat-rocm-rdna4](https://github.com/charyang-ai/gsplat-rocm-rdna4):
`trissim`, `triraster`, `triisect`. Two cautions if you pull from there rather
than from this repo:

- The `trissim` branch forks from `fb108a4`, one commit *behind* `main`. It is
  missing `3fc6b66` (TriRaster's tiled optimization). Rebase before merging or
  you will revert that work.
- Six files overlap across the three branches: `.gitignore`, `README.md`,
  `tests/collect_psnr_matrix.py`, `tests/profile_trainer.py`,
  `tests/run_psnr_matrix.sh`, and `triraster/src/triraster/_core.py`. This kit
  has already resolved them by taking the triisect working copy, which is the
  most integrated of the three; `_core.py` was identical in both copies, and the
  three test scripts are newest there.
