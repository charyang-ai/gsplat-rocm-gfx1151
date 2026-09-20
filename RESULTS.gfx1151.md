# gfx1151 results

Measured on `halo4`: AMD Ryzen AI MAX+ 395 / Radeon 8060S (RDNA 3.5, `gfx1151`, 40 CU,
LPDDR5X-8000 → 256 GB/s), Ubuntu 24.04, Linux 7.0.0-31, ROCm 7.2.1, torch
2.9.1+rocm7.2.1, HIP 7.2.53211. Image `gsplat-rocm:gfx1151` built from
`Dockerfile.gfx1151`; every command below is `./run.sh <cmd>`.

The comparison column throughout is `BASELINE.md` (R9700, RDNA 4, `gfx1201`, 64 CU,
640 GB/s). **gfx1151 has 62.5% of the CUs and 40.0% of the bandwidth**, which turns out
to explain most of what moved.

## Phase 1 — build

`csrc.so` carries `hipv4-amdgcn-amd-amdhsa--gfx1151` in all 20 bundles and no gfx942, so
`setup.py.gfx1151.patch` did its job. No ROCm #5853 null-dereference on first dispatch,
as expected from ROCm 7.2.1 on Linux 7.0.0.

Six things the shipped `Dockerfile.gfx1151` needed before it would build at all; see the
git history of that file. Briefly: `COPY triraster` pointed at a path that does not exist
in this layout (`kernels/triraster`), triisect was never installed at all, and
`patches/pycolmap_numpy2.patch` was present but never applied while
`examples/requirements.txt` pinned `numpy<2.0.0` underneath the base image's torch.

## Phase 2 — correctness gates

All pass. In the order MIGRATION.md ranks them by informativeness:

| Gate | Result |
|---|---|
| `isect_correctness_test.py` | 20/20 shapes exactly equal (`torch.equal`) |
| `isect_correctness_test.py --render --exact` | 12/12 renders bit-identical; culling `safe`+`subset` on every shape |
| `rasbwd_correctness_test.py --verify-configs` | 12/12 autotune candidates pass |
| `correctness_test.py` | forward, backward and finite-difference gradcheck all OK |
| `smoke_test.py rasterize` | `RASTERIZE+BACKWARD OK` |

Two cross-checks worth keeping. The intersection list is not merely correct, it is
*identical to R9700's*: 34,462,420 pairs at 1920×1080 tile 8, the same figure BASELINE
records. And exact ellipse culling drops 13.4–24.4% of pairs here (small shapes only —
the oracle is brute force), against BASELINE's 27% at 1080p tile 8.

**One number moved against us.** Worst relative gradient error across all autotune
candidates is **1.05e-05**, where R9700 measured 8.3e-7 — about 12× worse. It is still
four orders of magnitude inside the gate, and the pattern says why it is not a defect:
the large errors land on `v_colors` only, and only for particular `(SPLIT, num_warps)`
pairs, which is fp32 atomic accumulation order changing with the reduction shape.

## Phase 3 — recalibration

### tile_size still prefers 16, and the whole workload tracks bandwidth

`bench_table1.py`, 500k Gaussians, 1920×1080, SH 3, 5 runs × 30 iterations
(`results/table1_results.json`):

| Stage | R9700 τ=16 | gfx1151 τ=16 | slowdown | gfx1151 τ=8/τ=16 | R9700 τ=8/τ=16 |
|---|---|---|---|---|---|
| Tile intersection | 51.95 ms | 78.1 ms | 1.50× | 3.37× | 6.41× |
| **Radix sort** | 85.61 ms | 245.4 ms | **2.87×** | 3.17× | 3.23× |
| Rasterization forward | 17.76 ms | 36.2 ms | 2.04× | 1.70× | **0.86×** |
| Rasterization backward | 295.20 ms | 578.6 ms | 1.96× | 2.22× | 1.23× |
| **Total GPU** | **967.29 ms** | **2418.6 ms** | **2.50×** | **1.62×** | **1.58×**

Three things to read off this.

**The tile_size preference reproduces**, 1.62× for τ=16 against 1.58× on R9700, so the
`--tile-size 16` default carries over and needs no refit. That agrees with the Ada
result BASELINE cites, and it is the least surprising row here.

**The 2.50× total slowdown is almost exactly 1/0.40**, the inverse of the bandwidth
ratio, not the 1/0.625 the CU ratio would give. The whole training step on this part is
bandwidth-limited end to end.

**The radix sort degraded worst, at 2.87×**, above the 2.50× average — and the sort is
precisely the stage TriIsect's depth pre-sort exists to shrink. This is direct support
for MIGRATION.md's prediction that *TriIsect should do relatively better here*, arrived
at before TriIsect was switched on.

One qualitative reversal: the rasterization forward preferred τ=8 on R9700 (0.86×) and
prefers τ=16 here (1.70×).

### The SPLIT rule survives; its threshold halves

`rasbwd_correctness_test.py --tune-report`, 640×480 / 20k Gaussians, num_warps=1:

| tile_size | SPLIT | px/prog | px/lane | ms | |
|---|---|---|---|---|---|
| 8 | 1 | 64 | **2** | 1.043 | winner |
| 16 | 1 | 256 | 8 | 6.490 | worst of the sweep |
| 16 | 2 | 128 | 4 | 1.833 | R9700's winner |
| 16 | 4 | 64 | **2** | 1.640 | winner |

R9700's rule is "smallest SPLIT that keeps px/lane at most 4", which picks SPLIT=2 at
tile 16. gfx1151 wants **2 px/lane at both tile sizes**, i.e. SPLIT=4 at tile 16 — so
`SPLIT = TILE²/(64·num_warps)` here against `/(128·num_warps)` on gfx1201. RDNA 3.5
spills where RDNA 4 still fits, which is what a smaller per-SIMD VGPR budget predicts.
`triton.autotune` finds this unaided; the reason it matters is that carrying RDNA 4's
threshold over as a hand-picked default would cost 3.96× at tile 16, the entire spread
of the sweep. Recorded in `kernels/triraster/src/triraster/_core.py`.

Autotune spread narrowed at tile 8 (2.74× here, 5.80× on R9700) and the winner is
unchanged there (`SPLIT=1, num_warps=1`).

### `_PEAK_GBPS` corrected 640 → 256

`tests/isect_sortbits.py` hardcoded R9700's peak, which only rescales the "% of peak"
column but would have made every gfx1151 efficiency figure wrong. Now 256.0, and exposed
as `--peak-gbps`. Against the corrected peak the sort reaches **155–168 GB/s, 60–66% of
theoretical** — healthy for a radix sort. The staircase confirms the mechanism TriIsect
trades on: 8 bits per pass, so the 14-bit tile key at 1080p/τ=16 costs 2 passes where the
46-bit key costs 6.

### TriSSIM reproduces at a smaller multiple

`ssim_bench_repeat.py`, 1×3×1080×1920, `padding="valid"`, fp32, 5 runs × 100 iterations:

| | R9700 fwd+bwd | gfx1151 fwd+bwd | slowdown |
|---|---|---|---|
| Baseline (5× grouped conv2d → MIOpen) | 25.89 ms | 52.49 ± 0.03 ms | 2.03× |
| **Triton + bi-autotune** | **7.21 ms** | **21.86 ± 0.08 ms** | **3.03×** |
| speedup | 3.59× | **2.40×** | |

Forward only: 7.47 ± 0.02 ms, 3.97× (R9700: 2.48 ms, 6.11×).

TriSSIM degraded *more than the baseline it replaces* (3.03× against 2.03×), which is why
the speedup fell from 3.59× to 2.40×. That is the expected direction: TriSSIM's win comes
from replacing MIOpen's grouped convolutions, which BASELINE's own negative result
identifies as "issue and ALU" bound, with fused Triton passes that are closer to
bandwidth bound — and bandwidth is the axis this part is short on. A 2.40× on the loss is
still the single largest win in the stack.

`bench_table2.py`, whole training step, 5 paired profiler runs
(`results/table2_results.json`):

| | R9700 | gfx1151 |
|---|---|---|
| SSIM fwd+bwd, baseline | 1070.9 ms (30.8% of step) | 1956.5 ± 0.4 ms (23.8%) |
| SSIM fwd+bwd, TriSSIM | 87.3 ms (3.52%) | 271.8 ± 0.5 ms (4.3%) |
| SSIM speedup | 12.3× | **7.20×** |
| Step speedup | 1.40× | **1.30×** |

The loss is a smaller share of the step here (23.8% against 30.8%) because the stages
around it degraded more, which is the same bandwidth story from a different angle.

### Dispatch rule refitted: `min_pairs` 3M → 2M, `min_density` unchanged

Fitted on **90 timed operating points** from the five scenes at step 7000
(`isect_realscene.py --bench` and `--res`, scored by `isect_fit_dispatch.py`;
`results/realscene/*.json`). 71 of the 90 are wins for the Triton path.

| rule | of oracle |
|---|---|
| always HIP | 44.3% |
| always Triton | 95.8% |
| shipped `P/N ≥ 1.5 and P ≥ 3M` | 98.0% |
| **refitted `P/N ≥ 1.5 and P ≥ 2M`** | **99.9%** |
| least-squares cost model | 99.8% |

Sensitivity, percent of oracle:

```
  P >=       1M      2M      3M      4M      5M      6M
  P/N >= 1.25    99.5    99.7    97.8    95.9    94.9    90.7
  P/N >= 1.50    99.8    99.9    98.0    96.0    95.0    90.8   <- peak in every column
  P/N >= 1.75    99.6    99.7    97.7    95.8    94.8    90.9
```

**`min_density` did not move, and that is the more interesting half.** MIGRATION.md
predicted lower bandwidth would let the Triton path win at *lower* densities too, but the
sweep peaks at exactly 1.5 in every column. The fitted cost model agrees without being
told to: its break-even density is `a/b = 1.2583/0.8387 = 1.50`, from the timings alone.
The density term is a ratio of two costs that evidently scale together across these two
parts; the fixed per-call cost, which `min_pairs` stands in for, does not. So the
prediction was right that something was too conservative, and wrong about which term.

Only `min_pairs` changed, in `kernels/triisect/src/triisect/_core.py`.

Two notes on getting here. The synthetic sweep alone **cannot** support this fit: 15 of
its 16 cells are Triton wins, so "always Triton" already scores 99.9% and the cost model
collapses to 56.2% sign agreement with a 24.8 ms median error. Against the full 90 real
points those become 95.8% and 93.3% / 0.516 ms. And a 20-point intermediate fit (`--bench`
only) pointed at `(1.75, 2M)`; adding the 60 resolution points moved `min_density` back to
1.5. Either smaller set would have produced a different, worse answer.

`tests/isect_correctness_test.py --dispatch` restated `1.5, 3_000_000` as its own literals
and passed them explicitly, so it would have gone on passing while testing a rule the
package no longer shipped. It now reads both off `inspect.signature(triisect.isect_tiles)`.

### Superseded: the synthetic-only dispatch fit

Preliminary, from the synthetic sweep only (`isect_ablations.py --sweep` →
`isect_fit_dispatch.py`, `results/sweep.json`). The shipped rule is
`n_isects >= 1.5·n_elements and n_isects >= 3_000_000`, and it leaves two cells on the
table:

| shape | tile | P/N | n_pairs | shipped picks | forced Triton | lost |
|---|---|---|---|---|---|---|
| synthetic | 8 | 4.47 | 2,235,948 | HIP, 1.00× | **1.60×** | 0.780 ms |
| synthetic | 16 | 3.90 | 1,949,204 | HIP, 1.00× | **1.49×** | 0.564 ms |

The binding constraint is `min_pairs`, exactly as MIGRATION.md predicted ("lower
bandwidth should let the Triton path win at *lower* densities, so this is likely too
conservative here"). The sensitivity table reaches 99.9–100.0% of oracle at P ≥ 1M
against 99.3% at P ≥ 3M, and is flat in `min_density` across 1.00–3.00.

This pointed the right way on `min_pairs` but could not resolve `min_density`; see the
real-scene fit above, which supersedes it.

## Phase 4 — the cumulative stack

`profile_trainer.py`, BASELINE's stated workload (500k Gaussians, 1920×1080, SH degree 3 —
*not* profile_trainer's own defaults of 200k / SH 0), 30 profiled iterations after 10
warmup. Each row adds one substitution to the row above. Every run's header was checked to
confirm it took the intended path rather than a silent fallback.

| Stack, tile 8 | R9700 ms/step | R9700 vs stock | gfx1151 ms/step | gfx1151 vs stock |
|---|---|---|---|---|
| stock ROCm gsplat | 70.36 | 1.00× | 148.1 | 1.00× |
| + TriSSIM | 49.80 | 1.41× | 110.8 | 1.34× |
| + TriRaster | 43.06 | 1.63× | 99.3 | 1.49× |
| + TriIsect (presort) | 25.92 | 2.71× | 69.0 | 2.15× |
| + exact ellipse culling | 24.16 | 2.91× | **62.3** | **2.38×** |

At tile 16: stock 98.7 ms/step, full stack **51.3 ms/step, 1.92×** — against R9700's
1.20× at the same tile size. And 51.3 ms is the best absolute configuration on this part,
so tile 16 remains the right default here, by a wider margin than on R9700.

Per substitution, as its own multiplier on the row above:

| | R9700 | gfx1151 | RTX 4090 (BASELINE) |
|---|---|---|---|
| TriSSIM | 1.413× | 1.337× | reproduces |
| **TriRaster** | 1.157× | **1.116×** | **0.77× — regression** |
| TriIsect (presort) | 1.661× | 1.439× | 1.31× |
| exact ellipse culling | 1.073× | **1.108×** | — |

**TriRaster pays here.** That was MIGRATION.md's stated open question — "the one that
might not pay", with Ada's 0.77× as the cautionary case — and gfx1151 keeps almost all of
RDNA 4's gain. Read together with the autotune result, the reason is that what had to
move was the *SPLIT threshold*, not the design: once SPLIT is refitted to 2 px/lane the
atomics-reduction argument carries over intact. The claim that some of these wins belong
to `gsplat` and some to RDNA 4 is *not* supported by TriRaster on this part; if anything
gfx1151 says the win belongs to wave32 RDNA generally, and Ada's regression is about Ada.

**The bandwidth prediction splits.** MIGRATION.md predicted TriIsect would do relatively
better here because its wins are pure memory-traffic reductions against 40% of the
bandwidth. The exact-culling half confirms it — 1.108× here against 1.073× on R9700, the
only substitution that improves relative to R9700. The depth-presort half does not:
1.439× against 1.661×. So "memory-traffic reductions travel to a bandwidth-poor part" holds
for the one that removes work outright, and not for the one that narrows a sort.

### Confirmed at 30k

All five scenes were then trained to 30,000 steps (11.5 h total) and the fit re-derived on
a second set of 90 operating points, where the populations are 1.4–2.5× larger and the
densities correspondingly lower. Percent of oracle:

| rule | 7k (90 cells) | 30k (90 cells) | combined (180 cells) |
|---|---|---|---|
| `P/N ≥ 1.5, P ≥ 1M` | 99.8% | **100.0%** | **99.9%** |
| **`P/N ≥ 1.5, P ≥ 2M`** (adopted) | **99.9%** | 99.8% | **99.9%** |
| `P/N ≥ 1.5, P ≥ 3M` (gfx1201) | 98.0% | 99.7% | 99.0% |
| cost model | 99.8% | 100.0% | 99.9% |

`2_000_000` is within 0.2% of optimal at both population sizes and optimal on the combined
set; `1_000_000` is indistinguishable from it and would have been equally defensible. The
old `3_000_000` is the only one of the three that is clearly worse, and only at 7k — which
is worth noting, because had the port been done at 30k alone the original constant would
have looked adequate at 99.7% and the refit would have been skipped.

`min_density` peaks at exactly 1.50 on all three sets, and the cost model's independent
break-even lands at 1.50 / 1.53 / 1.52 respectively. That constant is stable across two
architectures and a 2.5× change in population size.

### PSNR cross-check

Not a planned deliverable, but the 30k runs supply one for free. bicycle at
`data_factor=4`, tile 16, 30k steps: **24.594 dB** here against BASELINE's **24.739 dB**
on R9700 — 0.145 dB apart, inside both its 0.200 dB pooled σ and its 0.507 dB within-arm
range. Final PSNR / population for all five: bicycle 24.594 / 7.75M, counter 29.640 /
1.58M, garden 27.693 / 6.62M, room 32.077 / 2.28M, stump 26.759 / 5.63M.

## Status

Phases 0 through 4 are complete, including the dispatch refit and its confirmation at 30k.

One item outstanding: **the PSNR noise floor.** BASELINE is explicit that its 0.200 dB
pooled σ must be re-derived here rather than carried over. Two things make that expensive
here rather than merely slow:

- `tests/run_psnr_matrix.sh` is written for an 8-GPU host — it dispatches in waves of
  `NGPU` so no two runs share a device, and defaults to `GPUS="0 1 2 3 4 5 6 7"`. On this
  single-GPU box the same 24 runs (6 repeats × 4 arms) serialize.
- One 30k bicycle run at `data_factor=4` takes ~4.5 h here, so the matrix as specified is
  four to five days of continuous GPU time.

The harness itself does work here, which was not a given: it drives
`examples/simple_trainer.py`, and `train_realscene.py` exists precisely because that
trainer could not run on the gfx1201 host (`torchmetrics.image.lpip` needs torchvision,
which had no wheel for that torch build). This image has torchvision, and
`run_simple_trainer.py default --help` resolves. Three host-specific defaults need
overriding: `GPUS="0"`, `REPO=/opt/gsplat`, and `DATA_DIR`.

Its `data_factor` should not be traded down for speed: the script's own header rejects
factor 8 on the grounds that bicycle then carries ~28 Gaussians per pixel, where the
survival of any individual Gaussian is chaotically sensitive — i.e. factor 8 would inflate
the very noise floor being measured.
