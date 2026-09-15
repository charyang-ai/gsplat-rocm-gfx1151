# TriSSIM — A Highly Optimized Triton-based Fused SSIM Library

**TriSSIM** is a fully-Triton, **autotuned forward *and* backward** fused SSIM loss for
Gaussian-Splatting training on AMD ROCm (RDNA4 / gfx1201) and NVIDIA CUDA.

It is a drop-in replacement for the gsplat `fused_ssim` loss. The fast path fuses all five
Gaussian-blur quantities (`mu1`, `mu2`, `img1²`, `img2²`, `img1·img2`) into **two** autotuned
Triton passes for the forward, and applies the blur's adjoint to all five upstream gradients
in **two more** autotuned Triton passes for the backward — replacing the five
`conv_transpose2d` (MIOpen) calls that previously dominated the train step.

## Highlights

- **Fully fused forward + backward** — 4 Triton kernels total, no `conv2d`/`conv_transpose2d`.
- **Autotuned** — block size and warp count are selected by `triton.autotune` per shape.
- **Cross-vendor** — runs on AMD ROCm and NVIDIA CUDA from the same code.
- **Graceful fallback** — transparently drops to a pure-torch separable path when the
  fast-path preconditions aren't met (or Triton is unavailable).
- **Numerically faithful** — gradients match the pure-torch reference to ~1e-12.

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch with a working GPU backend (ROCm or CUDA)
- Triton (ships with the ROCm/CUDA PyTorch build; optional — CPU falls back to torch)

> `torch` and `triton` are intentionally **not** declared as dependencies. On ROCm they ship
> with the pre-installed PyTorch build, and pinning them risks pip pulling a CUDA wheel and
> clobbering the ROCm install.

### Install from source

```bash
git clone <your-repo-url> trissim
cd trissim
pip install -e .
```

Or install directly from the project directory:

```bash
pip install -e fused_ssim_biauto
```

### Verify the install

```python
import torch
from fused_ssim_biauto import fused_ssim, _HAS_TRITON

print("Triton available:", _HAS_TRITON)
x = torch.rand(1, 3, 256, 256, device="cuda", requires_grad=True)
y = torch.rand(1, 3, 256, 256, device="cuda")
loss = 1.0 - fused_ssim(x, y, padding="valid")
loss.backward()
print("ok:", loss.item())
```

## Usage

```python
from fused_ssim_biauto import fused_ssim

# NCHW images in [0, 1]; use padding="valid" (matches gsplat simple_trainer)
loss = 1.0 - fused_ssim(pred, gt, padding="valid")
loss.backward()
```

The accelerated fast path requires **CUDA/ROCm + float32 + `padding="valid"`**; any other
configuration transparently falls back to the separable torch path (mathematically identical).

## Performance

Measured on an **AMD Radeon AI PRO R9700 (RDNA4 / gfx1201)** at 1080p, `padding="valid"`,
float32, forward + backward.

| Metric                         | Pure-torch baseline | TriSSIM  | Speedup |
|--------------------------------|---------------------|----------|---------|
| Forward + backward step time   | 25.7 ms             | 7.2 ms   | ~3.5×   |
| Gradient error vs. reference   | —                   | ~1e-12   | —       |

The gain comes from eliminating the five `conv_transpose2d` (MIOpen) calls in the backward
pass, which previously dominated the SSIM step. TriSSIM replaces the entire forward+backward
with four fused, autotuned Triton kernels, turning a forward-only optimization into a full
forward+backward speedup.

## How it works

**Forward** — two fused separable Gaussian-blur passes compute `(mu1, mu2, s11, s22, s12)`
for all five quantities at once (one ROW kernel + one COL kernel), both autotuned.

**Backward** — the blur is linear, so the adjoint `Bᵀ` of the valid separable blur (a "full"
separable correlation) is applied to all five upstream gradients via two fused, autotuned
Triton kernels, then combined by the chain rule:

```
grad_img1 = Bᵀ(g_mu1) + 2·img1·Bᵀ(g_s11) + img2·Bᵀ(g_s12)
grad_img2 = Bᵀ(g_mu2) + 2·img2·Bᵀ(g_s22) + img1·Bᵀ(g_s12)
```

## License

See repository license.
