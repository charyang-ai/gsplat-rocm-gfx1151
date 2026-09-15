"""fused_ssim_tbwd: fully-Triton (autotuned) fused SSIM for gsplat training on ROCm/CUDA.

Usage (drop-in for the gsplat fused_ssim loss):

    from fused_ssim_tbwd import fused_ssim
    loss = 1.0 - fused_ssim(pred, gt, padding="valid")
"""
from ._core import (
    fused_ssim,
    _fused_ssim_separable,
    _gaussian_1d,
    _HAS_TRITON,
)

__all__ = ["fused_ssim", "_fused_ssim_separable", "_gaussian_1d", "_HAS_TRITON"]
__version__ = "0.1.0"
