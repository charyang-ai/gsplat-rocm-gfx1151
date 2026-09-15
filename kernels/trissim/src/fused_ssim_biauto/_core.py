"""Fully-Triton `fused_ssim`: autotuned fused forward AND autotuned fused backward.

Self-contained (no cross-module imports). Fast path is CUDA + float32 + padding="valid"
(the gsplat training path); everything else transparently falls back to a pure-torch
separable implementation.

Forward: two fused separable Gaussian-blur passes computing (mu1, mu2, s11, s22, s12)
for all five quantities at once (one ROW kernel + one COL kernel), both autotuned.

Backward: the blur is linear, so the adjoint Bt of the valid separable blur (= a "full"
separable correlation) is applied to all five upstream gradients via two fused,
autotuned Triton kernels, then combined by the chain rule:
  grad_img1 = Bt(g_mu1) + 2*img1*Bt(g_s11) + img2*Bt(g_s12)
  grad_img2 = Bt(g_mu2) + 2*img2*Bt(g_s22) + img1*Bt(g_s12)

This replaces the five conv_transpose2d (MIOpen) calls that used to dominate the
train-step cost, giving a full forward+backward speedup instead of forward-only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton missing / CPU-only env
    _HAS_TRITON = False


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------
def _gaussian_1d(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()  # [w]


def _fused_ssim_separable(img1: torch.Tensor, img2: torch.Tensor,
                          padding: str = "same", train: bool = True) -> torch.Tensor:
    """Pure-torch separable-conv SSIM (fallback path). Mathematically identical."""
    C = img1.shape[-3]
    g = _gaussian_1d(11, 1.5).to(device=img1.device, dtype=img1.dtype)
    k_v = g.view(1, 1, 11, 1).expand(C, 1, 11, 1).contiguous()
    k_h = g.view(1, 1, 1, 11).expand(C, 1, 1, 11).contiguous()
    pad = 0 if padding == "valid" else 11 // 2

    def blur(x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, k_v, padding=(pad, 0), groups=C)
        x = F.conv2d(x, k_h, padding=(0, pad), groups=C)
        return x

    mu1, mu2 = blur(img1), blur(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = blur(img1 * img1) - mu1_sq
    sigma2_sq = blur(img2 * img2) - mu2_sq
    sigma12 = blur(img1 * img2) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


# --------------------------------------------------------------------------------------
# Triton kernels (forward blur + backward adjoint), all autotuned
# --------------------------------------------------------------------------------------
if _HAS_TRITON:

    _CONFIGS = [
        triton.Config({"BLOCK": block}, num_warps=warps)
        for block in (64, 128, 256, 512, 1024)
        for warps in (1, 2, 4, 8)
    ]

    # ---- forward: horizontal (row) pass ----
    @triton.autotune(configs=_CONFIGS, key=["H", "W", "Wout"])
    @triton.jit
    def _row_kernel(
        i1_ptr, i2_ptr,
        o_mu1, o_mu2, o_s11, o_s22, o_s12,
        H, W, Wout,
        w_ptr, K: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        plane = pid // H
        row = pid % H
        cb = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cb < Wout
        in_base = plane * (H * W) + row * W
        out_base = plane * (H * Wout) + row * Wout

        a_mu1 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_mu2 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s11 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s22 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s12 = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(0, K):
            wv = tl.load(w_ptr + k)
            c = cb + k
            cm = mask & (c < W)
            i1 = tl.load(i1_ptr + in_base + c, mask=cm, other=0.0)
            i2 = tl.load(i2_ptr + in_base + c, mask=cm, other=0.0)
            a_mu1 += wv * i1
            a_mu2 += wv * i2
            a_s11 += wv * i1 * i1
            a_s22 += wv * i2 * i2
            a_s12 += wv * i1 * i2
        tl.store(o_mu1 + out_base + cb, a_mu1, mask=mask)
        tl.store(o_mu2 + out_base + cb, a_mu2, mask=mask)
        tl.store(o_s11 + out_base + cb, a_s11, mask=mask)
        tl.store(o_s22 + out_base + cb, a_s22, mask=mask)
        tl.store(o_s12 + out_base + cb, a_s12, mask=mask)

    # ---- forward: vertical (col) pass ----
    @triton.autotune(configs=_CONFIGS, key=["H", "Wout", "Hout"])
    @triton.jit
    def _col_kernel(
        m1_ptr, m2_ptr, s11_ptr, s22_ptr, s12_ptr,
        o_mu1, o_mu2, o_s11, o_s22, o_s12,
        H, Wout, Hout,
        w_ptr, K: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        plane = pid // Hout
        orow = pid % Hout
        cb = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cb < Wout
        out_base = plane * (Hout * Wout) + orow * Wout

        a_mu1 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_mu2 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s11 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s22 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s12 = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(0, K):
            wv = tl.load(w_ptr + k)
            r = orow + k
            base = plane * (H * Wout) + r * Wout
            a_mu1 += wv * tl.load(m1_ptr + base + cb, mask=mask, other=0.0)
            a_mu2 += wv * tl.load(m2_ptr + base + cb, mask=mask, other=0.0)
            a_s11 += wv * tl.load(s11_ptr + base + cb, mask=mask, other=0.0)
            a_s22 += wv * tl.load(s22_ptr + base + cb, mask=mask, other=0.0)
            a_s12 += wv * tl.load(s12_ptr + base + cb, mask=mask, other=0.0)
        tl.store(o_mu1 + out_base + cb, a_mu1, mask=mask)
        tl.store(o_mu2 + out_base + cb, a_mu2, mask=mask)
        tl.store(o_s11 + out_base + cb, a_s11, mask=mask)
        tl.store(o_s22 + out_base + cb, a_s22, mask=mask)
        tl.store(o_s12 + out_base + cb, a_s12, mask=mask)

    # ---- backward: vertical full-correlation (adjoint of valid col conv) ----
    @triton.autotune(configs=_CONFIGS, key=["Hout", "H", "Wout"])
    @triton.jit
    def _adj_v_kernel(
        g1, g2, g3, g4, g5,
        o1, o2, o3, o4, o5,
        Hout, H, Wout,
        w_ptr, K: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)          # over P*H
        plane = pid // H
        orow = pid % H
        cb = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cb < Wout
        out_base = plane * (H * Wout) + orow * Wout

        a1 = tl.zeros((BLOCK,), dtype=tl.float32)
        a2 = tl.zeros((BLOCK,), dtype=tl.float32)
        a3 = tl.zeros((BLOCK,), dtype=tl.float32)
        a4 = tl.zeros((BLOCK,), dtype=tl.float32)
        a5 = tl.zeros((BLOCK,), dtype=tl.float32)
        for kv in tl.static_range(0, K):
            wv = tl.load(w_ptr + kv)
            src = orow - kv                        # adjoint of valid conv
            m = mask & (src >= 0) & (src < Hout)
            base = plane * (Hout * Wout) + src * Wout
            a1 += wv * tl.load(g1 + base + cb, mask=m, other=0.0)
            a2 += wv * tl.load(g2 + base + cb, mask=m, other=0.0)
            a3 += wv * tl.load(g3 + base + cb, mask=m, other=0.0)
            a4 += wv * tl.load(g4 + base + cb, mask=m, other=0.0)
            a5 += wv * tl.load(g5 + base + cb, mask=m, other=0.0)
        tl.store(o1 + out_base + cb, a1, mask=mask)
        tl.store(o2 + out_base + cb, a2, mask=mask)
        tl.store(o3 + out_base + cb, a3, mask=mask)
        tl.store(o4 + out_base + cb, a4, mask=mask)
        tl.store(o5 + out_base + cb, a5, mask=mask)

    # ---- backward: horizontal full-correlation (adjoint of valid row conv) ----
    @triton.autotune(configs=_CONFIGS, key=["H", "Wout", "W"])
    @triton.jit
    def _adj_h_kernel(
        g1, g2, g3, g4, g5,
        o1, o2, o3, o4, o5,
        H, Wout, W,
        w_ptr, K: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)          # over P*H
        plane = pid // H
        row = pid % H
        cb = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)   # out cols in [0,W)
        mask = cb < W
        out_base = plane * (H * W) + row * W
        in_base = plane * (H * Wout) + row * Wout

        a1 = tl.zeros((BLOCK,), dtype=tl.float32)
        a2 = tl.zeros((BLOCK,), dtype=tl.float32)
        a3 = tl.zeros((BLOCK,), dtype=tl.float32)
        a4 = tl.zeros((BLOCK,), dtype=tl.float32)
        a5 = tl.zeros((BLOCK,), dtype=tl.float32)
        for kh in tl.static_range(0, K):
            wh = tl.load(w_ptr + kh)
            src = cb - kh                          # adjoint of valid conv
            m = mask & (src >= 0) & (src < Wout)
            a1 += wh * tl.load(g1 + in_base + src, mask=m, other=0.0)
            a2 += wh * tl.load(g2 + in_base + src, mask=m, other=0.0)
            a3 += wh * tl.load(g3 + in_base + src, mask=m, other=0.0)
            a4 += wh * tl.load(g4 + in_base + src, mask=m, other=0.0)
            a5 += wh * tl.load(g5 + in_base + src, mask=m, other=0.0)
        tl.store(o1 + out_base + cb, a1, mask=mask)
        tl.store(o2 + out_base + cb, a2, mask=mask)
        tl.store(o3 + out_base + cb, a3, mask=mask)
        tl.store(o4 + out_base + cb, a4, mask=mask)
        tl.store(o5 + out_base + cb, a5, mask=mask)


def _fused_blur5_forward(img1: torch.Tensor, img2: torch.Tensor,
                         weight: torch.Tensor):
    """(mu1, mu2, s11, s22, s12) via two autotuned Triton passes. Inputs [P,H,W]."""
    P, H, W = img1.shape
    K = weight.numel()
    R = K // 2
    Wout, Hout = W - 2 * R, H - 2 * R

    row = [torch.empty((P, H, Wout), device=img1.device, dtype=torch.float32)
           for _ in range(5)]
    grid_row = lambda meta: (P * H, triton.cdiv(Wout, meta["BLOCK"]))  # noqa: E731
    _row_kernel[grid_row](img1, img2, row[0], row[1], row[2], row[3], row[4],
                          H, W, Wout, weight, K=K)

    out = [torch.empty((P, Hout, Wout), device=img1.device, dtype=torch.float32)
           for _ in range(5)]
    grid_col = lambda meta: (P * Hout, triton.cdiv(Wout, meta["BLOCK"]))  # noqa: E731
    _col_kernel[grid_col](row[0], row[1], row[2], row[3], row[4],
                          out[0], out[1], out[2], out[3], out[4],
                          H, Wout, Hout, weight, K=K)
    return out[0], out[1], out[2], out[3], out[4]


def _adjoint_blur5(grads, H, W, weight):
    """Apply Bt to five [P,Hout,Wout] grad tensors -> five [P,H,W] via 2 fused kernels.

    BLOCK / num_warps are chosen by triton.autotune (grids are meta-dependent)."""
    P, Hout, Wout = grads[0].shape
    K = weight.numel()
    g = [x.contiguous() for x in grads]

    mid = [torch.empty((P, H, Wout), device=g[0].device, dtype=torch.float32)
           for _ in range(5)]
    grid_v = lambda meta: (P * H, triton.cdiv(Wout, meta["BLOCK"]))  # noqa: E731
    _adj_v_kernel[grid_v](g[0], g[1], g[2], g[3], g[4],
                          mid[0], mid[1], mid[2], mid[3], mid[4],
                          Hout, H, Wout, weight, K=K)

    out = [torch.empty((P, H, W), device=g[0].device, dtype=torch.float32)
           for _ in range(5)]
    grid_h = lambda meta: (P * H, triton.cdiv(W, meta["BLOCK"]))  # noqa: E731
    _adj_h_kernel[grid_h](mid[0], mid[1], mid[2], mid[3], mid[4],
                          out[0], out[1], out[2], out[3], out[4],
                          H, Wout, W, weight, K=K)
    return out  # [a_mu1, a_mu2, a_s11, a_s22, a_s12]


class _FusedBlur5TBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1, img2, weight):
        mu1, mu2, s11, s22, s12 = _fused_blur5_forward(img1, img2, weight)
        ctx.save_for_backward(img1, img2, weight)
        ctx.hw = (img1.shape[1], img1.shape[2])
        return mu1, mu2, s11, s22, s12

    @staticmethod
    def backward(ctx, g_mu1, g_mu2, g_s11, g_s22, g_s12):
        img1, img2, weight = ctx.saved_tensors
        H, W = ctx.hw
        a_mu1, a_mu2, a_s11, a_s22, a_s12 = _adjoint_blur5(
            [g_mu1, g_mu2, g_s11, g_s22, g_s12], H, W, weight)
        grad_img1 = a_mu1 + 2.0 * img1 * a_s11 + img2 * a_s12
        grad_img2 = a_mu2 + 2.0 * img2 * a_s22 + img1 * a_s12
        return grad_img1, grad_img2, None


def _fused_ssim_tbwd(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    N, C, H, W = img1.shape
    weight = _gaussian_1d(11, 1.5).to(device=img1.device, dtype=torch.float32).contiguous()
    p1 = img1.reshape(N * C, H, W).contiguous()
    p2 = img2.reshape(N * C, H, W).contiguous()

    mu1, mu2, s11, s22, s12 = _FusedBlur5TBwd.apply(p1, p2, weight)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = s11 - mu1_sq
    sigma2_sq = s22 - mu2_sq
    sigma12 = s12 - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def fused_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    padding: str = "same",
    train: bool = True,
) -> torch.Tensor:
    """SSIM between two NCHW images in [0,1]; returns the mean SSIM (scalar).

    Fully-Triton (autotuned) forward+backward on the CUDA + float32 + padding="valid"
    fast path; otherwise falls back to the pure-torch separable implementation.
    """
    assert img1.shape == img2.shape, (img1.shape, img2.shape)
    fast = (
        _HAS_TRITON
        and img1.is_cuda
        and img1.dtype == torch.float32
        and padding == "valid"
    )
    if not fast:
        return _fused_ssim_separable(img1, img2, padding=padding, train=train)
    return _fused_ssim_tbwd(img1, img2)
