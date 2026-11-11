# -*- coding: utf-8 -*-
"""
Convolutional neural quantum state with open or periodic boundaries.

boundary="open"      zero padding at the edges (an edge site sees no neighbor beyond the boundary),
                     centered kernels, and a separate visible bias a_i for every site. No false
                     translation symmetry: edge sites can behave differently from bulk sites.
boundary="periodic"  circular convolution and a single visible bias (exact translation invariance).

Drop-in: CNN(θ, N, α) keeps working inside sNQS / VMC. Geometry and boundary for those calls come
from `configure(...)`, which you call once at the start of a run, e.g.
    configure(Lx=10, Ly=1, boundary="open")          # chain with open boundaries
    configure(Lx=3, Ly=3, boundary="open")           # 3x3 grid with open boundaries
Site index convention: i = x * Ly + y (same as before).
"""
import torch as tc
from typing import Sequence

__all__ = ["CNN", "random_θ", "random_θ_jq", "num_cnn_params", "configure"]

_DEFAULTS = {"Lx": None, "Ly": None, "kernel_size": None, "boundary": "open"}


def configure(*, Lx: int | None = None, Ly: int | None = None,
              kernel_size: int | tuple[int, int] | None = None, boundary: str = "open") -> None:
    """Set the geometry used whenever CNN / random_θ / num_cnn_params are called without these kwargs."""
    if boundary not in ("open", "periodic"):
        raise ValueError("boundary must be 'open' or 'periodic'.")
    _DEFAULTS.update(Lx=Lx, Ly=Ly, kernel_size=kernel_size, boundary=boundary)


def _resolve(N, α, Lx, Ly, kernel_size, boundary):
    """Normalize α to channel counts and resolve (Lx, Ly, Kx, Ky, boundary)."""
    Lx = _DEFAULTS["Lx"] if Lx is None else Lx
    Ly = _DEFAULTS["Ly"] if Ly is None else Ly
    kernel_size = _DEFAULTS["kernel_size"] if kernel_size is None else kernel_size
    boundary = _DEFAULTS["boundary"] if boundary is None else boundary
    if boundary not in ("open", "periodic"):
        raise ValueError("boundary must be 'open' or 'periodic'.")

    channels = (int(α),) if isinstance(α, int) else tuple(int(c) for c in α)

    if Lx is None and Ly is None:
        Lx, Ly = N, 1
    elif Lx is None:
        Lx = N // Ly
    elif Ly is None:
        Ly = N // Lx
    assert Lx * Ly == N, f"Lx * Ly ({Lx}*{Ly}) must equal N ({N})."

    # Largest useful kernel: periodic -> L (all distinct shifts); open -> 2L-1 (every site sees every site)
    kx_max, ky_max = (Lx, Ly) if boundary == "periodic" else (2 * Lx - 1, 2 * Ly - 1)
    if kernel_size is None:
        Kx, Ky = kx_max, ky_max
    elif isinstance(kernel_size, int):
        Kx, Ky = min(kernel_size, kx_max), min(kernel_size, ky_max)
    else:
        Kx, Ky = min(int(kernel_size[0]), kx_max), min(int(kernel_size[1]), ky_max)
    return channels, Lx, Ly, Kx, Ky, boundary


def num_cnn_params(N: int, α: int | Sequence[int], *, Lx: int | None = None, Ly: int | None = None,
                   kernel_size: int | tuple[int, int] | None = None, boundary: str | None = None) -> int:
    channels, Lx, Ly, Kx, Ky, boundary = _resolve(N, α, Lx, Ly, kernel_size, boundary)
    Np = N if boundary == "open" else 1          # visible biases: one per site (open) or one shared (periodic)
    c_in = 1
    for c_out in channels:
        Np += c_out + c_out * c_in * Kx * Ky
        c_in = c_out
    return Np


class CNN:
    r"""
    ln ψ(s) = Σ_i a_i s_i  (open)  or  a Σ_i s_i  (periodic)
            + Σ_{c,r} x^{(L)}_{c,r},   x^{(l+1)} = ln 2cosh( W_l * x^{(l)} + b_l ),  x^{(0)} = s.

    Single layer (α = int) or multi-layer (α = [c1, c2, ...]). d_lnPsi is computed analytically.
    """

    def __init__(self, θ: tc.Tensor, N: int, α: int | Sequence[int], *, Lx: int | None = None,
                 Ly: int | None = None, kernel_size: int | tuple[int, int] | None = None,
                 boundary: str | None = None) -> None:
        self.channels, self.Lx, self.Ly, self.Kx, self.Ky, self.boundary = _resolve(
            N, α, Lx, Ly, kernel_size, boundary)
        self.N, self.α = N, α
        self.open = self.boundary == "open"
        self.n_vis = N if self.open else 1
        self.Np = num_cnn_params(N, α, Lx=self.Lx, Ly=self.Ly, kernel_size=(self.Kx, self.Ky),
                                 boundary=self.boundary)
        assert θ.ndim == 1 and θ.numel() == self.Np, (
            f"θ has {θ.numel()} elements, expected Np={self.Np} for N={N}, α={α}, boundary={self.boundary}.")
        assert θ.is_complex(), "θ must be complex."
        self.θ, self.device, self.dtype = θ, θ.device, θ.dtype

        self._layer_specs: list[tuple[slice, slice, int, int]] = []
        offset, c_in = self.n_vis, 1
        for c_out in self.channels:
            b_slice = slice(offset, offset + c_out)
            offset += c_out
            w_len = c_out * c_in * self.Kx * self.Ky
            self._layer_specs.append((b_slice, slice(offset, offset + w_len), c_out, c_in))
            offset += w_len
            c_in = c_out

        self._patch_idx = self._build_patch_indices()    # (Kx*Ky, N); value N = zero "phantom" site (open)

    # ---------------- geometry ----------------
    def _build_patch_indices(self) -> tc.Tensor:
        dev, Lx, Ly, Kx, Ky = self.device, self.Lx, self.Ly, self.Kx, self.Ky
        x = tc.arange(Lx, device=dev).view(1, 1, Lx, 1)
        y = tc.arange(Ly, device=dev).view(1, 1, 1, Ly)
        if self.open:                                    # centered offsets: -(K//2) ... K-1-K//2
            dx = (tc.arange(Kx, device=dev) - Kx // 2).view(Kx, 1, 1, 1)
            dy = (tc.arange(Ky, device=dev) - Ky // 2).view(1, Ky, 1, 1)
        else:
            dx = tc.arange(Kx, device=dev).view(Kx, 1, 1, 1)
            dy = tc.arange(Ky, device=dev).view(1, Ky, 1, 1)
        px = (x + dx).expand(Kx, Ky, Lx, Ly)
        py = (y + dy).expand(Kx, Ky, Lx, Ly)
        if self.open:
            valid = (px >= 0) & (px < Lx) & (py >= 0) & (py < Ly)
            idx = px.clamp(0, Lx - 1) * Ly + py.clamp(0, Ly - 1)
            idx = tc.where(valid, idx, tc.full_like(idx, self.N))
        else:
            idx = (px % Lx) * Ly + (py % Ly)
        return idx.reshape(Kx * Ky, self.N)

    def _pad(self, x: tc.Tensor) -> tc.Tensor:
        """Append the zero phantom site (open boundaries only)."""
        if not self.open:
            return x
        zeros = tc.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
        return tc.cat([x, zeros], dim=-1)

    def _extract_patches(self, x: tc.Tensor) -> tc.Tensor:
        """x: (M, C_in, N) -> patches (M, C_in * Kx * Ky, N)."""
        M, C_in, _ = x.shape
        return self._pad(x)[:, :, self._patch_idx].reshape(M, C_in * self.Kx * self.Ky, self.N)

    # ---------------- parameters ----------------
    @property
    def a(self) -> tc.Tensor:
        return self.θ[:self.n_vis]                       # (N,) open, (1,) periodic

    def _unpack_layers(self) -> list[tuple[tc.Tensor, tc.Tensor]]:
        K_area = self.Kx * self.Ky
        return [(self.θ[b], self.θ[w].reshape(c_out, c_in * K_area))
                for b, w, c_out, c_in in self._layer_specs]

    # ---------------- network ----------------
    def lnPsi(self, s_mn: tc.Tensor) -> tc.Tensor:
        *m, n = s_mn.shape
        s = s_mn.reshape(-1, n).to(dtype=self.dtype)
        part1 = s @ self.a if self.open else s.sum(dim=1) * self.a[0]
        x = s.unsqueeze(1)
        for b_l, W_l in self._unpack_layers():
            z = tc.matmul(W_l, self._extract_patches(x)) + b_l[None, :, None]
            x = tc.log(2.0 * tc.cosh(z))
        return (part1 + x.sum(dim=(1, 2))).reshape(*m)

    def probability(self, s_mn: tc.Tensor) -> tc.Tensor:
        logw = 2.0 * self.lnPsi(s_mn).real.to(dtype=tc.float64)
        flat = logw.reshape(-1)
        if flat.numel() == 0:
            raise ValueError("Input states must be non-empty.")
        return tc.exp(flat - tc.logsumexp(flat, dim=0)).reshape(logw.shape)

    def d_lnPsi(self, s_mn: tc.Tensor) -> tc.Tensor:
        """Per-sample log-derivatives O_{m,j} = ∂ ln ψ(s_m) / ∂θ_j (holomorphic in θ)."""
        *m, n = s_mn.shape
        s = s_mn.reshape(-1, n).to(dtype=self.dtype)
        M, K_area, N = s.shape[0], self.Kx * self.Ky, self.N
        layers = self._unpack_layers()
        L = len(layers)

        patches_cache, tanh_cache = [], []
        x = s.unsqueeze(1)
        for l, (b_l, W_l) in enumerate(layers):
            patches = self._extract_patches(x)
            patches_cache.append(patches)
            z = tc.matmul(W_l, patches) + b_l[None, :, None]
            tanh_cache.append(tc.tanh(z))
            if l < L - 1:
                x = tc.log(2.0 * tc.cosh(z))

        grads: list[tc.Tensor] = [None] * L
        delta = tanh_cache[-1]                           # ∂ ln ψ / ∂ z^{(L-1)}
        n_buf = N + 1 if self.open else N                # include the phantom site, then drop it
        for l in range(L - 1, -1, -1):
            b_l, W_l = layers[l]
            patches_l = patches_cache[l]
            c_in = patches_l.shape[1] // K_area
            grad_b = delta.sum(dim=2)
            grad_W = tc.bmm(delta, patches_l.transpose(1, 2)).reshape(M, -1)
            grads[l] = tc.cat([grad_b, grad_W], dim=1)
            if l > 0:
                d_patches = tc.matmul(W_l.T, delta).reshape(M, c_in, K_area * N)
                idx = self._patch_idx.reshape(1, 1, K_area * N).expand(M, c_in, K_area * N)
                d_x = tc.zeros((M, c_in, n_buf), dtype=self.dtype, device=self.device)
                d_x.scatter_add_(dim=2, index=idx, src=d_patches)
                delta = tanh_cache[l - 1] * d_x[:, :, :N]

        grad_a = s if self.open else s.sum(dim=1, keepdim=True)
        return tc.cat([grad_a] + grads, dim=1).reshape(*m, self.Np)


def random_θ(N: int, α: int | Sequence[int], device: str, *, Lx: int | None = None, Ly: int | None = None,
             kernel_size: int | tuple[int, int] | None = None, boundary: str | None = None) -> tc.Tensor:
    Np = num_cnn_params(N, α, Lx=Lx, Ly=Ly, kernel_size=kernel_size, boundary=boundary)
    return (tc.randn(Np, dtype=tc.float64, device=device)
            + 1j * tc.randn(Np, dtype=tc.float64, device=device)) * 1.0e-3


def random_θ_jq(Q: int, N: int, α: int | Sequence[int], device: str, *, Lx: int | None = None,
                Ly: int | None = None, kernel_size: int | tuple[int, int] | None = None,
                boundary: str | None = None) -> tc.Tensor:
    Np = num_cnn_params(N, α, Lx=Lx, Ly=Ly, kernel_size=kernel_size, boundary=boundary)
    return (tc.randn((Np, Q), dtype=tc.float64, device=device)
            + 1j * tc.randn((Np, Q), dtype=tc.float64, device=device)) * 1.0e-3