# -*- coding: utf-8 -*-
# @Author: trleung
# @Date:   2026-08-17 19:23:43
# @Last Modified by:   trleung
# @Last Modified time: 2026-11-14 09:42:00

import torch as tc
import torch.nn.functional as F

__all__ = ["CNN", "random_θ", "random_θ_jq"]

class CNN():
    def __init__(self, θ: tc.Tensor, Lx: int, Ly: int,  α,
                 kernel_size: tuple = (3, 1), open_boundary: bool=True) -> None:
        assert θ.is_complex()
        self.θ = θ
        self.Lx = Lx
        self.Ly = Ly
        self.N = Lx * Ly
        self.α = [α] if isinstance(α, int) else list(α)

        cap = (lambda L: 2 * L - 1) if open_boundary else (lambda L: L)
        self.Kx, self.Ky = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.Kx, self.Ky = min(self.Kx, cap(self.Lx)), min(self.Ky, cap(self.Ly))
        self.n_vis = self.N if open_boundary else 1  # one visible bias per site (open) or shared (periodic)

        self.open = open_boundary

        self.device = θ.device
        self.dtype = θ.dtype

    @property
    def a_n(self):
        return self.θ[:self.n_vis]

    @property
    def layers(self):
        channels = [1] + self.α
        layers = []
        off = self.n_vis  # running position in θ
        for c in range(len(channels) - 1):  # one iteration per layer
            c_in, c_out = channels[c], channels[c + 1]
            b = self.θ[off: off + c_out]
            off += c_out  # past biases
            n_W = c_out * c_in * self.Kx * self.Ky
            W = self.θ[off: off + n_W].reshape(c_out, c_in, self.Kx, self.Ky)
            off += n_W  # past filters
            layers.append((b, W))
        assert off == self.θ.numel(), f"θ has {self.θ.numel()} entries, layers used {off}"
        return layers

    # handles 1d chains and 2d lattices
    def lnPsi(self, s_mn: tc.Tensor) -> tc.Tensor:
        *m, n = s_mn.shape
        s = s_mn.reshape(-1, n).real.to(tc.float64)  # (M, N)
        M = s.shape[0]
        Kx, Ky, Lx, Ly = self.Kx, self.Ky, self.Lx, self.Ly

        # boundary handling
        if self.open:
            pad_amounts, mode = (Ky // 2, Ky - 1 - Ky // 2, Kx // 2, Kx - 1 - Kx // 2), "constant"
        else:  # wrap around
            pad_amounts, mode = (0, Ky - 1, 0, Kx - 1), "circular"

        # layer input as real and imaginary parts
        xr_mcxy = s.reshape(M, 1, Lx, Ly)
        xi_mcxy = tc.zeros_like(xr_mcxy)

        for b_c, W_cckk in self.layers:  # W: (c_out, c_in, Kx, Ky)
            pr_mcxy = F.pad(xr_mcxy, pad_amounts, mode=mode)
            pi_mcxy = F.pad(xi_mcxy, pad_amounts, mode=mode)
            zr = F.conv2d(pr_mcxy, W_cckk.real) - F.conv2d(pi_mcxy, W_cckk.imag) + b_c.real.view(1, -1, 1, 1)
            zi = F.conv2d(pr_mcxy, W_cckk.imag) + F.conv2d(pi_mcxy, W_cckk.real) + b_c.imag.view(1, -1, 1, 1)
            z_mcxy = tc.complex(zr, zi)
            x_mcxy = log2cosh(z_mcxy)
            xr_mcxy, xi_mcxy = x_mcxy.real, x_mcxy.imag

        vis = s.to(tc.complex128) @ self.a_n if self.open else s.sum(dim=1).to(tc.complex128) * self.a_n[0]
        ln_psi = vis + tc.complex(xr_mcxy, xi_mcxy).sum(dim=(1, 2, 3))
        return ln_psi.reshape(*m)

    def d_lnPsi(self, s_mn: tc.Tensor) -> tc.Tensor:
        *m, n = s_mn.shape
        s_mn = s_mn.reshape(-1, n).real.to(tc.float64)  # (M, N)
        M = s_mn.shape[0]
        Kx, Ky, Lx, Ly = self.Kx, self.Ky, self.Lx, self.Ly
        if self.open:
            pad_amounts, mode = (Ky // 2, Ky - 1 - Ky // 2, Kx // 2, Kx - 1 - Kx // 2), "constant"
        else:
            pad_amounts, mode = (0, Ky - 1, 0, Kx - 1), "circular"
        layers = self.layers

        # visible biases: same as the RBM
        grad_a_mn = s_mn if self.open else s_mn.sum(dim=1, keepdim=True)

        # forward pass, caching each layer's padded input (real, imag) and z
        xr_mcxy = s_mn.reshape(M, 1, Lx, Ly)
        xi_mcxy = tc.zeros_like(xr_mcxy)
        cache = []
        for b_c, W_cckk in layers:  # W: (c_out, c_in, Kx, Ky)
            pr_mcxy = F.pad(xr_mcxy, pad_amounts, mode=mode)
            pi_mcxy = F.pad(xi_mcxy, pad_amounts, mode=mode)
            zr = F.conv2d(pr_mcxy, W_cckk.real) - F.conv2d(pi_mcxy, W_cckk.imag) + b_c.real.view(1, -1, 1, 1)
            zi = F.conv2d(pr_mcxy, W_cckk.imag) + F.conv2d(pi_mcxy, W_cckk.real) + b_c.imag.view(1, -1, 1, 1)
            z_mcxy = tc.complex(zr, zi)
            cache.append([pr_mcxy, pi_mcxy, z_mcxy])
            x_mcxy = log2cosh(z_mcxy)
            xr_mcxy, xi_mcxy = x_mcxy.real, x_mcxy.imag

        # backward pass: δ = ∂lnψ/∂z, starting from the last layer
        grad_layers = []
        _, _, z_last = cache[-1]
        del_mcxy = tc.tanh(z_last)
        for l in reversed(range(len(layers))):
            pr_mcxy, pi_mcxy, z_mcxy = cache[l]
            b_c, W_cckk = layers[l]

            grad_b_mc = del_mcxy.sum(dim=(2, 3))  # (M, c_out)
            patch_mkn = tc.complex(F.unfold(pr_mcxy, (Kx, Ky)), F.unfold(pi_mcxy, (Kx, Ky)))  # (M, c_in*Kx*Ky, Lx*Ly)
            grad_W_mck = del_mcxy.reshape(M, del_mcxy.shape[1], -1) @ patch_mkn.transpose(1, 2)  # (M, c_out, c_in*Kx*Ky)
            grad_layers.insert(0, [grad_b_mc, grad_W_mck.reshape(M, -1)])

            if l > 0:
                del_i, del_r = del_mcxy.imag, del_mcxy.real
                gr = F.conv_transpose2d(del_r, W_cckk.real) - F.conv_transpose2d(del_i, W_cckk.imag)
                gi = F.conv_transpose2d(del_r, W_cckk.imag) + F.conv_transpose2d(del_i, W_cckk.real)
                g_mcxy = tc.complex(gr, gi)

                # unpadding back
                if self.open:
                    g_mcxy = g_mcxy[..., Kx // 2: Kx // 2 + Lx, Ky // 2: Ky // 2 + Ly]
                else:
                    d = g_mcxy[..., :Lx, :].clone()
                    d[..., :Kx - 1, :] += g_mcxy[..., Lx:, :]
                    g_mcxy = d[..., :Ly].clone()
                    g_mcxy[..., :Ky - 1] += d[..., Ly:]
                del_mcxy = tc.tanh(cache[l - 1][2]) * g_mcxy

        grad_θ_mj = tc.cat([grad_a_mn.to(tc.complex128)] + [g for pair in grad_layers for g in pair], dim=1)
        return grad_θ_mj.reshape(*m, -1)

# needed to avoid overflow
def log2cosh(z):
    sign = tc.where(z.real >= 0, 1.0, -1.0).to(z.real.dtype)
    zz = sign * z  # Re(zz) >= 0
    return zz + tc.log(1 + tc.exp(-2 * zz))  # = ln(e^z + e^-z)

def random_θ(N: int, α: list, Kx: int, Ky: int, device: str,
             vis_scale: float = 1.e-3, last_scale: float = 1.e-2) -> tc.Tensor:
    α = [α] if isinstance(α, int) else list(α)
    ch = [1] + α

    def randc(n, scale):
        return (tc.randn(n, dtype=tc.float64, device=device)
                + 1j * tc.randn(n, dtype=tc.float64, device=device)) * scale

    parts = [randc(N, vis_scale)]                                   # visible biases a_n
    for l in range(len(α)):
        c_in, c_out = ch[l], ch[l + 1]
        fan_in = c_in * Kx * Ky
        w_scale = last_scale if l == len(α) - 1 else 1. / (2 * fan_in) ** 0.5
        parts.append(randc(c_out, 1.e-3))                           # biases b
        parts.append(randc(c_out * fan_in, w_scale))                # filters W
    return tc.cat(parts)


def random_θ_jq(Q: int, N: int, α: list, Kx: int, Ky: int, device: str) -> tc.Tensor:
    θ0 = random_θ(N, α, Kx, Ky, device)                             # constant column: layer-scaled
    θ = (tc.randn((θ0.numel(), Q), dtype=tc.float64, device=device)
         + 1j * tc.randn((θ0.numel(), Q), dtype=tc.float64, device=device)) * 1.e-3
    θ[:, 0] = θ0                                                    # time-dependent columns stay at 1e-3
    return θ

