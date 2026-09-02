# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-06 18:53:53
# @Last Modified by:   dzwang
# @Last Modified time: 2026-04-20 16:32:15
import torch as tc


__all__ = ["RBM", "random_θ", "random_θ_jq"]


class RBM():
    def __init__(self, θ:tc.Tensor, N:int, α:int) -> None:
        assert θ.ndim == 1 and θ.numel() == N + α*N + N*α*N
        assert θ.is_complex()
        self.θ = θ
        self.N = N
        self.α = α
        
        self.device = θ.device
        self.dtype = θ.dtype
    
    @property
    def a_n(self) -> tc.Tensor:
        return self.θ[:self.N].reshape(-1).contiguous()
    
    @property
    def b_αn(self) -> tc.Tensor:
        return self.θ[self.N: self.N + self.α*self.N].reshape(-1).contiguous()

    @property
    def W_nαn(self) -> tc.Tensor:
        return self.θ[self.N + self.α*self.N:].reshape(self.N, self.α*self.N).contiguous()

    def lnPsi(self, s_mn:tc.Tensor) -> tc.Tensor:
        a_n, b_αn, W_nαn = self.a_n, self.b_αn, self.W_nαn
        *m, n = s_mn.shape
        s_mn = s_mn.reshape(-1, n)
        part1 = s_mn @ a_n
        part2 = tc.sum(tc.log(2*tc.cosh(b_αn + s_mn @ W_nαn)), dim=1)
        return (part1 + part2).reshape(*m)
    
    def probability(self, s_mn:tc.Tensor) -> tc.Tensor:
        logw = 2.0 * self.lnPsi(s_mn).real.to(dtype=tc.float64)
        flat_logw = logw.reshape(-1)
        if flat_logw.numel() == 0:
            raise ValueError("Input states must be non-empty.")
        log_norm = tc.logsumexp(flat_logw, dim=0)
        return tc.exp(flat_logw - log_norm).reshape(logw.shape)
    
    def d_lnPsi(self, s_mn:tc.Tensor) -> tc.Tensor:
        b_αn, W_nαn = self.b_αn, self.W_nαn
        *m, n = s_mn.shape
        s_mn = s_mn.reshape(-1, n)
        grad_a_mn = s_mn
        grad_b_mαn = tc.tanh(b_αn + s_mn @ W_nαn)
        grad_W_mnαn = s_mn.unsqueeze(2) * grad_b_mαn.unsqueeze(1)
        grad_θ_mj = tc.cat([grad_a_mn, grad_b_mαn, grad_W_mnαn.reshape(s_mn.shape[0], -1)], dim=1)
        return grad_θ_mj.reshape(*m, -1)
    

def random_θ(N:int, α:int, device:str) -> tc.Tensor:
    Np = N + α*N + N*α*N
    real_part = tc.randn(Np, dtype=tc.float64, device=device)
    imag_part = tc.randn(Np, dtype=tc.float64, device=device)
    θ = (real_part + 1j*imag_part) * 1.e-3
    return θ


def random_θ_jq(Q:int, N:int, α:int, device:str) -> tc.Tensor:
    Np = N + α*N + N*α*N
    real_part = tc.randn((Np, Q), dtype=tc.float64, device=device)
    imag_part = tc.randn((Np, Q), dtype=tc.float64, device=device)
    θ = (real_part + 1j*imag_part) * 1.e-3
    return θ
