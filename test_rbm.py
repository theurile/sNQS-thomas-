# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-07 15:37:47
# @Last Modified by:   dzwang
# @Last Modified time: 2026-04-20 15:58:21
import torch as tc
from rbm import RBM, random_θ, random_θ_jq
from sampler import random_samples
from exact import enumerate_spin_states
 
 
# * Test parameters
device = "cuda" if tc.cuda.is_available() else "cpu"
## Parameters to generate RBM instance
N = tc.randint(low=1, high=10, size=(1,))
α = tc.randint(low=1, high=5, size=(1,))
## Parameters to generate Samples
M = tc.randint(low=1, high=1000, size=(1,)) * 6  # must be a multiple of 6 for testing lnPsi shape using MC


def test_random_θ() -> None:
    θ = random_θ(N, α, device)
    assert θ.ndim == 1
    assert θ.shape[0] == N + α*N + N*α*N
    assert θ.dtype == tc.complex128
    assert θ.device.type == device


def test_random_θ_jq() -> None:
    Q = tc.randint(low=2, high=10, size=(1,))
    θ_jq = random_θ_jq(Q, N, α, device)
    assert θ_jq.ndim == 2
    assert θ_jq.shape[1] == Q
    assert θ_jq.shape[0] == N + α*N + N*α*N
    assert θ_jq.dtype == tc.complex128
    assert θ_jq.device.type == device
    

def test_rbm_parameters_shape() -> None:
    θ = random_θ(N, α, device)
    nqs = RBM(θ, N, α)
    assert nqs.a_n.ndim == 1 and nqs.a_n.shape[0] == N
    assert nqs.b_αn.ndim == 1 and nqs.b_αn.shape[0] == α*N
    assert nqs.W_nαn.ndim == 2 and nqs.W_nαn.shape == (N, α*N)


def test_rbm_lnPsi() -> None:
    θ = random_θ(N, α, device)
    nqs = RBM(θ, N, α)
    s_mn = random_samples(M, N, device).reshape(2, 3, -1, N)
    lnPsi_m = nqs.lnPsi(s_mn)
    assert lnPsi_m.shape == (2, 3, M//6)
    # benchmark with einsum
    a_n, b_αn, W_nαn = nqs.a_n, nqs.b_αn, nqs.W_nαn
    *m, n = s_mn.shape
    s_mn = s_mn.reshape(-1, n)
    part1 = tc.einsum("n,mn->m", a_n, s_mn)
    part2 = tc.sum(tc.log(2*tc.cosh(b_αn + s_mn @ W_nαn)), dim=1)
    lnPsi_einsum_m = (part1 + part2).reshape(*m)
    assert tc.allclose(lnPsi_m, lnPsi_einsum_m)


def test_rbm_d_lnPsi() -> None:
    θ = random_θ(N, α, device)
    Np = θ.numel()
    nqs = RBM(θ, N, α)
    s_mn = random_samples(M, N, device)
    d_lnPsi_mj = nqs.d_lnPsi(s_mn)
    assert d_lnPsi_mj.shape == (M, Np)
    # benchmark with einsum
    b_αn, W_nαn = nqs.b_αn, nqs.W_nαn
    *m, n = s_mn.shape
    s_mn = s_mn.reshape(-1, n)
    grad_a_mn = s_mn
    grad_b_mαn = tc.tanh(b_αn + s_mn @ W_nαn)
    grad_W_mnαn = tc.einsum("mn,mj->mnj", s_mn, grad_b_mαn)
    grad_θ_mj = tc.cat([grad_a_mn, grad_b_mαn, grad_W_mnαn.reshape(s_mn.shape[0], -1)], dim=1)
    d_lnPsi_mj_einsum = grad_θ_mj.reshape(*m, -1)
    assert tc.allclose(d_lnPsi_mj, d_lnPsi_mj_einsum)
    # benchmark with aotugrad
    θ = nqs.θ.clone().requires_grad_(True)
    nqs = RBM(θ, N, α)
    lnPsi = nqs.lnPsi(s_mn)
    auto_d_lnPsi_mj = tc.empty((M, Np), dtype=tc.complex128, device=device)
    unit = tc.ones((), dtype=tc.complex128, device=device)
    for i in range(lnPsi.shape[0]):
        grad_θ, = tc.autograd.grad(lnPsi[i], θ, grad_outputs=unit, retain_graph=True)
        auto_d_lnPsi_mj[i,:] = grad_θ
    assert tc.allclose(d_lnPsi_mj, auto_d_lnPsi_mj.conj())


def test_rbm_probability() -> None:
    N = 4
    α = 2
    θ = random_θ(N, α, device)
    nqs = RBM(θ, N, α)
    states = enumerate_spin_states(N, device=device)
    prob = nqs.probability(states)

    # Benchmark the normalized exact probability with a direct formula.
    lnpsi = nqs.lnPsi(states)
    weight = tc.exp(2.0 * lnpsi.real)
    expected = weight / weight.sum()

    assert prob.shape == (2**N,)
    assert prob.dtype == tc.float64
    assert tc.allclose(prob.sum(), tc.tensor(1.0, dtype=tc.float64, device=device))
    assert tc.all(prob >= 0.0)
    assert tc.allclose(prob, expected)
    


    
