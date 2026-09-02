# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-07 16:14:58
# @Last Modified by:   dzwang
# @Last Modified time: 2026-02-03 19:12:17
import pytest
import itertools
import torch as tc
from rbm import RBM, random_θ
from sampler import random_samples, Metropolis
device = "cuda" if tc.cuda.is_available() else "cpu"


M = tc.randint(low=1, high=10000, size=(1,))
N = tc.randint(low=2, high=100, size=(1,))
     
 
def test_random_samples() -> None:
    s_mn = random_samples(M, N, device)
    assert s_mn.ndim == 2 and s_mn.shape == (M, N)
    assert s_mn.dtype == tc.complex128
    assert s_mn.device.type == device
    assert tc.all(tc.isin(s_mn.real, tc.tensor([-1, 1], device=device)))
    s_mn = s_mn.real.to(tc.float64)
    p_plus = (s_mn > 0).float().mean().item()
    assert abs(p_plus - 0.5) < 0.05


# def test_Metropolis() -> None:
N, α = 4, 1
Np = N + N*α + N*α*N
real_part = tc.linspace(start=0.1, end=0.9, steps=Np)
imag_part = tc.linspace(start=-0.9, end=-0.1, steps=Np)
θ = (0.01 * real_part + 1.j * imag_part).to(device=device, dtype=tc.complex128)
ψ = RBM(θ, N, α)
# exact probability distribution
all_states = tc.tensor(list(itertools.product([-1, 1], repeat=N)), dtype=tc.complex128, device=device)
print(all_states.real)
lnPsi_all = ψ.lnPsi(all_states)
P_all = (lnPsi_all.exp().abs() ** 2).real
P_all /= P_all.sum()
print(P_all)
# Metropolis probability distribution
M = 1000
s_mn = random_samples(M, N, device)
s_mn, acc_rate = Metropolis(s_mn, ψ, sweep=100*N, ret_rate=True)
print(acc_rate)
counts = {}
for row in s_mn.cpu().numpy():
    key = tuple(int(x.real) for x in row)
    counts[key] = counts.get(key, 0) + 1
emp_prob = {k: v / M for k, v in counts.items()}
print(emp_prob)
# benchmark
for state, p_exact in zip(all_states.cpu().numpy(), P_all.cpu().numpy()):
    key = tuple(int(x.real) for x in state)
    p_emp = emp_prob.get(key, 0.0)
    assert abs(p_emp - p_exact) < 1./M + 0.02















