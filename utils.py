# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-09 17:28:58
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-27 14:05:59
from __future__ import annotations

import math
import numpy as np
import torch as tc
from tqdm import tqdm
from sampler import random_samples
from vmc import VMC

from rbm import RBM, random_θ

# utils.py
import torch as tc

def get_g_qt(t:float|np.ndarray, Q:int, device="cpu", basis_type="simple", model=None, config=None) -> tc.Tensor:
    if basis_type == "simple":
        return _simple_polynomial(t, Q=Q, device=device)
    elif basis_type == "fourier":
        return _fourier_mode(t, Q=Q, model=model, config=config, device=device, omega_init='energy_spread')
    elif basis_type == "fourier_linear":
        return _fourier_mode_linear(t, Q=Q, model=model, config=config, device=device, omega_init='energy_spread')


def _simple_polynomial(t, Q, device="cpu") -> tc.Tensor:
    """
    time basis function should be a column vector
    .. math:: 
        ._    _.
        |g_1(t)|
        |g_2(t)|
        |  ... |
        |g_Q(t)|
        `-    -`
    time points should be a row vector
    .. math::
        ._                _.
        |t_1, t_2, ..., t_K|
        `-                -`
        -> time propagating
    """
    dtype = tc.complex128
    t = tc.as_tensor(t, dtype=dtype, device=device)
    qs = tc.arange(Q, device=device)
    if t.ndim == 0:
        g_qt = tc.empty(Q, dtype=dtype, device=device)
        g_qt[0] = 1.0 + 0.0j
        if Q > 1:
            g_qt[1:] = t ** qs[1:]
        return g_qt, None
    elif t.ndim == 1:
        Nt = t.numel()
        g_qt = tc.empty((Q, Nt), dtype=dtype, device=device)
        g_qt[0, :] = 1.0 + 0.0j
        if Q > 1:
            g_qt[1:, :] = t[None, :] ** qs[1:, None]
        return g_qt, None

def _time_rescaling(t, t0, t1):
    rescaled = ((t - t[0]) * (t1-t0) / (t[-1] - t[0])) + t0
    return rescaled

def _fourier_mode(t, Q, model, config, device="cpu", omega_init='energy_linspace'):
    dtype = tc.complex128
    t = tc.as_tensor(t, dtype=tc.float64, device=device)
    qs = tc.arange(1, Q, device='cpu')

    if omega_init == 'energy_linspace':
        theta_rand = random_θ(N=config.N, α=config.alpha, device=device)
        S_rand = random_samples(config.M, config.N, device=device)
        vmc_min = VMC(theta_rand, config.Lx, config.Ly, config.alpha, model=model)
        print("VMC model:", vmc_min.model.J, vmc_min.model.hx, vmc_min.model.hz)
        _, S_min = vmc_min.train(
            S_rand,
            config.batch,
            steps=config.vmc_steps,
            lr=config.vmc_lr,
            log_interval=config.vmc_steps,
        )

        epsilon_min, _, _ = vmc_min.energy_Sx_Sz(S_min, model.bonds(config.Lx, config.Ly))

        theta_rand = random_θ(N=config.N, α=config.alpha, device=device)
        model = TIM(-config.J, -config.hx, -config.hz)
        vmc_max = VMC(theta_rand, config.Lx, config.Ly, config.alpha, model=model)
        _, S_max = vmc_max.train(
            S_rand,
            config.batch,
            steps=config.vmc_steps,
            lr=config.vmc_lr,
            log_interval=config.vmc_steps,
        )

        epsilon_max, _, _ = vmc_max.energy_Sx_Sz(S_max, model.bonds(config.Lx, config.Ly))
        epsilon_max = -epsilon_max # negating to get the max energy
        step = (epsilon_max - epsilon_min) / (qs.shape[0] - 1)
        omega_q = epsilon_min + (step * (qs - 1.0))
        omega_q = tc.as_tensor(omega_q, dtype=tc.float64, device=t.device)
    elif omega_init == 'energy_spread':
        D0 = 2 * math.sqrt(config.hx ** 2 + config.hz ** 2)  # single-spin gap
        qs = tc.arange(1, (Q - 1) // 2 + 1, dtype=tc.float64)
        omega_q = tc.cat([-qs.flip(0), qs]) * D0  # ±Δ0, ±2Δ0, ... (Q odd)
    elif omega_init == 'integer_harmonics':
        omega_q = qs
        omega_q = tc.as_tensor(omega_q, dtype=tc.float64, device=t.device)
    elif omega_init == 'window_harmonics':
        epsilon_min = 2 * math.pi / (2.0)
        omega_q = epsilon_min * qs
        omega_q = tc.as_tensor(omega_q, dtype=tc.float64, device=t.device)
    elif omega_init == 'window_linspace':
        dw = 2 * math.pi / 1.0  # spacing set by the window
        n_pairs = max(1, int(0.3 / config.dt / 2.0 + 0.5))  # as many pairs as fit under the ceiling
        w_edge = (n_pairs - 0.5) * dw
        omega_q = tc.linspace(-w_edge, w_edge, 2 * n_pairs, dtype=tc.float64)  # ±dw/2, ±3dw/2, ...
    else:
        print("Unsupported omega initialization scheme.")

    if t.ndim == 0:
        g_qt = tc.empty(Q, dtype=dtype, device=device)
        g_qt[0] = 1.0 + 0.0j
        if Q > 1:
            g_qt[1:] = (tc.exp(-1j * t * omega_q))
        return g_qt, omega_q
    elif t.ndim == 1:
        Nt = t.numel()
        g_qt = tc.empty((Q, Nt) if t.ndim else Q, dtype=dtype, device=device)
        g_qt[0] = 1.0 + 0.0j
        if Q > 1:
            g_qt[1:] = tc.exp(-1j * omega_q[:, None] * t[None, :])
        return g_qt, omega_q

def _fourier_mode_linear(t, Q, model, config, device="cpu", omega_init='energy_linspace') -> tc.Tensor:
    dtype = tc.complex128
    t = tc.as_tensor(t, dtype=tc.float64, device=device)
    qs = tc.arange(2, Q, device='cpu')

    if omega_init == 'energy_linspace':
        theta_rand = random_θ(N=config.N, α=config.alpha, device=device)
        S_rand = random_samples(config.M, config.N, device=device)
        vmc_min = VMC(theta_rand, config.Lx, config.Ly, config.alpha, model=model)
        print("VMC model:", vmc_min.model.J, vmc_min.model.hx, vmc_min.model.hz)
        _, S_min = vmc_min.train(
            S_rand,
            config.batch,
            steps=config.vmc_steps,
            lr=config.vmc_lr,
            log_interval=config.vmc_steps,
        )

        epsilon_min, _, _ = vmc_min.energy_Sx_Sz(S_min, model.bonds(config.Lx, config.Ly))

        theta_rand = random_θ(N=config.N, α=config.alpha, device=device)
        model = TIM(-config.J, -config.hx, -config.hz)
        vmc_max = VMC(theta_rand, config.Lx, config.Ly, config.alpha, model=model)
        _, S_max = vmc_max.train(
            S_rand,
            config.batch,
            steps=config.vmc_steps,
            lr=config.vmc_lr,
            log_interval=config.vmc_steps,
        )

        epsilon_max, _, _ = vmc_max.energy_Sx_Sz(S_max, model.bonds(config.Lx, config.Ly))
        epsilon_max = -epsilon_max # negating to get the max energy
        step = (epsilon_max - epsilon_min) / (qs.shape[0] - 1)
        omega_q = epsilon_min + (step * (qs - 1.0))
        omega_q = tc.as_tensor(omega_q, dtype=tc.float64, device=t.device)
    elif omega_init == 'energy_spread':
        n_modes = Q - 2  # rows 0, 1 are constant and linear
        if n_modes % 2:
            raise ValueError(f"Q - 2 = {n_modes} must be even (use an even Q).")
        D0 = 2 * math.sqrt(config.hx ** 2 + config.hz ** 2)  # single-spin gap
        qs = tc.arange(1, n_modes // 2 + 1, dtype=tc.float64, device=t.device)
        omega_q = tc.cat([-qs.flip(0), qs]) * D0  # ±Δ0, ±2Δ0, ..., ±(n_modes/2)Δ0
    elif omega_init == 'integer_harmonics':
        omega_q = qs
        omega_q = tc.as_tensor(omega_q, dtype=tc.float64, device=t.device)
    elif omega_init == 'window_harmonics':
        epsilon_min = 2 * math.pi / (2.0)
        omega_q = epsilon_min * qs
        omega_q = tc.as_tensor(omega_q, dtype=tc.float64, device=t.device)
    elif omega_init == 'window_linspace':
        dw = 2 * math.pi / 1.0  # spacing set by the window
        n_pairs = max(1, int(0.3 / config.dt / 2.0 + 0.5))  # as many pairs as fit under the ceiling
        w_edge = (n_pairs - 0.5) * dw
        omega_q = tc.linspace(-w_edge, w_edge, 2 * n_pairs, dtype=tc.float64)  # ±dw/2, ±3dw/2, ...
    else:
        print("Unsupported omega initialization scheme.")

    if t.ndim == 0:
        g_qt = tc.empty(Q, dtype=dtype, device=device)
        g_qt[0] = 1.0 + 0.0j
        g_qt[1] = t + 0.0j
        if Q > 1:
            g_qt[2:] = (tc.exp(-1j * t * omega_q))
        return g_qt, omega_q
    elif t.ndim == 1:
        Nt = t.numel()
        g_qt = tc.empty((Q, Nt) if t.ndim else Q, dtype=dtype, device=device)
        g_qt[0] = 1.0 + 0.0j
        g_qt[1] = t + 0.0j
        if Q > 1:
            g_qt[2:] = tc.exp(-1j * omega_q[:, None] * t[None, :])
        return g_qt, omega_q

# here is some code I wrote to do the derivative matching between windows

def basis_derivatives(t0: float, Q: int, omega, device='cpu') -> tc.Tensor:

    temp = omega.tolist()
    temp.insert(0, 1e-20)
    omega = tc.tensor(temp)

    n_max = Q
    qs = tc.arange(Q, device=device)
    n = tc.arange(n_max, device=device).reshape(-1, 1)
    lam = (-1j * omega).to(tc.complex128)  # <- force complex128
    g_t0 = tc.exp(lam * t0).to(tc.complex128)
    D = ((lam.reshape(1, -1) ** n) * g_t0.reshape(1, -1)).to(tc.complex128)
    return D

def match_theta_at_boundary(theta_prev: tc.Tensor, D_prev: tc.Tensor, D_next: tc.Tensor) -> tc.Tensor:
    n_match, Q = D_prev.shape
    rhs = theta_prev @ D_prev.T
    return tc.linalg.lstsq(D_next, rhs.T).solution.T

def get_LPE_coeffs(order: int) -> np.ndarray:
    poly_desc = [1 / math.factorial(k) for k in range(order, -1, -1)]
    roots = np.roots(poly_desc)
    a_ms = -1.0 / roots
    return a_ms

def get_LPE_time_grid(
    t0:float,
    tK:float,
    dt:float,
    a_ms:tc.Tensor,
    *,
    device="cpu",
    dtype=tc.complex128,
    node_type:str="real",
    return_coeff_nodes:bool=False,
) -> tuple[tc.Tensor, tc.Tensor, list[int]] | tuple[tc.Tensor, tc.Tensor, list[int], tc.Tensor]:
    ### LPE coefficients
    a_ms = tc.as_tensor(a_ms, dtype=dtype, device=device).reshape(-1)
    ### LPE order
    s = a_ms.numel()
    ### number of physical time points
    Nt_phys = int(round((tK - t0) / dt))  
    if abs(t0 + Nt_phys * dt - tK) > 1e-12:
        raise ValueError("t_end must lie on the physical dt grid.")
    
    c_seq = tc.zeros(s + 1, dtype=dtype, device=device)
    c_seq[1:] = tc.cumsum(a_ms, dim=0)
    
    # check sum(a_ms)=1
    one = tc.tensor(1.0, dtype=dtype, device=device)
    if not tc.allclose(c_seq[-1], one, atol=1e-12, rtol=1e-12):
        raise ValueError("LPE coefficients must satisfy sum(a_ms)=1.")
    
    t0_c = tc.tensor(t0, dtype=dtype, device=device)
    dt_c = tc.tensor(dt, dtype=dtype, device=device)
    s_c = tc.tensor(float(s), dtype=dtype, device=device)

    coeff_t_nodes = [t0_c]  # original LPE coefficient-based (possibly complex) nodes
    real_t_nodes = [t0_c]   # uniform real nodes within each physical interval
    a_links = []  ## LPE coefficients linking intermediate time points
    phy_idx = [0]  ## indices of physical time points
    
    current_idx = 0
    for n in range(Nt_phys):
        t_base = t0_c + n * dt_c
        for m in range(1, s + 1):
            coeff_t_nodes.append(t_base + c_seq[m] * dt_c)
            real_t_nodes.append(t_base + tc.tensor(float(m), dtype=dtype, device=device) * dt_c / s_c)
            a_links.append(a_ms[m - 1])
        current_idx += s
        phy_idx.append(current_idx)

    coeff_t_nodes = tc.stack(coeff_t_nodes)  # (M,)
    real_t_nodes = tc.stack(real_t_nodes)    # (M,)

    print("----------- COEFFS -----------")
    print(coeff_t_nodes)
    print("----------- REALS -----------")
    print( real_t_nodes )

    a_links = tc.stack(a_links)   # (M-1,)
    if node_type == "real":
        t_nodes = real_t_nodes
    elif node_type == "coeff":
        t_nodes = coeff_t_nodes
    else:
        raise ValueError("t_node_type must be either 'real' or 'coeff'.")

    if return_coeff_nodes:
        return t_nodes, a_links, phy_idx, coeff_t_nodes
    return t_nodes, a_links, phy_idx


def time_function(Q:int, t0:float, tK:float, Δt:float, tW:float, *, device) -> tc.Tensor:
    N_times = int(round((tK - t0) / Δt)) + 1
    ts = t0 + Δt * tc.arange(N_times, device=device, dtype=tc.float64)
    xs = (2.0 * ts / tW) - 1.0
    xs = xs.clamp(-1.0, 1.0)
    θ = tc.arccos(xs)
    q = tc.arange(Q, device=device, dtype=tc.float64)
    g_qt = tc.cos(q[:, None] * θ[None, :])
    return ts, g_qt.to(tc.complex128)


def Ilocal(bar:RBM, ket:RBM, s_mn:tc.Tensor) -> tc.Tensor:
    return (ket.lnPsi(s_mn) - bar.lnPsi(s_mn)).exp()


def Hlocal(bar:RBM, ket:RBM, s_mn:tc.Tensor, model:dict):
    J, hx, hz = model["J"], model["hx"], model["hz"]
    bonds, flip_tn = model["bonds"], model["flip_tn"]
    Hz = hz * s_mn.sum(dim=1)
    Hzz = J * (s_mn[:, bonds[:,0]] * s_mn[:, bonds[:,1]]).sum(dim=1)
    Eloc_diag = (Hz + Hzz) * ...


def H2local(bar:RBM, ket:RBM, s_mn:tc.Tensor, model:dict):
    J, hx, hz = model["J"], model["hx"], model["hz"]

import math
import numpy as np
import torch as tc
from exact import (build_exact_spin_basis, diagonal_tim_energy,
                   tim_hamiltonian_action, tim_exact_observables)
from model import TIM

@tc.no_grad()
def exact_dynamics(Lx, Ly, J, hx, hz, t_max, dt_out=0.1, dt_int=0.01, order=12, device="cpu"):
    N = Lx * Ly
    model = TIM(J=J, hx=hx, hz=hz)
    basis = build_exact_spin_basis(N, device=device, dtype=tc.complex128)
    bonds = model.bonds(Lx, Ly).to(device=device, dtype=tc.long)
    Ediag = diagonal_tim_energy(basis.states, bonds, J=J, hz=hz)

    def H(v):
        return tim_hamiltonian_action(v, basis, bonds, J=J, hx=hx, hz=hz, Ediag_s=Ediag)

    def step(v, dt):
        out, term = v.clone(), v.clone()
        for k in range(1, order + 1):
            term = (-1j * dt / k) * H(term)
            out = out + term
        return out

    psi = tc.full((2 ** N,), 1.0 / math.sqrt(2 ** N), dtype=tc.complex128, device=device)
    n_sub = int(round(dt_out / dt_int))
    ts, E_t, Sx_t, Sz_t = [], [], [], []
    t = 0.0
    while t <= t_max + 1e-12:
        E, Sx, Sz = tim_exact_observables(psi, basis, bonds, J=J, hx=hx, hz=hz, Ediag_s=Ediag)
        ts.append(t); E_t.append(float(E.real) / N); Sx_t.append(float(Sx.real) / N); Sz_t.append(float(Sz.real) / N)
        for _ in range(n_sub):
            psi = step(psi, dt_int)
        psi = psi / psi.norm()
        t += dt_out
    return np.array(ts), np.array(Sx_t), np.array(Sz_t), np.array(E_t)

