# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-04-20 15:18:52
# @Last Modified by:   dzwang
# @Last Modified time: 2026-04-20 17:19:16
import itertools

import torch as tc

from exact import (
    build_exact_spin_basis,
    diagonal_tim_energy,
    enumerate_spin_states,
    rbm_state_vector,
    single_flip_indices,
    tim_exact_observables,
    tim_hamiltonian_action,
)
from model import TIM
from rbm import RBM, random_θ


device = "cuda" if tc.cuda.is_available() else "cpu"


def test_enumerate_spin_states_matches_lexicographic_order() -> None:
    states = enumerate_spin_states(3, device=device)
    expected = tc.tensor(
        list(itertools.product([-1, 1], repeat=3)),
        dtype=tc.complex128,
        device=device,
    )
    assert tc.equal(states, expected)


def test_single_flip_indices_matches_state_flips() -> None:
    basis = build_exact_spin_basis(4, device=device)
    flipped_states = basis.states[basis.flip1_idx]
    expected_states = basis.states[:, None, :].expand(-1, 4, -1).clone()
    site_idx = tc.arange(4, device=device)
    expected_states[:, site_idx, site_idx] *= -1
    assert tc.equal(flipped_states, expected_states)


def test_diagonal_tim_energy_matches_manual_formula() -> None:
    tim = TIM(J=-1., hx=-0.5, hz=-0.5)
    bonds = tim.bonds(Lx=3, Ly=1)
    states = enumerate_spin_states(3, device=device)
    energy = diagonal_tim_energy(states, bonds, J=tim.J, hz=tim.hz)
    manual = []
    for state in states.cpu():
        s0, s1, s2 = [int(x.real.item()) for x in state]
        manual.append(tim.J * (s0 * s1 + s1 * s2) + tim.hz * (s0 + s1 + s2))
    expected = tc.tensor(manual, dtype=tc.float64, device=device)
    assert tc.allclose(energy, expected)


def test_build_exact_spin_basis_shapes() -> None:
    basis = build_exact_spin_basis(5, device=device)
    assert basis.states.shape == (32, 5)
    assert basis.flip1_idx.shape == (32, 5)
    assert basis.num_states == 32
    assert basis.num_spins == 5


def _dense_tim_hamiltonian(basis, tim: TIM, bonds: tc.Tensor) -> tc.Tensor:
    nstates = basis.num_states
    H = tc.zeros((nstates, nstates), dtype=tc.complex128, device=device)
    row_idx = tc.arange(nstates, device=device)[:, None].expand(-1, basis.num_spins)
    diag_idx = tc.arange(nstates, device=device)
    Ediag = diagonal_tim_energy(basis.states, bonds, J=tim.J, hz=tim.hz)
    H[diag_idx, diag_idx] = Ediag.to(tc.complex128)
    H[row_idx, basis.flip1_idx] += tc.tensor(tim.hx, dtype=tc.complex128, device=device)
    return H


def _dense_total_sx(basis) -> tc.Tensor:
    nstates = basis.num_states
    X = tc.zeros((nstates, nstates), dtype=tc.complex128, device=device)
    row_idx = tc.arange(nstates, device=device)[:, None].expand(-1, basis.num_spins)
    X[row_idx, basis.flip1_idx] += 1.0 + 0.0j
    return X


def test_tim_hamiltonian_action_matches_dense_matrix() -> None:
    tim = TIM(J=-0.8, hx=0.25, hz=0.4)
    basis = build_exact_spin_basis(3, device=device)
    bonds = tim.bonds(Lx=3, Ly=1).to(device=device)
    rbm = RBM(random_θ(3, 2, device), N=3, α=2)
    psi_s = rbm_state_vector(rbm, basis)
    ## benchmark one
    H_dense = _dense_tim_hamiltonian(basis, tim, bonds)
    expected = H_dense @ psi_s
    ## need to test
    actual = tim_hamiltonian_action(psi_s, basis, bonds, J=tim.J, hx=tim.hx, hz=tim.hz)
    assert tc.allclose(actual, expected)


def test_tim_exact_observables_match_dense_expectation_values() -> None:
    tim = TIM(J=-1.1, hx=-0.3, hz=0.2)
    basis = build_exact_spin_basis(4, device=device)
    bonds = tim.bonds(Lx=4, Ly=1).to(device=device)
    rbm = RBM(random_θ(4, 2, device), N=4, α=2)
    psi_s = rbm_state_vector(rbm, basis)
    ## benchmark one
    H_dense = _dense_tim_hamiltonian(basis, tim, bonds)
    X_dense = _dense_total_sx(basis)
    Sz_diag = basis.states.real.sum(dim=1).to(dtype=tc.complex128)
    norm = psi_s.abs().square().sum()
    E_dense = (psi_s.conj() * (H_dense @ psi_s)).sum() / norm
    Sx_dense = (psi_s.conj() * (X_dense @ psi_s)).sum() / norm
    Sz_dense = (psi_s.conj() * (Sz_diag * psi_s)).sum() / norm
    ## need to test
    E_exact, Sx_exact, Sz_exact = tim_exact_observables(
        psi_s, basis, bonds, J=tim.J, hx=tim.hx, hz=tim.hz
    )
    assert tc.allclose(E_exact, E_dense)
    assert tc.allclose(Sx_exact, Sx_dense)
    assert tc.allclose(Sz_exact, Sz_dense)
