# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-04-20 15:18:52
# @Last Modified by:   dzwang
# @Last Modified time: 2026-04-20 16:37:58

import torch as tc
from dataclasses import dataclass

import torch as tc

from rbm import RBM


__all__ = [
    "ExactSpinBasis",
    "enumerate_spin_states",
    "single_flip_indices",
    "diagonal_tim_energy",
    "build_exact_spin_basis",
    "rbm_state_vector",
    "tim_hamiltonian_action",
    "tim_local_energy",
    "tim_total_sx_local",
    "tim_total_sz_local",
    "tim_exact_observables",
]


@dataclass(frozen=True)
class ExactSpinBasis:
    states: tc.Tensor
    flip1_idx: tc.Tensor

    @property
    def num_states(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_spins(self) -> int:
        return int(self.states.shape[1])


def _bit_shifts(N: int, device: str | tc.device) -> tc.Tensor:
    if N < 1:
        raise ValueError("N must be a positive integer.")
    return tc.arange(N - 1, -1, -1, device=device, dtype=tc.long)


def enumerate_spin_states(
    N: int,
    *,
    device: str | tc.device = "cpu",
    dtype: tc.dtype = tc.complex128,
) -> tc.Tensor:
    """
    Enumerate the full spin basis in lexicographic order.

    Parameters
    ----------
    N : int
        Number of spins.
    device : str | tc.device, optional
        Target device.
    dtype : tc.dtype, optional
        Output tensor dtype.

    Returns
    -------
    tc.Tensor
        Tensor of shape ``(2**N, N)`` with entries in ``{-1, +1}``.
    """
    shifts = _bit_shifts(N, device)
    nstates = 1 << N
    state_idx = tc.arange(nstates, device=device, dtype=tc.long)
    bits = ((state_idx[:, None] >> shifts[None, :]) & 1).to(tc.int8)
    return (2 * bits - 1).to(dtype=dtype)


def single_flip_indices(
    N: int,
    *,
    device: str | tc.device = "cpu",
) -> tc.Tensor:
    """
    Build the state index table for all single-spin flips.

    Parameters
    ----------
    N : int
        Number of spins.
    device : str | tc.device, optional
        Target device.

    Returns
    -------
    tc.Tensor
        Integer tensor of shape ``(2**N, N)``.
    """
    shifts = _bit_shifts(N, device)
    nstates = 1 << N
    state_idx = tc.arange(nstates, device=device, dtype=tc.long)
    masks = 1 << shifts
    return state_idx[:, None] ^ masks[None, :]


def diagonal_tim_energy(
    states: tc.Tensor,
    bonds: tc.Tensor,
    J: float,
    hz: float,
) -> tc.Tensor:
    """
    Evaluate the diagonal TIM energy on a batch of spin configurations.

    Parameters
    ----------
    states : tc.Tensor
        Spin configurations with shape ``(M, N)``.
    bonds : tc.Tensor
        Bond list with shape ``(Nb, 2)``.
    J : float
        ZZ coupling strength.
    hz : float
        Longitudinal field strength.

    Returns
    -------
    tc.Tensor
        Real tensor with shape ``(M,)``.
    """
    spins = states.real.to(dtype=tc.float64)
    energy = hz * spins.sum(dim=1)

    if bonds.numel() == 0:
        return energy

    bonds = bonds.to(device=states.device, dtype=tc.long).reshape(-1, 2)
    zz = spins[:, bonds[:, 0]] * spins[:, bonds[:, 1]]
    return energy + J * zz.sum(dim=1)


def build_exact_spin_basis(
    N: int,
    *,
    device: str | tc.device = "cpu",
    dtype: tc.dtype = tc.complex128,
) -> ExactSpinBasis:
    """
    Construct the reusable full-summation basis object.

    Parameters
    ----------
    N : int
        Number of spins.
    device : str | tc.device, optional
        Target device.
    dtype : tc.dtype, optional
        State tensor dtype.

    Returns
    -------
    ExactSpinBasis
        Basis states and single-flip lookup table.
    """
    states = enumerate_spin_states(N, device=device, dtype=dtype)
    flip1_idx = single_flip_indices(N, device=device)
    return ExactSpinBasis(states=states, flip1_idx=flip1_idx)


def _as_state_vector(psi_s: tc.Tensor, basis: ExactSpinBasis) -> tc.Tensor:
    psi_s = tc.as_tensor(psi_s)
    if psi_s.ndim != 1 or psi_s.shape[0] != basis.num_states:
        raise ValueError("psi_s must have shape (2**N,).")
    if psi_s.device != basis.states.device:
        raise ValueError("psi_s and basis must live on the same device.")
    if not psi_s.is_complex():
        psi_s = psi_s.to(dtype=tc.complex128)
    return psi_s


def _safe_local_ratio(numerator_s: tc.Tensor, denominator_s: tc.Tensor, cutoff: float = 1e-30) -> tc.Tensor:
    out = tc.zeros_like(numerator_s)
    mask = denominator_s.abs() > cutoff
    out[mask] = numerator_s[mask] / denominator_s[mask]
    return out


def rbm_state_vector(psi: RBM, basis: ExactSpinBasis | tc.Tensor) -> tc.Tensor:
    """
    Evaluate an RBM wavefunction on the full basis.

    Parameters
    ----------
    psi : RBM
        RBM wavefunction.
    basis : ExactSpinBasis | tc.Tensor
        Full basis object or explicit state tensor.

    Returns
    -------
    tc.Tensor
        Complex state vector with shape ``(2**N,)``.
    """
    states = basis.states if isinstance(basis, ExactSpinBasis) else basis
    return tc.exp(psi.lnPsi(states))


def tim_hamiltonian_action(
    psi_s: tc.Tensor,
    basis: ExactSpinBasis,
    bonds: tc.Tensor,
    J: float,
    hx: float,
    hz: float,
    *,
    Ediag_s: tc.Tensor | None = None,
) -> tc.Tensor:
    """
    Apply the transverse-field Ising Hamiltonian on a full state vector.

    Parameters
    ----------
    psi_s : tc.Tensor
        State vector with shape ``(2**N,)``.
    basis : ExactSpinBasis
        Full basis object.
    bonds : tc.Tensor
        Bond list with shape ``(Nb, 2)``.
    J : float
        ZZ coupling strength.
    hx : float
        Transverse field strength.
    hz : float
        Longitudinal field strength.
    Ediag_s : tc.Tensor | None, optional
        Cached diagonal energy.

    Returns
    -------
    tc.Tensor
        Complex tensor with shape ``(2**N,)``.
    """
    psi_s = _as_state_vector(psi_s, basis)
    if Ediag_s is None:
        Ediag_s = diagonal_tim_energy(basis.states, bonds, J=J, hz=hz)
    flip_sum_s = psi_s[basis.flip1_idx].sum(dim=1)
    return Ediag_s.to(dtype=psi_s.dtype) * psi_s + tc.as_tensor(hx, dtype=psi_s.dtype, device=psi_s.device) * flip_sum_s


def tim_local_energy(
    psi_s: tc.Tensor,
    basis: ExactSpinBasis,
    bonds: tc.Tensor,
    J: float,
    hx: float,
    hz: float,
    *,
    Ediag_s: tc.Tensor | None = None,
) -> tc.Tensor:
    """
    Compute the local energy on the full basis.

    Parameters
    ----------
    psi_s : tc.Tensor
        State vector with shape ``(2**N,)``.
    basis : ExactSpinBasis
        Full basis object.
    bonds : tc.Tensor
        Bond list with shape ``(Nb, 2)``.
    J : float
        ZZ coupling strength.
    hx : float
        Transverse field strength.
    hz : float
        Longitudinal field strength.
    Ediag_s : tc.Tensor | None, optional
        Cached diagonal energy.

    Returns
    -------
    tc.Tensor
        Complex tensor with shape ``(2**N,)``.
    """
    psi_s = _as_state_vector(psi_s, basis)
    Hpsi_s = tim_hamiltonian_action(psi_s, basis, bonds, J=J, hx=hx, hz=hz, Ediag_s=Ediag_s)
    return _safe_local_ratio(Hpsi_s, psi_s)


def tim_total_sx_local(psi_s: tc.Tensor, basis: ExactSpinBasis) -> tc.Tensor:
    """
    Compute the local estimator of the total ``sum_i sigma_i^x`` operator.

    Parameters
    ----------
    psi_s : tc.Tensor
        State vector with shape ``(2**N,)``.
    basis : ExactSpinBasis
        Full basis object.

    Returns
    -------
    tc.Tensor
        Complex tensor with shape ``(2**N,)``.
    """
    psi_s = _as_state_vector(psi_s, basis)
    flip_sum_s = psi_s[basis.flip1_idx].sum(dim=1)
    return _safe_local_ratio(flip_sum_s, psi_s)


def tim_total_sz_local(basis: ExactSpinBasis) -> tc.Tensor:
    """
    Compute the local value of the total ``sum_i sigma_i^z`` operator.

    Parameters
    ----------
    basis : ExactSpinBasis
        Full basis object.

    Returns
    -------
    tc.Tensor
        Real tensor with shape ``(2**N,)``.
    """
    return basis.states.real.to(dtype=tc.float64).sum(dim=1)


def tim_exact_observables(
    psi_s: tc.Tensor,
    basis: ExactSpinBasis,
    bonds: tc.Tensor,
    J: float,
    hx: float,
    hz: float,
    *,
    Ediag_s: tc.Tensor | None = None,
) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
    """
    Evaluate exact energy, total Sx, and total Sz by full summation.

    Parameters
    ----------
    psi_s : tc.Tensor
        State vector with shape ``(2**N,)``.
    basis : ExactSpinBasis
        Full basis object.
    bonds : tc.Tensor
        Bond list with shape ``(Nb, 2)``.
    J : float
        ZZ coupling strength.
    hx : float
        Transverse field strength.
    hz : float
        Longitudinal field strength.
    Ediag_s : tc.Tensor | None, optional
        Cached diagonal energy.

    Returns
    -------
    tuple[tc.Tensor, tc.Tensor, tc.Tensor]
        Exact expectation values ``(E, Sx, Sz)``.
    """
    psi_s = _as_state_vector(psi_s, basis)
    prob_s = psi_s.abs().square()
    prob_s = prob_s / prob_s.sum()
    Eloc_s = tim_local_energy(psi_s, basis, bonds, J=J, hx=hx, hz=hz, Ediag_s=Ediag_s)
    Sxloc_s = tim_total_sx_local(psi_s, basis)
    Szloc_s = tim_total_sz_local(basis).to(dtype=psi_s.dtype)
    E = (prob_s * Eloc_s).sum()
    Sx = (prob_s * Sxloc_s).sum()
    Sz = (prob_s * Szloc_s).sum()
    return E, Sx, Sz
