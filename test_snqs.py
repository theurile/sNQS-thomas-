# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-09 21:00:18
# @Last Modified by:   dzwang
# @Last Modified time: 2026-04-21 03:29:29
from utils import *
import torch as tc
import pytest
from exact import rbm_state_vector, tim_exact_observables, tim_hamiltonian_action
from model import TIM
from rbm import RBM, random_θ, random_θ_jq
from snqs import sNQS_rbm
device = "cuda" if tc.cuda.is_available() else "cpu"


# def test_chebyshev_recurrence() -> None:
#     Q = tc.randint(3, 10, (1,)).item()
#     t0, tK, tW, Δt = 0.3, 0.5, 2., 0.0025
#     t, g_qt = time_function(Q, t0, tK, Δt, tW, device=device)
#     x = (2.0 * t / tW) - 1.0
#     left = g_qt[2:]
#     right = 2.0 * x * g_qt[1:-1] - g_qt[:-2]
#     assert tc.allclose(left, right)


def test_sNQS_rbm_ψs() -> None:
    Δt = 0.01
    N = 10
    α = 5
    Q = 4
    t = np.array([2, 3])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    snqs = sNQS_rbm(θ_jq, g_qt, Lx=N, Ly=1, α=α, Δt=Δt, model= TIM(J=-1, hx=-0.3, hz=-0.3))
    ψs = snqs.ψs
    assert len(ψs) == g_qt.shape[1]
    for ψk in ψs:
        assert isinstance(ψk, RBM) and ψk.N == N and ψk.α == α and ψk.θ.shape == (N + α*N + N*α*N,)
    # change θ_jq should change ψS automatically
    snqs.θ_jq = tc.randn_like(θ_jq)
    ψS_ = snqs.ψs
    for ψk, ψk_ in zip(ψs, ψS_):
        assert not tc.allclose(ψk.θ, ψk_.θ)


def test_sNQS_rbm_exact_expectation_value() -> None:
    Δt = 0.01
    N = 4
    α = 2
    Q = 3
    t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    model = TIM(J=-1.0, hx=-0.3, hz=0.2)
    snqs = sNQS_rbm(
        θ_jq,
        g_qt,
        Lx=N,
        Ly=1,
        α=α,
        Δt=Δt,
        model=model,
        backend="exact",
    )
    ## need to test
    E_k, Sx_k, Sz_k = snqs.expectation_value(Ss=None, batch=1)
    ## benckmark one
    basis = snqs.exact_basis
    bonds = snqs.bonds
    Ediag_s = snqs.exact_Ediag
    expected_E, expected_Sx, expected_Sz = [], [], []
    for ψk in snqs.ψs:
        psi_s = rbm_state_vector(ψk, basis)
        E, Sx, Sz = tim_exact_observables(
            psi_s,
            basis,
            bonds,
            J=model.J,
            hx=model.hx,
            hz=model.hz,
            Ediag_s=Ediag_s,
        )
        expected_E.append(float(E.real))
        expected_Sx.append(float(Sx.real))
        expected_Sz.append(float(Sz.real))
        
    assert np.allclose(E_k, expected_E)
    assert np.allclose(Sx_k, expected_Sx)
    assert np.allclose(Sz_k, expected_Sz)


def test_sNQS_rbm_exact_backend_size_guard() -> None:
    Δt = 0.01
    N = 20
    α = 1
    Q = 2
    t = np.array([0.0, 0.1])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    snqs = sNQS_rbm(
        θ_jq,
        g_qt,
        Lx=N,
        Ly=1,
        α=α,
        Δt=Δt,
        model=TIM(J=-1.0, hx=-0.3, hz=0.2),
        backend="exact",
        allow_large_exact=False,
    )

    with pytest.raises(ValueError, match="max_exact_spins=16"):
        _ = snqs.exact_basis


def test_sNQS_rbm_exact_backend_size_guard_can_be_overridden() -> None:
    Δt = 0.01
    N = 17
    α = 1
    Q = 2
    t = np.array([0.0, 0.1])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    snqs = sNQS_rbm(
        θ_jq,
        g_qt,
        Lx=N,
        Ly=1,
        α=α,
        Δt=Δt,
        model=TIM(J=-1.0, hx=-0.3, hz=0.2),
        backend="exact",
        allow_large_exact=True,
    )

    basis = snqs.exact_basis
    assert basis.num_spins == N


def _safe_ratio(numerator_s: tc.Tensor, denominator_s: tc.Tensor, cutoff: float = 1e-30) -> tc.Tensor:
    out = tc.zeros_like(numerator_s)
    mask = denominator_s.abs() > cutoff
    out[mask] = numerator_s[mask] / denominator_s[mask]
    return out


def _dense_weighted_stats(ψk: RBM, states: tc.Tensor, prob_s: tc.Tensor, value_s: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
    O_sj = ψk.d_lnPsi(states).conj()
    mean_O = (prob_s[:, None] * O_sj).sum(dim=0)
    mean_OV = ((prob_s * value_s)[:, None] * O_sj).sum(dim=0)
    mean_value = (prob_s * value_s).sum()
    return mean_O, mean_OV, mean_value


def test_sNQS_rbm_exact_grad_matches_dense_formula() -> None:
    Δt = 0.02
    N = 3
    α = 1
    Q = 2
    t = np.array([0.0, 0.1, 0.2])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    ψini = RBM(random_θ(N, α, device), N=N, α=α)
    model = TIM(J=-1.0, hx=-0.25, hz=0.15)
    snqs = sNQS_rbm(
        θ_jq,
        g_qt,
        Lx=N,
        Ly=1,
        α=α,
        Δt=Δt,
        model=model,
        backend="exact",
        max_exact_spins=8,
    )
    ψs = snqs.ψs
    ## need to test
    grad_exact, loss_exact = snqs.grad_jq(ψini, ψs, Ss=None, batch=1)
    ## benchmark
    basis = snqs.exact_basis
    bonds = snqs.bonds
    Ediag_s = snqs.exact_Ediag
    psiinit_s = rbm_state_vector(ψini, basis)
    psi_vecs = [rbm_state_vector(ψk, basis) for ψk in ψs]

    grad_dense = tc.zeros_like(θ_jq)
    loss_dense = tc.tensor(1.0 + 0.0j)

    prob0_s = psi_vecs[0].abs().square()
    prob0_s = prob0_s / prob0_s.sum()
    r0_s = _safe_ratio(psiinit_s, psi_vecs[0])
    mean_O0, mean_O0r0, mean_r0 = _dense_weighted_stats(ψs[0], basis.states, prob0_s, r0_s)
    overlap = (psiinit_s.conj() * psi_vecs[0]).sum()
    norm_init = psiinit_s.abs().square().sum()
    G0 = (mean_O0r0 - mean_O0 * mean_r0) / mean_r0
    C0 = (overlap / norm_init) * mean_r0
    grad_dense += G0.reshape(-1, 1) @ g_qt[:, 0].reshape(1, -1)
    loss_dense *= C0

    for k in range(snqs.K):
        psi_k_s = psi_vecs[k]
        probk_s = psi_k_s.abs().square()
        probk_s = probk_s / probk_s.sum()

        if k > 0:
            H_prev = tim_hamiltonian_action(psi_vecs[k - 1], basis, bonds, J=model.J, hx=model.hx, hz=model.hz, Ediag_s=Ediag_s)
            H2_prev = tim_hamiltonian_action(H_prev, basis, bonds, J=model.J, hx=model.hx, hz=model.hz, Ediag_s=Ediag_s)
            Upsi_prev = psi_vecs[k - 1] - 1j * Δt * H_prev - 0.5 * (Δt ** 2) * H2_prev
            Uloc_prev = _safe_ratio(Upsi_prev, psi_k_s)
            mean_O, mean_OU, mean_U = _dense_weighted_stats(ψs[k], basis.states, probk_s, Uloc_prev)
            G_prev = (mean_OU - mean_O * mean_U) / mean_U
            grad_dense += G_prev.reshape(-1, 1) @ g_qt[:, k].reshape(1, -1)
            loss_dense *= mean_U

        if k < snqs.K - 1:
            H_next = tim_hamiltonian_action(psi_vecs[k + 1], basis, bonds, J=model.J, hx=model.hx, hz=model.hz, Ediag_s=Ediag_s)
            H2_next = tim_hamiltonian_action(H_next, basis, bonds, J=model.J, hx=model.hx, hz=model.hz, Ediag_s=Ediag_s)
            Udagpsi_next = psi_vecs[k + 1] + 1j * Δt * H_next - 0.5 * (Δt ** 2) * H2_next
            Uloc_next = _safe_ratio(Udagpsi_next, psi_k_s)
            mean_O, mean_OU, mean_U = _dense_weighted_stats(ψs[k], basis.states, probk_s, Uloc_next)
            G_next = (mean_OU - mean_O * mean_U) / mean_U
            grad_dense += G_next.reshape(-1, 1) @ g_qt[:, k].reshape(1, -1)
            loss_dense *= mean_U

    assert tc.allclose(grad_exact, grad_dense)
    assert np.isclose(loss_exact, float(np.abs(1.0 - loss_dense.item())), atol=1.0e-6, rtol=1.0e-6)


def test_sNQS_rbm_exact_train_smoke() -> None:
    Δt = 0.02
    N = 3
    α = 1
    Q = 2
    t = np.array([0.0, 0.1, 0.2])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    ψini = RBM(random_θ(N, α, device), N=N, α=α)
    snqs = sNQS_rbm(
        θ_jq,
        g_qt,
        Lx=N,
        Ly=1,
        α=α,
        Δt=Δt,
        model=TIM(J=-1.0, hx=-0.25, hz=0.15),
        backend="exact",
        max_exact_spins=8,
    )

    θ_out, Ss, losses, ψfinal = snqs.train(
        ψini,
        Sini=None,
        batch=1,
        steps=2,
        lr=1.0e-3,
        log_interval=1,
    )

    assert θ_out.shape == θ_jq.shape
    assert Ss is None
    assert len(losses) == 2
    assert np.isfinite(losses).all()
    assert isinstance(ψfinal, RBM)


def test_sNQS_rbm_exact_train_returns_time_losses_when_requested() -> None:
    Δt = 0.02
    N = 3
    α = 1
    Q = 2
    t = np.array([0.0, 0.1, 0.2])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    ψini = RBM(random_θ(N, α, device), N=N, α=α)
    snqs = sNQS_rbm(
        θ_jq,
        g_qt,
        Lx=N,
        Ly=1,
        α=α,
        Δt=Δt,
        model=TIM(J=-1.0, hx=-0.25, hz=0.15),
        backend="exact",
        max_exact_spins=8,
    )

    θ_out, Ss, losses, losses_by_time, ψfinal = snqs.train(
        ψini,
        Sini=None,
        batch=1,
        steps=2,
        lr=1.0e-3,
        log_interval=1,
        return_time_losses=True,
    )

    assert θ_out.shape == θ_jq.shape
    assert Ss is None
    assert losses.shape == (2,)
    assert losses_by_time.shape == (2, snqs.K)
    assert np.isfinite(losses).all()
    assert np.isfinite(losses_by_time).all()
    assert isinstance(ψfinal, RBM)


def test_sNQS_rbm_exact_train_link_fidelity_objective_smoke() -> None:
    Δt = 0.02
    N = 3
    α = 1
    Q = 2
    t = np.array([0.0, 0.1, 0.2])
    g_qt = get_g_qt(t, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    ψini = RBM(random_θ(N, α, device), N=N, α=α)
    snqs = sNQS_rbm(
        θ_jq,
        g_qt,
        Lx=N,
        Ly=1,
        α=α,
        Δt=Δt,
        model=TIM(J=-1.0, hx=-0.25, hz=0.15),
        backend="exact",
        max_exact_spins=8,
    )

    θ_out, Ss, losses, losses_by_time, ψfinal = snqs.train(
        ψini,
        Sini=None,
        batch=1,
        steps=2,
        lr=1.0e-3,
        log_interval=1,
        objective="link_fidelity",
        return_time_losses=True,
    )

    assert θ_out.shape == θ_jq.shape
    assert Ss is None
    assert losses.shape == (2,)
    assert losses_by_time.shape == (2, snqs.K)
    assert np.isfinite(losses).all()
    assert np.isfinite(losses_by_time).all()
    assert isinstance(ψfinal, RBM)


def test_sNQS_rbm_exact_lpe_grad_matches_dense_formula() -> None:
    Δt = 0.1
    N = 3
    α = 1
    Q = 3
    order = 2
    a_ms = get_LPE_coeffs(order=order)
    t_nodes, a_links, phy_idx = get_LPE_time_grid(
        0.0,
        0.2,
        dt=Δt,
        a_ms=a_ms,
        device=device,
        node_type="coeff",
    )
    g_qt = get_g_qt(t_nodes, Q, device, basis_type="simple")
    θ_jq = random_θ_jq(Q, N, α, device=device)
    ψini = RBM(random_θ(N, α, device), N=N, α=α)
    model = TIM(J=-1.0, hx=-0.25, hz=0.15)
    snqs = sNQS_rbm(
        θ_jq,
        g_qt,
        Lx=N,
        Ly=1,
        α=α,
        Δt=Δt,
        model=model,
        backend="exact",
        max_exact_spins=8,
        scheme="lpe",
        a_links=a_links,
        phy_idx=phy_idx,
    )

    ψs = snqs.ψs
    grad_exact, loss_exact, loss_by_time = snqs.grad_jq(
        ψini,
        ψs,
        Ss=None,
        batch=1,
        return_time_losses=True,
    )
    assert loss_by_time.shape == (snqs.K,)
    assert np.isfinite(loss_by_time).all()

    basis = snqs.exact_basis
    bonds = snqs.bonds
    Ediag_s = snqs.exact_Ediag
    psiinit_s = rbm_state_vector(ψini, basis)
    psi_vecs = [rbm_state_vector(ψk, basis) for ψk in ψs]

    grad_dense = tc.zeros_like(θ_jq)
    loss_dense = tc.tensor(1.0 + 0.0j)

    prob0_s = psi_vecs[0].abs().square()
    prob0_s = prob0_s / prob0_s.sum()
    r0_s = _safe_ratio(psiinit_s, psi_vecs[0])
    mean_O0, mean_O0r0, mean_r0 = _dense_weighted_stats(ψs[0], basis.states, prob0_s, r0_s)
    overlap = (psiinit_s.conj() * psi_vecs[0]).sum()
    norm_init = psiinit_s.abs().square().sum()
    G0 = (mean_O0r0 - mean_O0 * mean_r0) / mean_r0
    C0 = (overlap / norm_init) * mean_r0
    grad_dense += G0.reshape(-1, 1) @ g_qt[:, 0].reshape(1, -1)
    loss_dense *= C0

    for k in range(snqs.K):
        psi_k_s = psi_vecs[k]
        probk_s = psi_k_s.abs().square()
        probk_s = probk_s / probk_s.sum()

        if k > 0:
            H_prev = tim_hamiltonian_action(psi_vecs[k - 1], basis, bonds, J=model.J, hx=model.hx, hz=model.hz, Ediag_s=Ediag_s)
            coeff_prev = (-1j * Δt) * tc.as_tensor(a_links[k - 1], dtype=tc.complex128, device=device)
            Upsi_prev = psi_vecs[k - 1] + coeff_prev * H_prev
            Uloc_prev = _safe_ratio(Upsi_prev, psi_k_s)
            mean_O, mean_OU, mean_U = _dense_weighted_stats(ψs[k], basis.states, probk_s, Uloc_prev)
            G_prev = (mean_OU - mean_O * mean_U) / mean_U
            grad_dense += G_prev.reshape(-1, 1) @ g_qt[:, k].reshape(1, -1)
            loss_dense *= mean_U

        if k < snqs.K - 1:
            H_next = tim_hamiltonian_action(psi_vecs[k + 1], basis, bonds, J=model.J, hx=model.hx, hz=model.hz, Ediag_s=Ediag_s)
            coeff_next = (+1j * Δt) * tc.conj(tc.as_tensor(a_links[k], dtype=tc.complex128, device=device))
            Udagpsi_next = psi_vecs[k + 1] + coeff_next * H_next
            Uloc_next = _safe_ratio(Udagpsi_next, psi_k_s)
            mean_O, mean_OU, mean_U = _dense_weighted_stats(ψs[k], basis.states, probk_s, Uloc_next)
            G_next = (mean_OU - mean_O * mean_U) / mean_U
            grad_dense += G_next.reshape(-1, 1) @ g_qt[:, k].reshape(1, -1)
            loss_dense *= mean_U

    assert tc.allclose(grad_exact, grad_dense)
    assert np.isclose(loss_exact, float(np.abs(1.0 - loss_dense.item())), atol=1.0e-6, rtol=1.0e-6)
