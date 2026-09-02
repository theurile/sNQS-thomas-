# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-03-06 21:45:50
# @Last Modified by:   dzwang
# @Last Modified time: 2026-03-27 14:08:35
import pytest
import numpy as np
import torch as tc
from utils import get_g_qt, get_LPE_coeffs, get_LPE_time_grid
device = "cuda" if tc.cuda.is_available() else "cpu"


def test_get_g_qt() -> None:
    ## single time point case
    t = 2
    g_qt = get_g_qt(t=2, Q=4, device=device, basis_type="simple")
    expected = tc.tensor([1, 2, 4, 8], dtype=tc.complex128, device=device)
    assert tc.allclose(g_qt, expected)
    ## list time points case
    t = np.array([2, 3])
    g_qt = get_g_qt(t=t, Q=3, device=device, basis_type="simple")
    expected = tc.tensor([[1, 1],
                          [2, 3],
                          [4, 9]], dtype=tc.complex128, device=device)
    assert tc.allclose(g_qt, expected)


def test_get_LPE_coeffs() -> None:
    order = 2
    a_ms = get_LPE_coeffs(order)
    assert np.allclose(np.sum(a_ms), 1.0)
    order = 3
    a_ms = get_LPE_coeffs(order)
    assert np.allclose(np.sum(a_ms), 1.0)
    order = 4
    a_ms = get_LPE_coeffs(order)
    assert np.allclose(np.sum(a_ms), 1.0)


def test_get_LPE_time_grid_real_nodes_order2() -> None:
    a_ms = get_LPE_coeffs(order=2)
    print(f"a_ms: {a_ms}")
    t0, tK = 0.0, 0.1
    t_nodes, a_links, phy_idx = get_LPE_time_grid(t0, tK, dt=0.1, a_ms=a_ms, node_type="real")
    expected = tc.tensor([0.0, 0.05, 0.1], dtype=tc.complex128)
    assert tc.allclose(t_nodes.cpu(), expected, atol=1e-12, rtol=1e-12)
    assert a_links.numel() == 2
    assert phy_idx == [0, 2]


@pytest.mark.parametrize(
    "order, expected",
    [
        (3, [0.0, 0.1/3., 0.2/3., 0.1]),
        (4, [0.0, 0.1/4., 0.2/4., 0.3/4., 0.1]),
    ],
)
def test_get_LPE_time_grid_real_nodes_higher_order(order:int, expected:list[float]) -> None:
    a_ms = get_LPE_coeffs(order=order)
    t0, tK = 0.0, 0.1
    t_nodes, _, _ = get_LPE_time_grid(t0, tK, dt=0.1, a_ms=a_ms, node_type="real")
    expected_t = tc.tensor(expected, dtype=tc.complex128)
    assert tc.allclose(t_nodes.cpu(), expected_t, atol=1e-12, rtol=1e-12)


def test_get_LPE_time_grid_preserve_coeff_nodes() -> None:
    a_ms = get_LPE_coeffs(order=2)
    t0, tK = 0.0, 0.1
    t_nodes, _, _, coeff_nodes = get_LPE_time_grid(
        t0, tK, dt=0.1, a_ms=a_ms, return_coeff_nodes=True
    )

    # Default output should be real, while coeff_nodes keeps the original complex grid.
    assert tc.allclose(t_nodes.cpu(), tc.tensor([0.0, 0.05, 0.1], dtype=tc.complex128), atol=1e-12, rtol=1e-12)
    assert tc.is_complex(coeff_nodes)
    assert not tc.allclose(coeff_nodes.cpu(), t_nodes.cpu(), atol=1e-12, rtol=1e-12)

