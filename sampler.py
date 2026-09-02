# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-06 20:13:09
# @Last Modified by:   dzwang
# @Last Modified time: 2026-02-03 19:12:09
import torch as tc
from rbm import RBM
  

__all__ = ["random_samples", "Metropolis"]


def random_samples(M:int, N:int, device:str) -> tc.Tensor:
    s_mn = tc.empty((M, N), device=device, dtype=tc.int8).random_(2)
    s_mn.mul_(2).add_(-1)
    return s_mn.to(dtype=tc.complex128)


@tc.no_grad()
def Metropolis(s_mn:tc.Tensor, ψ:RBM, sweep:int, ret_rate:bool=False) -> tc.Tensor:
    assert s_mn.ndim == 2 and s_mn.shape[1] == ψ.N
    device = s_mn.device
    m, n = s_mn.shape
    inx_m = tc.arange(m, device=device)
    
    old_lnPsi_m = ψ.lnPsi(s_mn)
    acc_sum = 0.
    for _ in range(sweep):
        new_s_mn = s_mn.clone()
        j = tc.randint(n, (m,), device=device)
        new_s_mn[inx_m, j] *= -1
        new_lnPsi_m = ψ.lnPsi(new_s_mn)
        # Metropolis Hastings acceptance ratio
        log_ratio = 2.*(new_lnPsi_m.real - old_lnPsi_m.real).to(dtype=tc.float64)
        log_randm = tc.log(tc.rand(m, device=device, dtype=tc.float64))
        accept_m = log_randm < log_ratio
        # update samples
        if accept_m.any():
            s_mn[accept_m] = new_s_mn[accept_m]
            old_lnPsi_m[accept_m] = new_lnPsi_m[accept_m]
        
        acc_sum += accept_m.float().mean()
    if ret_rate:
        return s_mn, (acc_sum / sweep).item()
    return s_mn

