# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-15 13:32:57
# @Last Modified by:   dzwang
# @Last Modified time: 2026-02-03 19:17:17
import torch as tc
from rbm import RBM
from tqdm import tqdm
from sampler import Metropolis
from model import TIM


class VMC:
    def __init__(self, θ:tc.Tensor, Lx:int, Ly:int, α:int, model:TIM) -> None:
        self.θ = θ
        self.Lx = Lx
        self.Ly = Ly
        self.N = Lx * Ly
        self.α = α
        self.model = model
        assert model.J == model.hz == 0
        
        self.Np = θ.numel()
        self.device = θ.device
    
    @property
    def ψ(self) -> RBM:
        return RBM(self.θ, self.N, self.α)
    
    def train(self, S0:tc.Tensor, batch:int, *, steps:int, lr:float, log_interval:int) -> tuple[RBM, tc.Tensor]:
        device = self.device
        
        S = S0.clone().to(device)
        S = Metropolis(S, self.ψ, sweep=10*self.N)
        for epoch in tqdm(range(1, steps+1)):
            ψ = self.ψ
            S = Metropolis(S, ψ, sweep=self.N)
            delta, E = self.grad(S, batch=batch)
            self.θ -= lr * delta
            if epoch%log_interval==0 or epoch == 1 or epoch == steps:
                print(f"Epoch {epoch:4d}/{steps}, ⟨H⟩={E:.10f}")
            if abs(E - self.model.hx*self.N) < 1.e-12:
                E, Sx, Sz = self.energy_Sx_Sz(S, self.model.bonds(self.Lx, self.Ly))
                print(f"Epoch {epoch:4d}/{steps}, ⟨H⟩={E:.10f}, ⟨Sx⟩={Sx:.8f}, ⟨Sz⟩={Sz:.8f}")
                break 
        return self.ψ, S
    
    def grad(self, s_mn:tc.Tensor, batch) -> tuple[tc.Tensor, float]:
        device = self.device
        ψ = self.ψ
        Np = self.Np
        sum_OO = tc.zeros((Np, Np), device=device, dtype=tc.complex128)
        sum_OE = tc.zeros(Np, device=device, dtype=tc.complex128)
        sum_O = tc.zeros(Np, device=device, dtype=tc.complex128)
        sum_E = tc.zeros((), device=device, dtype=tc.complex128)
        
        s_mn = s_mn.clone()
        for _ in range(batch):
            s_mn = Metropolis(s_mn, ψ, sweep=self.N)
            O_mn = ψ.d_lnPsi(s_mn)
            E_m = self.Eloc(s_mn)
            sum_OO += O_mn.conj().T @ O_mn
            sum_OE += (O_mn.conj() * E_m[:, None]).sum(dim=0)
            sum_O += O_mn.sum(dim=0)
            sum_E += E_m.sum()
        
        Nmc = s_mn.shape[0] * batch
        mean_O = sum_O / Nmc
        mean_E = sum_E / Nmc
        mean_OO = sum_OO / Nmc
        mean_OE = sum_OE / Nmc
        
        F = mean_OE - mean_O.conj() * mean_E
        S_mat = mean_OO - mean_O.conj()[:, None] * mean_O[None, :]
        S_mat = S_mat + 1.e-4 * tc.eye(Np, device=device, dtype=tc.complex128)
        S_mat = 0.5 * (S_mat + S_mat.conj().T)
        scale = S_mat.real.diag().mean().clamp_min(1e-12)
        S_mat = S_mat + (1.e-3 * scale).to(tc.complex128) * tc.eye(Np, dtype=tc.complex128, device=device)
        delta = tc.linalg.solve(S_mat, F)
        return delta, mean_E.real.item()
    
    def Eloc(self, s_mn:tc.Tensor) -> tc.Tensor:
        device = s_mn.device
        ψ = self.ψ
        Nmc, N = s_mn.shape
        hx = self.model.hx
        lnψ_m = ψ.lnPsi(s_mn)
        # r_i(s)
        s_mnn = s_mn[:, None, :].expand(Nmc, N, N).clone()
        idx = tc.arange(N, device=device)
        s_mnn[:, idx, idx] *= -1
        S2 = s_mnn.reshape(-1, N)
        lnψ_mn = ψ.lnPsi(S2).reshape(Nmc, N)
        r_mn = tc.exp(lnψ_mn - lnψ_m[:, None])
        Eloc = hx * r_mn.sum(dim=1)
        return Eloc
      
    def energy_Sx_Sz(self, s_mn:tc.Tensor, bonds:tc.Tensor) -> tuple[float, float, float]:
        device = self.device
        ψ = self.ψ
        Mmc, N = s_mn.shape
        J, hx, hz = self.model.J, self.model.hx, self.model.hz
        lnψ_m = ψ.lnPsi(s_mn)  # (Mmc,)
        ## r_i(s)
        S_flip = s_mn[:, None, :].expand(Mmc, N, N).clone()  # (Mmc, N, N)
        idx = tc.arange(N, device=device)
        S_flip[:, idx, idx] *= -1
        lnψk_mn = ψ.lnPsi(S_flip.reshape(-1, N)).reshape(Mmc, N)  # (Mmc, N)
        r_si = tc.exp(lnψk_mn - lnψ_m[:,None])  # (Mmc, N)
        
        ## sigmaX
        Sx_m = r_si.sum(dim=1)  # (Mmc,)
        ## sigmaZ
        Sz_m = s_mn.sum(dim=1)  # (Mmc,)
        ## energy
        E_m = J*(s_mn[:,bonds[:,0]] * s_mn[:,bonds[:,1]]).sum(dim=1) + hz*Sz_m + hx*Sx_m  # (Mmc,)
        return E_m.mean().real.item(), Sx_m.mean().real.item(), Sz_m.mean().real.item()
    



