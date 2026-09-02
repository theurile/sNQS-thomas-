# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-09-06 20:12:55
# @Last Modified by:   dzwang
# @Last Modified time: 2026-04-27 22:19:55
import quante
import numpy as np
import torch as tc
from exact import (
    ExactSpinBasis,
    build_exact_spin_basis,
    diagonal_tim_energy,
    rbm_state_vector,
    tim_exact_observables,
    tim_hamiltonian_action,
)
from rbm import RBM
from model import TIM
from sampler import Metropolis


__all__ = ["sNQS_rbm", "update_Ss"]


class sNQS_rbm:
    r"""
    The network parameters θ should always be column vector
    .. math::
        ._   _.
        |θ_{1}|
        |θ_{2}|
        |  ...|
        |θ_{J}|
        `-   -`

    Within sNQS_rbm algorithm, we construct network paramters as different time points like
    .. math::
        ._      _.     ._                         _.   ._    _.
        |θ_{1}(t)|     |θ_{1,1} θ_{1,2} ... θ_{1,Q}|   |g_1(t)|
        |θ_{2}(t)|     |θ_{2,1} θ_{2,2} ... θ_{2,Q}|   |g_2(t)|
        |  ...   |  =  |                ...            | ...  |
        |θ_{J}(t)|     |θ_{J,1} θ_{J,2} ... θ_{J,Q}|   |g_Q(t)|
        `-      -`     `-                         -`   `_    _`                
    """
    
    def __init__(self, θ_jq: tc.Tensor, g_qt: tc.Tensor, Lx: int, Ly: int, α: int, Δt: float, model: TIM,
        *,
        backend: str = "mc",
        max_exact_spins: int = 16,
        allow_large_exact: bool = False,
        scheme: str = "taylor",
        a_links: tc.Tensor | None = None,
        phy_idx: list[int] | None = None) -> None:
        self.θ_jq = θ_jq
        self.g_qt = g_qt
        self.Lx = Lx
        self.Ly = Ly
        self.N = Lx * Ly
        self.α = α
        self.Δt = Δt
        self.model = model
        
        self.Np = θ_jq.shape[0]  # number of paramters in each network
        self.K = g_qt.shape[1]
        self.device = θ_jq.device
        
        if backend not in {"mc", "exact"}:
            raise ValueError("backend must be either 'mc' or 'exact'.")
        if max_exact_spins < 1:
            raise ValueError("max_exact_spins must be a positive integer.")
        self.backend = backend
        self.max_exact_spins = max_exact_spins
        self.allow_large_exact = allow_large_exact
        self.scheme = scheme
        self.a_links = a_links
        self.phy_idx = phy_idx
        self._exact_basis = None
        self._exact_Ediag = None
    
    @property
    def ψs(self) -> list[RBM]:  # 's' labels time
        θ_jt = self.θ_jq @ self.g_qt
        return [RBM(θ_jt[:, k].contiguous(), self.N, self.α) for k in range(self.K)]

    @property
    def bonds(self) -> tc.Tensor:
        return self.model.bonds(self.Lx, self.Ly).to(device=self.device, dtype=tc.long)

    def _validate_exact_backend_size(self) -> None:
        if self.backend != "exact" or self.allow_large_exact:
            return
        if self.N <= self.max_exact_spins:
            return
        num_states = 1 << self.N
        raise ValueError(
            "Exact backend is intended for small systems; "
            f"got N={self.N} ({num_states} states), "
            f"which exceeds max_exact_spins={self.max_exact_spins}. "
            "Set allow_large_exact=True to override this safety check."
        )

    @property
    def exact_basis(self) -> ExactSpinBasis:
        self._validate_exact_backend_size()
        if self._exact_basis is None:
            self._exact_basis = build_exact_spin_basis(self.N, device=self.device, dtype=tc.complex128)
        return self._exact_basis

    @property
    def exact_Ediag(self) -> tc.Tensor:
        self._validate_exact_backend_size()
        if self._exact_Ediag is None:
            self._exact_Ediag = diagonal_tim_energy(
                self.exact_basis.states,
                self.bonds,
                J=self.model.J,
                hz=self.model.hz,
            )
        return self._exact_Ediag
    
    def train(self, ψini:RBM, Sini:tc.Tensor | None, batch:int, 
              *,
              steps:int, lr:float, ema_alpha:float=0.95, log_interval:int,
              objective: str="global_product",
              return_time_losses: bool=False,
              optimizer_name: str="adamw",
              ) -> tuple[tc.Tensor, list[tc.Tensor] | None, np.ndarray, RBM] | tuple[tc.Tensor, list[tc.Tensor] | None, np.ndarray, np.ndarray, RBM]:
        if objective not in {"global_product", "link_fidelity"}:
            raise ValueError("objective must be either 'global_product' or 'link_fidelity'.")
        if objective == "link_fidelity" and self.backend != "exact":
            raise NotImplementedError("objective='link_fidelity' is currently implemented only for backend='exact'.")

        ψs = self.ψs
        if self.backend == "exact":
            self._validate_exact_backend_size()
            Ss = None
        else:
            if Sini is None:
                raise ValueError("Sini must be provided when backend='mc'.")
            Ss = [Sini.clone() for _ in range(self.K)]
            Ss = update_Ss(ψs, Ss, sweep=50*self.N)
        
        # make sure 'θ_jq' is a leaf param and define the optimizer
        if not isinstance(self.θ_jq, tc.nn.Parameter):
            self.θ_jq = tc.nn.Parameter(self.θ_jq, requires_grad=True)
        param = self.θ_jq
        assert id(param) == id(self.θ_jq)
        optimizer_key = optimizer_name.lower()
        if optimizer_key == "adamw":
            optimizer = tc.optim.AdamW([param], lr=lr, weight_decay=1e-5, amsgrad=True)
        elif optimizer_key == "adam":
            optimizer = tc.optim.Adam([param], lr=lr, weight_decay=1e-5, amsgrad=True)
        else:
            raise ValueError("optimizer_name must be either 'adamw' or 'adam'.")
        scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=300, cooldown=41, factor=0.5, min_lr=1.e-6)
        # scheduler = tc.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=1.1, last_epoch=-1)
        
        losses = []
        time_losses = [] if return_time_losses else None
        ema = None
        for epoch in range(1, steps+1):
            if objective == "link_fidelity":
                ψs = self.ψs
                loss_tensor, loss_by_time = self.exact_link_fidelity_loss(ψini, ψs)
                loss = float(loss_tensor.detach().cpu())
                loss_tensor.backward()
            else:
                with tc.no_grad():
                    ψs = self.ψs
                    if self.backend == "mc":
                        Ss = update_Ss(ψs, Ss, sweep=self.N)
                    if return_time_losses:
                        grad, loss, loss_by_time = self.grad_jq(ψini, ψs, Ss, batch, return_time_losses=True)
                    else:
                        grad, loss = self.grad_jq(ψini, ψs, Ss, batch)  # grad: complex (Np, Q); loss: float
                    if grad.dtype != param.dtype or grad.device != param.device:
                        grad = grad.to(dtype=param.dtype, device=param.device)
                    
                    # initialize/overwrite param.grad (no graph)
                    if param.grad is None: 
                        param.grad = tc.zeros_like(param)
                    param.grad = (-grad).clone()

            # one optimizer step and scheduler step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema = loss if ema is None else (ema_alpha*ema + (1.-ema_alpha)*loss)
            scheduler.step(ema)
            
            # Record
            losses.append(loss)
            if return_time_losses:
                time_losses.append(loss_by_time)
            if epoch%log_interval==0 or epoch==1 or epoch==steps:
                print(f"[{epoch:6d}/{steps}] "
                      f"optimizer={optimizer_key} "
                      f"loss={loss:.6g}, ema={ema:.6g} "
                      f"lr={optimizer.param_groups[0]['lr']:.3g}")
                if return_time_losses:
                    loss_label = "normalized_link_loss_by_time" if objective == "link_fidelity" else "link_loss_by_time"
                    print(
                        f"    {loss_label}="
                        f"{np.array2string(loss_by_time, precision=3, suppress_small=False, max_line_width=140)}"
                    )
        
        ψfinal = self.ψs[-1]
        if return_time_losses:
            return self.θ_jq.detach().clone(), Ss, np.array(losses), np.array(time_losses), ψfinal
        return self.θ_jq.detach().clone(), Ss, np.array(losses), ψfinal
    
    @tc.no_grad()
    def grad_jq(
        self,
        ψini:RBM,
        ψs:list[RBM],
        Ss:list[tc.Tensor] | None,
        batch:int,
        *,
        return_time_losses: bool=False,
    ) -> tuple[tc.Tensor, float] | tuple[tc.Tensor, float, np.ndarray]:
        if self.backend == "exact":
            return self.grad_jq_exact(ψini, ψs, return_time_losses=return_time_losses)

        g_qt = self.g_qt
        grad = tc.zeros_like(self.θ_jq, device=self.device)   # (Np, Q)
        loss = 1.
        C_0 = None
        C_prev_by_time = [None] * self.K if return_time_losses else None
        C_next_by_time = [None] * self.K if return_time_losses else None
        
        ## first term
        G_0, C_0 = self.dC_init(Ss[0], ψs[0], ψini, batch)
        grad += G_0.reshape(-1, 1) @ g_qt[:, 0].reshape(1, -1)
        loss *= C_0
        
        ## other terms 
        for k in range(self.K):
            ψkm1 = ψs[k-1] if k > 0 else None
            ψkp1 = ψs[k+1] if k < self.K-1 else None
            a_prev = self.a_links[k-1] if (self.scheme == "lpe" and k > 0) else None
            a_next = self.a_links[k]   if (self.scheme == "lpe" and k < self.K-1) else None
            G_prev, G_next, C_prev, C_next = self.dC_pair(Ss[k], ψs[k], ψkm1, ψkp1, batch, a_prev=a_prev, a_next=a_next)
            if G_prev is not None:
                # grad += tc.outer(g_qt[:, k], G_prev)
                grad += G_prev.reshape(-1, 1) @ g_qt[:, k].reshape(1, -1)
                loss *= C_prev
                if C_prev_by_time is not None:
                    C_prev_by_time[k] = C_prev
            if G_next is not None:
                # grad += tc.outer(g_qt[:, k], G_next)
                grad += G_next.reshape(-1, 1) @ g_qt[:, k].reshape(1, -1)
                loss *= C_next
                if C_next_by_time is not None:
                    C_next_by_time[k] = C_next
        loss = np.abs(1. - loss)
        if C_prev_by_time is not None and C_next_by_time is not None:
            return grad, loss, self._loss_by_time_from_link_products(C_0, C_prev_by_time, C_next_by_time)
        return grad, loss

    def _loss_by_time_from_link_products(
        self,
        C0: complex,
        C_prev_by_time: list[complex | None],
        C_next_by_time: list[complex | None],
    ) -> np.ndarray:
        loss_by_time = np.full(self.K, np.nan, dtype=np.float64)
        loss_by_time[0] = float(np.abs(1.0 - C0))
        for k in range(self.K - 1):
            C_next = C_next_by_time[k]
            C_prev = C_prev_by_time[k + 1]
            if C_next is not None and C_prev is not None:
                loss_by_time[k + 1] = float(np.abs(1.0 - C_next * C_prev))
        return loss_by_time
    
    @tc.no_grad()
    def dC_init(self, S0:tc.Tensor, ψ0:RBM, ψinit:RBM, batch:int) -> tuple[tc.Tensor, float]:
        sum_O = tc.zeros(self.Np, dtype=tc.complex128, device=self.device)
        sum_Or0 = tc.zeros(self.Np, dtype=tc.complex128, device=self.device)
        sum_r0 = tc.zeros((), dtype=tc.complex128, device=self.device)
        _sum_r0 = tc.zeros((), dtype=tc.complex128, device=self.device)
        
        S0 = S0.clone()
        for _ in range(batch):
            S0 = Metropolis(S0, ψ0, sweep=self.N)
            O_mj = ψ0.d_lnPsi(S0).conj()  # (Nmc, Np)
            ##        <ψ0|ψinit>
            ## r0_m = ----------
            ##        <ψ0|ψ0>
            r0_m = tc.exp(ψinit.lnPsi(S0) - ψ0.lnPsi(S0))  # (Nmc,)
            sum_O += O_mj.sum(dim=0)  # (, Np)
            sum_r0 += r0_m.sum()  # (, 1)
            sum_Or0 += (O_mj * r0_m[:, None]).sum(dim=0) # (, Np)
            ##         <ψinit|ψ0>
            ## _r0_m = ------------
            ##         <ψinit|ψinit>
            _r0_m = tc.exp(ψ0.lnPsi(S0) - ψinit.lnPsi(S0))  # (Nmc,)
            _sum_r0 += _r0_m.sum()  # (, 1)
        
        Nmc = S0.shape[0] * batch
        mean_O = sum_O / Nmc
        mean_Or0 = sum_Or0 / Nmc
        mean_r0 = sum_r0 / Nmc
        _mean_r0 = _sum_r0 / Nmc
        
        cov = mean_Or0 - mean_O * mean_r0
        G0  = cov / mean_r0
        C0 = (mean_r0 * _mean_r0).item()
        return G0, C0
    
    @tc.no_grad()
    def dC_pair(
        self,
        Sk: tc.Tensor,
        ψk: RBM,
        ψkm1: RBM | None,
        ψkp1: RBM | None,
        batch: int,
        *,
        a_prev=None,
        a_next=None,
    ) -> tuple[tc.Tensor | None, tc.Tensor | None]:
        Np = self.Np
        device = self.device
        sum_O = tc.zeros(Np, dtype=tc.complex128, device=device)
        sum_OU_prev = tc.zeros(Np, dtype=tc.complex128, device=device)
        sum_OU_next = tc.zeros(Np, dtype=tc.complex128, device=device)
        sum_U_prev  = tc.zeros((), dtype=tc.complex128, device=device)
        sum_U_next  = tc.zeros((), dtype=tc.complex128, device=device)
        Sk = Sk.clone()
        for _ in range(batch):
            Sk = Metropolis(Sk, ψk, sweep=self.N)
            O_mj = ψk.d_lnPsi(Sk).conj()  # (Nmc, Np)
            # U_prev, U_next = self.Uloc_pair(Sk, ψk, ψkm1, ψkp1)  # (Nmc,)
            U_prev, U_next = self.Uloc_pair(Sk, ψk, ψkm1, ψkp1, a_prev=a_prev, a_next=a_next)
            
            sum_O += O_mj.sum(dim=0)
            if ψkm1 is not None:
                sum_OU_prev += (O_mj * U_prev[:,None]).sum(dim=0)
                sum_U_prev += U_prev.sum()
            if ψkp1 is not None:
                sum_OU_next += (O_mj * U_next[:,None]).sum(dim=0)
                sum_U_next += U_next.sum()
        
        Nmc = Sk.shape[0] * batch
        mean_O = sum_O / Nmc
        
        def finalize(sum_OU:tc.Tensor, sum_U:tc.Tensor) -> tc.Tensor:
            mean_OU = sum_OU / Nmc
            mean_U = sum_U  / Nmc
            cov = mean_OU - mean_O * mean_U
            G = cov / mean_U
            C = mean_U.item()
            return G, C
        
        G_prev, C_prev = finalize(sum_OU_prev, sum_U_prev) if ψkm1 is not None else (None, None)
        G_next, C_next = finalize(sum_OU_next, sum_U_next) if ψkp1 is not None else (None, None)
        
        return G_prev, G_next, C_prev, C_next
    
    @tc.no_grad()
    def Uloc_pair(self, Sk:tc.Tensor, ψk:RBM, ψkm1:RBM, ψkp1:RBM, *, a_prev=None, a_next=None) -> tuple[tc.Tensor, tc.Tensor]:
        if self.scheme == "taylor":
            return self.Uloc_pair_taylor(Sk, ψk, ψkm1, ψkp1)
        elif self.scheme == "lpe":
            return self.Uloc_pair_lpe(Sk, ψk, ψkm1, ψkp1, a_prev=a_prev, a_next=a_next)
    
    @tc.no_grad()
    def Uloc_pair_lpe(
        self, Sk:tc.Tensor, ψk:RBM, ψkm1:RBM, ψkp1:RBM,
        *,
        a_prev=None, a_next=None
    ) -> tuple[tc.Tensor, tc.Tensor]:
        
        Nmc, N = Sk.shape  # number of MC samples; number of spins
        Δt, model, device = self.Δt, self.model, self.device
        J, hx, hz = model.J, model.hx, model.hz
        bonds = model.bonds(self.Lx, self.Ly).to(device=device, dtype=tc.long)
        
        def _Ediag(s_mn: tc.Tensor) -> tc.Tensor:
            s_mn = s_mn.real.to(tc.float64)
            i, j = bonds[:, 0], bonds[:, 1]
            zz_pair = s_mn[:, i] * s_mn[:, j]
            return J * zz_pair.sum(dim=1) + hz * s_mn.sum(dim=1)   # (Nmc,)
        
        ### current time point 'tk' for ψ
        lnψk_m = ψk.lnPsi(Sk)  # (Nmc,)
        Ediag_m = _Ediag(Sk)  # (Nmc,)
        ### s_mnn
        S_flip = Sk[:, None, :].expand(Nmc, N, N).clone()
        idx = tc.arange(N, device=device)
        S_flip[:, idx, idx] *= -1
        S_flip2d = S_flip.reshape(-1, N)  # (Nmc*N, N)

        def build(ψj: RBM, coeff) -> tc.Tensor | None:
            if ψj is None or coeff is None:
                return None
            lnψj_m = ψj.lnPsi(Sk)  # (Nmc,)
            r_m = tc.exp(lnψj_m - lnψk_m)  # (Nmc,)
            lnψj_mn = ψj.lnPsi(S_flip2d).reshape(Nmc, N)  # (Nmc, N)
            r_mi = tc.exp(lnψj_mn - lnψk_m[:, None])  # (Nmc, N)
            H_cross = Ediag_m * r_m + hx * r_mi.sum(dim=1) # (Nmc,)
            return r_m + coeff * H_cross

        coeff_prev = None
        if a_prev is not None:
            a_prev = tc.as_tensor(a_prev, dtype=tc.complex128, device=device)
            coeff_prev = (-1j * Δt) * a_prev
        coeff_next = None
        if a_next is not None:
            a_next = tc.as_tensor(a_next, dtype=tc.complex128, device=device)
            coeff_next = (+1j * Δt) * tc.conj(a_next)
        
        U_prev = build(ψkm1, coeff_prev)   # <ψ_k | T_{a_prev} | ψ_{k-1}>
        U_next = build(ψkp1, coeff_next)   # <ψ_k | T_{a_next}^\dagger | ψ_{k+1}>
        return U_prev, U_next
    
    @tc.no_grad()
    def Uloc_pair_taylor(self, Sk:tc.Tensor, ψk:RBM, ψkm1:RBM, ψkp1:RBM) -> tuple[tc.Tensor, tc.Tensor]:
        Nmc, N = Sk.shape  # Nmc: number of MC samples; N: number of spins
        Δt, model, device = self.Δt, self.model, self.device
        J, hx, hz = model.J, model.hx, model.hz
        bonds = model.bonds(self.Lx, self.Ly).to(device=device, dtype=tc.long)
        
        def _Ediag_and_Ediag_i(s_mn:tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
            s_mn = s_mn.real.to(tc.float64)  # (Nmc, N)
            i, j = bonds[:,0], bonds[:,1]  # (N-1,) for OBC, (N,) for PBC
            ## E_diag(s)
            zz_pair = s_mn[:,i] * s_mn[:,j]
            Ediag_m = J*zz_pair.sum(dim=1) + hz*s_mn.sum(dim=1) # (Nmc,)
            ## E_diag(s^(i))
            local_fields = hz * tc.ones_like(s_mn, dtype=tc.float64, device=device)  # (Nmc, N)
            local_fields[:, i] += J * s_mn[:, j]
            local_fields[:, j] += J * s_mn[:, i]
            Ediag_mi = Ediag_m[:, None]- 2. * s_mn * local_fields  # (Nmc, N)
            return Ediag_m, Ediag_mi
        
        def _r_ij(s_mn:tc.Tensor, ψj:RBM) -> tc.Tensor|None:
            r_ij = tc.zeros(Nmc, dtype=tc.complex128, device=device)  # (Nmc,)
            for j in range(1, N):
                Sj = s_mn.clone()
                Sj[:, j] *= -1
                block = Sj[:, None, :].expand(Nmc, j, N).clone()  # (Nmc, j, N)
                idx = tc.arange(j, device=device)
                block[:, idx, idx] *= -1
                Sj2d = block.reshape(-1, N)  # (Nmc*j, N)
                lnψj_blk = ψj.lnPsi(Sj2d).reshape(Nmc, j)  # (Nmc, j)
                r_ij += tc.exp(lnψj_blk - lnψk_m[:, None]).sum(dim=1)  # (Nmc,)
            return r_ij
        
        lnψk_m = ψk.lnPsi(Sk)  # (Nmc,)
        Ediag_m, Ediag_mi = _Ediag_and_Ediag_i(Sk)  # (Nmc,), (Nmc, N)
        # s_mnn
        S_flip = Sk[:,None,:].expand(Nmc, N, N).clone()
        idx = tc.arange(N, device=device)
        S_flip[:, idx, idx] *= -1
        S_flip2d = S_flip.reshape(-1, N)  
        
        def build(ψj:RBM, dagger:bool) -> tc.Tensor|None:
            if ψj is None:
                return None
            lnψj_m = ψj.lnPsi(Sk)  # (Nmc,)
            # 0th order 
            r_m = tc.exp(lnψj_m - lnψk_m)  # (Nmc,)
            # 1st order: (+ or -) iΔtH 
            lnψj_mn = ψj.lnPsi(S_flip2d).reshape(Nmc, N)  # (Nmc, N)
            r_mi = tc.exp(lnψj_mn - lnψk_m[:, None])  # (Nmc, N)
            sign = +1.0 if dagger else -1.0
            first = (sign * 1j * Δt) * (Ediag_m * r_m + hx * r_mi.sum(dim=1))
            # 2nd order: -1/2 Δt^2 H^2
            part_DD = (Ediag_m**2) * r_m
            part_DX_XD = hx * ((Ediag_m[:, None] + Ediag_mi) * r_mi).sum(dim=1)
            part_XX = hx**2 * (N*r_m + 2*_r_ij(Sk, ψj))
            second = (-0.5 * Δt**2) * (part_DD + part_DX_XD + part_XX)
            return r_m + first + second
        
        U_prev = build(ψkm1, dagger=False)  # <ψ_k| U |ψ_{k-1}>
        U_next = build(ψkp1, dagger=True)  # <ψ_k| U† |ψ_{k+1}>
        return U_prev, U_next

    @tc.no_grad()
    def grad_jq_exact(
        self,
        ψini: RBM,
        ψs: list[RBM],
        *,
        return_time_losses: bool=False,
    ) -> tuple[tc.Tensor, float] | tuple[tc.Tensor, float, np.ndarray]:
        self._validate_exact_backend_size()
        grad = tc.zeros_like(self.θ_jq, device=self.device)
        loss = 1.0 + 0.0j
        C_prev_by_time = [None] * self.K if return_time_losses else None
        C_next_by_time = [None] * self.K if return_time_losses else None

        psiinit_s = rbm_state_vector(ψini, self.exact_basis)
        psi_vecs = [rbm_state_vector(ψk, self.exact_basis) for ψk in ψs]

        G0, C0 = self.dC_init_exact(ψs[0], psi_vecs[0], ψini, psiinit_s)
        grad += G0.reshape(-1, 1) @ self.g_qt[:, 0].reshape(1, -1)
        loss *= C0

        for k in range(self.K):
            psi_km1_s = psi_vecs[k - 1] if k > 0 else None
            psi_kp1_s = psi_vecs[k + 1] if k < self.K - 1 else None
            a_prev = self.a_links[k - 1] if (self.scheme == "lpe" and k > 0) else None
            a_next = self.a_links[k] if (self.scheme == "lpe" and k < self.K - 1) else None
            G_prev, G_next, C_prev, C_next = self.dC_pair_exact(
                ψs[k],
                psi_vecs[k],
                psi_km1_s,
                psi_kp1_s,
                a_prev=a_prev,
                a_next=a_next,
            )
            if G_prev is not None:
                grad += G_prev.reshape(-1, 1) @ self.g_qt[:, k].reshape(1, -1)
                loss *= C_prev
                if C_prev_by_time is not None:
                    C_prev_by_time[k] = C_prev
            if G_next is not None:
                grad += G_next.reshape(-1, 1) @ self.g_qt[:, k].reshape(1, -1)
                loss *= C_next
                if C_next_by_time is not None:
                    C_next_by_time[k] = C_next

        if C_prev_by_time is not None and C_next_by_time is not None:
            return grad, float(np.abs(1.0 - loss)), self._loss_by_time_from_link_products(C0, C_prev_by_time, C_next_by_time)
        return grad, float(np.abs(1.0 - loss))

    def exact_link_fidelity_loss(self, ψini: RBM, ψs: list[RBM]) -> tuple[tc.Tensor, np.ndarray]:
        self._validate_exact_backend_size()
        psiinit_s = rbm_state_vector(ψini, self.exact_basis).detach()
        psi_vecs = [rbm_state_vector(ψk, self.exact_basis) for ψk in ψs]

        one = tc.tensor(1.0, dtype=tc.float64, device=self.device)
        losses = []

        psi0_s = psi_vecs[0]
        init_overlap = (psiinit_s.conj() * psi0_s).sum()
        init_den = psiinit_s.abs().square().sum() * psi0_s.abs().square().sum()
        F0 = init_overlap.abs().square() / init_den.real
        losses.append(one - F0.real)

        for k in range(self.K - 1):
            prop_s = self._exact_forward_state(psi_vecs[k], link_idx=k)
            target_s = psi_vecs[k + 1]
            overlap = (target_s.conj() * prop_s).sum()
            den = prop_s.abs().square().sum() * target_s.abs().square().sum()
            F = overlap.abs().square() / den.real
            losses.append(one - F.real)

        loss_by_time_t = tc.stack(losses)
        return loss_by_time_t.mean(), loss_by_time_t.detach().cpu().numpy()

    def _exact_forward_state(self, psi_s: tc.Tensor, *, link_idx: int) -> tc.Tensor:
        if self.scheme == "taylor":
            return self._exact_taylor_state(psi_s, dagger=False)
        if self.scheme == "lpe":
            coeff = (-1j * self.Δt) * tc.as_tensor(self.a_links[link_idx], dtype=tc.complex128, device=self.device)
            return self._exact_lpe_state(psi_s, coeff)
        raise ValueError("Unsupported scheme.")
    
    @tc.no_grad()
    def dC_init_exact(self, ψ0: RBM, psi0_s: tc.Tensor, ψinit: RBM, psiinit_s: tc.Tensor) -> tuple[tc.Tensor, complex]:
        prob0_s = psi0_s.abs().square()
        prob0_s = prob0_s / prob0_s.sum()
        r0_s = self._safe_local_ratio(psiinit_s, psi0_s)
        mean_O, mean_Or0, mean_r0 = self._exact_weighted_stats(ψ0, prob0_s, r0_s)
        overlap = (psiinit_s.conj() * psi0_s).sum()
        norm_init = psiinit_s.abs().square().sum()
        C0 = (overlap / norm_init) * mean_r0
        G0 = (mean_Or0 - mean_O * mean_r0) / mean_r0
        return G0, C0.item()

    def _exact_weighted_stats(self, ψk: RBM, prob_s: tc.Tensor, value_s: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
        chunk_size = self._exact_chunk_size()
        mean_O = tc.zeros(self.Np, dtype=tc.complex128, device=self.device)
        mean_OV = tc.zeros(self.Np, dtype=tc.complex128, device=self.device)
        mean_value = (prob_s * value_s).sum()

        for start in range(0, self.exact_basis.num_states, chunk_size):
            stop = min(start + chunk_size, self.exact_basis.num_states)
            states_chunk = self.exact_basis.states[start:stop]
            O_chunk = ψk.d_lnPsi(states_chunk).conj()
            prob_chunk = prob_s[start:stop]
            value_chunk = value_s[start:stop]
            mean_O += (prob_chunk[:, None] * O_chunk).sum(dim=0)
            mean_OV += ((prob_chunk * value_chunk)[:, None] * O_chunk).sum(dim=0)
        return mean_O, mean_OV, mean_value
    
    def _exact_chunk_size(self) -> int:
        return min(self.exact_basis.num_states, 4096)
    
    @tc.no_grad()
    def dC_pair_exact(
        self,
        ψk: RBM,
        psi_k_s: tc.Tensor,
        psi_km1_s: tc.Tensor | None,
        psi_kp1_s: tc.Tensor | None,
        *,
        a_prev=None,
        a_next=None,
    ) -> tuple[tc.Tensor | None, tc.Tensor | None, complex | None, complex | None]:
        probk_s = psi_k_s.abs().square()
        probk_s = probk_s / probk_s.sum()
        U_prev, U_next = self.Uloc_pair_exact(
            psi_k_s,
            psi_km1_s,
            psi_kp1_s,
            a_prev=a_prev,
            a_next=a_next,
        )

        def finalize(value_s: tc.Tensor | None) -> tuple[tc.Tensor | None, complex | None]:
            if value_s is None:
                return None, None
            mean_O, mean_OU, mean_value = self._exact_weighted_stats(ψk, probk_s, value_s)
            G = (mean_OU - mean_O * mean_value) / mean_value
            return G, mean_value.item()

        G_prev, C_prev = finalize(U_prev)
        G_next, C_next = finalize(U_next)
        return G_prev, G_next, C_prev, C_next
    
    def Uloc_pair_exact(
        self,
        psi_k_s: tc.Tensor,
        psi_km1_s: tc.Tensor | None,
        psi_kp1_s: tc.Tensor | None,
        *,
        a_prev=None,
        a_next=None,
    ) -> tuple[tc.Tensor | None, tc.Tensor | None]:
        U_prev = None
        if psi_km1_s is not None:
            if self.scheme == "taylor":
                numer_prev = self._exact_taylor_state(psi_km1_s, dagger=False)
            elif self.scheme == "lpe":
                coeff_prev = (-1j * self.Δt) * tc.as_tensor(a_prev, dtype=tc.complex128, device=self.device)
                numer_prev = self._exact_lpe_state(psi_km1_s, coeff_prev)
            else:
                raise ValueError("Unsupported scheme.")
            U_prev = self._safe_local_ratio(numer_prev, psi_k_s)

        U_next = None
        if psi_kp1_s is not None:
            if self.scheme == "taylor":
                numer_next = self._exact_taylor_state(psi_kp1_s, dagger=True)
            elif self.scheme == "lpe":
                coeff_next = (+1j * self.Δt) * tc.conj(tc.as_tensor(a_next, dtype=tc.complex128, device=self.device))
                numer_next = self._exact_lpe_state(psi_kp1_s, coeff_next)
            else:
                raise ValueError("Unsupported scheme.")
            U_next = self._safe_local_ratio(numer_next, psi_k_s)
        return U_prev, U_next
    
    def _exact_taylor_state(self, psi_s: tc.Tensor, *, dagger: bool) -> tc.Tensor:
        Hpsi_s = tim_hamiltonian_action(
            psi_s,
            self.exact_basis,
            self.bonds,
            J=self.model.J,
            hx=self.model.hx,
            hz=self.model.hz,
            Ediag_s=self.exact_Ediag,
        )
        H2psi_s = tim_hamiltonian_action(
            Hpsi_s,
            self.exact_basis,
            self.bonds,
            J=self.model.J,
            hx=self.model.hx,
            hz=self.model.hz,
            Ediag_s=self.exact_Ediag,
        )
        sign = +1.0 if dagger else -1.0
        return psi_s + (sign * 1j * self.Δt) * Hpsi_s - 0.5 * (self.Δt ** 2) * H2psi_s

    def _exact_lpe_state(self, psi_s: tc.Tensor, coeff: tc.Tensor) -> tc.Tensor:
        Hpsi_s = tim_hamiltonian_action(
            psi_s,
            self.exact_basis,
            self.bonds,
            J=self.model.J,
            hx=self.model.hx,
            hz=self.model.hz,
            Ediag_s=self.exact_Ediag,
        )
        return psi_s + coeff * Hpsi_s

    def _safe_local_ratio(self, numerator_s: tc.Tensor, denominator_s: tc.Tensor, cutoff: float = 1e-30) -> tc.Tensor:
        out = tc.zeros_like(numerator_s)
        mask = denominator_s.abs() > cutoff
        out[mask] = numerator_s[mask] / denominator_s[mask]
        return out
    
    @tc.no_grad()
    def expectation_value(self, Ss:list[tc.Tensor] | None, batch:int) -> tuple[list, list, list]:
        if self.backend == "exact":
            return self.expectation_value_exact()

        K = self.K
        ψs = self.ψs
        assert len(Ss) == len(ψs) == K
        bonds = self.bonds
        
        Nmc = Ss[0].shape[0] * batch
        E_k, Sx_k, Sz_k = [], [], []

        for k in range(self.K):
            Sk, ψk = Ss[k], ψs[k]
            sum_E, sum_Sx, sum_Sz = 0., 0., 0.
        
            # before the inner accumulation loop
            for _ in range(getattr(self, "burn_in", 20)):  # 5~20 sweeps 通常足够
                Sk = Metropolis(Sk, ψk, sweep=self.N)
            for _ in range(batch):
                Sk = Metropolis(Sk, ψk, sweep=self.N)
                E, Sx, Sz = self.energy_Sx_Sz(Sk, ψk, bonds)
                sum_E += E
                sum_Sx += Sx
                sum_Sz += Sz

            E_k.append(float((sum_E/Nmc).real))
            Sx_k.append(float((sum_Sx/Nmc).real))
            Sz_k.append(float((sum_Sz/Nmc).real))
            
            Ss[k] = Sk
        return E_k, Sx_k, Sz_k

    @tc.no_grad()
    def expectation_value_exact(self) -> tuple[list, list, list]:
        self._validate_exact_backend_size()
        ψs = self.ψs
        basis = self.exact_basis
        bonds = self.bonds
        Ediag_s = self.exact_Ediag

        E_k, Sx_k, Sz_k = [], [], []
        for ψk in ψs:
            psi_s = rbm_state_vector(ψk, basis)
            E, Sx, Sz = tim_exact_observables(
                psi_s,
                basis,
                bonds,
                J=self.model.J,
                hx=self.model.hx,
                hz=self.model.hz,
                Ediag_s=Ediag_s,
            )
            E_k.append(float(E.real))
            Sx_k.append(float(Sx.real))
            Sz_k.append(float(Sz.real))
        return E_k, Sx_k, Sz_k
    
    @tc.no_grad()
    def energy_Sx_Sz(self, Sk:tc.Tensor, ψk:RBM, bonds:tc.Tensor) -> tuple[float, float, float]:
        Mmc, N = Sk.shape
        device = self.device
        J, hx, hz = self.model.J, self.model.hx, self.model.hz
        lnψk_m = ψk.lnPsi(Sk)  # (Mmc,)
        ## r_i(s)
        S_flip = Sk[:, None, :].expand(Mmc, N, N).clone()  # (Mmc, N, N)
        idx = tc.arange(N, device=device)
        S_flip[:, idx, idx] *= -1
        lnψk_mn = ψk.lnPsi(S_flip.reshape(-1, N)).reshape(Mmc, N)  # (Mmc, N)
        r_si = tc.exp(lnψk_mn - lnψk_m[:,None])  # (Mmc, N)
        
        ## sigmaX
        Sx_m = r_si.sum(dim=1)  # (Mmc,)
        ## sigmaZ
        Sz_m = Sk.sum(dim=1)  # (Mmc,)
        ## energy
        E_m = J*(Sk[:,bonds[:,0]] * Sk[:,bonds[:,1]]).sum(dim=1) + hz*Sz_m + hx*Sx_m  # (Mmc,)
        return E_m.sum(), Sx_m.sum(), Sz_m.sum()


@tc.no_grad()
def update_Ss(ψs:list[RBM], Ss:list[tc.Tensor], sweep:int) -> list[tc.Tensor]:
    K = len(Ss)
    assert len(Ss) == len(ψs)
    return [Metropolis(Ss[k], ψs[k], sweep) for k in range(K)]
