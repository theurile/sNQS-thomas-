import torch as tc
from rbm import RBM, random_θ


class RBM_jastrow(RBM):
    r"""
    RBM multiplied by a pairwise Jastrow factor:
        ln ψ(s) = ln ψ_RBM(s) + Σ_{i<j} J_ij s_i s_j

    θ layout: [ RBM parameters (a, b, W) as in RBM | J_ij for i<j (N(N-1)/2 entries) ].
    Same call signature as RBM: RBM_jastrow(θ, N, α).
    """

    def __init__(self, θ: tc.Tensor, N: int, α: int) -> None:
        self.n_rbm = N + α * N + α * N * N
        self.n_pairs = N * (N - 1) // 2
        assert θ.ndim == 1 and θ.numel() == self.n_rbm + self.n_pairs, (
            f"θ has {θ.numel()} entries, expected {self.n_rbm} (RBM) + {self.n_pairs} (Jastrow).")
        super().__init__(θ[:self.n_rbm], N, α)       # parent sees only its own parameters
        self.θ_full = θ                              # full vector (the RBM part is a view of it)
        self.Np = θ.numel()
        self.J = θ[self.n_rbm:]
        self.iu = tc.triu_indices(N, N, offset=1, device=θ.device)

    def _pairs(self, s_mn: tc.Tensor):
        *m, n = s_mn.shape
        s = s_mn.reshape(-1, n).to(self.J.dtype)
        return s[:, self.iu[0]] * s[:, self.iu[1]], m      # (M, N(N-1)/2)

    def lnPsi(self, s_mn: tc.Tensor) -> tc.Tensor:
        pairs, m = self._pairs(s_mn)
        return super().lnPsi(s_mn) + (pairs @ self.J).reshape(*m)

    def d_lnPsi(self, s_mn: tc.Tensor) -> tc.Tensor:
        pairs, m = self._pairs(s_mn)
        O_rbm = super().d_lnPsi(s_mn).reshape(-1, self.n_rbm)
        return tc.cat([O_rbm, pairs], dim=1).reshape(*m, self.Np)   # ∂ ln ψ / ∂J_ij = s_i s_j


def random_θ_jastrow(N: int, α: int, device: str, J_scale: float = 0.0) -> tc.Tensor:
    """RBM parameters as usual, Jastrow couplings zero (or small random with J_scale > 0)."""
    n_pairs = N * (N - 1) // 2
    J = J_scale * (tc.randn(n_pairs, dtype=tc.float64, device=device)
                   + 1j * tc.randn(n_pairs, dtype=tc.float64, device=device))
    return tc.cat([random_θ(N=N, α=α, device=device), J.to(tc.complex128)])
