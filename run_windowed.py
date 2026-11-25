import math
from dataclasses import replace

from exact import tim_exact_observables, rbm_state_vector
from run_lpe_job import LPEJobConfig, run_lpe_job, parse_args, _set_seed, plot_lpe_result, _n10_ed_benchmark
from utils import basis_derivatives, match_theta_at_boundary, get_LPE_coeffs, get_LPE_time_grid, get_g_qt
from rbm import RBM

import numpy as np
import torch as tc
import matplotlib.pyplot as plt

from pathlib import Path
from quante.basicfun import save_hdf5

from model import TIM
from rbm import random_θ
from sampler import random_samples
from snqs import sNQS_rbm
from utils import get_LPE_coeffs, get_LPE_time_grid, get_g_qt
from vmc import VMC


def update_cumulative_data(data: dict, cumulative_data: dict):
    important = ['time', 'observables', 'training']
    for outer, inner_dict in data.items():
        if outer not in important:
            continue

        for inner, v in inner_dict.items():
            if outer not in cumulative_data:
                cumulative_data[outer] = {}

            if inner not in cumulative_data[outer]:
                cumulative_data[outer][inner] = np.array(v)
            else:
                cumulative_data[outer][inner] = np.concatenate((cumulative_data[outer][inner], v))


def run_windowed(base_config, tW: float, n_windows: int):
    """
    Consecutive windows [i*tW, (i+1)*tW]. Each window is warm-started from the previous
    window's θ_jq, final state and (for the Fourier basis) trained ω_q. Because g_qt is
    built on absolute times, θ_jq_prev evaluated on the new window's nodes is exactly the
    continuation of the previous parameter trajectory.
    """
    _set_seed(base_config.seed)
    cumulative_data = {}

    theta_prev, psi_prev, omega_q_prev = None, None, None
    i, t0, tK = 0, 0.0, 0.0  # defined even if n_windows == 0, for the final plot config below

    for i in range(n_windows):
        t0, tK = i * tW, (i + 1) * tW
        config = replace(base_config, t0=t0, tK=tK,
                         output_file=f"windowed_{base_config.output_file}_win{i}.h5")

        theta_jq, psi_final, data, omega_q = run_lpe_job(
            config,
            omega_q0_override=omega_q_prev,  # None on first window -> default frequency grid
            theta_jq0_override=theta_prev,
            psi_initial_override=psi_prev,
        )


        update_cumulative_data(data, cumulative_data)
        theta_prev, psi_prev = theta_jq, psi_final
        omega_q_prev = None if omega_q is None else tc.as_tensor(omega_q).detach().clone()

    cfg = replace(base_config, t0=t0, tK=tK,
                  output_file=f"windowed_{base_config.output_file}_Q{base_config.Q}_win{i}.h5")
    plot_lpe_result(cumulative_data, cfg)
    return theta_prev, psi_prev, cumulative_data



# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_numpy(value: tc.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def group_name(config: LPEJobConfig) -> str:
    return f"dt={config.dt:g}/tK={config.tK:g}"


def output_path(config: LPEJobConfig) -> Path:
    return Path(config.output_dir) / config.output_file


def _slug(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def job_label(config: LPEJobConfig) -> str:
    return f"lpe_dt{_slug(config.dt)}_tK{_slug(config.tK)}"


def _n10_ed_benchmark_hz0() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """N=10, J=-1, hx=-0.3, hz=0. OBC chain (TIM.bonds(Lx=10, Ly=1) convention),
    initial state |+x>^⊗N, site-averaged <σx>,<σz>."""
    ts_exact = np.arange(0.0, 2.1, 0.1)
    sx_exact = np.array([
        1.0, 0.96448027, 0.86354556, 0.71306099, 0.53630046,
        0.35983241, 0.20890346, 0.10318494, 0.0536608, 0.06119931,
        0.11701358, 0.20484834, 0.3044058, 0.39530611, 0.46080727,
        0.49059463, 0.48216755, 0.44065087, 0.37717724, 0.30625719,
        0.24272499,
    ])
    sz_exact = np.zeros_like(ts_exact)  # exactly 0 at machine precision -- Z2 symmetry unbroken at hz=0
    return ts_exact, sx_exact, sz_exact, 0.0

def main() -> None:
    args = parse_args()
    config = LPEJobConfig(
                dt=args.dt,
                tK=args.tK,
                n_windows=args.n_windows,
                hx=args.hx,
                hz=args.hz,
                J=args.J,
                epochs=args.epochs,
                lr=args.lr,
                lpe_order=args.lpe_order,
                optimizer_name=args.optimizer,
                basis_type=args.basis_type,
                omega_init=args.omega_init,
                omega_trainable=args.omega_train,
                network_type=args.network_type,
                node_type="real",
                scheme='taylor',
                objective=args.objective,
                Q=args.Q,
                Lx=args.Lx,
                Ly=args.Ly,
                alpha=args.alpha,
                M=args.M,
                batch=args.batch,
                vmc_steps=args.vmc_steps,
                loss_log_interval=args.loss_log_interval,
                seed=args.seed,
                output_dir=args.output_dir,
                output_file=args.output_file,
            )
        # run_lpe_untrained_diagnostic(config)
    run_windowed(config, args.tK, args.n_windows)
if __name__ == "__main__":
    main()