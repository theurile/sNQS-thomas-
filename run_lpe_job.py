# -*- coding: utf-8 -*-
"""Run one exact-backend LPE sNQS job and save the result as HDF5."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch as tc
from quante.basicfun import save_hdf5

from model import TIM
from rbm import random_θ
from sampler import random_samples
from snqs import sNQS_rbm
from utils import get_LPE_coeffs, get_LPE_time_grid, get_g_qt
from vmc import VMC

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "lpe_grid"


@dataclass(frozen=True)
class LPEJobConfig:
    dt: float
    tK: float
    t0: float = 0.0
    Lx: int = 10
    Ly: int = 1
    alpha: int = 5
    Q: int = 4
    lpe_order: int = 4
    epochs: int = 2000
    lr: float = 1e-3
    basis_type: str = "fourier"
    scheme: str = "taylor"
    node_type: str = "real"
    optimizer_name: str = "adamw"
    backend: str = "exact"
    objective: str = "link_fidelity"
    J: float = -1.0
    hx: float = -0.3
    hz: float = -1.0
    M: int = 5000
    batch: int = 1
    vmc_steps: int = 100
    vmc_lr: float = 1.0e-2
    seed: int = 1234
    loss_log_interval: int = 100
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_file: str = "N10_lpe_order4_epochs2000_adamw.h5"

    @property
    def N(self) -> int:
        return self.Lx * self.Ly


def _slug(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def group_name(config: LPEJobConfig) -> str:
    return f"dt={config.dt:g}/tK={config.tK:g}"


def output_path(config: LPEJobConfig) -> Path:
    return Path(config.output_dir) / (f"figures{config.Q}/" + (f"dt={config.dt:g}/tK={config.tK:g}_" + config.output_file))


def job_label(config: LPEJobConfig) -> str:
    if config.scheme == 'lpe':
        return f"lpe_dt{_slug(config.dt)}_tK{_slug(config.tK)}"
    elif config.scheme =='taylor':
        return f"taylor_dt{_slug(config.dt)}_tK{_slug(config.tK)}"


def _n10_ed_benchmark() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ts_exact = np.arange(0.0, 2.1, 0.1)
    sx_exact = np.array([
        1.0, 0.96273264, 0.85715787, 0.70076436, 0.51896288,
        0.34027325, 0.19100578, 0.09050201, 0.04787478, 0.06086643,
        0.11699507, 0.19668197, 0.2776524, 0.33966117, 0.36855907,
        0.35888529, 0.31450154, 0.24720419, 0.17367351, 0.11146064,
        0.07490269,
    ])
    sz_exact = np.array([
        0.0, 0.00177, 0.00673126, 0.01391918, 0.02197988,
        0.02949107, 0.03530452, 0.03883094, 0.04019972, 0.04025321,
        0.04037221, 0.04216846, 0.0471106, 0.05616646, 0.06954257,
        0.086582, 0.10584876, 0.12538797, 0.14311282, 0.15724053,
        0.16668596,
    ])
    return ts_exact, sx_exact, sz_exact, -0.3


def _has_n10_benchmark(config: LPEJobConfig) -> bool:
    return (
        config.N == 10
        and np.isclose(config.J, -1.0)
        and np.isclose(config.hx, -0.3)
        and np.isclose(config.hz, -0.3)
    )


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)


def _to_numpy(value: tc.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _interval_loss_values(losses_by_time: np.ndarray, phy_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    interval_means = []
    interval_maxes = []
    for start, stop in zip(phy_idx[:-1], phy_idx[1:]):
        interval_block = losses_by_time[:, start + 1 : stop + 1]
        interval_means.append(interval_block.mean(axis=1))
        interval_maxes.append(interval_block.max(axis=1))
    if not interval_means:
        empty = np.empty((losses_by_time.shape[0], 0), dtype=losses_by_time.dtype)
        return empty, empty
    return np.stack(interval_means, axis=1), np.stack(interval_maxes, axis=1)


def build_hdf5_data(
    config: LPEJobConfig,
    *,
    theta_initial: tc.Tensor,
    theta_jq: tc.Tensor,
    t_nodes: tc.Tensor,
    a_ms: np.ndarray,
    a_links: tc.Tensor,
    phy_idx: list[int],
    losses: np.ndarray,
    losses_by_time: np.ndarray,
    E: list[float],
    Sx: list[float],
    Sz: list[float],
) -> dict:
    phy_idx_arr = np.asarray(phy_idx, dtype=np.int64)
    t_nodes_np = _to_numpy(t_nodes)
    a_links_np = _to_numpy(a_links)
    t_physical = t_nodes_np[phy_idx_arr]
    interval_mean, interval_max = _interval_loss_values(losses_by_time, phy_idx_arr)

    E_all = np.asarray(E, dtype=np.float64)
    Sx_all = np.asarray(Sx, dtype=np.float64)
    Sz_all = np.asarray(Sz, dtype=np.float64)
    normalizer = float(config.N)

    return {
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "job_label": job_label(config),
            "group": group_name(config),
        },
        "time": {
            "t_nodes": t_nodes_np,
            "t_nodes_real": t_nodes_np.real,
            "t_nodes_imag": t_nodes_np.imag,
            "t_physical": t_physical,
            "t_physical_real": t_physical.real,
            "phy_idx": phy_idx_arr,
            "a_ms": np.asarray(a_ms),
            "a_links": a_links_np,
        },
        "observables": {
            "E_all": E_all,
            "Sx_all": Sx_all,
            "Sz_all": Sz_all,
            "E_per_site_all": E_all / normalizer,
            "Sx_per_site_all": Sx_all / normalizer,
            "Sz_per_site_all": Sz_all / normalizer,
            "E_physical": E_all[phy_idx_arr],
            "Sx_physical": Sx_all[phy_idx_arr],
            "Sz_physical": Sz_all[phy_idx_arr],
            "E_per_site_physical": E_all[phy_idx_arr] / normalizer,
            "Sx_per_site_physical": Sx_all[phy_idx_arr] / normalizer,
            "Sz_per_site_physical": Sz_all[phy_idx_arr] / normalizer,
        },
        "training": {
            "loss": losses,
            "loss_value_by_lpe_node": losses_by_time,
            "loss_value_by_physical_time": losses_by_time[:, phy_idx_arr],
            "loss_value_by_interval_mean": interval_mean,
            "loss_value_by_interval_max": interval_max,
            "final_loss_value_by_lpe_node": losses_by_time[-1],
            "final_loss_value_by_physical_time": losses_by_time[-1, phy_idx_arr],
            "final_loss_value_by_interval_mean": interval_mean[-1],
            "final_loss_value_by_interval_max": interval_max[-1],
        },
        "parameters": {
            "theta_initial": _to_numpy(theta_initial),
            "theta_jq": _to_numpy(theta_jq),
        },
    }


def plot_lpe_result(data: dict, config: LPEJobConfig) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(config.output_dir) / f"figures{config.Q}"
    figures_dir.mkdir(parents=True, exist_ok=True)

    label = job_label(config)
    t_physical = np.asarray(data["time"]["t_physical_real"], dtype=np.float64)
    loss = np.asarray(data["training"]["loss"], dtype=np.float64)
    final_loss_physical = np.maximum(
        np.asarray(data["training"]["final_loss_value_by_physical_time"], dtype=np.float64),
        1.0e-16,
    )
    epochs = np.arange(1, loss.shape[0] + 1)

    observables = data["observables"]
    fig = plt.figure(figsize=(14.0, 1.9))
    axes = [fig.add_subplot(1, 4, idx + 1) for idx in range(4)]

    # this displays exact dynamics for any given TFIM Hamiltonians
    from utils import exact_dynamics
    t_max = 2.0
    ts_exact, sx_exact, sz_exact, energy_exact = exact_dynamics(config.Lx, config.Ly, config.J,
                                                                config.hx, config.hz,
                                                                t_max,
                                                                dt_out=0.1, dt_int=0.01, order=12, device="cpu")
    '''
    if n10_benchmark:
        ts_exact, sx_exact, sz_exact, energy_exact = _n10_ed_benchmark()
    else:
        ts_exact = sx_exact = sz_exact = None
        energy_exact = None
    '''

    axes[0].plot(t_physical, observables["Sx_per_site_physical"], "-", label="sNQS LPE exact")
    axes[0].plot(ts_exact, sx_exact, ".", label="ED")
    axes[0].set_xlim(0.0, t_max)
    #else:
     #   axes[0].set_xlim(float(t_physical.min()), float(t_physical.max()))
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$\langle \sigma_x \rangle$")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].legend()

    axes[1].plot(t_physical, observables["Sz_per_site_physical"], "-", label="sNQS LPE exact")
    axes[1].plot(ts_exact, sz_exact, ".", label="ED" )
    axes[1].set_xlim(0.0, t_max)
    # else:
      #  axes[1].set_xlim(float(t_physical.min()), float(t_physical.max()))
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$\langle \sigma_z \rangle$")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].legend()

    axes[2].plot(t_physical, observables["E_per_site_physical"], "-", label="sNQS LPE exact")
    #if n10_benchmark:
    axes[2].axhline(y=energy_exact[0], color="k", linestyle="--", label="Exact")
    axes[2].set_xlim(0.0, t_max)
    axes[2].set_ylim(min(float(np.min(observables["E_per_site_physical"])), energy_exact[0]) - 0.01, -0.2)
    #else:
        #axes[2].set_xlim(float(t_physical.min()), float(t_physical.max()))
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Energy per site")
    axes[2].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[2].legend()

    axes[3].plot(epochs, np.maximum(loss, 1.0e-16), ".-")
    axes[3].set_xlim(0, max(config.epochs, int(epochs[-1])))
    axes[3].set_xlabel("Training step")
    axes[3].set_ylabel("Loss")
    axes[3].set_yscale("log")

    fig.tight_layout()
    summary_path = figures_dir / f"{label}_summary.png"
    fig.savefig(summary_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig_loss, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(t_physical, final_loss_physical, ".-", linewidth=1.2, markersize=2)
    ax.set_xlim(0.0, t_max)
    #else:
     #   ax.set_xlim(float(t_physical.min()), float(t_physical.max()))
    ax.set_xlabel("Physical time")
    ax.set_ylabel("Final local loss value")
    ax.set_yscale("log")
    ax.set_title("Final epoch local loss on physical time points")
    fig_loss.tight_layout()
    loss_path = figures_dir / f"{label}_loss_values.png"
    fig_loss.savefig(loss_path, dpi=220, bbox_inches="tight")
    plt.close(fig_loss)

    return summary_path, loss_path


def run_lpe_job(config, *, dry_run=False, theta_jq0_override=None, omega_q0_override=None, psi_initial_override=None):
    filename = output_path(config)
    group = group_name(config)
    print(f"job={job_label(config)}")
    print(f"output={filename}::{group}")
    print(
        "parameters: "
        f"N={config.N}, LPE order={config.lpe_order}, epochs={config.epochs}, "
        f"dt={config.dt:g}, tK={config.tK:g}, optimizer={config.optimizer_name}"
    )
    if dry_run:
        return filename, group

    _set_seed(config.seed)
    device = "cuda" if tc.cuda.is_available() else "cpu"
    model = TIM(J=config.J, hx=config.hx, hz=config.hz)

    print("-" * 20)
    print("Getting initial state...")


    if psi_initial_override is not None:
        psi_initial = psi_initial_override
    else:
        theta_rand = random_θ(N=config.N, α=config.alpha, device=device)
        S_rand = random_samples(config.M, config.N, device=device)
        vmc = VMC(theta_rand, config.Lx, config.Ly, config.alpha, model=TIM(0.0, -1.0, 0.0))
        psi_initial, S = vmc.train(
            S_rand,
            config.batch,
            steps=config.vmc_steps,
            lr=config.vmc_lr,
            log_interval=config.vmc_steps,
        )


    if theta_jq0_override is not None:
        theta_jq0 = theta_jq0_override
    else:
        theta_jq0 = tc.zeros((psi_initial.θ.numel(), config.Q), dtype=tc.complex128, device=device)
        theta_jq0[:, 0] = psi_initial.θ.detach().clone()

    if config.scheme == 'lpe':
        a_ms = get_LPE_coeffs(order=config.lpe_order)
        t_nodes, a_links, phy_idx = get_LPE_time_grid(
            config.t0, config.tK, dt=config.dt, a_ms=a_ms, device=device, node_type=config.node_type,
        )
    else:
        t_nodes = tc.arange(config.t0, config.tK + 0.5 * config.dt, config.dt, device=device, dtype=tc.float64)
        phy_idx = [0, 0, 0, 0]
        a_ms, a_links = tc.empty(5), tc.empty(5)

    g_qt, omega_q = get_g_qt(t_nodes, config.Q, device, basis_type=config.basis_type, model=model, config=config)
    if config.basis_type == "fourier" and omega_q0_override is not None:
        print("Previous omega detected, overriding window initialization.. ")
        omega_q = tc.as_tensor(omega_q0_override, device=device).detach().real.to(tc.float64)
        g_qt = tc.cat([
            g_qt[:1],  # constant mode
            tc.exp(-1j * omega_q.to(tc.complex128)[:, None] * t_nodes.real.to(tc.complex128)[None, :]),
        ], dim=0)

    alpha = 0.3
    g_t0 = g_qt[:, 0]  # basis values at the new window's t0
    theta_jq0[:, 1:] *= alpha  # shrink the inherited oscillating parts
    # re-adjust the constant column so ϑ(t0) still equals the previous end point exactly
    theta_jq0[:, 0] = (theta_jq0 @ g_t0 - theta_jq0[:, 1:] @ g_t0[1:]) / g_t0[0]

    snqs = sNQS_rbm(
        theta_jq0,
        g_qt,
        config.Lx,
        config.Ly,
        config.alpha,
        config.dt,
        model,
        omega_q=omega_q,
        omega_trainable=True,
        t_nodes=t_nodes,
        backend=config.backend,
        scheme="taylor",
        a_links=a_links,
        phy_idx=phy_idx,
    )

    print("-" * 20)
    print(f"Running sNQS_rbm with backend={config.backend}, objective={config.objective}...")
    print(f"a_ms: {a_ms}")
    print(f"t_nodes: {_to_numpy(t_nodes)}")
    theta_jq, Ss, losses, losses_by_time, ψfinal = snqs.train(
        psi_initial,
        Sini=None if config.backend == "exact" else S_rand,
        batch=config.batch,
        steps=config.epochs,
        lr=config.lr,
        log_interval=config.loss_log_interval,
        objective=config.objective,
        return_time_losses=True,
        optimizer_name=config.optimizer_name,
    )

    E, Sx, Sz = snqs.expectation_value(Ss, batch=20 * config.batch)
    if config.scheme == 'lpe':
        data = build_hdf5_data(
            config,
            theta_initial=psi_initial.θ.detach().clone(),
            theta_jq=theta_jq,
            t_nodes=t_nodes,
            a_ms=a_ms,
            a_links=a_links,
            phy_idx=phy_idx,
            losses=losses,
            losses_by_time=losses_by_time,
            E=E,
            Sx=Sx,
            Sz=Sz,
        )
    else:
        data = build_hdf5_data(
            config,
            theta_initial=psi_initial.θ.detach().clone(),
            theta_jq=theta_jq,
            t_nodes=t_nodes,
            a_ms=tc.tensor([]),
            a_links = tc.tensor([]),
            phy_idx = list(range(len(t_nodes))),
            losses=losses,
            losses_by_time=losses_by_time,
            E=E,
            Sx=Sx,
            Sz=Sz,
        )
    filename.parent.mkdir(parents=True, exist_ok=True)
    save_hdf5(filename=str(filename), group=group, mode="a", data=data)
    summary_path, loss_path = plot_lpe_result(data, config)
    print(f"saved HDF5 group {filename}::{group}")
    print(f"saved summary figure {summary_path}")
    print(f"saved local-loss figure {loss_path}")
    return theta_jq, ψfinal, data, snqs.omega_q.detach().clone()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--tK", type=float, required=True)
    parser.add_argument("--n-windows", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=LPEJobConfig.epochs)
    parser.add_argument("--lpe-order", type=int, default=LPEJobConfig.lpe_order)
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default=LPEJobConfig.optimizer_name)
    parser.add_argument("--Q", type=int, default=LPEJobConfig.Q)
    parser.add_argument("--Lx", type=int, default=LPEJobConfig.Lx)
    parser.add_argument("--Ly", type=int, default=LPEJobConfig.Ly)
    parser.add_argument("--alpha", type=int, default=LPEJobConfig.alpha)
    parser.add_argument("--M", type=int, default=LPEJobConfig.M)
    parser.add_argument("--batch", type=int, default=LPEJobConfig.batch)
    parser.add_argument("--vmc-steps", type=int, default=LPEJobConfig.vmc_steps)
    parser.add_argument("--loss-log-interval", type=int, default=LPEJobConfig.loss_log_interval)
    parser.add_argument("--seed", type=int, default=LPEJobConfig.seed)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", default=LPEJobConfig.output_file)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LPEJobConfig(
        dt=args.dt,
        tK=args.tK,
        epochs=args.epochs,
        lpe_order=args.lpe_order,
        optimizer_name=args.optimizer,
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
    run_lpe_job(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
