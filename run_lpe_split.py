# -*- coding: utf-8 -*-
"""Run split-interval exact-backend LPE sNQS with warm-started intervals."""

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

from run_lpe_job import _has_n10_benchmark, _n10_ed_benchmark, _slug, _to_numpy


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "lpe_split"


@dataclass(frozen=True)
class SplitLPEConfig:
    dt: float
    total_time: float
    interval_length: float = 0.1
    t0: float = 0.0
    Lx: int = 10
    Ly: int = 1
    alpha: int = 3
    Q: int = 4
    lpe_order: int = 4
    epochs: int = 500
    lr: float = 1.0e-3
    optimizer_name: str = "adamw"
    backend: str = "exact"
    objective: str = "link_fidelity"
    J: float = -1.0
    hx: float = -0.3
    hz: float = -0.3
    M: int = 500
    batch: int = 1
    vmc_steps: int = 100
    vmc_lr: float = 1.0e-2
    seed: int = 1234
    loss_log_interval: int = 100
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_file: str = "N10_split_lpe_order4_adamw.h5"

    @property
    def N(self) -> int:
        return self.Lx * self.Ly

    @property
    def num_intervals(self) -> int:
        n_intervals = int(round((self.total_time - self.t0) / self.interval_length))
        if abs(self.t0 + n_intervals * self.interval_length - self.total_time) > 1.0e-12:
            raise ValueError("total_time must lie on the interval_length grid.")
        return n_intervals


def split_group_name(config: SplitLPEConfig) -> str:
    return (
        f"dt={config.dt:g}/interval={config.interval_length:g}/"
        f"T={config.total_time:g}/epochs={config.epochs:d}"
    )


def split_job_label(config: SplitLPEConfig) -> str:
    return (
        f"split_dt{_slug(config.dt)}_interval{_slug(config.interval_length)}_"
        f"T{_slug(config.total_time)}_e{config.epochs:d}"
    )


def output_path(config: SplitLPEConfig) -> Path:
    return Path(config.output_dir) / config.output_file


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)


def _initial_theta_jq(psi_initial, Q: int, previous_theta_jq: tc.Tensor | None) -> tc.Tensor:
    if previous_theta_jq is None:
        theta_jq = tc.zeros((psi_initial.θ.numel(), Q), dtype=tc.complex128, device=psi_initial.device)
    else:
        theta_jq = previous_theta_jq.detach().clone()
    theta_jq[:, 0] = psi_initial.θ.detach().clone()
    return theta_jq


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


def _plot_split_result(data: dict, config: SplitLPEConfig) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(config.output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    label = split_job_label(config)

    time = data["aggregate"]["time_physical"]
    observables = data["aggregate"]["observables"]
    loss_steps = data["aggregate"]["training_step"]
    loss = data["aggregate"]["loss"]
    local_loss_time = data["aggregate"]["local_loss_time"]
    local_loss_value = np.maximum(data["aggregate"]["local_loss_value"], 1.0e-16)

    n10_benchmark = _has_n10_benchmark(config)
    if n10_benchmark:
        ts_exact, sx_exact, sz_exact, energy_exact = _n10_ed_benchmark()
    else:
        ts_exact = sx_exact = sz_exact = None
        energy_exact = None

    fig = plt.figure(figsize=(14.0, 1.9))
    axes = [fig.add_subplot(1, 4, idx + 1) for idx in range(4)]

    axes[0].plot(time, observables["Sx_per_site"], ".-", label="sNQS LPE split")
    if n10_benchmark:
        axes[0].plot(ts_exact, sx_exact, ".", label="ED")
        axes[0].set_xlim(0.0, 2.0)
    else:
        axes[0].set_xlim(float(time.min()), float(time.max()))
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$\langle \sigma_x \rangle$")
    axes[0].legend()

    axes[1].plot(time, observables["Sz_per_site"], ".-", label="sNQS LPE split")
    if n10_benchmark:
        axes[1].plot(ts_exact, sz_exact, ".", label="ED")
        axes[1].set_xlim(0.0, 2.0)
    else:
        axes[1].set_xlim(float(time.min()), float(time.max()))
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$\langle \sigma_z \rangle$")
    axes[1].legend()

    axes[2].plot(time, observables["E_per_site"], ".-", label="sNQS LPE split")
    if n10_benchmark:
        axes[2].axhline(y=energy_exact, color="k", linestyle="--", label="Exact")
        axes[2].set_xlim(0.0, 2.0)
        axes[2].set_ylim(min(float(observables["E_per_site"].min()), energy_exact) - 0.01, -0.2)
    else:
        axes[2].set_xlim(float(time.min()), float(time.max()))
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Energy per site")
    axes[2].legend()

    axes[3].plot(loss_steps, np.maximum(loss, 1.0e-16), ".-")
    axes[3].set_xlabel("Training step")
    axes[3].set_ylabel("Loss")
    axes[3].set_yscale("log")

    fig.tight_layout()
    summary_path = figures_dir / f"{label}_summary.png"
    fig.savefig(summary_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig_loss, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(local_loss_time, local_loss_value, ".-", linewidth=1.2, markersize=4.5)
    if n10_benchmark:
        ax.set_xlim(0.0, 2.0)
    else:
        ax.set_xlim(float(local_loss_time.min()), float(local_loss_time.max()))
    ax.set_xlabel("Physical time")
    ax.set_ylabel("Final local loss value")
    ax.set_yscale("log")
    ax.set_title("Final epoch local loss on physical time points")
    fig_loss.tight_layout()
    loss_path = figures_dir / f"{label}_loss_values.png"
    fig_loss.savefig(loss_path, dpi=220, bbox_inches="tight")
    plt.close(fig_loss)

    return summary_path, loss_path


def run_lpe_split(config: SplitLPEConfig, *, dry_run: bool = False) -> tuple[Path, str]:
    filename = output_path(config)
    group = split_group_name(config)
    print(f"job={split_job_label(config)}")
    print(f"output={filename}::{group}")
    print(
        "parameters: "
        f"N={config.N}, LPE order={config.lpe_order}, epochs={config.epochs}, "
        f"dt={config.dt:g}, interval={config.interval_length:g}, "
        f"T={config.total_time:g}, optimizer={config.optimizer_name}"
    )
    if dry_run:
        return filename, group

    _set_seed(config.seed)
    device = "cuda" if tc.cuda.is_available() else "cpu"
    model = TIM(J=config.J, hx=config.hx, hz=config.hz)

    print("-" * 20)
    print("Getting initial state...")
    theta_rand = random_θ(N=config.N, α=config.alpha, device=device)
    S_rand = random_samples(config.M, config.N, device=device)
    vmc = VMC(theta_rand, config.Lx, config.Ly, config.alpha, model=TIM(0.0, -1.0, 0.0))
    psi_current, _ = vmc.train(
        S_rand,
        config.batch,
        steps=config.vmc_steps,
        lr=config.vmc_lr,
        log_interval=config.vmc_steps,
    )

    a_ms = get_LPE_coeffs(order=config.lpe_order)
    previous_theta_jq = None
    interval_data = {}

    aggregate_time = []
    aggregate_E = []
    aggregate_Sx = []
    aggregate_Sz = []
    aggregate_loss = []
    aggregate_loss_step = []
    aggregate_local_loss_time = []
    aggregate_local_loss_value = []
    global_step_offset = 0

    for interval_idx in range(config.num_intervals):
        start_t = config.t0 + interval_idx * config.interval_length
        end_t = start_t + config.interval_length
        print("-" * 20)
        print(f"Interval {interval_idx + 1}/{config.num_intervals}: [{start_t:g}, {end_t:g}]")

        theta_jq0 = _initial_theta_jq(psi_current, config.Q, previous_theta_jq)
        t_nodes, a_links, phy_idx = get_LPE_time_grid(
            0.0,
            config.interval_length,
            dt=config.dt,
            a_ms=a_ms,
            device=device,
            node_type="coeff",
        )
        g_qt = get_g_qt(t_nodes, config.Q, device, basis_type="simple")
        snqs = sNQS_rbm(
            theta_jq0,
            g_qt,
            config.Lx,
            config.Ly,
            config.alpha,
            config.dt,
            model,
            backend=config.backend,
            scheme="lpe",
            a_links=a_links,
            phy_idx=phy_idx,
        )

        theta_jq, Ss, losses, losses_by_time, psi_final = snqs.train(
            psi_current,
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

        phy_idx_arr = np.asarray(phy_idx, dtype=np.int64)
        t_nodes_np = _to_numpy(t_nodes)
        t_physical_local = t_nodes_np.real[phy_idx_arr]
        t_physical_abs = start_t + t_physical_local
        E_arr = np.asarray(E, dtype=np.float64)
        Sx_arr = np.asarray(Sx, dtype=np.float64)
        Sz_arr = np.asarray(Sz, dtype=np.float64)
        interval_mean, interval_max = _interval_loss_values(losses_by_time, phy_idx_arr)

        keep = slice(None) if interval_idx == 0 else slice(1, None)
        aggregate_time.append(t_physical_abs[keep])
        aggregate_E.append(E_arr[phy_idx_arr][keep])
        aggregate_Sx.append(Sx_arr[phy_idx_arr][keep])
        aggregate_Sz.append(Sz_arr[phy_idx_arr][keep])
        aggregate_loss.append(losses)
        aggregate_loss_step.append(global_step_offset + np.arange(1, losses.shape[0] + 1))
        aggregate_local_loss_time.append(t_physical_abs[keep])
        aggregate_local_loss_value.append(losses_by_time[-1, phy_idx_arr][keep])

        interval_data[f"interval_{interval_idx:03d}"] = {
            "start_t": start_t,
            "end_t": end_t,
            "time": {
                "t_nodes_local": t_nodes_np,
                "t_nodes_local_real": t_nodes_np.real,
                "t_nodes_local_imag": t_nodes_np.imag,
                "t_physical": t_physical_abs,
                "phy_idx": phy_idx_arr,
                "a_ms": np.asarray(a_ms),
                "a_links": _to_numpy(a_links),
            },
            "observables": {
                "E_physical": E_arr[phy_idx_arr],
                "Sx_physical": Sx_arr[phy_idx_arr],
                "Sz_physical": Sz_arr[phy_idx_arr],
                "E_per_site_physical": E_arr[phy_idx_arr] / config.N,
                "Sx_per_site_physical": Sx_arr[phy_idx_arr] / config.N,
                "Sz_per_site_physical": Sz_arr[phy_idx_arr] / config.N,
            },
            "training": {
                "loss": losses,
                "loss_value_by_lpe_node": losses_by_time,
                "loss_value_by_physical_time": losses_by_time[:, phy_idx_arr],
                "loss_value_by_interval_mean": interval_mean,
                "loss_value_by_interval_max": interval_max,
                "final_loss_value_by_physical_time": losses_by_time[-1, phy_idx_arr],
            },
            "parameters": {
                "theta_jq": _to_numpy(theta_jq),
                "theta_initial": _to_numpy(psi_current.θ),
                "theta_final": _to_numpy(psi_final.θ),
            },
        }

        previous_theta_jq = theta_jq
        psi_current = psi_final
        global_step_offset += config.epochs

    time_all = np.concatenate(aggregate_time)
    E_all = np.concatenate(aggregate_E)
    Sx_all = np.concatenate(aggregate_Sx)
    Sz_all = np.concatenate(aggregate_Sz)
    loss_all = np.concatenate(aggregate_loss)
    loss_step_all = np.concatenate(aggregate_loss_step)
    local_loss_time = np.concatenate(aggregate_local_loss_time)
    local_loss_value = np.concatenate(aggregate_local_loss_value)

    data = {
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "job_label": split_job_label(config),
            "group": group,
            "num_intervals": config.num_intervals,
        },
        "aggregate": {
            "time_physical": time_all,
            "training_step": loss_step_all,
            "loss": loss_all,
            "local_loss_time": local_loss_time,
            "local_loss_value": local_loss_value,
            "observables": {
                "E": E_all,
                "Sx": Sx_all,
                "Sz": Sz_all,
                "E_per_site": E_all / config.N,
                "Sx_per_site": Sx_all / config.N,
                "Sz_per_site": Sz_all / config.N,
            },
        },
        "intervals": interval_data,
    }

    filename.parent.mkdir(parents=True, exist_ok=True)
    save_hdf5(filename=str(filename), group=group, mode="a", data=data)
    summary_path, loss_path = _plot_split_result(data, config)
    print(f"saved HDF5 group {filename}::{group}")
    print(f"saved summary figure {summary_path}")
    print(f"saved local-loss figure {loss_path}")
    return filename, group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--total-time", type=float, required=True)
    parser.add_argument("--interval-length", type=float, default=SplitLPEConfig.interval_length)
    parser.add_argument("--epochs", type=int, default=SplitLPEConfig.epochs)
    parser.add_argument("--lpe-order", type=int, default=SplitLPEConfig.lpe_order)
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default=SplitLPEConfig.optimizer_name)
    parser.add_argument("--Q", type=int, default=SplitLPEConfig.Q)
    parser.add_argument("--Lx", type=int, default=SplitLPEConfig.Lx)
    parser.add_argument("--Ly", type=int, default=SplitLPEConfig.Ly)
    parser.add_argument("--alpha", type=int, default=SplitLPEConfig.alpha)
    parser.add_argument("--M", type=int, default=SplitLPEConfig.M)
    parser.add_argument("--batch", type=int, default=SplitLPEConfig.batch)
    parser.add_argument("--vmc-steps", type=int, default=SplitLPEConfig.vmc_steps)
    parser.add_argument("--loss-log-interval", type=int, default=SplitLPEConfig.loss_log_interval)
    parser.add_argument("--seed", type=int, default=SplitLPEConfig.seed)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", default=SplitLPEConfig.output_file)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SplitLPEConfig(
        dt=args.dt,
        total_time=args.total_time,
        interval_length=args.interval_length,
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
    run_lpe_split(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
