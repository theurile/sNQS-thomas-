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
    print(base_config.J, base_config.hx, base_config.hz)
    cumulative_data = {}

    theta_prev, psi_prev, omega_q_prev = None, None, None
    i, t0, tK = 0, 0.0, 0.0  # defined even if n_windows == 0, for the final plot config below

    for i in range(n_windows):
        print("new run", i)
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


def _fourier_basis(t, omega_q, const_row) -> tc.Tensor:
    """Evaluate the Fourier basis (row 0 = constant mode, rows 1.. = exp(-i ω_q t)) at times t
    with the given ω. Same convention as sNQS_rbm; used to continue with trained frequencies."""
    t = tc.as_tensor(t, device=omega_q.device)
    t = (t.real if t.is_complex() else t).to(tc.float64).reshape(-1)
    w = omega_q.to(tc.float64).reshape(-1)
    const = const_row.reshape(-1)[:1].to(tc.complex128).expand(1, t.numel())
    rows = tc.exp(-1j * w.to(tc.complex128)[:, None] * t.to(tc.complex128)[None, :])
    return tc.cat([const, rows], dim=0)


# ---------------------------------------------------------------------------
# Untrained-extrapolation diagnostic
# ---------------------------------------------------------------------------
def run_lpe_untrained_diagnostic(
        config,
        ed_benchmark_fn=_n10_ed_benchmark,
        *,
        dry_run=False,
        theta_jq0_override=None,
        psi_initial_override=None,
        omega_q0_override=None,
        extrapolation_factor: float = 4.0,
        plot: bool = True,
):
    """
    Same setup as run_lpe_job: builds the model, initial state, theta_jq0, time grid,
    and sNQS_rbm, then trains once on [t0, tK]. It then evaluates the SAME trained
    theta_jq / omega_q (no further optimization) on an extended grid past tK and
    compares against ED -- the s-NQS paper's Fig. 2a test.
    """
    filename = output_path(config)
    group = group_name(config)
    print(f"job={job_label(config)}  [UNTRAINED-EXTRAPOLATION DIAGNOSTIC]")
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

    S_rand = random_samples(config.M, config.N, device=device)
    if psi_initial_override is not None:
        psi_initial = psi_initial_override
    else:
        theta_rand = random_θ(N=config.N, α=config.alpha, device=device)
        print("vmc-ing ")
        vmc = VMC(theta_rand, config.Lx, config.Ly, config.alpha, model=TIM(0.0, -1.0, 0.0))
        psi_initial, _ = vmc.train(
            S_rand,
            config.batch,
            steps=config.vmc_steps,
            lr=config.vmc_lr,
            log_interval=config.vmc_steps,
        )
    if theta_jq0_override is not None:
        theta_jq0 = theta_jq0_override
    else:
        print("assigning theta ")
        theta_jq0 = tc.zeros((psi_initial.θ.numel(), config.Q), dtype=tc.complex128, device=device)
        theta_jq0[:, 0] = psi_initial.θ.detach().clone()

    print("getting coeffs ")
    if config.scheme == 'lpe':
        a_ms = get_LPE_coeffs(order=config.lpe_order)
        t_nodes, a_links, phy_idx = get_LPE_time_grid(
            config.t0, config.tK, dt=config.dt, a_ms=a_ms, device=device, node_type=config.node_type,
        )
    else:
        t_nodes = tc.arange(config.t0, config.tK + 0.5 * config.dt, config.dt, device=device, dtype=tc.float64)
        phy_idx = [0, 0, 0, 0]
        a_ms, a_links = tc.empty(5), tc.empty(5)

    # g_qt is built outside the object (absolute times, so warm starts are true continuations)
    g_qt, omega_q = get_g_qt(t_nodes, config.Q, device, basis_type=config.basis_type, model=model)
    if config.basis_type == "fourier" and omega_q0_override is not None:
        # continue from given (e.g. previously trained) frequencies: rebuild the Fourier rows with them
        omega_q = tc.as_tensor(omega_q0_override, device=device).detach()
        omega_q = (omega_q.real if omega_q.is_complex() else omega_q).to(tc.float64)
        g_qt = _fourier_basis(t_nodes, omega_q, g_qt[:1])
    if config.basis_type != "fourier":
        omega_q = None

    print("defining snqs model ")
    snqs = sNQS_rbm(
        theta_jq0,
        g_qt,
        config.Lx,
        config.Ly,
        config.alpha,
        config.dt,
        model,
        backend=config.backend,
        scheme=config.scheme,
        a_links=a_links,
        phy_idx=phy_idx,
        omega_q=omega_q,  # None -> non-Fourier basis
        omega_trainable=bool(getattr(config, "omega_trainable", False)) and omega_q is not None,
        n_fixed_rows=2 if config.basis_type == "fourier_linear" else 1,
        t_nodes=t_nodes,
    )

    print("-" * 20)
    print(f"Running sNQS_rbm with backend={config.backend}, objective={config.objective}...")
    print(f"a_ms: {a_ms}")
    print(f"t_nodes: {_to_numpy(t_nodes)}")
    if snqs.is_fourier:
        print(f"initial omega_q: {snqs.omega_q.detach().cpu().numpy().round(3)} "
              f"(trainable={snqs.omega_trainable})")
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

    # --- untrained-extrapolation diagnostic: reuse trained theta_jq/omega_q, no further optimization ---
    print("-" * 20)
    print(f"Evaluating trained parameters on an extended grid ({extrapolation_factor}x the window), "
          "no additional training...")

    t0, tK = float(config.t0), float(config.tK)
    T_span = tK - t0
    n_points = max(int(round(T_span * extrapolation_factor / float(config.dt))), 2) + 1
    t_ext = np.linspace(t0, t0 + T_span * extrapolation_factor, n_points, dtype=np.float64)  # absolute times

    if snqs.is_fourier:
        omega_q = snqs.omega_q.detach()  # trained (or fixed) frequencies
        g_qt_ext = _fourier_basis(t_ext, omega_q, snqs.g_qt[:1, :1].detach())
    else:
        omega_q = None
        print("WARNING: non-Fourier basis. get_g_qt maps the basis onto the grid it is given; for "
              "Chebyshev this rescales r(t) to the extended grid, which is NOT the trained basis. "
              "Evaluate the basis with the training window's [t0, tK] mapping for a valid test.")
        out = get_g_qt(t_ext, config.Q, device, basis_type=config.basis_type, model=model)
        g_qt_ext = out[0] if isinstance(out, tuple) else out

    theta_jt_ext = theta_jq.detach().to(g_qt_ext.dtype) @ g_qt_ext  # (Np, len(t_ext)) -- NO retraining

    sx_pred, sz_pred = [], []
    for k in range(len(t_ext)):
        psi_k = RBM(theta_jt_ext[:, k].contiguous(), snqs.N, snqs.α)
        psi_s = rbm_state_vector(psi_k, snqs.exact_basis)
        E, Sx, Sz = tim_exact_observables(
            psi_s, snqs.exact_basis, snqs.bonds,
            J=config.J, hx=config.hx, hz=config.hz,
            Ediag_s=snqs.exact_Ediag,
        )
        sx_pred.append(float(Sx.real) / snqs.N)  # site-averaged, matches ED benchmark convention
        sz_pred.append(float(Sz.real) / snqs.N)
    sx_pred, sz_pred = np.array(sx_pred), np.array(sz_pred)

    ts_ed, sx_ed, sz_ed, _ = ed_benchmark_fn()

    in_range = (ts_ed >= t_ext[0] - 1e-9) & (ts_ed <= t_ext[-1] + 1e-9)  # avoid np.interp clamping
    mask_train = in_range & (ts_ed >= t0 - 1e-9) & (ts_ed <= tK + 1e-9)
    mask_extrap = in_range & (ts_ed > tK + 1e-9)
    sx_pred_on_ed = np.interp(ts_ed, t_ext, sx_pred)
    sz_pred_on_ed = np.interp(ts_ed, t_ext, sz_pred)

    def mean_abs_err(pred, ed, mask):
        return float(np.mean(np.abs(pred[mask] - ed[mask]))) if mask.any() else float("nan")

    err_sx_train = mean_abs_err(sx_pred_on_ed, sx_ed, mask_train)
    err_sx_extrap = mean_abs_err(sx_pred_on_ed, sx_ed, mask_extrap)
    err_sz_train = mean_abs_err(sz_pred_on_ed, sz_ed, mask_train)
    err_sz_extrap = mean_abs_err(sz_pred_on_ed, sz_ed, mask_extrap)

    print(f"training window = [{t0:.4f}, {tK:.4f}]")
    if omega_q is not None:
        print(f"trained omega_q: {omega_q.cpu().numpy().round(3)}")
    print(f"  <sx> mean |err|  inside window: {err_sx_train:.4f}   outside window: {err_sx_extrap:.4f}")
    print(f"  <sz> mean |err|  inside window: {err_sz_train:.4f}   outside window: {err_sz_extrap:.4f}")
    if not np.isnan(err_sx_extrap) and err_sx_train > 0:
        print(f"  degradation ratio (sx): {err_sx_extrap / max(err_sx_train, 1e-12):.1f}x worse outside vs inside")

    diagnostic = {
        "t_ext": t_ext, "sx_pred": sx_pred, "sz_pred": sz_pred,
        "ts_ed": ts_ed, "sx_ed": sx_ed, "sz_ed": sz_ed,
        "t0": t0, "tK": tK, "omega_q": None if omega_q is None else omega_q.cpu().numpy(),
        "err_sx_train": err_sx_train, "err_sx_extrap": err_sx_extrap,
        "err_sz_train": err_sz_train, "err_sz_extrap": err_sz_extrap,
        "theta_jq": theta_jq, "losses": losses, "losses_by_time": losses_by_time,
    }

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for a, pred, ed, name in [(ax[0], sx_pred, sx_ed, "<sigma_x>"), (ax[1], sz_pred, sz_ed, "<sigma_z>")]:
            a.plot(t_ext, pred, '-', label="s-NQS (trained once, no retrain)")
            a.plot(ts_ed, ed, 'o', markersize=3, label="ED")
            a.axvspan(t0, tK, color='gray', alpha=0.12, label="training window")
            a.set_title(f"{name}  {job_label(config)}")
            a.set_xlabel("t")
            a.legend(fontsize=8)
        plt.tight_layout()
        diag_path = filename.parent / f"{filename.stem}_untrained_diagnostic.png"
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(diag_path, dpi=120)
        plt.show()
        print(f"saved untrained-diagnostic figure {diag_path}")

    return theta_jq, ψfinal, diagnostic


def main() -> None:
    args = parse_args()
    # run_lpe_untrained_diagnostic(config)
    for i in range(5, 13, 2):
        args.Q = i
        config = LPEJobConfig(
            dt=args.dt,
            tK=args.tK,
            epochs=5000,
            lr=8e-3,
            lpe_order=args.lpe_order,
            optimizer_name=args.optimizer,
            basis_type=args.basis_type,
            omega_init=args.omega_init,
            node_type="real",
            scheme='taylor',
            objective='link_fidelity',
            Q=args.Q,
            Lx=args.Lx,
            Ly=args.Ly,
            alpha=args.alpha,
            M=args.M,
            batch=args.batch,
            vmc_steps=100,
            loss_log_interval=args.loss_log_interval,
            seed=args.seed,
            output_dir=args.output_dir,
            output_file=args.output_file,
        )
        run_windowed(config, args.tK, args.n_windows)
if __name__ == "__main__":
    main()