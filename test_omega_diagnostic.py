import numpy as np
import torch as tc


@tc.no_grad()
def test_omega_sensitivity(
    model,
    ψini,
    Ss,
    batch: int,
    *,
    q_indices: list[int] | None = None,
    sweep_values: list[float] | None = None,
) -> dict:
    """
    Diagnostic: for each mode q in q_indices, sweep omega_q[q] over sweep_values
    (holding everything else fixed) and record the resulting loss.

    Interpretation:
      - loss roughly FLAT across the sweep  -> gauge freedom / the sign or magnitude
        of this mode doesn't actually matter to the fit; drift during training is
        harmless wandering along a flat direction, not a bug.
      - loss clearly BEST near the current trained value, worse elsewhere
        -> the optimizer is doing real work; wherever it's pushing omega_q is a
        genuine improvement, not an artifact.
      - loss WORSE at the current/trending value than at other values nearby
        -> something is wrong (bad gradient sign, wrong step size, chain-rule bug) --
        re-run the finite-difference gradient check on this mode.

    Returns a dict: {q_idx: {"values": [...], "losses": [...]}}
    """
    if q_indices is None:
        q_indices = list(range(model.omega_q.numel()))
    if sweep_values is None:
        current = model.omega_q.detach().cpu().numpy()
        spread = max(np.abs(current).max(), 1.0) * 1.5
        sweep_values = np.linspace(-spread, spread, 11).tolist()

    omega_snapshot = model.omega_q.detach().clone()
    ψs = model.ψs
    g_qt_current = model.g_qt

    results = {}
    for q_idx in q_indices:
        losses = []
        for val in sweep_values:
            model.omega_q = omega_snapshot.clone()
            model.omega_q[q_idx] = val
            g_qt_test = model.g_qt  # rebuilt from the perturbed omega_q

            if model.backend == "exact":
                _, _, loss_test = model.grad_jq_exact(ψini, ψs, g_qt_test)
            else:
                _, _, loss_test = model.grad_jq(ψini, ψs, Ss, batch, g_qt_test)
            losses.append(float(loss_test))

        results[q_idx] = {"values": list(sweep_values), "losses": losses}

        current_val = float(omega_snapshot[q_idx])
        best_idx = int(np.argmin(losses))
        best_val = sweep_values[best_idx]
        print(f"q={q_idx:2d}  current omega={current_val:+.3f}  "
              f"loss@current-nearest-grid vs loss@sweep-min(omega={best_val:+.3f})="
              f"{losses[np.argmin(np.abs(np.array(sweep_values)-current_val))]:.4g} vs {losses[best_idx]:.4g}  "
              f"flatness(max-min)={max(losses)-min(losses):.4g}")

    model.omega_q = omega_snapshot  # restore exactly, no side effects
    return results


def plot_omega_sensitivity(results: dict, title: str = ""):
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)
    for ax, (q_idx, data) in zip(axes[0], results.items()):
        ax.plot(data["values"], data["losses"], "o-")
        ax.set_title(f"omega_q[{q_idx}] sweep {title}")
        ax.set_xlabel("omega value")
        ax.set_ylabel("loss")
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/omega_sensitivity.png", dpi=120)
    plt.show()
