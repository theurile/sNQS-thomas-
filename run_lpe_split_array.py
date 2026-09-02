# -*- coding: utf-8 -*-
"""Dispatch the 8-job split-interval LPE grid from local shell or NSCC PBS."""

from __future__ import annotations

import argparse
import os

from run_lpe_split import SplitLPEConfig, run_lpe_split


TOTAL_TIME = 2.0
EPOCHS = 2000
OUTPUT_FILE = "N10_split_lpe_order4_epochs2000_adamw.h5"

JOB_GRID: tuple[tuple[float, float], ...] = (
    (0.10, 0.10),
    (0.05, 0.10),
    (0.02, 0.10),
    (0.01, 0.10),
    (0.10, 0.20),
    (0.05, 0.20),
    (0.02, 0.20),
    (0.01, 0.20),
)


def _env_job_id() -> int | None:
    for name in ("PBS_ARRAY_INDEX", "PBS_ARRAYID", "SLURM_ARRAY_TASK_ID"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return None


def _print_grid() -> None:
    for idx, (dt, interval_length) in enumerate(JOB_GRID):
        print(
            f"{idx}: dt={dt:g}, interval_length={interval_length:g}, "
            f"T={TOTAL_TIME:g}, epochs={EPOCHS:d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        _print_grid()
        return

    job_id = args.job_id if args.job_id is not None else _env_job_id()
    if job_id is None:
        raise SystemExit("Provide --job-id or run inside a PBS/SLURM array job.")
    if job_id < 0 or job_id >= len(JOB_GRID):
        raise SystemExit(f"job id must be in [0, {len(JOB_GRID) - 1}], got {job_id}.")

    dt, interval_length = JOB_GRID[job_id]
    config = SplitLPEConfig(
        dt=dt,
        interval_length=interval_length,
        total_time=TOTAL_TIME,
        epochs=EPOCHS,
        output_file=OUTPUT_FILE,
    )
    run_lpe_split(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
