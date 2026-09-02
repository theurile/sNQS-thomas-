# -*- coding: utf-8 -*-
"""Job 7: LPE exact sNQS, dt=0.01, tK=0.2."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_lpe_job import LPEJobConfig, run_lpe_job  # noqa: E402


if __name__ == "__main__":
    run_lpe_job(LPEJobConfig(dt=0.01, tK=0.20))
