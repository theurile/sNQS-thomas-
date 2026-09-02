# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2026-04-21 04:04:09
# @Last Modified by:   dzwang
# @Last Modified time: 2026-05-14 10:55:00

"""Default exact-backend LPE run for the current 8-job grid."""

from run_lpe_job import LPEJobConfig, run_lpe_job


def main() -> None:
    run_lpe_job(LPEJobConfig(dt=0.10, tK=0.20))


if __name__ == "__main__":
    main()
