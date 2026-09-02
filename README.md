# sNQS_rbm: Smooth Neural Quantum States for Real-Time Evolution

This repository provides the implementation of the **s-NQS** (smooth Neural Quantum States) method. 

s-NQS introduces a continuous-time variational ansatz for real-time quantum dynamics using Chebyshev interpolation of neural network parameters. 
The method enables stable, global optimization of neural quantum states via Monte Carlo sampling.

## Features

- Real-time evolution of quantum many-body systems via NQS
- Smooth parametrization using Chebyshev basis with global optimization
- Monte Carlo sampling with Metropolis-Hastings algorithm
- Optimizer: AdamW with PyTorch backend

## Repository Structure

- `model.py` — Defines system Hamiltonians
- `rbm.py` — Restricted Boltzmann Machine architecture
- `sNQS_rbm.py` — s-NQS evolution with Chebyshev basis
- `sampler.py` — MCMC sampler implementation
- `utils.py` — Utility functions
- `vmc.py` — Variational Monte Carlo utilities
- `test_*.py` — Unit tests

## Dependencies

- Python 3.12+
- NumPy
- PyTorch
- Matplotlib
- Quante (https://github.com/zhuhaodavid/quante.git)

## Running the Code

To run one exact-backend LPE job and save the result with `quante.basicfun.save_hdf5`:

```bash
python run_lpe_job.py --dt 0.1 --tK 0.2
```

The default 8-job grid uses `N=10`, `LPE order=4`, `epochs=2000`, AdamW, and
`dt = 0.1, 0.05, 0.02, 0.01` crossed with `tK = 0.1, 0.2`.

Run one grid member locally:

```bash
python run_lpe_array.py --job-id 0
```

or run the corresponding small wrapper:

```bash
python jobs/run_lpe_dt0p1_tK0p1.py
```

On NSCC/PBS, submit the array from the repository root:

```bash
qsub submit_lpe_grid.pbs
```

The HDF5 output stores the global training curve as `training/loss`.
`training/loss_value_by_lpe_node` keeps the raw local loss on every LPE internal
node; use `training/loss_value_by_physical_time` or
`training/loss_value_by_interval_mean` when plotting one value per physical time
or physical interval.
