# ECC'07 Supermarket Refrigeration Benchmark

Implementation of the ECC'07 supermarket refrigeration benchmark from Larsen et al. (2007). 

Note: not all details are present for a perfect reproduction of the figures shown in the paper. For example, the capacity quantization and unit dispatch policy (midpoint thresholds and optional lead/lag alternation), the exact switch-counting convention, the initial conditions (Psuc and case states), and the VFD min-capacity behavior are not fully specified in the paper and are therefore defined explicitly here, along with small numerical guards (anti-windup, dρ/dP floor, optional load noise).

Larsen, L. F. S., Izadi-Zamanabadi, R., & Wisniewski, R. (2007, July 2-5). Supermarket Refrigeration System - Benchmark for Hybrid System Control. *Proceedings of the European Control Conference 2007*, Kos, Greece. TuA03.5.

## Setup

```bash
# Clone repository
git clone git@github.com:pierrelux/larsen-supermarket-benchmark.git
cd larsen-supermarket-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib

# Run benchmark
python supermarket.py
```

## Usage

Default configuration runs clean baseline:
```bash
python supermarket.py
```

Enable baseline alignment features by editing flags in `supermarket.py`:
```python
ENABLE_ALTERNATION = True    # Lead/lag unit rotation
ENABLE_HYSTERESIS = True     # Per-unit hysteresis bands
LOAD_NOISE_STD = 100.0       # Load noise [J/s]
```

## PID Tuning with Bayesian Optimization

The codebase now includes a complete Bayesian Optimization interface for automatic PID parameter tuning.

### Quick Start

```bash
# Install Ax platform
pip install ax-platform

# Run BO tuning
python tune_pid_with_bo.py
```

### Features

- **Automatic parameter search**: Optimizes Kp, τ_I, and DB using Bayesian Optimization
- **Scalar objective**: Combines γ_con (constraint violation), γ_switch (switching rate), and γ_pow (power) with configurable weights
- **Hard constraint penalties**: Prevents unsafe solutions that violate temperature/pressure bounds
- **Multi-seed support**: Robust optimization under stochastic load noise
- **Both scenarios**: Works with 2d-2c and 3d-3c configurations

### API Example

```python
from supermarket import evaluate_bo_objective, compute_reference_metrics, run_bo_tuning

# Quick evaluation of custom PID parameters
ref_metrics = compute_reference_metrics(scenario='2d-2c', seed=42)
theta = (-100.0, 75.0, 0.15)  # (Kp, tau_I, DB)
loss = evaluate_bo_objective(theta, scenario='2d-2c', ref_metrics=ref_metrics)

# Full BO optimization (50 trials)
best_params, best_loss = run_bo_tuning(scenario='2d-2c', n_trials=50)
print(f"Optimal Kp={best_params['Kp']:.2f}, tau_I={best_params['tau_I']:.2f}, DB={best_params['DB']:.3f}")
```

See [`BO_TUNING_GUIDE.md`](BO_TUNING_GUIDE.md) for detailed documentation, configuration tips, and advanced usage.
