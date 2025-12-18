# Bayesian Optimization Guide for PID Tuning

This guide explains how to use Bayesian Optimization (BO) to automatically tune the PID controller parameters in the supermarket refrigeration benchmark.

## Overview

The refactored `supermarket.py` now includes a complete BO interface that allows you to:

1. **Define tunable parameters**: Kp (proportional gain), τ_I (integral time), DB (dead band)
2. **Compute scalar loss**: Combines γ_con, γ_switch, γ_pow with configurable weights
3. **Handle hard constraints**: Adds penalties for safety violations
4. **Support multiple scenarios**: Both 2d-2c and 3d-3c
5. **Enable robustness**: Multiple seed evaluations for noisy objectives

## Installation

First, install the Ax platform for Bayesian Optimization:

```bash
pip install ax-platform
```

## Quick Start

### Option 1: Use the standalone script

```bash
python tune_pid_with_bo.py
```

This will run a complete BO session with sensible defaults.

### Option 2: Use the API directly

```python
from supermarket import evaluate_bo_objective, compute_reference_metrics, run_bo_tuning

# Quick example: evaluate a single configuration
ref_metrics = compute_reference_metrics(scenario='2d-2c', seed=42)
theta = (-100.0, 75.0, 0.15)  # (Kp, tau_I, DB)
loss = evaluate_bo_objective(theta, scenario='2d-2c', ref_metrics=ref_metrics, verbose=True)
print(f"Loss: {loss:.3f}")

# Full BO optimization
best_params, best_loss = run_bo_tuning(scenario='2d-2c', n_trials=50, n_seeds=1)
```

## Key Functions

### 1. `compute_reference_metrics()`

Computes baseline metrics for normalization.

```python
ref_metrics = compute_reference_metrics(
    scenario='2d-2c',    # '2d-2c' or '3d-3c'
    Kp=-75.0,            # Baseline Kp
    tau_I=50.0,          # Baseline tau_I
    DB=0.20,             # Baseline DB
    seed=42              # Random seed
)
# Returns: dict with 'gcon_day', 'gsw_day', 'gpow_day', 'gcon_night', 'gsw_night', 'gpow_night'
```

### 2. `evaluate_bo_objective()`

Main objective function for BO. Takes PID parameters, runs simulation, returns scalar loss.

```python
loss = evaluate_bo_objective(
    theta=(Kp, tau_I, DB),       # Parameters to evaluate
    scenario='2d-2c',             # Which scenario
    ref_metrics=ref_metrics,      # Reference for normalization
    alpha=0.5,                    # Day/night blend weight
    w_con=1.0,                    # Weight for constraint violation
    w_pow=0.3,                    # Weight for power
    w_sw=0.2,                     # Weight for switching
    lambda_pen=10.0,              # Hard violation penalty multiplier
    seeds=(42,),                  # Seeds for repeated evaluations
    enable_alternation=False,     # Controller features
    enable_hysteresis=False,
    enable_antiwindup=False,
    vfd_min_hold=True,
    verbose=False                 # Print details
)
```

**Loss formula:**
```
L(θ) = w_con·γ̃_con + w_sw·γ̃_switch + w_pow·γ̃_pow + λ·Φ(θ)
```

Where:
- γ̃ = normalized metrics (γ / γ_baseline)
- Φ(θ) = hard constraint violation count

### 3. `detect_hard_violations()`

Detects safety violations beyond soft bounds.

```python
penalty_count = detect_hard_violations(
    time, T_air, P_suc,
    T_min=1.5,                # Hard lower bound [°C]
    T_max=5.5,                # Hard upper bound [°C]
    P_max=2.05,               # Hard pressure bound [bar]
    violation_duration=60.0   # Minimum duration to count [s]
)
```

### 4. `run_bo_tuning()`

Complete BO optimization loop using Ax.

```python
best_params, best_loss = run_bo_tuning(
    scenario='2d-2c',      # Which scenario
    n_trials=50,           # Number of BO iterations
    n_seeds=1,             # Seeds per evaluation (1-3 typical)
    verbose=True           # Print progress
)
# Returns: (dict with Kp, tau_I, DB), best_loss
```

## Search Space

The default search space is:

| Parameter | Range | Scale | Description |
|-----------|-------|-------|-------------|
| **Kp** | [-150, -10] | Linear | Proportional gain (negative) |
| **τ_I** | [10, 200] s | **Log** | Integral time constant |
| **DB** | [0.05, 0.35] bar | Linear | Dead band |

The log scale for τ_I helps explore both fast (10s) and slow (200s) integral actions efficiently.

## Objective Function Details

### Metrics Computed

For each simulation run, the objective computes:

1. **Constraint violation** (γ_con): From Eq. (16) in the paper
   - Penalizes P_suc > P_ref and T_air outside [2, 5]°C
   - Units: [°C²]

2. **Switching rate** (γ_switch): From Eq. (19) in the paper
   - Counts compressor and valve switches
   - Units: [switches/s]

3. **Average power** (γ_pow): From Eq. (20) in the paper
   - Mean power consumption
   - Units: [kW]

### Normalization

All metrics are normalized by the baseline reference:
```
γ̃ = γ / γ_baseline
```

This ensures all metrics are on comparable scales (~1.0 for baseline).

### Hard Constraint Penalties

Beyond the soft metrics, the objective adds **large penalties** (λ=10 by default) for:

- **Temperature violations**: T_air < 1.5°C or > 5.5°C for >60s
- **Pressure violations**: P_suc > 2.05 bar for >60s

This creates a safety buffer around the soft bounds [2,5]°C and prevents BO from finding dangerous solutions.

### Day/Night Blending

Metrics are computed separately for day (0-7200s) and night (7200-14400s), then blended:
```
γ̃ = α·γ̃_day + (1-α)·γ̃_night
```

Default α=0.5 weights both periods equally.

## Configuration Tips

### For fast exploration (initial testing):
```python
best_params, _ = run_bo_tuning(
    scenario='2d-2c',
    n_trials=20,       # Quick
    n_seeds=1,         # Single seed
    verbose=True
)
```

### For robust optimization:
```python
best_params, _ = run_bo_tuning(
    scenario='2d-2c',
    n_trials=100,      # More exploration
    n_seeds=3,         # Multiple seeds → averaged loss
    verbose=True
)
```

### Custom objective weights:

```python
# Prioritize constraint satisfaction over power
loss = evaluate_bo_objective(
    theta,
    scenario='2d-2c',
    ref_metrics=ref_metrics,
    w_con=2.0,         # Double weight on safety
    w_pow=0.1,         # Less weight on power
    w_sw=0.5,          # Moderate weight on switching
    lambda_pen=20.0    # Strict hard penalties
)
```

### For noisy objectives (with load noise):

In `supermarket.py`, set:
```python
LOAD_NOISE_STD = 100.0  # J/s
```

Then use multiple seeds:
```python
best_params, _ = run_bo_tuning(
    scenario='2d-2c',
    n_trials=80,
    n_seeds=3,         # Average over 3 random seeds
    verbose=True
)
```

## Multi-Fidelity Speedup (Advanced)

To accelerate BO, you can implement a two-stage approach:

**Stage 1**: Quick exploration with reduced fidelity
- Shorter duration (7200s instead of 14400s)
- Coarser time step (dt=2s instead of 1s)
- Broad search with 30-50 trials

**Stage 2**: Refinement with full fidelity
- Full 14400s simulation
- Fine time step (dt=1s)
- Top 5-10 candidates from Stage 1

This isn't built-in but can be implemented by modifying `evaluate_bo_objective()` to accept `duration` and `dt` parameters.

## Interpreting Results

After BO completes, you'll get output like:

```
Best parameters found:
  Kp:    -98.234
  tau_I: 63.521 s
  DB:    0.142 bar
  Loss:  0.847
```

### What does the loss mean?

- **Loss ~1.0**: Similar to baseline
- **Loss <1.0**: Better than baseline (lower is better)
- **Loss >1.0**: Worse than baseline
- **Loss >>10**: Hard constraint violations occurred

### Validating the solution

1. **Check for violations**: Look at the verbose output during evaluation
2. **Run full simulation**: Use the optimized parameters in the main script
3. **Compare plots**: Visually inspect T_air, P_suc, switching behavior
4. **Test robustness**: Re-evaluate with different seeds

Example validation:
```python
# After getting best_params from BO
from supermarket import run_scenario, plot_results

# You'd need to modify run_scenario to accept custom PID params, or
# temporarily update ControlParams defaults in supermarket.py:
# K_p: float = best_params['Kp']
# tau_I: float = best_params['tau_I']
# DB: float = best_params['DB']

# Then run and plot
time, T_air, P_suc, P_ref, comp_cap, power, valve_states, comp_switches, _, n_cases = \
    run_scenario('2d-2c', duration=14400, seed=999)

fig = plot_results(time, T_air, P_suc, P_ref, comp_cap, power, n_cases, '2d-2c (Optimized)')
plt.show()
```

## Alternative Optimization Strategies

### 1. ε-Constraint Optimization

Instead of scalarization, minimize power subject to hard constraints:

```python
minimize: γ_pow
subject to: γ_con ≤ ε_con
            γ_switch ≤ ε_switch
```

Not directly supported, but can be implemented by returning `float('inf')` when constraints are violated.

### 2. Pareto Multi-Objective BO

Find the Pareto frontier over (γ_con, γ_pow, γ_switch) using `ax.service.managed_loop` with multiple objectives. Requires more Ax setup but provides a set of trade-off solutions.

### 3. Grid Search (baseline)

For comparison, you can implement a simple grid search:

```python
import itertools
from supermarket import evaluate_bo_objective, compute_reference_metrics

ref_metrics = compute_reference_metrics(scenario='2d-2c', seed=42)

Kp_grid = [-120, -100, -80, -60, -40]
tau_I_grid = [20, 35, 50, 75, 100, 150]
DB_grid = [0.10, 0.15, 0.20, 0.25, 0.30]

best_loss = float('inf')
best_theta = None

for Kp, tau_I, DB in itertools.product(Kp_grid, tau_I_grid, DB_grid):
    theta = (Kp, tau_I, DB)
    loss = evaluate_bo_objective(theta, scenario='2d-2c', ref_metrics=ref_metrics)
    if loss < best_loss:
        best_loss = loss
        best_theta = theta
    print(f"θ={theta} → loss={loss:.3f}")

print(f"\nBest: {best_theta} with loss {best_loss:.3f}")
```

This explores 5×6×5=150 configurations exhaustively but doesn't adapt like BO.

## Troubleshooting

**Problem**: `ImportError: No module named 'ax'`
- **Solution**: Install with `pip install ax-platform`

**Problem**: BO finds parameters that violate constraints
- **Solution**: Increase `lambda_pen` (e.g., to 20 or 50) to make violations more costly

**Problem**: Loss is always ~1.0, no improvement
- **Solution**: Try wider search ranges or different weight combinations (w_con, w_pow, w_sw)

**Problem**: BO is too slow
- **Solution**: 
  - Reduce `n_trials` (e.g., 20-30 for quick tests)
  - Use `n_seeds=1` instead of multiple seeds
  - Consider multi-fidelity approach (shorter simulations initially)

**Problem**: Results not reproducible
- **Solution**: Set fixed seeds in `evaluate_bo_objective(..., seeds=(42, 43, 44))`

## Advanced: Custom Objective Functions

You can create your own objective function for specialized needs:

```python
import numpy as np
from supermarket import RefrigerationSystem, calculate_performance

def custom_objective(Kp, tau_I, DB):
    """Custom objective focusing only on night period power"""
    
    # Setup
    system = RefrigerationSystem(n_cases=2, comp_capacities=[50.0, 50.0], 
                                  V_sl=0.08, has_vfd=False)
    system.controller.cp.K_p = Kp
    system.controller.cp.tau_I = tau_I
    system.controller.cp.DB = DB
    
    # Initial conditions
    system.cases[0].state = np.array([2.0, 0.0, 5.1, 0.0])
    system.cases[1].state = np.array([2.0, 0.0, 0.0, 1.0])
    system.P_suc = 1.40
    
    # Simulate
    dt, duration = 1.0, 14400
    n_steps = int(duration / dt)
    power_night = []
    
    system.set_day_mode()
    for i in range(n_steps):
        t = i * dt
        if t >= 7200:
            system.set_night_mode()
        
        _, _, pwr, _ = system.simulate_step(dt, t)
        
        if t >= 7200:  # Only care about night
            power_night.append(pwr)
    
    # Return average night power
    return np.mean(power_night) / 1000  # kW

# Use with Ax by wrapping in a function that Ax can call
```

## References

- **Original paper**: Larsen et al., "A Benchmark Test Case for Supermarket Refrigeration," ECC 2007
- **Ax documentation**: https://ax.dev/
- **Bayesian Optimization overview**: https://arxiv.org/abs/1807.02811

## Contact & Contributions

For questions or improvements to the BO interface, please open an issue or PR in the repository.

