# Supermarket Refrigeration Benchmark

This repository implements the ECC'07 supermarket refrigeration benchmark from Larsen et al. (2007). It provides a PID baseline controller and three Model Predictive Control (MPC) algorithms for comparison.

## Benchmark Description

The benchmark models a supermarket refrigeration system with display cases and a compressor rack. The system is hybrid: temperatures and pressure evolve continuously, but actuators are discrete (valves are binary, compressors switch between fixed capacity levels).

Two scenarios from the paper are implemented:

| Scenario | Display Cases | Compressors | VFD |
|----------|---------------|-------------|-----|
| `2d-2c`  | 2             | 2 × 50%     | No  |
| `3d-3c`  | 3             | 40% + 2×30% | Yes |

The 3d-3c scenario includes a Variable Frequency Drive (VFD) on the first compressor, allowing continuous control in the [10%, 40%] range.

## Controllers

The repository includes four controllers:

1. **PID Baseline**: Hysteresis-based valve control with PI pressure regulation, as described in the benchmark paper.

2. **MS-SLSQP**: Multiple Shooting MPC using Sequential Quadratic Programming. Uses JAX for automatic differentiation.

3. **MPPI**: Model Predictive Path Integral control. A sampling-based method that does not require gradients.

4. **RTI-MPC**: Real-Time Iteration MPC with linearized dynamics and quadratic programming.

The MPC controllers output continuous controls that are discretized through a hysteresis layer. This addresses the mismatch between smooth optimization outputs and discrete actuator constraints.

| Variable   | Turn ON threshold | Turn OFF threshold |
|------------|-------------------|-------------------|
| Valves     | T_air > 4.5°C     | T_air < 3.5°C     |
| Compressor | P_suc > P_ref     | P_suc < P_ref - 0.3 bar |

## Results

![Controller Comparison](controller_comparison.png)

The figure shows a 4-hour simulation with a day/night transition at t = 2 hours. The MPC controllers maintain temperature within the [2°C, 5°C] bounds and keep suction pressure below the reference. The PID baseline exhibits temperature undershoot during cooling cycles and pressure constraint violations during the night period.

| Controller | Temperature Constraints | Pressure Constraints |
|------------|------------------------|---------------------|
| MS-SLSQP   | Satisfied              | Satisfied           |
| MPPI       | Satisfied              | Satisfied           |
| RTI-MPC    | Satisfied              | Satisfied           |
| PID        | Violated (undershoot)  | Violated            |

## Installation

```bash
git clone https://github.com/pierrelux/larsen-supermarket-benchmark.git
cd larsen-supermarket-benchmark

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

Run the controller comparison:

```bash
python compare_controllers.py
```

This executes a 4-hour simulation and generates `controller_comparison.png`.

To run the PID baseline alone:

```python
from supermarket import run_scenario

# 2d-2c scenario
time, T_air, P_suc, P_ref, comp_cap, power, valve_states, comp_switches, _, n_cases = \
    run_scenario('2d-2c', duration=14400, dt=1.0)

# 3d-3c scenario
time, T_air, P_suc, P_ref, comp_cap, power, valve_states, comp_switches, _, n_cases = \
    run_scenario('3d-3c', duration=14400, dt=1.0)
```

To use the MPC controllers:

```python
from mpc_controller import optimize_full_trajectory

results = optimize_full_trajectory(duration=14400, window_size=180, dt=10.0)
```

## File Structure

```
supermarket.py           # System dynamics (Python and JAX implementations)
mpc_controller.py        # MS-SLSQP controller
mppi_controller.py       # MPPI controller
rti_mpc.py               # RTI-MPC controller
compare_controllers.py   # Comparison script
requirements.txt         # Dependencies
extras/bo_tuning/        # Bayesian Optimization for PID tuning (optional)
```

The dynamics are implemented twice: a Python version with explicit control flow for the PID baseline, and a JAX version using `jnp.where` for differentiability. Both implementations produce identical results.

## Scenario Parameters

| Parameter     | Day Mode  | Night Mode |
|---------------|-----------|------------|
| Q_airload     | 3000 W    | 1800 W     |
| m_ref_const   | 0.2 kg/s  | 0.0 kg/s   |
| P_ref         | 1.5 bar   | 1.7 bar    |

The day/night transition occurs at t = 7200 s (2 hours).

## Dependencies

- Python 3.10+
- JAX (with float64 enabled)
- NumPy
- SciPy
- Matplotlib

## Reference

Larsen, L. F. S., Izadi-Zamanabadi, R., & Wisniewski, R. (2007). Supermarket Refrigeration System - Benchmark for Hybrid System Control. Proceedings of the European Control Conference 2007, Kos, Greece.

## License

MIT
