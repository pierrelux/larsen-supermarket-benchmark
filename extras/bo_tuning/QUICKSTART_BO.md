# Bayesian Optimization Quickstart

Get started with PID tuning in 5 minutes!

## Step 1: Verify Installation (30 seconds)

```bash
cd /Users/pierre-luc.bacon/Documents/compressor_rack
source venv/bin/activate
python test_bo_interface.py
```

**Expected output**: ✓ All 5 tests passed

---

## Step 2: Manual Exploration (2-3 minutes)

```bash
python manual_pid_comparison.py
```

This compares 5 pre-configured PID settings and shows you:
- Which performs best
- Trade-offs between configurations
- Visual plots

**Customize**: Edit `CONFIGURATIONS` in the script to try your own parameters.

---

## Step 3: Install Ax (30 seconds)

```bash
pip install ax-platform
```

---

## Step 4: Run Bayesian Optimization (5-30 minutes)

```bash
python tune_pid_with_bo.py
```

**Quick test** (5 min): Edit script to set `N_TRIALS = 20`  
**Full optimization** (30 min): Use default `N_TRIALS = 30` or increase to 50-100

---

## What You Get

After BO completes, you'll see:

```
Best parameters found:
  Kp:    -98.234
  tau_I: 63.521 s
  DB:    0.142 bar
  Loss:  0.847
```

**Loss < 1.0** = Better than baseline ✓  
**Loss > 1.0** = Worse than baseline ✗

---

## Using Results

### Option 1: Quick validation

```python
from supermarket import evaluate_bo_objective, compute_reference_metrics

ref = compute_reference_metrics(scenario='2d-2c', seed=42)
theta = (-98.234, 63.521, 0.142)  # Your optimized params

loss = evaluate_bo_objective(theta, scenario='2d-2c', 
                              ref_metrics=ref, verbose=True)
```

### Option 2: Full simulation

Edit `supermarket.py`, update `ControlParams`:

```python
@dataclass
class ControlParams:
    """Parameters for traditional control"""
    T_air_min: float = 2.0
    T_air_max: float = 5.0
    K_p: float = -98.234        # ← Updated
    tau_I: float = 63.521       # ← Updated
    DB: float = 0.142           # ← Updated
```

Then run:
```bash
python supermarket.py
```

Compare the plots with baseline!

---

## Customizing the Objective

Want to emphasize different goals? Edit objective weights:

```python
from supermarket import evaluate_bo_objective, compute_reference_metrics

ref = compute_reference_metrics(scenario='2d-2c', seed=42)

# Prioritize power reduction over switching
loss = evaluate_bo_objective(
    theta=(-80.0, 60.0, 0.15),
    scenario='2d-2c',
    ref_metrics=ref,
    w_con=1.0,      # Constraint violation (keep high for safety)
    w_pow=1.0,      # Power (increase to emphasize energy)
    w_sw=0.05,      # Switching (decrease if not important)
    lambda_pen=10.0 # Hard violation penalty
)
```

Then run BO with these custom weights (requires modifying `run_bo_tuning()` or calling Ax directly).

---

## Troubleshooting

### "Command not found: python"
→ Activate venv: `source venv/bin/activate`

### "ImportError: No module named 'ax'"
→ Install: `pip install ax-platform`

### "BO finds unsafe parameters (loss >> 10)"
→ Increase penalty: In objective, use `lambda_pen=20.0` or higher

### "Loss always ~1.0, no improvement"
→ Try wider search ranges or different weight combinations

### "BO is too slow"
→ Reduce `n_trials` (try 20 for quick tests)  
→ Use `n_seeds=1` instead of multiple seeds

---

## More Information

- **Detailed guide**: `BO_TUNING_GUIDE.md`
- **What changed**: `CHANGELOG_BO.md`
- **File reference**: `BO_FILES_SUMMARY.md`
- **Main README**: `README.md`

---

## API Quick Reference

```python
from supermarket import (
    compute_reference_metrics,    # Get baseline
    evaluate_bo_objective,         # Evaluate PID config
    run_bo_tuning                  # Full BO optimization
)

# 1. Compute baseline
ref = compute_reference_metrics(scenario='2d-2c', seed=42)

# 2. Evaluate single config
theta = (-100.0, 75.0, 0.15)  # (Kp, tau_I, DB)
loss = evaluate_bo_objective(theta, scenario='2d-2c', ref_metrics=ref)

# 3. Run full BO
best_params, best_loss = run_bo_tuning(scenario='2d-2c', n_trials=50)
```

---

## Scenarios

| Scenario | Display Cases | Compressors | VFD? |
|----------|--------------|-------------|------|
| `'2d-2c'` | 2 | 2×50% | No |
| `'3d-3c'` | 3 | 40%+30%+30% | Yes (first) |

---

**Ready to optimize? Start with Step 1 above! 🚀**

