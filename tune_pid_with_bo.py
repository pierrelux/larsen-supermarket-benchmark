#!/usr/bin/env python3
"""
Bayesian Optimization for PID Tuning
=====================================

This script demonstrates how to use the BO interface in supermarket.py
to automatically tune the PID controller parameters.

Requirements:
    pip install ax-platform

Usage:
    python tune_pid_with_bo.py

The script will:
1. Compute baseline reference metrics
2. Run Bayesian Optimization to find optimal (Kp, tau_I, DB)
3. Validate the best parameters found
4. Plot comparison between baseline and optimized controllers
"""

import numpy as np
import matplotlib.pyplot as plt
from supermarket import (
    run_scenario, 
    plot_results,
    compute_reference_metrics,
    evaluate_bo_objective,
    run_bo_tuning,
    calculate_performance
)

def main():
    print("=" * 80)
    print("BAYESIAN OPTIMIZATION FOR PID TUNING")
    print("Supermarket Refrigeration System Benchmark")
    print("=" * 80)
    
    # Configuration
    SCENARIO = '2d-2c'        # or '3d-3c'
    N_TRIALS = 30             # Number of BO iterations (increase to 50-100 for better results)
    N_SEEDS = 1               # Number of seeds per evaluation (use 2-3 for robustness)
    BASELINE_SEED = 42
    VALIDATION_SEED = 999
    
    print(f"\nConfiguration:")
    print(f"  Scenario:    {SCENARIO}")
    print(f"  BO trials:   {N_TRIALS}")
    print(f"  Seeds/eval:  {N_SEEDS}")
    print()
    
    # Step 1: Compute baseline reference metrics
    print("\n" + "=" * 80)
    print("STEP 1: Computing baseline reference metrics")
    print("=" * 80)
    print("\nRunning baseline simulation with default PID parameters...")
    print("  Kp = -75.0, tau_I = 50.0 s, DB = 0.20 bar")
    
    ref_metrics = compute_reference_metrics(scenario=SCENARIO, seed=BASELINE_SEED)
    
    print("\nBaseline metrics:")
    print(f"  Day:   γ_con={ref_metrics['gcon_day']:.3f}, "
          f"γ_switch={ref_metrics['gsw_day']:.6f}, "
          f"γ_pow={ref_metrics['gpow_day']:.2f} kW")
    print(f"  Night: γ_con={ref_metrics['gcon_night']:.3f}, "
          f"γ_switch={ref_metrics['gsw_night']:.6f}, "
          f"γ_pow={ref_metrics['gpow_night']:.2f} kW")
    
    # Step 2: Run Bayesian Optimization
    print("\n" + "=" * 80)
    print("STEP 2: Running Bayesian Optimization")
    print("=" * 80)
    print("\nSearch space:")
    print("  Kp:    [-150.0, -10.0]  (proportional gain)")
    print("  tau_I: [10.0, 200.0] s  (integral time constant, log scale)")
    print("  DB:    [0.05, 0.35] bar (dead band)")
    print()
    
    best_params, best_loss = run_bo_tuning(
        scenario=SCENARIO,
        n_trials=N_TRIALS,
        n_seeds=N_SEEDS,
        verbose=True
    )
    
    if best_params is None:
        print("\nERROR: Bayesian Optimization failed. Is ax-platform installed?")
        print("Install with: pip install ax-platform")
        return
    
    # Step 3: Compare baseline vs optimized
    print("\n" + "=" * 80)
    print("STEP 3: Comparing baseline vs optimized")
    print("=" * 80)
    
    # Evaluate baseline
    print("\nEvaluating baseline configuration...")
    theta_baseline = (-75.0, 50.0, 0.20)
    loss_baseline = evaluate_bo_objective(
        theta_baseline,
        scenario=SCENARIO,
        ref_metrics=ref_metrics,
        seeds=(VALIDATION_SEED,),
        verbose=False
    )
    
    # Evaluate optimized
    print("\nEvaluating optimized configuration...")
    theta_optimized = (best_params['Kp'], best_params['tau_I'], best_params['DB'])
    loss_optimized = evaluate_bo_objective(
        theta_optimized,
        scenario=SCENARIO,
        ref_metrics=ref_metrics,
        seeds=(VALIDATION_SEED,),
        verbose=True
    )
    
    print("\n" + "-" * 80)
    print("COMPARISON:")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Kp':<12} {'tau_I':<12} {'DB':<12} {'Loss':<12}")
    print("-" * 80)
    print(f"{'Baseline':<20} {theta_baseline[0]:<12.2f} {theta_baseline[1]:<12.2f} "
          f"{theta_baseline[2]:<12.3f} {loss_baseline:<12.3f}")
    print(f"{'Optimized':<20} {theta_optimized[0]:<12.2f} {theta_optimized[1]:<12.2f} "
          f"{theta_optimized[2]:<12.3f} {loss_optimized:<12.3f}")
    print("-" * 80)
    improvement = (loss_baseline - loss_optimized) / loss_baseline * 100
    print(f"Improvement: {improvement:+.1f}%")
    print("-" * 80)
    
    # Step 4: Visualize both controllers
    print("\n" + "=" * 80)
    print("STEP 4: Generating comparison plots")
    print("=" * 80)
    
    # This would require modifying run_scenario to accept custom PID params
    # For now, just show the concept
    print("\nTo fully validate, you can:")
    print("1. Update ControlParams defaults in supermarket.py to use optimized values")
    print("2. Re-run the main simulation")
    print("3. Compare the plots visually")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest parameters found:")
    print(f"  Kp    = {best_params['Kp']:.3f}")
    print(f"  tau_I = {best_params['tau_I']:.3f} s")
    print(f"  DB    = {best_params['DB']:.3f} bar")
    print(f"  Loss  = {best_loss:.3f}")
    print("\nTo use these parameters, update the ControlParams defaults in supermarket.py:")
    print(f"  K_p: float = {best_params['Kp']:.3f}  # Optimized by BO")
    print(f"  tau_I: float = {best_params['tau_I']:.3f}  # Optimized by BO")
    print(f"  DB: float = {best_params['DB']:.3f}  # Optimized by BO")
    print("=" * 80)

if __name__ == "__main__":
    main()

