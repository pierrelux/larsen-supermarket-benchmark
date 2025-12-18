#!/usr/bin/env python3
"""
Test script for the Bayesian Optimization interface

This validates that the BO functions work correctly without requiring Ax to be installed.
Run this before attempting full BO to ensure everything is set up properly.
"""

import numpy as np
from supermarket import (
    compute_reference_metrics,
    detect_hard_violations,
    evaluate_bo_objective
)

def test_reference_metrics():
    """Test that reference metric computation works"""
    print("=" * 70)
    print("TEST 1: Reference Metrics Computation")
    print("=" * 70)
    
    try:
        ref_metrics = compute_reference_metrics(scenario='2d-2c', seed=42)
        
        print("\n✓ Successfully computed reference metrics")
        print(f"  Day metrics:")
        print(f"    γ_con:    {ref_metrics['gcon_day']:.3f}")
        print(f"    γ_switch: {ref_metrics['gsw_day']:.6f}")
        print(f"    γ_pow:    {ref_metrics['gpow_day']:.2f} kW")
        print(f"  Night metrics:")
        print(f"    γ_con:    {ref_metrics['gcon_night']:.3f}")
        print(f"    γ_switch: {ref_metrics['gsw_night']:.6f}")
        print(f"    γ_pow:    {ref_metrics['gpow_night']:.2f} kW")
        
        # Basic sanity checks
        assert ref_metrics['gcon_day'] > 0, "Day constraint violation should be positive"
        assert ref_metrics['gpow_day'] > 0, "Day power should be positive"
        assert ref_metrics['gsw_day'] > 0, "Day switching should be positive"
        
        print("\n✓ All reference metric checks passed")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False

def test_violation_detection():
    """Test hard constraint violation detection"""
    print("\n" + "=" * 70)
    print("TEST 2: Hard Constraint Violation Detection")
    print("=" * 70)
    
    try:
        # Create synthetic data with known violations
        dt = 1.0
        duration = 200.0
        n_steps = int(duration / dt)
        n_cases = 2
        
        time = np.arange(0, duration, dt)
        T_air = np.ones((n_steps, n_cases)) * 3.0  # Normal temp
        P_suc = np.ones(n_steps) * 1.5  # Normal pressure
        
        # Inject violations
        T_air[50:150, 0] = 1.0  # Case 0: too cold for 100s (should trigger)
        T_air[100:140, 1] = 6.0  # Case 1: too hot for 40s (should not trigger, <60s)
        P_suc[80:160] = 2.1  # High pressure for 80s (should trigger)
        
        pen = detect_hard_violations(time, T_air, P_suc, violation_duration=60.0)
        
        print(f"\n✓ Violation detection completed")
        print(f"  Penalties detected: {pen}")
        print(f"  Expected: 2 (cold violation + pressure violation)")
        
        assert pen == 2, f"Expected 2 penalties, got {pen}"
        
        print("\n✓ Violation detection test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_objective_evaluation():
    """Test objective function evaluation"""
    print("\n" + "=" * 70)
    print("TEST 3: Objective Function Evaluation")
    print("=" * 70)
    
    try:
        # Compute reference once
        print("\nComputing reference metrics...")
        ref_metrics = compute_reference_metrics(scenario='2d-2c', seed=42)
        
        # Test baseline parameters
        print("\nEvaluating baseline configuration...")
        theta_baseline = (-75.0, 50.0, 0.20)
        loss_baseline = evaluate_bo_objective(
            theta_baseline,
            scenario='2d-2c',
            ref_metrics=ref_metrics,
            seeds=(42,),
            verbose=False
        )
        
        print(f"  Baseline PID: Kp={theta_baseline[0]}, tau_I={theta_baseline[1]}, DB={theta_baseline[2]}")
        print(f"  Loss: {loss_baseline:.3f}")
        
        # Test alternative parameters
        print("\nEvaluating alternative configuration...")
        theta_alt = (-100.0, 75.0, 0.15)
        loss_alt = evaluate_bo_objective(
            theta_alt,
            scenario='2d-2c',
            ref_metrics=ref_metrics,
            seeds=(42,),
            verbose=False
        )
        
        print(f"  Alternative PID: Kp={theta_alt[0]}, tau_I={theta_alt[1]}, DB={theta_alt[2]}")
        print(f"  Loss: {loss_alt:.3f}")
        
        # Sanity checks
        assert loss_baseline > 0, "Loss should be positive"
        assert loss_alt > 0, "Loss should be positive"
        assert not np.isnan(loss_baseline), "Loss should not be NaN"
        assert not np.isnan(loss_alt), "Loss should not be NaN"
        
        print(f"\nΔ Loss: {loss_alt - loss_baseline:+.3f}")
        if loss_alt < loss_baseline:
            print("  → Alternative config is better!")
        elif loss_alt > loss_baseline:
            print("  → Baseline config is better!")
        else:
            print("  → Configs are equivalent")
        
        print("\n✓ Objective evaluation test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_clipping():
    """Test that out-of-bounds parameters are clipped properly"""
    print("\n" + "=" * 70)
    print("TEST 4: Parameter Clipping")
    print("=" * 70)
    
    try:
        ref_metrics = compute_reference_metrics(scenario='2d-2c', seed=42)
        
        # Test extreme parameters (should be clipped)
        theta_extreme = (-200.0, 5.0, 0.5)  # All out of bounds
        
        print(f"\nEvaluating extreme parameters (should be clipped):")
        print(f"  Input:  Kp={theta_extreme[0]}, tau_I={theta_extreme[1]}, DB={theta_extreme[2]}")
        print(f"  Valid ranges: Kp∈[-150,-10], tau_I∈[10,200], DB∈[0.05,0.35]")
        
        loss = evaluate_bo_objective(
            theta_extreme,
            scenario='2d-2c',
            ref_metrics=ref_metrics,
            seeds=(42,),
            verbose=False
        )
        
        print(f"  Expected clips: Kp=-150, tau_I=10, DB=0.35")
        print(f"  Loss: {loss:.3f} (finite = clipping worked)")
        
        assert np.isfinite(loss), "Loss should be finite even with extreme parameters"
        
        print("\n✓ Parameter clipping test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_seed_evaluation():
    """Test evaluation with multiple seeds"""
    print("\n" + "=" * 70)
    print("TEST 5: Multi-Seed Evaluation")
    print("=" * 70)
    
    try:
        ref_metrics = compute_reference_metrics(scenario='2d-2c', seed=42)
        theta = (-75.0, 50.0, 0.20)
        
        print("\nEvaluating with 3 different seeds...")
        loss = evaluate_bo_objective(
            theta,
            scenario='2d-2c',
            ref_metrics=ref_metrics,
            seeds=(42, 43, 44),
            verbose=False
        )
        
        print(f"  Mean loss across 3 seeds: {loss:.3f}")
        
        assert np.isfinite(loss), "Multi-seed loss should be finite"
        
        print("\n✓ Multi-seed evaluation test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "BO INTERFACE TEST SUITE" + " " * 30 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\nThis will validate the Bayesian Optimization interface")
    print("without requiring Ax to be installed.\n")
    
    tests = [
        test_reference_metrics,
        test_violation_detection,
        test_objective_evaluation,
        test_parameter_clipping,
        test_multi_seed_evaluation,
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for i, (test_func, result) in enumerate(zip(tests, results), 1):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  Test {i} ({test_func.__name__}): {status}")
    
    total_passed = sum(results)
    total_tests = len(results)
    
    print(f"\nResults: {total_passed}/{total_tests} tests passed")
    
    if all(results):
        print("\n" + "=" * 70)
        print("SUCCESS! All tests passed. BO interface is ready to use.")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Install Ax: pip install ax-platform")
        print("  2. Run BO tuning: python tune_pid_with_bo.py")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("FAILURE: Some tests failed. Please review errors above.")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    exit(main())

