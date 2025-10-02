#!/usr/bin/env python3
"""
Manual PID Parameter Comparison

This script lets you manually compare different PID configurations
without needing Ax installed. Useful for:
- Quick what-if analysis
- Understanding the objective function
- Validating BO results

Usage:
    python manual_pid_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from supermarket import (
    compute_reference_metrics,
    evaluate_bo_objective,
    run_scenario,
    plot_results
)

# Define configurations to compare
CONFIGURATIONS = [
    # (Name, Kp, tau_I, DB)
    ("Paper Baseline", -75.0, 50.0, 0.20),
    ("Aggressive", -100.0, 30.0, 0.10),
    ("Conservative", -50.0, 100.0, 0.30),
    ("Fast Integral", -75.0, 25.0, 0.20),
    ("Slow Integral", -75.0, 150.0, 0.20),
]

SCENARIO = '2d-2c'  # or '3d-3c'
SEED = 42

def main():
    print("=" * 80)
    print("MANUAL PID PARAMETER COMPARISON")
    print("=" * 80)
    print(f"\nScenario: {SCENARIO}")
    print(f"Configurations to compare: {len(CONFIGURATIONS)}")
    print()
    
    # Compute reference metrics once
    print("Computing reference metrics...")
    ref_metrics = compute_reference_metrics(scenario=SCENARIO, seed=SEED)
    print(f"  Baseline: γ_con={ref_metrics['gcon_day']:.3f}, "
          f"γ_switch={ref_metrics['gsw_day']:.6f}, "
          f"γ_pow={ref_metrics['gpow_day']:.2f} kW")
    
    # Evaluate all configurations
    print("\n" + "=" * 80)
    print("EVALUATING CONFIGURATIONS")
    print("=" * 80)
    
    results = []
    for name, Kp, tau_I, DB in CONFIGURATIONS:
        print(f"\n[{name}]")
        print(f"  Parameters: Kp={Kp:.2f}, tau_I={tau_I:.2f}s, DB={DB:.3f} bar")
        
        theta = (Kp, tau_I, DB)
        loss = evaluate_bo_objective(
            theta,
            scenario=SCENARIO,
            ref_metrics=ref_metrics,
            seeds=(SEED,),
            verbose=True
        )
        
        results.append((name, Kp, tau_I, DB, loss))
    
    # Sort by loss
    results.sort(key=lambda x: x[4])
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS (sorted by loss)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Configuration':<20} {'Kp':<10} {'tau_I':<10} {'DB':<10} {'Loss':<10}")
    print("-" * 80)
    
    for rank, (name, Kp, tau_I, DB, loss) in enumerate(results, 1):
        marker = " ★" if rank == 1 else ""
        print(f"{rank:<6} {name:<20} {Kp:<10.2f} {tau_I:<10.2f} {DB:<10.3f} {loss:<10.3f}{marker}")
    
    print("-" * 80)
    
    best_name, best_Kp, best_tau_I, best_DB, best_loss = results[0]
    worst_name, worst_Kp, worst_tau_I, worst_DB, worst_loss = results[-1]
    
    print(f"\nBest configuration: {best_name}")
    print(f"  Kp={best_Kp:.2f}, tau_I={best_tau_I:.2f}s, DB={best_DB:.3f} bar")
    print(f"  Loss: {best_loss:.3f}")
    
    improvement = (worst_loss - best_loss) / worst_loss * 100
    print(f"\nImprovement over worst: {improvement:.1f}%")
    
    # Visualize trade-offs
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)
    
    # Create scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Kp vs tau_I, colored by loss
    losses = [r[4] for r in results]
    Kps = [r[1] for r in results]
    tau_Is = [r[2] for r in results]
    names = [r[0] for r in results]
    
    scatter = ax1.scatter(Kps, tau_Is, c=losses, s=200, cmap='RdYlGn_r', 
                         edgecolors='black', linewidths=2, alpha=0.8)
    
    for name, kp, ti in zip(names, Kps, tau_Is):
        ax1.annotate(name, (kp, ti), fontsize=8, ha='center', va='bottom')
    
    ax1.set_xlabel('Kp (Proportional Gain)', fontsize=12)
    ax1.set_ylabel('tau_I (Integral Time) [s]', fontsize=12)
    ax1.set_title('PID Parameter Space', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Loss (lower is better)', fontsize=10)
    
    # Plot 2: Bar chart of losses
    colors = ['green' if i == 0 else 'orange' if i < len(results)-1 else 'red' 
              for i in range(len(results))]
    
    ax2.barh(range(len(results)), losses, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(results)))
    ax2.set_yticklabels(names)
    ax2.set_xlabel('Loss', fontsize=12)
    ax2.set_title('Configuration Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, loss in enumerate(losses):
        ax2.text(loss + 0.5, i, f'{loss:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'pid_comparison_{SCENARIO}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Add your own configurations to CONFIGURATIONS in this script")
    print("  2. Run full simulations with the best parameters")
    print("  3. Use Bayesian Optimization to explore automatically:")
    print("     python tune_pid_with_bo.py")
    print("=" * 80)

if __name__ == "__main__":
    main()

