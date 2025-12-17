"""
Controller Comparison Script

Runs all three MPC controllers (MS-SLSQP, MPPI, RTI-MPC) against the PID baseline
on a 4-hour scenario with day/night transition.

Usage:
    python compare_controllers.py
    
Output:
    controller_comparison.png - Trajectory comparison plot
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import time

from mpc_controller import optimize_full_trajectory
from mppi_controller import run_mppi_scenario
from rti_mpc import run_rti_scenario
import supermarket as sm

def run_comparison(duration=14400, dt=10.0):
    print("="*80)
    print("CONTROLLER COMPARISON: SLSQP vs MPPI vs RTI-MPC")
    print("="*80)
    
    # 1. SLSQP (proven robust baseline)
    print("\nRunning SLSQP (Single Shooting)...")
    res_slsqp = optimize_full_trajectory(duration=duration, dt=dt, max_iter=50)
    
    # 2. MPPI (Sampling based)
    print("\nRunning MPPI...")
    t_m, T_m, P_m, Pow_m = run_mppi_scenario(duration=duration, dt=dt)
    
    # 3. RTI-MPC (Linearized QP)
    print("\nRunning RTI-MPC...")
    t_r, T_r, P_r, Pow_r = run_rti_scenario(duration=duration, dt=dt)
    
    # 4. PID Baseline
    print("\nRunning PID Baseline...")
    t_b, T_b, P_b, _, _, Pow_b, _, _, _, _ = sm.run_scenario('2d-2c', duration=duration, dt=dt, seed=42)
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Temperature
    axes[0].plot(t_b/3600, T_b[:, 0], 'k--', alpha=0.3, label='PID Baseline')
    axes[0].plot(res_slsqp['time_opt']/3600, res_slsqp['T_air_opt'][:, 0], color='#6B5B95', label='MS-SLSQP')
    axes[0].plot(t_m/3600, T_m[:, 0], color='#E8734A', label='MPPI')
    axes[0].plot(t_r/3600, T_r[:, 0], color='#45B69C', label='RTI-MPC')
    axes[0].axhspan(2.0, 5.0, alpha=0.1, color='green')
    axes[0].set_ylabel('T_air [°C]')
    axes[0].legend(loc='upper right', ncol=4)
    axes[0].set_title('Controller Trajectory Comparison (Case 1)')
    
    # Pressure
    axes[1].plot(t_b/3600, P_b, 'k--', alpha=0.3)
    axes[1].plot(res_slsqp['time_opt']/3600, res_slsqp['P_suc_opt'], color='#6B5B95')
    axes[1].plot(t_m/3600, P_m, color='#E8734A')
    axes[1].plot(t_r/3600, P_r, color='#45B69C')
    P_ref_traj = np.where(t_b >= 7200.0, 1.7, 1.5)
    axes[1].plot(t_b/3600, P_ref_traj, 'r--', label='P_ref')
    axes[1].set_ylabel('P_suc [bar]')
    
    # Power
    axes[2].plot(t_b/3600, Pow_b/1000, 'k--', alpha=0.3)
    axes[2].plot(res_slsqp['time_opt']/3600, res_slsqp['power_opt']/1000, color='#6B5B95')
    axes[2].plot(t_m/3600, Pow_m/1000, color='#E8734A')
    axes[2].plot(t_r/3600, Pow_r/1000, color='#45B69C')
    axes[2].set_ylabel('Power [kW]')
    axes[2].set_xlabel('Time [hours]')
    
    plt.tight_layout()
    plt.savefig('controller_comparison.png', dpi=300)
    print("\nComparison plot saved to controller_comparison.png")

if __name__ == "__main__":
    run_comparison()
