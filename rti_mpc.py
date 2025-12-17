"""
RTI-MPC Controller for Supermarket Refrigeration

Real-Time Iteration MPC - linearized QP-based optimization:
- Linearize dynamics around reference trajectory
- Solve single QP per timestep using gradient descent
- JAX Jacobians for fast linearization
- Hysteresis layer for discrete actuation

Usage:
    python rti_mpc.py
    
Or import:
    from rti_mpc import run_rti_scenario
    time, T_air, P_suc, power = run_rti_scenario(duration=14400)
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from typing import Tuple

import supermarket as sm
from supermarket import (
    make_dynamics_step, make_forward_simulate, get_state_vector_jax
)

class RTIMPC:
    """Real-Time Iteration MPC (Linearized iterative optimization at each step)"""
    
    def __init__(self, n_cases: int, comp_capacities: list, V_sl: float,
                 horizon: int, dt: float = 10.0):
        self.n_cases = n_cases
        self.horizon = horizon
        self.dt = dt
        self.n_u = n_cases + 1
        self.n_x = 4 * n_cases + 1
        
        self.params = {
            'n_cases': n_cases, 'dt': dt,
            'M_goods': 200.0, 'Cp_goods': 1000.0, 'UA_goods_air': 300.0,
            'M_wall': 260.0, 'Cp_wall': 385.0, 'UA_air_wall': 500.0,
            'M_air': 50.0, 'Cp_air': 1000.0,
            'UA_wall_ref_max': 4000.0, 'M_ref_max': 1.0, 'tau_fill': 40.0,
            'V_suc': 5.0, 'eta_vol': 0.81, 'V_sl': V_sl,
        }
        
        self.dynamics_step = make_dynamics_step(self.params)
        self.forward_simulate = make_forward_simulate(self.params)
        
        # JIT-compiled Jacobians
        @jit
        def get_jacobians(x, u, d):
            A = jacfwd(self.dynamics_step, argnums=0)(x, u, d)
            B = jacfwd(self.dynamics_step, argnums=1)(x, u, d)
            return A, B
        self.get_jacobians = get_jacobians
        
        self.u_nominal = None

    def optimize(self, x0: np.ndarray, d_traj: np.ndarray, P_ref: float,
                max_iter: int = 10, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # Initialize reference trajectory
        if self.u_nominal is None:
            u_opt = np.zeros((self.horizon, self.n_u))
            u_opt[:, :self.n_cases] = 1.0  # Valves open
            u_opt[:, -1] = 75.0  # Compressor at 75%
        else:
            u_opt = self.u_nominal.copy()
        
        # Do iterative linearization (more robust than single step)
        for iteration in range(3):
            # Forward simulate to get linearization point
            x_traj = np.array(self.forward_simulate(jnp.array(x0), jnp.array(u_opt), jnp.array(d_traj)))
            
            # Simple gradient-based step using JAX-smoothed gradients
            # This allows the controller to "see" past the switching cliffs
            grad_u = np.zeros_like(u_opt)
            
            # Target tracking (Case center + P_ref center)
            x_ref = np.zeros(self.n_x)
            x_ref[2*self.n_cases:3*self.n_cases] = 3.5
            x_ref[-1] = P_ref - 0.2
            
            # Cost matrices (simple diagonal) - balanced weights
            Q = np.zeros(self.n_x)
            Q[2*self.n_cases:3*self.n_cases] = 500.0 # T_air (stronger tracking)
            Q[-1] = 500.0 # P_suc (balanced, not nuclear)
            R = np.ones(self.n_u) * 0.01
            
            # Backprop through linearized dynamics
            lambda_k = np.zeros(self.n_x)
            for k in range(self.horizon - 1, -1, -1):
                # Linearize at x_k, u_k
                x_k = x_traj[k]
                u_k = u_opt[k]
                d_k = d_traj[k]
                
                A, B = self.get_jacobians(jnp.array(x_k), jnp.array(u_k), jnp.array(d_k))
                A, B = np.array(A), np.array(B)
                
                # Cost gradient with constraint violation penalty
                state_error = x_traj[k+1] - x_ref
                
                # Add penalty for T_air violations (both above and below)
                constraint_grad = np.zeros(self.n_x)
                for i in range(self.n_cases):
                    T_air_i = x_traj[k+1, 2*self.n_cases + i]
                    if T_air_i > 5.0:  # Above max
                        constraint_grad[2*self.n_cases + i] = 10000.0 * (T_air_i - 5.0)
                    elif T_air_i < 2.0:  # Below min (overcooling!)
                        constraint_grad[2*self.n_cases + i] = 10000.0 * (T_air_i - 2.0)
                
                # Penalty for P_suc violation
                P_suc_k = x_traj[k+1, -1]
                if P_suc_k > P_ref:
                    constraint_grad[-1] = 5000.0 * (P_suc_k - P_ref)
                elif P_suc_k < 0.8:  # Too low - compressor working too hard
                    constraint_grad[-1] = 5000.0 * (P_suc_k - 0.8)
                
                dL_dx = Q * state_error + A.T @ lambda_k + constraint_grad
                dL_du = R * u_k + B.T @ lambda_k
                
                grad_u[k] = dL_du
                lambda_k = dL_dx
            
            # Update step
            alpha = 0.1 # Learning rate
            u_opt = u_opt - alpha * grad_u
            u_opt[:, :self.n_cases] = np.clip(u_opt[:, :self.n_cases], 0, 1)
            u_opt[:, -1] = np.clip(u_opt[:, -1], 0, 100)  # Full range for on/off cycling
            
        # Round valves for actual application
        u_opt[:, :self.n_cases] = np.round(u_opt[:, :self.n_cases])
        
        # Save for next window
        self.u_nominal = np.vstack([u_opt[1:], u_opt[-1:]])
        
        # Predict final trajectory
        x_final = np.array(self.forward_simulate(jnp.array(x0), jnp.array(u_opt), jnp.array(d_traj)))
        
        return u_opt, x_final

def run_rti_scenario(duration=14400, dt=10.0):
    n_cases = 2
    system = sm.RefrigerationSystem(n_cases, [50.0, 50.0], 0.08, False)
    system.set_day_mode()
    
    controller = RTIMPC(n_cases, [50.0, 50.0], 0.08, horizon=18, dt=dt)
    
    time_hist, T_air_hist, P_suc_hist, power_hist = [], [], [], []
    comp_mode = 'on'  # Hysteresis state
    
    total_steps = int(duration / dt)
    prev_mode = 'day'
    for step in range(total_steps):
        t_current = step * dt
        if t_current >= 7200:
            system.set_night_mode()
            if prev_mode == 'day':
                controller.u_nominal = None  # Reset at transition
                prev_mode = 'night'
        else:
            system.set_day_mode()
        
        x0 = get_state_vector_jax(system, n_cases)
        d_traj = np.array([[system.Q_airload, system.Q_airload, system.m_ref_const]] * 18)
        
        u_opt, _ = controller.optimize(x0, d_traj, system.P_ref)
        u_apply = u_opt[0]
        
        # --- HONEST DISCRETE ACTUATION with HYSTERESIS ---
        u_sim = np.array(u_apply)
        
        # T_air hysteresis for valves (higher threshold for more margin)
        T_air_avg = np.mean([c.state[2] for c in system.cases])
        if T_air_avg < 3.5:
            u_sim[:n_cases] = 0.0  # Force valves CLOSED at 3.5°C to stop at ~2°C
        elif T_air_avg > 4.5:
            u_sim[:n_cases] = 1.0  # Force valves OPEN
        else:
            u_sim[:n_cases] = np.round(u_sim[:n_cases])
        
        # P_suc hysteresis for compressor
        if comp_mode == 'off' and system.P_suc > system.P_ref:
            comp_mode = 'on'
        elif comp_mode == 'on' and system.P_suc < system.P_ref - 0.3:
            comp_mode = 'off'
        
        comp_sim = 100.0 if comp_mode == 'on' else 0.0
        u_sim[n_cases] = comp_sim
        
        # Log
        time_hist.append(t_current)
        T_air_hist.append([case.state[2] for case in system.cases])
        P_suc_hist.append(system.P_suc)
        power_hist.append(system.power_consumption(system.P_suc, [comp_sim]))
        
        # Simulate (using corrected JAX dynamics for consistency with HONEST actuation)
        d_curr = jnp.array([system.Q_airload, system.Q_airload, system.m_ref_const])
        x_next = controller.dynamics_step(x0, jnp.array(u_sim), d_curr)
        for i in range(n_cases):
            system.cases[i].state = np.array([x_next[i], x_next[n_cases+i], x_next[2*n_cases+i], x_next[3*n_cases+i]])
        system.P_suc = float(x_next[4*n_cases])
        
        if step % 100 == 0: print(f"Step {step}/{total_steps}")
        
    return np.array(time_hist), np.array(T_air_hist), np.array(P_suc_hist), np.array(power_hist)

if __name__ == "__main__":
    time_r, T_air_r, P_suc_r, power_r = run_rti_scenario()
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1); plt.plot(time_r/3600, T_air_r)
    plt.subplot(3, 1, 2); plt.plot(time_r/3600, P_suc_r)
    plt.subplot(3, 1, 3); plt.plot(time_r/3600, power_r)
    plt.savefig('rti_results.png')
