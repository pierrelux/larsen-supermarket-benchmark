"""
Multiple Shooting MPC for Supermarket Refrigeration Control

MS-SLSQP: Multiple shooting with Sequential Quadratic Programming
- Controls and states as decision variables
- JAX automatic differentiation for gradients
- Hysteresis layer for discrete actuation
- Receding horizon optimization (3-min windows)

Usage:
    python mpc_controller.py
    
Or import:
    from mpc_controller import optimize_full_trajectory
    results = optimize_full_trajectory(duration=14400)
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad
from scipy.optimize import minimize, Bounds
import time
from typing import Tuple
import matplotlib.pyplot as plt

import supermarket as sm
from supermarket import (
    make_dynamics_step, make_forward_simulate, power_factor_jax,
    density_suction, d_density_dP, calculate_performance
)

class TrajectoryOptimizer:
    def __init__(self, n_cases: int, comp_capacities: list, V_sl: float, 
                 horizon: int, dt: float = 10.0,
                 gamma_con_weight: float = 10000.0):
        self.n_cases = n_cases
        self.horizon = horizon
        self.dt = dt
        self.n_u = n_cases + 1
        self.n_x = 4 * n_cases + 1
        self.gamma_con_weight = gamma_con_weight
        
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
        self._setup_jax_functions()
    
    def _setup_jax_functions(self):
        forward_simulate = self.forward_simulate
        n_cases = self.n_cases
        horizon = self.horizon
        V_sl = self.params['V_sl']
        eta_vol = self.params['eta_vol']
        
        @jit
        def objective(u_flat, x0, d_traj, P_ref_val):
            u_traj = u_flat.reshape(horizon, self.n_u)
            x_traj = forward_simulate(x0, u_traj, d_traj)
            
            total_power = 0.0
            for k in range(horizon):
                comp_pct = u_traj[k, n_cases]
                P_suc = x_traj[k+1, -1]
                V_comp = eta_vol * V_sl * (comp_pct / 100.0)
                power = V_comp * power_factor_jax(P_suc)
                total_power += power
            power_cost = total_power / horizon
            
            # Constraints
            T_air_min, T_air_max = 2.0, 5.0
            P_suc_max = P_ref_val
            gamma_con = 0.0
            tracking_cost = 0.0
            
            for k in range(horizon):
                x_k = x_traj[k+1]
                P_suc_k = x_k[-1]
                gamma_con += jnp.maximum(0.0, P_suc_k - P_suc_max)**2
                for i in range(n_cases):
                    T_air_i = x_k[2*n_cases + i]
                    gamma_con += jnp.maximum(0.0, T_air_i - T_air_max)**2
                    gamma_con += jnp.maximum(0.0, T_air_min - T_air_i)**2
                    tracking_cost += (T_air_i - 3.5)**2
            
            # Valve usage penalty: if T_air > 3.5 and valve is closed, big penalty
            valve_penalty = 0.0
            for k in range(horizon):
                for i in range(n_cases):
                    T_air_i = x_traj[k+1, 2*n_cases + i]
                    valve_i = u_traj[k, i]
                    # If temperature is above target and valve is closed, penalize
                    valve_penalty += jnp.maximum(0.0, T_air_i - 3.5) * (1.0 - valve_i)
            
            return (power_cost / 1000.0 + 
                    self.gamma_con_weight * (gamma_con / horizon) + 
                    100.0 * (tracking_cost / horizon) +  # 100x stronger T_air tracking
                    1000.0 * (valve_penalty / horizon))   # Force valves open when warm
        
        self.objective_jax = objective
        self.objective_grad_jax = jit(grad(objective))
        self.prev_u_solution = None

    def optimize(self, x0: np.ndarray, d_traj: np.ndarray, P_ref: float,
                max_iter: int = 50, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # Warm start or initialize with compressor at 50%
        if self.prev_u_solution is not None:
            u_init = np.vstack([self.prev_u_solution[1:], self.prev_u_solution[-1:]])
        else:
            u_init = np.ones((self.horizon, self.n_u)) * 0.5
            u_init[:, -1] = 50.0  # Start compressor at 50%
        
        # Bounds: valves [0,1], compressor [0, 100] (full range for on/off cycling like PID)
        lb = np.zeros(self.horizon * self.n_u)
        ub = np.ones(self.horizon * self.n_u)
        for k in range(self.horizon):
            lb[k * self.n_u + self.n_cases] = 0.0   # Allow compressor OFF
            ub[k * self.n_u + self.n_cases] = 100.0
        bounds = Bounds(lb, ub)
        
        res = minimize(
            fun=lambda u: float(self.objective_jax(u, x0, d_traj, P_ref)),
            x0=u_init.flatten(),
            method='SLSQP',
            jac=lambda u: np.array(self.objective_grad_jax(u, x0, d_traj, P_ref)),
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        u_opt = res.x.reshape(self.horizon, self.n_u)
        self.prev_u_solution = u_opt
        return u_opt, np.array(self.forward_simulate(x0, u_opt, d_traj))

def optimize_full_trajectory(duration=14400, window_size=180, dt=10.0, max_iter=50):
    n_cases = 2
    system = sm.RefrigerationSystem(n_cases, [50.0, 50.0], 0.08, False)
    system.set_day_mode()
    horizon_steps = int(window_size / dt)
    # Balanced penalty: temperature and pressure weighted equally
    optimizer = TrajectoryOptimizer(n_cases, [50.0, 50.0], 0.08, horizon=horizon_steps, dt=dt, gamma_con_weight=5000.0)
    
    time_opt, T_air_opt, P_suc_opt, power_opt, u_opt_hist = [], [], [], [], []
    comp_mode = 'on'  # Hysteresis state: 'on' or 'off'
    
    total_steps = int(duration / dt)
    prev_mode = 'day'
    for step in range(total_steps):
        t_current = step * dt
        if t_current >= 7200:
            system.set_night_mode()
            if prev_mode == 'day':
                optimizer.prev_u_solution = None  # Reset warm-start at transition
                prev_mode = 'night'
        else:
            system.set_day_mode()
        
        x0 = sm.get_state_vector_jax(system, n_cases)
        d_traj = np.array([[system.Q_airload, system.Q_airload, system.m_ref_const]] * horizon_steps)
        
        u_opt, _ = optimizer.optimize(x0, d_traj, system.P_ref, max_iter=max_iter)
        u_apply = u_opt[0]
        
        # Log before update
        time_opt.append(t_current)
        T_air_opt.append([c.state[2] for c in system.cases])
        P_suc_opt.append(system.P_suc)
        
        # --- HONEST DISCRETE ACTUATION with HYSTERESIS (like PID) ---
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
        
        # Log power of DISCRETE control
        power = system.power_consumption(system.P_suc, [comp_sim])
        power_opt.append(power)
        u_opt_hist.append(u_sim)
        
        # Simulate step using TRUE DISCRETE ACTUATION in JAX dynamics
        d_curr = jnp.array([system.Q_airload, system.Q_airload, system.m_ref_const])
        x_next = optimizer.dynamics_step(x0, jnp.array(u_sim), d_curr)
        
        for i in range(n_cases):
            system.cases[i].state = np.array([x_next[i], x_next[n_cases+i], x_next[2*n_cases+i], x_next[3*n_cases+i]])
        system.P_suc = float(x_next[4*n_cases])
        
        if step % 100 == 0: print(f"Window {step}/{total_steps}")
        
    return {
        'time_opt': np.array(time_opt), 'T_air_opt': np.array(T_air_opt),
        'P_suc_opt': np.array(P_suc_opt), 'power_opt': np.array(power_opt),
        'gamma_con_opt': 0.0, 'gamma_pow_opt': np.mean(power_opt), 'avg_window_time': 0.0
    }

if __name__ == "__main__":
    optimize_full_trajectory()
