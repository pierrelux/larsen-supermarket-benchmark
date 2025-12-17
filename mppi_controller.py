"""
MPPI Controller for Supermarket Refrigeration

Model Predictive Path Integral (MPPI) - sampling-based trajectory optimization:
- 1024 parallel rollouts using JAX vmap
- Elite sampling (top-k selection)
- Hysteresis layer for discrete actuation
- Gradient-free: handles discrete switching naturally

Usage:
    python mppi_controller.py
    
Or import:
    from mppi_controller import run_mppi_scenario
    time, T_air, P_suc, power = run_mppi_scenario(duration=14400)
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple

import supermarket as sm
from supermarket import (
    make_dynamics_step, make_forward_simulate, power_factor_jax,
    calculate_performance, get_state_vector_jax
)

class MPPIController:
    """MPPI controller with JAX parallel rollouts"""
    
    def __init__(self, n_cases: int, comp_capacities: list, V_sl: float,
                 horizon: int, dt: float = 10.0,
                 n_samples: int = 512, temperature: float = 0.1,
                 noise_sigma_valve: float = 0.3, noise_sigma_comp: float = 15.0,
                 gamma_con_weight: float = 5000.0):
        self.n_cases = n_cases
        self.horizon = horizon
        self.dt = dt
        self.n_samples = n_samples
        self.temperature = temperature
        self.noise_sigma_valve = noise_sigma_valve
        self.noise_sigma_comp = noise_sigma_comp
        self.gamma_con_weight = gamma_con_weight
        
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
        
        # Nominal control (warm start)
        self.u_nominal = None
        
        # Random key for sampling
        self.rng_key = random.PRNGKey(42)
        
        self._setup_jax_functions()
    
    def _setup_jax_functions(self):
        """Setup JIT-compiled MPPI functions"""
        dynamics_step = self.dynamics_step
        forward_simulate = self.forward_simulate
        n_cases = self.n_cases
        n_x = self.n_x
        n_u = self.n_u
        horizon = self.horizon
        eta_vol = self.params['eta_vol']
        V_sl = self.params['V_sl']
        gamma_con_weight = self.gamma_con_weight
        
        @jit
        def rollout_cost(u_traj, x0, d_traj, P_ref):
            """Compute cost of a single trajectory"""
            x_traj = forward_simulate(x0, u_traj, d_traj)
            
            total_power = 0.0
            for k in range(horizon):
                comp_pct = u_traj[k, n_cases]
                P_suc = x_traj[k+1, -1]
                V_comp = eta_vol * V_sl * (comp_pct / 100.0)
                power = V_comp * power_factor_jax(P_suc)
                total_power += power
            power_cost = total_power / horizon
            
            # Constraint violation (gamma_con)
            T_air_min, T_air_max = 2.0, 5.0
            P_suc_max = P_ref
            
            gamma_con = 0.0
            tracking_cost = 0.0
            discrete_penalty = 0.0
            
            for k in range(horizon):
                x_k = x_traj[k+1]
                P_suc_k = x_k[-1]
                
                # Penalize violations
                gamma_con += jnp.maximum(0.0, P_suc_k - P_suc_max)**2
                for i in range(n_cases):
                    T_air_i = x_k[2*n_cases + i]
                    gamma_con += jnp.maximum(0.0, T_air_i - T_air_max)**2
                    gamma_con += jnp.maximum(0.0, T_air_min - T_air_i)**2
                    
                    # Tracking to center of range
                    tracking_cost += (T_air_i - 3.5)**2
                    
                    # Penalty for non-discrete valves: valve * (1-valve) is max at 0.5, 0 at 0 and 1
                    discrete_penalty += u_traj[k, i] * (1.0 - u_traj[k, i])
            
            # Total cost
            return (power_cost / 1000.0 + 
                    gamma_con_weight * (gamma_con / horizon) + 
                    1.0 * (tracking_cost / horizon) +
                    10.0 * (discrete_penalty / horizon))
        
        # Vectorize over samples
        self.batch_rollout_cost = jit(vmap(rollout_cost, in_axes=(0, None, None, None)))
        self.single_rollout_cost = jit(rollout_cost)
    
    def optimize(self, x0: np.ndarray, d_traj: np.ndarray, P_ref: float,
                 verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        x0_jax = jnp.array(x0)
        d_traj_jax = jnp.array(d_traj)
        P_ref_jax = jnp.array(P_ref)
        
        # Initialize nominal if not set
        if self.u_nominal is None:
            self.u_nominal = jnp.zeros((self.horizon, self.n_u))
            self.u_nominal = self.u_nominal.at[:, :self.n_cases].set(0.5)
            self.u_nominal = self.u_nominal.at[:, self.n_cases].set(50.0)
        
        # Sample perturbations
        self.rng_key, subkey = random.split(self.rng_key)
        noise_valve = random.normal(subkey, (self.n_samples, self.horizon, self.n_cases)) * self.noise_sigma_valve
        self.rng_key, subkey = random.split(self.rng_key)
        noise_comp = random.normal(subkey, (self.n_samples, self.horizon, 1)) * self.noise_sigma_comp
        noise = jnp.concatenate([noise_valve, noise_comp], axis=-1)
        
        # Generate samples (centered around current nominal)
        u_samples = self.u_nominal + noise
        # Clip valves to [0, 1], compressor to [0, 100] (full range for on/off cycling)
        valve_samples = jnp.clip(u_samples[:, :, :self.n_cases], 0.0, 1.0)
        comp_samples = jnp.clip(u_samples[:, :, self.n_cases:], 0.0, 100.0)
        u_samples = jnp.concatenate([valve_samples, comp_samples], axis=-1)
        
        # Compute costs
        costs = self.batch_rollout_cost(u_samples, x0_jax, d_traj_jax, P_ref_jax)
        
        # ELITE SELECTION: Pick the single best sample or tight average
        # This solves the "averaging to garbage" problem
        best_idx = jnp.argmin(costs)
        u_optimal = u_samples[best_idx]
        
        # Update nominal for next step (selection + slight smoothing)
        alpha = 0.8 # Smoothing factor
        self.u_nominal = alpha * jnp.concatenate([u_optimal[1:], u_optimal[-1:]], axis=0) + (1-alpha) * self.u_nominal
        
        x_optimal = np.array(self.forward_simulate(x0_jax, u_optimal, d_traj_jax))
        return np.array(u_optimal), x_optimal

def run_mppi_scenario(duration=14400, dt=10.0, window_size=180):
    n_cases = 2
    system = sm.RefrigerationSystem(n_cases, [50.0, 50.0], 0.08, False)
    system.set_day_mode()
    
    horizon_steps = int(window_size / dt)
    # Balanced penalty for constraints
    controller = MPPIController(n_cases, [50.0, 50.0], 0.08, horizon=horizon_steps, dt=dt, 
                                n_samples=1024, gamma_con_weight=5000.0)
    
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
        d_traj = np.array([[system.Q_airload, system.Q_airload, system.m_ref_const]] * horizon_steps)
        
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
        
        # Log before step
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
    time_m, T_air_m, P_suc_m, power_m = run_mppi_scenario()
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_m/3600, T_air_m)
    plt.subplot(3, 1, 2)
    plt.plot(time_m/3600, P_suc_m)
    plt.subplot(3, 1, 3)
    plt.plot(time_m/3600, power_m)
    plt.savefig('mppi_results.png')
