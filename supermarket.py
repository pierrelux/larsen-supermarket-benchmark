"""
Supermarket Refrigeration System Dynamics

This file contains BOTH implementations of the refrigeration system:

┌─────────────────────────────────────────────────────────────────┐
│ PART 1: Python Classes (lines 70-290)                           │
│ - RefrigerationSystem: Complete simulator with PID controller   │
│ - Used for: Baseline comparison, closed-loop simulation         │
│ - Style: Object-oriented, stateful                              │
├─────────────────────────────────────────────────────────────────┤
│ PART 2: JAX Functions (lines 291-400)                           │
│ - make_dynamics_step(): Fast gradient-compatible dynamics       │
│ - Used for: Trajectory optimization with autodiff               │
│ - Style: Functional, pure (no side effects)                     │
│ - Speed: 10-100x faster gradient computation                    │
└─────────────────────────────────────────────────────────────────┘

The dynamics are mathematically identical - JAX version uses functional
style (jnp.where instead of if/else) for automatic differentiation.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from dataclasses import dataclass
from typing import List

# ============================================================================
# REFRIGERANT PROPERTIES (R134a) - SHARED BY BOTH IMPLEMENTATIONS
# ============================================================================

def evaporation_temp(P):
    """Evaporation temperature as function of pressure [bar] -> [°C]"""
    return -4.3544 * P**2 + 29.2240 * P - 51.2005

def latent_heat(P):
    """Latent heat as function of pressure [bar] -> [J/kg]"""
    return (0.0217 * P**2 - 0.1704 * P + 2.2988) * 1e5

def density_suction(P):
    """Density in suction manifold [bar] -> [kg/m³]"""
    return 4.6073 * P + 0.3798

def d_density_dP(P):
    """Derivative of density w.r.t. pressure [bar] -> [kg/(m³·bar)]"""
    return -0.0329 * P**3 + 0.2161 * P**2 - 0.4742 * P + 5.4817

def power_factor(P):
    """ρ(h_oc - h_ic) for power calculation [bar] -> [J/m³]"""
    return (0.0265 * P**3 - 0.4346 * P**2 + 2.4923 * P + 1.2189) * 1e5

# JAX versions (same equations, JAX arrays)
@jit
def evaporation_temp_jax(P):
    return -4.3544 * P**2 + 29.2240 * P - 51.2005

@jit
def latent_heat_jax(P):
    return (0.0217 * P**2 - 0.1704 * P + 2.2988) * 1e5

@jit
def density_suction_jax(P):
    return 4.6073 * P + 0.3798

@jit
def d_density_dP_jax(P):
    return -0.0329 * P**3 + 0.2161 * P**2 - 0.4742 * P + 5.4817

@jit
def power_factor_jax(P):
    return (0.0265 * P**3 - 0.4346 * P**2 + 2.4923 * P + 1.2189) * 1e5


# ============================================================================
# PART 1: PYTHON IMPLEMENTATION (Reference Simulator + PID Baseline)
# ============================================================================

@dataclass
class DisplayCaseParams:
    """Parameters for a single display case"""
    M_goods: float = 200.0
    Cp_goods: float = 1000.0
    UA_goods_air: float = 300.0
    M_wall: float = 260.0
    Cp_wall: float = 385.0
    UA_air_wall: float = 500.0
    M_air: float = 50.0
    Cp_air: float = 1000.0
    UA_wall_ref_max: float = 4000.0
    M_ref_max: float = 1.0
    tau_fill: float = 40.0
    T_SH: float = 10.0

@dataclass
class ControlParams:
    """Parameters for traditional PID control"""
    T_air_min: float = 2.0
    T_air_max: float = 5.0
    K_p: float = -75.0
    tau_I: float = 50.0
    DB: float = 0.20

class DisplayCase:
    def __init__(self, params: DisplayCaseParams, case_id: int):
        self.p = params
        self.id = case_id
        self.state = np.array([2.0, 0.0, 5.0, 0.0])  # [T_goods, T_wall, T_air, M_ref]
        self.valve = 0
        
    def heat_transfer_coeff(self, M_ref):
        """Heat transfer coefficient as function of refrigerant mass"""
        return self.p.UA_wall_ref_max * M_ref / self.p.M_ref_max
    
    def dynamics(self, state, P_suc, valve, Q_airload, dt):
        """Compute and integrate one step for display case"""
        T_goods, T_wall, T_air, M_ref = state
        
        T_e = evaporation_temp(P_suc)
        
        Q_goods_air = self.p.UA_goods_air * (T_goods - T_air)
        Q_air_wall = self.p.UA_air_wall * (T_air - T_wall)
        UA_wall_ref = self.heat_transfer_coeff(max(0, M_ref))
        Q_e = UA_wall_ref * (T_wall - T_e)
        
        dT_goods = -Q_goods_air / (self.p.M_goods * self.p.Cp_goods)
        dT_wall = (Q_air_wall - Q_e) / (self.p.M_wall * self.p.Cp_wall)
        dT_air = (Q_goods_air + Q_airload - Q_air_wall) / (self.p.M_air * self.p.Cp_air)
        
        # Refrigerant mass derivative (exact discrete logic)
        if valve == 1:
            dM_ref = (self.p.M_ref_max - M_ref) / self.p.tau_fill
        elif M_ref > 0:
            dM_ref = -Q_e / latent_heat(P_suc)
        else:
            dM_ref = 0.0
            
        new_state = state + np.array([dT_goods, dT_wall, dT_air, dM_ref]) * dt
        new_state[3] = np.clip(new_state[3], 0, self.p.M_ref_max)
        
        return new_state, Q_e
    
    def mass_flow_out(self, Q_e, P_suc):
        """Mass flow out of evaporator to suction manifold"""
        if Q_e > 0 and self.state[3] > 0:
            return Q_e / latent_heat(P_suc)
        return 0.0

class TraditionalController:
    """PID controller with valve hysteresis (from benchmark paper)"""
    def __init__(self, control_params: ControlParams, n_cases: int, comp_capacities: List[float],
                 has_vfd: bool = False):
        self.cp = control_params
        self.n_cases = n_cases
        self.comp_capacities = comp_capacities
        self.integral = 0.0
        self.valve_states = [0] * n_cases
        self.prev_comp_on = [0.0] * len(comp_capacities)
        self.has_vfd = has_vfd  # True if first compressor has Variable Frequency Drive
        
    def valve_control(self, T_air, case_idx):
        """Hysteresis controller for inlet valve"""
        if T_air > self.cp.T_air_max:
            self.valve_states[case_idx] = 1
        elif T_air < self.cp.T_air_min:
            self.valve_states[case_idx] = 0
        return self.valve_states[case_idx]
    
    def pressure_control(self, P_suc, P_ref, dt):
        """PI controller with dead band for compressor capacity"""
        error = P_ref - P_suc
        
        if abs(error) > self.cp.DB:
            self.integral += error * dt
        
        u_PI = self.cp.K_p * error + (self.cp.K_p / self.cp.tau_I) * self.integral
        
        # Quantize to available compressor capacities
        comp_on = [0.0] * len(self.comp_capacities)
        
        # For VFD scenario (3d-3c): comp1 is continuous [10,40], others are discrete
        if self.has_vfd:
            if u_PI <= 0:
                pass  # All off
            elif u_PI < 10:
                comp_on[0] = 10  # Hold at minimum
            elif u_PI < self.comp_capacities[0]:
                comp_on[0] = u_PI  # Continuous control
            else:
                comp_on[0] = self.comp_capacities[0]  # VFD at max
                # Add discrete units with midpoint thresholds
                cumsum = self.comp_capacities[0]
                for i in range(1, len(self.comp_capacities)):
                    threshold = cumsum + self.comp_capacities[i] / 2.0
                    if u_PI >= threshold:
                        comp_on[i] = self.comp_capacities[i]
                        cumsum += self.comp_capacities[i]
                    else:
                        break
            return comp_on
        
        # Standard discrete quantization (2d-2c scenario)
        cumsum = 0
        for i in range(len(self.comp_capacities)):
            threshold = cumsum + self.comp_capacities[i] / 2.0
            if u_PI >= threshold:
                comp_on[i] = self.comp_capacities[i]
                cumsum += self.comp_capacities[i]
            else:
                break
        
        return comp_on

class RefrigerationSystem:
    """Complete refrigeration system with PID baseline"""
    def __init__(self, n_cases: int, comp_capacities: List[float], V_sl: float = 0.08, has_vfd: bool = False):
        self.n_cases = n_cases
        self.cases = [DisplayCase(DisplayCaseParams(), i) for i in range(n_cases)]
        self.P_suc = 1.40
        self.V_suc = 5.0
        self.V_sl = V_sl
        self.eta_vol = 0.81
        self.comp_capacities = comp_capacities
        self.has_vfd = has_vfd
        self.controller = TraditionalController(ControlParams(), n_cases, comp_capacities, has_vfd=has_vfd)
        
        # Scenario parameters
        self.Q_airload = 3000.0
        self.m_ref_const = 0.2
        self.P_ref = 1.50
        
        # Control timing
        self.comp_control_time = 0.0
        self.comp_sample_time = 60.0
        self.current_comp_on = [0.0] * len(comp_capacities)
        
    def set_day_mode(self):
        self.Q_airload = 3000.0
        self.m_ref_const = 0.2
        self.P_ref = 1.50
        
    def set_night_mode(self):
        self.Q_airload = 1800.0
        self.m_ref_const = 0.0
        self.P_ref = 1.70
    
    def volume_flow(self, comp_capacities):
        """Total volume flow from compressors [m³/s]"""
        total_capacity = sum(comp_capacities)
        return (total_capacity / 100.0) * self.eta_vol * self.V_sl
    
    def power_consumption(self, P_suc, comp_capacities):
        """Power consumption [W]"""
        V_comp = self.volume_flow(comp_capacities)
        return V_comp * power_factor(P_suc)
    
    def simulate_step(self, dt, current_time):
        """Simulate one time step with PID control"""
        # Valve control
        valves = []
        for i, case in enumerate(self.cases):
            valve = self.controller.valve_control(case.state[2], i)
            valves.append(valve)
            case.valve = valve
        
        # Update display cases
        m_in_suc = 0.0
        for case in self.cases:
            new_state, Q_e = case.dynamics(case.state, self.P_suc, case.valve, self.Q_airload, dt)
            case.state = new_state
            m_in_suc += case.mass_flow_out(Q_e, self.P_suc)
        
        # Compressor control (60 sec sample time)
        comp_switches = 0
        if current_time - self.comp_control_time >= self.comp_sample_time - 1e-6:
            new_comp_on = self.controller.pressure_control(self.P_suc, self.P_ref, self.comp_sample_time)
            
            for i in range(len(self.comp_capacities)):
                prev_on = self.controller.prev_comp_on[i] > 0.1
                curr_on = new_comp_on[i] > 0.1
                # For VFD: only count 0↔on transitions, not adjustments within [10,40]
                if self.has_vfd and i == 0:
                    if prev_on != curr_on:
                        comp_switches += 1
                else:
                    if prev_on != curr_on:
                        comp_switches += 1
            
            self.controller.prev_comp_on = new_comp_on[:]
            self.current_comp_on = new_comp_on
            self.comp_control_time = current_time
        
        comp_on = self.current_comp_on
        V_comp = self.volume_flow(comp_on)
        
        # Update suction pressure
        rho = density_suction(self.P_suc)
        drho_dP = max(1e-3, d_density_dP(self.P_suc))
        dP = (m_in_suc + self.m_ref_const - V_comp * rho) / (self.V_suc * drho_dP)
        self.P_suc += dP * dt
        self.P_suc = np.clip(self.P_suc, 0.8, 3.0)
        
        power = self.power_consumption(self.P_suc, comp_on)
        
        return valves, comp_on, power, comp_switches


# ============================================================================
# PART 2: JAX IMPLEMENTATION (For Fast Gradient-Based Optimization)
# ============================================================================

def make_dynamics_step(params):
    """Factory function to create JIT-compiled dynamics with static params
    
    This JAX version is mathematically identical to the Python DisplayCase.dynamics()
    above, but uses functional style for autodiff compatibility.
    """
    
    # Extract static parameters
    n_cases = int(params['n_cases'])
    dt = float(params['dt'])
    M_goods = float(params['M_goods'])
    Cp_goods = float(params['Cp_goods'])
    UA_goods_air = float(params['UA_goods_air'])
    M_wall = float(params['M_wall'])
    Cp_wall = float(params['Cp_wall'])
    UA_air_wall = float(params['UA_air_wall'])
    M_air = float(params['M_air'])
    Cp_air = float(params['Cp_air'])
    UA_wall_ref_max = float(params['UA_wall_ref_max'])
    M_ref_max = float(params['M_ref_max'])
    tau_fill = float(params['tau_fill'])
    V_suc = float(params['V_suc'])
    
    @jit
    def dynamics_step(x, u, d):
        """Single dynamics step
        
        Args:
            x: state [4n+1] = [T_goods, T_wall, T_air, M_ref, P_suc]
            u: control [n+1] = [valves, comp]
            d: disturbance [n+1] = [Q_airload, m_ref_const]
        
        Returns:
            x_next: next state [4n+1]
        """
        
        # Extract states
        T_goods = x[0:n_cases]
        T_wall = x[n_cases:2*n_cases]
        T_air = x[2*n_cases:3*n_cases]
        M_ref = x[3*n_cases:4*n_cases]
        P_suc = x[4*n_cases]
        
        # Extract controls
        valves = u[:n_cases]
        comp = u[-1]
        
        # Extract disturbances
        Q_airload = d[:n_cases]
        m_ref_const = d[-1]
        
        # Evaporation temperature
        T_e = evaporation_temp_jax(P_suc)
        
        # Display case dynamics (vectorized where possible)
        Q_goods_air = UA_goods_air * (T_goods - T_air)
        Q_air_wall = UA_air_wall * (T_air - T_wall)
        UA_wall_ref = UA_wall_ref_max * jnp.maximum(0.0, M_ref) / M_ref_max
        Q_e = UA_wall_ref * (T_wall - T_e)
        
        # Temperature derivatives
        dT_goods = -Q_goods_air / (M_goods * Cp_goods)
        dT_wall = (Q_air_wall - Q_e) / (M_wall * Cp_wall)
        dT_air = (Q_goods_air + Q_airload - Q_air_wall) / (M_air * Cp_air)
        
        # Refrigerant mass derivative (exact discrete logic using jnp.where)
        dM_ref_fill = (M_ref_max - M_ref) / tau_fill
        dM_ref_evap = -Q_e / latent_heat_jax(P_suc)
        dM_ref = jnp.where(valves > 0.5,
                          dM_ref_fill,
                          jnp.where(M_ref > 0, dM_ref_evap, 0.0))
        
        # Integrate display cases
        T_goods_new = T_goods + dT_goods * dt
        T_wall_new = T_wall + dT_wall * dt
        T_air_new = T_air + dT_air * dt
        M_ref_new = jnp.clip(M_ref + dM_ref * dt, 0.0, M_ref_max)
        
        # Mass flow to suction manifold
        # Must use OLD M_ref to match Python: mass_flow_out() checks self.state[3] before update
        # Only positive Q_e contributes (matching Python: if Q_e > 0 and M_ref > 0)
        m_in_suc = jnp.sum(jnp.maximum(0.0, Q_e) / latent_heat_jax(P_suc) * (M_ref > 0))
        
        # Compressor volume flow
        V_comp = (comp / 100.0) * params['eta_vol'] * params['V_sl']
        
        # Suction pressure dynamics
        rho = density_suction_jax(P_suc)
        drho_dP = jnp.maximum(1e-3, d_density_dP_jax(P_suc))
        dP = (m_in_suc + m_ref_const - V_comp * rho) / (V_suc * drho_dP)
        P_suc_new = jnp.clip(P_suc + dP * dt, 0.8, 3.0)
        
        # Stack new state
        x_new = jnp.concatenate([T_goods_new, T_wall_new, T_air_new, M_ref_new, jnp.array([P_suc_new])])
        
        return x_new
    
    return dynamics_step


def make_forward_simulate(params):
    """Factory function to create JIT-compiled forward simulation"""
    
    dynamics_step = make_dynamics_step(params)
    
    @jit
    def forward_simulate(x0, u_traj, d_traj):
        """Forward simulate trajectory
        
        Args:
            x0: initial state [4n+1]
            u_traj: control trajectory [T, n+1]
            d_traj: disturbance trajectory [T, n+1]
        
        Returns:
            x_traj: state trajectory [T+1, 4n+1]
        """
        def scan_fn(x, inputs):
            u, d = inputs
            x_next = dynamics_step(x, u, d)
            return x_next, x_next
        
        _, x_traj = jax.lax.scan(scan_fn, x0, (u_traj, d_traj))
        
        return jnp.concatenate([x0[None, :], x_traj], axis=0)
    
    return forward_simulate


def get_state_vector_jax(system, n_cases):
    """Extract state as grouped vector [4n+1] for JAX dynamics"""
    T_goods = [system.cases[i].state[0] for i in range(n_cases)]
    T_wall = [system.cases[i].state[1] for i in range(n_cases)]
    T_air = [system.cases[i].state[2] for i in range(n_cases)]
    M_ref = [system.cases[i].state[3] for i in range(n_cases)]
    return jnp.array(T_goods + T_wall + T_air + M_ref + [system.P_suc], dtype=jnp.float64)


# ============================================================================
# PERFORMANCE METRICS (From Benchmark Paper)
# ============================================================================

def calculate_performance(time, T_air, P_suc, comp_switches, power, valve_states, P_ref):
    """Calculate performance metrics per Eq. (16), (19), (20) in paper"""
    dt = time[1] - time[0] if len(time) > 1 else 1.0
    n_cases = T_air.shape[1]
    # Allow time-varying pressure reference (array) or constant (scalar)
    P_ref_arr = np.asarray(P_ref)
    
    # Constraint violation per Eq. (16)
    eps_con = 0.0
    for i in range(len(time)):
        P_ref_i = P_ref_arr[i] if P_ref_arr.ndim > 0 else float(P_ref_arr)
        if P_suc[i] > P_ref_i:
            eps_P_sq = (P_suc[i] - P_ref_i)**2
        else:
            eps_P_sq = 0.0
        
        eps_T_sum_sq = 0.0
        for j in range(n_cases):
            if T_air[i, j] > 5.0:
                eps_T_sum_sq += (T_air[i, j] - 5.0)**2
            elif T_air[i, j] < 2.0:
                eps_T_sum_sq += (2.0 - T_air[i, j])**2
        
        eps_con += eps_P_sq + (1.0 / n_cases) * eps_T_sum_sq
    
    gamma_con = eps_con * dt / (time[-1] - time[0] + dt)
    
    # Switches per Eq. (19)
    n_comp_switches = np.sum(comp_switches)
    n_valve_switches = 0
    for j in range(n_cases):
        n_valve_switches += np.sum(np.abs(np.diff(valve_states[:, j])) > 0.1)
    
    gamma_switch = (n_comp_switches + n_valve_switches / 100.0) / (time[-1] - time[0] + dt)
    
    # Average power per Eq. (20)
    gamma_pow = np.mean(power)
    
    return gamma_con, gamma_switch, gamma_pow


# ============================================================================
# BASELINE SIMULATION FUNCTION
# ============================================================================

def run_scenario(scenario='2d-2c', duration=14400, dt=1.0, seed=None):
    """Run PID baseline simulation
    
    Args:
        scenario: '2d-2c' (2 cases, 2 compressors) or '3d-3c' (3 cases, 3 compressors with VFD)
        duration: Simulation duration [s]
        dt: Time step [s]
        seed: Random seed
        
    Returns:
        time, T_air, P_suc, P_ref, comp_capacity, power, valve_states, comp_switches, comp_states_log, n_cases
    """
    if seed is not None:
        np.random.seed(seed)
    
    if scenario == '2d-2c':
        n_cases = 2
        comp_capacities = [50.0, 50.0]
        V_sl = 0.08
        has_vfd = False
    elif scenario == '3d-3c':
        n_cases = 3
        comp_capacities = [40.0, 30.0, 30.0]  # comp1 is VFD: continuous [10,40]
        V_sl = 0.095
        has_vfd = True
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Use '2d-2c' or '3d-3c'")
    
    # Initialize system
    system = RefrigerationSystem(n_cases, comp_capacities, V_sl, has_vfd=has_vfd)
    
    # Set initial conditions per paper Appendix C
    if scenario == '2d-2c':
        system.cases[0].state = np.array([2.0, 0.0, 5.1, 0.0])
        system.cases[1].state = np.array([2.0, 0.0, 0.0, 1.0])
    else:  # 3d-3c
        system.cases[0].state = np.array([2.0, 0.0, 5.1, 0.0])
        system.cases[1].state = np.array([2.0, 0.0, 0.0, 1.0])
        system.cases[2].state = np.array([2.0, 0.0, 2.5, 0.5])
    system.P_suc = 1.40
    
    # Storage for results
    n_steps = int(duration / dt)
    time = np.zeros(n_steps)
    T_air = np.zeros((n_steps, n_cases))
    P_suc = np.zeros(n_steps)
    P_ref_profile = np.zeros(n_steps)
    comp_capacity = np.zeros(n_steps)
    power = np.zeros(n_steps)
    valve_states = np.zeros((n_steps, n_cases))
    comp_switches = np.zeros(n_steps)
    comp_states_log = []
    
    # Simulation loop
    system.set_day_mode()
    
    for i in range(n_steps):
        t = i * dt
        time[i] = t
        
        if t >= 7200:
            system.set_night_mode()
        
        valves, comp_on, pwr, comp_sw = system.simulate_step(dt, t)
        
        if comp_sw > 0 or i == 0:
            comp_states_log.append((t, comp_on[:]))
        
        for j in range(n_cases):
            T_air[i, j] = system.cases[j].state[2]
            valve_states[i, j] = valves[j]
        P_suc[i] = system.P_suc
        P_ref_profile[i] = system.P_ref
        comp_capacity[i] = sum(comp_on)
        power[i] = pwr
        comp_switches[i] = comp_sw
    
    return time, T_air, P_suc, P_ref_profile, comp_capacity, power, valve_states, comp_switches, comp_states_log, n_cases

