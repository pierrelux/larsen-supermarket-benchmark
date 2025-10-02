import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

# ============================================================================
# REFRIGERANT PROPERTIES (R134a)
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

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

@dataclass
class DisplayCaseParams:
    """Parameters for a single display case"""
    M_goods: float = 200.0      # kg
    Cp_goods: float = 1000.0    # J/(kg·K)
    UA_goods_air: float = 300.0 # J/(s·K)
    M_wall: float = 260.0       # kg
    Cp_wall: float = 385.0      # J/(kg·K)
    UA_air_wall: float = 500.0  # J/(s·K)
    M_air: float = 50.0         # kg
    Cp_air: float = 1000.0      # J/(kg·K)
    UA_wall_ref_max: float = 4000.0  # J/(s·K)
    M_ref_max: float = 1.0      # kg
    tau_fill: float = 40.0      # s
    T_SH: float = 10.0          # K (superheat)

@dataclass
class SuctionManifoldParams:
    """Parameters for suction manifold"""
    V_suc: float = 5.0  # m³

@dataclass
class CompressorParams:
    """Parameters for compressor rack"""
    V_sl: float = 0.08   # m³/s (total displacement volume)
    eta_vol: float = 0.81  # volumetric efficiency

@dataclass
class ControlParams:
    """Parameters for traditional control"""
    T_air_min: float = 2.0   # °C
    T_air_max: float = 5.0   # °C
    K_p: float = -75.0       # Proportional gain (negative per paper Appendix B)
    tau_I: float = 50.0      # Integral time constant [s]
    DB: float = 0.20         # Dead band [bar]
    
    # Baseline alignment options
    enable_alternation: bool = False  # Lead/lag unit alternation
    enable_hysteresis: bool = False   # Per-unit on/off hysteresis bands
    band_eps: float = 2.0             # Hysteresis band width [%]
    vfd_min_hold: bool = True         # Keep VFD at min 10% when on
    enable_antiwindup: bool = False   # PI anti-windup
    aw_gain: float = 1.0              # Anti-windup gain factor

# ============================================================================
# DISPLAY CASE MODEL
# ============================================================================

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
        
        # Evaporation temperature
        T_e = evaporation_temp(P_suc)
        
        # Heat transfers
        Q_goods_air = self.p.UA_goods_air * (T_goods - T_air)
        Q_air_wall = self.p.UA_air_wall * (T_air - T_wall)
        UA_wall_ref = self.heat_transfer_coeff(max(0, M_ref))
        Q_e = UA_wall_ref * (T_wall - T_e)
        
        # Temperature derivatives
        dT_goods = -Q_goods_air / (self.p.M_goods * self.p.Cp_goods)
        dT_wall = (Q_air_wall - Q_e) / (self.p.M_wall * self.p.Cp_wall)
        dT_air = (Q_goods_air + Q_airload - Q_air_wall) / (self.p.M_air * self.p.Cp_air)
        
        # Refrigerant mass derivative
        if valve == 1:
            dM_ref = (self.p.M_ref_max - M_ref) / self.p.tau_fill
        elif M_ref > 0:
            dM_ref = -Q_e / latent_heat(P_suc)
        else:
            dM_ref = 0.0
            
        # Euler integration
        new_state = state + np.array([dT_goods, dT_wall, dT_air, dM_ref]) * dt
        new_state[3] = np.clip(new_state[3], 0, self.p.M_ref_max)  # Keep M_ref in bounds
        
        return new_state, Q_e
    
    def mass_flow_out(self, Q_e, P_suc):
        """Mass flow out of evaporator to suction manifold"""
        if Q_e > 0 and self.state[3] > 0:
            return Q_e / latent_heat(P_suc)
        return 0.0

# ============================================================================
# TRADITIONAL HYSTERESIS CONTROLLER
# ============================================================================

class TraditionalController:
    def __init__(self, control_params: ControlParams, n_cases: int, comp_capacities: List[float], 
                 has_vfd: bool = False):
        self.cp = control_params
        self.n_cases = n_cases
        self.comp_capacities = comp_capacities
        self.integral = 0.0
        self.valve_states = [0] * n_cases
        self.has_vfd = has_vfd  # True if first compressor has Variable Frequency Drive
        # Track individual compressor on/off states for switch counting
        self.prev_comp_on = [0.0] * len(comp_capacities)
        # Lead/lag alternation state
        self.lead_idx = 0  # which discrete unit comes on first when entering a band
        
    def valve_control(self, T_air, case_idx):
        """Hysteresis controller for inlet valve"""
        if T_air > self.cp.T_air_max:
            self.valve_states[case_idx] = 1
        elif T_air < self.cp.T_air_min:
            self.valve_states[case_idx] = 0
        # else maintain previous state
        return self.valve_states[case_idx]
    
    def pressure_control(self, P_suc, P_ref, dt):
        """PI controller with dead band for compressor capacity
        
        Per paper Section IV and physics:
        - Compressors pump refrigerant OUT, so more capacity → lower pressure
        - When P_suc too high → need MORE capacity to reduce it
        
        With K_p = -75.0 (negative from Appendix) and error = P_ref - P_suc:
        - P_suc too high → error negative → u_PI positive → more capacity ✓
        - P_suc too low → error positive → u_PI negative → less capacity ✓
        """
        error = P_ref - P_suc
        
        # Dead band - only integrate if outside dead band
        if abs(error) > self.cp.DB:
            self.integral += error * dt
        
        # PI output
        u_PI = self.cp.K_p * error + (self.cp.K_p / self.cp.tau_I) * self.integral
        
        # Anti-windup (optional)
        if self.cp.enable_antiwindup:
            u_sat = np.clip(u_PI, 0, sum(self.comp_capacities))
            aw_gain = self.cp.aw_gain * self.cp.K_p / max(self.cp.tau_I, 1e-6)
            self.integral += (u_sat - u_PI) * aw_gain * dt
            u_PI = u_sat
        
        # Quantize to available compressor capacities based on Eq. (14)
        comp_on = [0.0] * len(self.comp_capacities)
        
        # For VFD scenario: comp1 is continuous [10,40], others are discrete
        # Per Eq. (14) - units activated in fixed order with midpoint thresholds
        if self.has_vfd:
            # VFD compressor first (continuous control)
            vfd_min = 10 if self.cp.vfd_min_hold else 0
            
            if u_PI <= 0:
                pass  # All off
            elif u_PI < vfd_min:
                if self.cp.vfd_min_hold:
                    comp_on[0] = vfd_min  # Hold at minimum
                else:
                    pass  # Allow VFD to turn off completely
            elif u_PI < self.comp_capacities[0]:
                comp_on[0] = u_PI  # Continuous control
            else:
                comp_on[0] = self.comp_capacities[0]  # VFD at max
                
                # Add discrete units with midpoint thresholds (+ optional hysteresis)
                cumsum = self.comp_capacities[0]
                for i in range(1, len(self.comp_capacities)):
                    threshold = cumsum + self.comp_capacities[i] / 2.0
                    
                    # Optional hysteresis
                    if self.cp.enable_hysteresis:
                        was_on = self.prev_comp_on[i] > 0.1
                        threshold += self.cp.band_eps if was_on else -self.cp.band_eps
                    
                    if u_PI >= threshold:
                        comp_on[i] = self.comp_capacities[i]
                        cumsum += self.comp_capacities[i]
                    else:
                        break
            return comp_on
        
        # Standard discrete quantization per Eq. (14)
        # Derive thresholds automatically from comp_capacities
        cumsum = 0
        for i in range(len(self.comp_capacities)):
            threshold = cumsum + self.comp_capacities[i] / 2.0
            
            # Optional hysteresis
            if self.cp.enable_hysteresis:
                was_on = self.prev_comp_on[i] > 0.1
                threshold += self.cp.band_eps if was_on else -self.cp.band_eps
            
            if u_PI >= threshold:
                comp_on[i] = self.comp_capacities[i]
                cumsum += self.comp_capacities[i]
            else:
                break
        
        # Optional lead/lag alternation for equal-capacity discrete units
        if self.cp.enable_alternation and len(self.comp_capacities) > 1:
            # Check if we're entering a single-unit region from off or full
            prev_total = sum(self.prev_comp_on)
            curr_total = sum(comp_on)
            
            # Are we in single-unit region with equal capacities?
            if (curr_total == self.comp_capacities[0] and 
                all(c == self.comp_capacities[0] for c in self.comp_capacities)):
                
                # Did we just enter this region?
                if prev_total == 0 or prev_total == sum(self.comp_capacities):
                    # Rotate which unit is on
                    comp_on_new = [0.0] * len(self.comp_capacities)
                    comp_on_new[self.lead_idx] = self.comp_capacities[self.lead_idx]
                    comp_on = comp_on_new
                    # Advance lead for next entry
                    self.lead_idx = (self.lead_idx + 1) % len(self.comp_capacities)
                
        return comp_on

# ============================================================================
# COMPLETE REFRIGERATION SYSTEM
# ============================================================================

class RefrigerationSystem:
    def __init__(self, n_cases: int, comp_capacities: List[float], V_sl: float = 0.08, has_vfd: bool = False):
        # Initialize components
        self.n_cases = n_cases
        self.cases = [DisplayCase(DisplayCaseParams(), i) for i in range(n_cases)]
        self.P_suc = 1.40  # Initial suction pressure [bar] (per Appendix C)
        self.V_suc = 5.0  # Suction manifold volume [m³]
        
        comp_params = CompressorParams(V_sl=V_sl)
        self.V_sl = V_sl
        self.eta_vol = comp_params.eta_vol
        self.comp_capacities = comp_capacities
        
        self.controller = TraditionalController(ControlParams(), n_cases, comp_capacities, has_vfd=has_vfd)
        
        # Scenario parameters (day mode initial)
        self.Q_airload = 3000.0  # J/s per display case
        self.m_ref_const = 0.2   # kg/s
        self.P_ref = 1.50        # bar (PI setpoint for controller)
        self.load_noise_std = 0.0  # J/s - set to ~100 to reproduce valve synchronization
        
        # Control timing (per paper Section V.A)
        self.comp_control_time = 0.0  # Last time compressor control was updated
        self.comp_sample_time = 60.0  # Compressor control sample time [s]
        self.current_comp_on = [0.0] * len(comp_capacities)  # Current compressor state
        self.has_vfd = has_vfd  # Store for switch counting logic
        
    def set_day_mode(self):
        """Set parameters for day operation"""
        self.Q_airload = 3000.0
        self.m_ref_const = 0.2
        self.P_ref = 1.50  # From figure plots
        
    def set_night_mode(self):
        """Set parameters for night operation"""
        self.Q_airload = 1800.0
        self.m_ref_const = 0.0
        self.P_ref = 1.70  # From figure plots
    
    def volume_flow(self, comp_capacities):
        """Total volume flow from compressors [m³/s]"""
        total_capacity = sum(comp_capacities)
        return (total_capacity / 100.0) * self.eta_vol * self.V_sl
    
    def power_consumption(self, P_suc, comp_capacities):
        """Power consumption [W]"""
        V_comp = self.volume_flow(comp_capacities)
        return V_comp * power_factor(P_suc)
    
    def simulate_step(self, dt, current_time):
        """Simulate one time step
        
        Per paper Section V.A:
        - Valve control operates at 1 sec sample time
        - Compressor control operates at 60 sec sample time
        """
        # Valve control (1 sec sample time)
        valves = []
        for i, case in enumerate(self.cases):
            valve = self.controller.valve_control(case.state[2], i)
            valves.append(valve)
            case.valve = valve
        
        # Update display cases
        m_in_suc = 0.0
        for case in self.cases:
            # Apply load noise if enabled
            Q_load = self.Q_airload
            if self.load_noise_std > 0:
                Q_load += np.random.randn() * self.load_noise_std
            
            # Update state
            new_state, Q_e = case.dynamics(case.state, self.P_suc, case.valve, Q_load, dt)
            case.state = new_state
            
            # Mass flow to suction manifold
            m_in_suc += case.mass_flow_out(Q_e, self.P_suc)
        
        # Compressor control (60 sec sample time)
        # Only update compressor control every 60 seconds
        comp_switches = 0
        if current_time - self.comp_control_time >= self.comp_sample_time - 1e-6:
            new_comp_on = self.controller.pressure_control(self.P_suc, self.P_ref, 
                                                           self.comp_sample_time)
            
            # Count individual compressor switches (start/stop events)
            for i in range(len(self.comp_capacities)):
                prev_on = self.controller.prev_comp_on[i] > 0.1
                curr_on = new_comp_on[i] > 0.1
                
                # For VFD: only count 0↔on transitions, not adjustments within [10,40]
                if self.has_vfd and i == 0:
                    # VFD compressor: only count if crossing 0
                    if prev_on != curr_on:
                        comp_switches += 1
                else:
                    # Discrete compressors: count any on/off toggle
                    if prev_on != curr_on:
                        comp_switches += 1
            
            self.controller.prev_comp_on = new_comp_on[:]
            self.current_comp_on = new_comp_on
            self.comp_control_time = current_time
        
        comp_on = self.current_comp_on
        V_comp = self.volume_flow(comp_on)
        
        # Update suction manifold pressure with numerical stability guards
        rho = density_suction(self.P_suc)
        drho_dP = max(1e-3, d_density_dP(self.P_suc))  # Guard against near-zero derivative
        
        dP = (m_in_suc + self.m_ref_const - V_comp * rho) / (self.V_suc * drho_dP)
        self.P_suc += dP * dt
        
        # Clamp pressure to reasonable bounds
        self.P_suc = np.clip(self.P_suc, 0.8, 3.0)
        
        # Calculate power
        power = self.power_consumption(self.P_suc, comp_on)
        
        return valves, comp_on, power, comp_switches

# ============================================================================
# SIMULATION AND PLOTTING
# ============================================================================

def run_scenario(scenario='2d-2c', duration=14400, dt=1.0, 
                 enable_alternation=False, enable_hysteresis=False, 
                 enable_antiwindup=False, load_noise_std=0.0,
                 vfd_min_hold=True, seed=None):
    """Run simulation scenario
    
    Args:
        scenario: '2d-2c' or '3d-3c'
        duration: Simulation duration [s]
        dt: Time step [s]
        enable_alternation: Enable lead/lag unit alternation
        enable_hysteresis: Enable per-unit on/off hysteresis
        enable_antiwindup: Enable PI anti-windup
        load_noise_std: Standard deviation of load noise [J/s]
        vfd_min_hold: Keep VFD at minimum 10% when on
        seed: Random seed for reproducibility (when using load noise)
    """
    
    # Set random seed if specified
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
        has_vfd = True  # First compressor has Variable Frequency Drive
    else:
        raise ValueError("Unknown scenario")
    
    # Initialize system with custom control parameters
    system = RefrigerationSystem(n_cases, comp_capacities, V_sl, has_vfd=has_vfd)
    
    # Update controller parameters
    system.controller.cp.enable_alternation = enable_alternation
    system.controller.cp.enable_hysteresis = enable_hysteresis
    system.controller.cp.enable_antiwindup = enable_antiwindup
    system.controller.cp.vfd_min_hold = vfd_min_hold
    
    # Set load noise
    system.load_noise_std = load_noise_std
    
    # Set initial conditions per Appendix C of the paper
    if scenario == '2d-2c':
        system.cases[0].state = np.array([2.0, 0.0, 5.1, 0.0])
        system.cases[1].state = np.array([2.0, 0.0, 0.0, 1.0])
    else:
        system.cases[0].state = np.array([2.0, 0.0, 5.1, 0.0])
        system.cases[1].state = np.array([2.0, 0.0, 0.0, 1.0])
        system.cases[2].state = np.array([2.0, 0.0, 2.5, 0.5])
    
    # Initial pressure per Appendix C (matching paper baseline scenarios)
    system.P_suc = 1.40
    
    # Storage for results
    n_steps = int(duration / dt)
    time = np.zeros(n_steps)
    T_air = np.zeros((n_steps, n_cases))
    P_suc = np.zeros(n_steps)
    P_ref_profile = np.zeros(n_steps)
    comp_capacity = np.zeros(n_steps)
    power = np.zeros(n_steps)
    valve_states = np.zeros((n_steps, n_cases))  # Track valve states for switch counting
    comp_switches = np.zeros(n_steps)  # Track compressor switches per time step
    comp_states_log = []  # Log (time, comp_on) for diagnostics
    
    # Simulation loop - start in day mode
    system.set_day_mode()
    
    for i in range(n_steps):
        t = i * dt
        time[i] = t
        
        # Switch to night mode at t=7200s (as specified in paper)
        if t >= 7200:
            system.set_night_mode()
        
        # Simulate step
        valves, comp_on, pwr, comp_sw = system.simulate_step(dt, t)
        
        # Log compressor state changes (every 60s when controller updates)
        if comp_sw > 0 or i == 0:
            comp_states_log.append((t, comp_on[:]))
        
        # Store results
        for j in range(n_cases):
            T_air[i, j] = system.cases[j].state[2]
            valve_states[i, j] = valves[j]
        P_suc[i] = system.P_suc
        P_ref_profile[i] = system.P_ref
        comp_capacity[i] = sum(comp_on)
        power[i] = pwr
        comp_switches[i] = comp_sw
    
    return time, T_air, P_suc, P_ref_profile, comp_capacity, power, valve_states, comp_switches, comp_states_log, n_cases

def plot_results(time, T_air, P_suc, P_ref, comp_capacity, power, n_cases, scenario):
    """Create comprehensive plots"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Air temperatures
    ax = axes[0]
    colors = ['blue', 'red', 'green']
    for i in range(n_cases):
        ax.plot(time, T_air[:, i], label=f'T_air,{i+1}', color=colors[i], linewidth=1.5)
    ax.axhline(y=2.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=5.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=7200, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('T_air [°C]')
    ax.set_xlim([0, time[-1]])
    ax.set_ylim([-4, 6])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title(f'Scenario {scenario} - Traditional Control')
    ax.text(3600, -3, 'Day', ha='center', fontsize=10)
    ax.text(10800, -3, 'Night', ha='center', fontsize=10)
    
    # Suction pressure
    ax = axes[1]
    ax.plot(time, P_suc, 'b-', linewidth=1.5, label='P_suc')
    ax.plot(time, P_ref, 'r--', linewidth=1.5, label='P_suc,ref')
    ax.axvline(x=7200, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('P_suc [bar]')
    ax.set_xlim([0, time[-1]])
    ax.set_ylim([1.0, 2.2])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Compressor capacity
    ax = axes[2]
    ax.plot(time, comp_capacity, 'g-', linewidth=1.5)
    ax.axvline(x=7200, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Comp. capacity [%]')
    ax.set_xlim([0, time[-1]])
    ax.set_ylim([-5, 105])
    ax.grid(True, alpha=0.3)
    
    # Power consumption
    ax = axes[3]
    ax.plot(time, power/1000, 'r-', linewidth=1.5)
    ax.axvline(x=7200, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Power [kW]')
    ax.set_xlabel('time [sec]')
    ax.set_xlim([0, time[-1]])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def calculate_performance(time, T_air, P_suc, comp_switches, power, valve_states, P_ref):
    """Calculate performance metrics per Eq. (16), (19), (20) in paper
    
    Args:
        comp_switches: Array tracking individual compressor unit switches per time step
    """
    dt = time[1] - time[0] if len(time) > 1 else 1.0
    n_cases = T_air.shape[1]
    
    # Constraint violation per Eq. (16): gamma_con = integral[eps_P^2 + (1/n)*sum(eps_T^2)] / duration
    eps_con = 0.0
    for i in range(len(time)):
        # Pressure constraint violation (only upper bound per Eq. (17))
        if P_suc[i] > P_ref:
            eps_P_sq = (P_suc[i] - P_ref)**2
        else:
            eps_P_sq = 0.0
        
        # Temperature constraint violations per Eq. (18)
        eps_T_sum_sq = 0.0
        for j in range(n_cases):
            if T_air[i, j] > 5.0:
                eps_T_sum_sq += (T_air[i, j] - 5.0)**2
            elif T_air[i, j] < 2.0:
                eps_T_sum_sq += (2.0 - T_air[i, j])**2
        
        eps_con += eps_P_sq + (1.0 / n_cases) * eps_T_sum_sq
    
    gamma_con = eps_con * dt / (time[-1] - time[0] + dt)
    
    # Switches per Eq. (19): count individual compressor unit switches AND valve switches
    # Compressor switches: sum of per-unit toggles (already counted in comp_switches array)
    n_comp_switches = np.sum(comp_switches)
    
    # Valve switches: count changes in each valve state, sum across all valves
    n_valve_switches = 0
    for j in range(n_cases):
        n_valve_switches += np.sum(np.abs(np.diff(valve_states[:, j])) > 0.1)
    
    # Per Eq. (19): 100 valve switches = 1 compressor switch
    gamma_switch = (n_comp_switches + n_valve_switches / 100.0) / (time[-1] - time[0] + dt)
    
    # Average power per Eq. (20)
    gamma_pow = np.mean(power)
    
    print(f"γ_con:    {gamma_con:.3f} [°C²]")
    print(f"γ_switch: {gamma_switch:.6f} [-]")
    print(f"γ_pow:    {gamma_pow/1000:.2f} [kW]")
    
    return gamma_con, gamma_switch, gamma_pow

def build_comparison_table(results_dict):
    """Build a comparison table of results vs paper baseline
    
    Args:
        results_dict: Dict with keys like ('2d-2c', 'day') -> (γ_con, γ_switch, γ_pow)
    """
    # Paper's baseline values from Appendix C
    paper_baseline = {
        ('2d-2c', 'day'):   (2.280, 0.026, 15.50),
        ('2d-2c', 'night'): (3.500, 0.030, 0.98),
        ('3d-3c', 'day'):   (2.249, 0.066, 13.95),
        ('3d-3c', 'night'): (4.157, 0.027, 1.32),
    }
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH PAPER BASELINE (ECC'07 Appendix C)")
    print("=" * 80)
    print(f"{'Scenario':<12} {'Period':<8} {'Metric':<12} {'Paper':<10} {'Ours':<10} {'Delta':<10}")
    print("-" * 80)
    
    for scenario in ['2d-2c', '3d-3c']:
        for period in ['day', 'night']:
            key = (scenario, period)
            if key in results_dict and key in paper_baseline:
                paper = paper_baseline[key]
                ours = results_dict[key]
                
                # γ_con
                print(f"{scenario:<12} {period:<8} {'γ_con':<12} {paper[0]:<10.3f} "
                      f"{ours[0]:<10.3f} {ours[0]-paper[0]:<+10.3f}")
                
                # γ_switch
                print(f"{'':12} {'':8} {'γ_switch':<12} {paper[1]:<10.6f} "
                      f"{ours[1]:<10.6f} {ours[1]-paper[1]:<+10.6f}")
                
                # γ_pow
                print(f"{'':12} {'':8} {'γ_pow [kW]':<12} {paper[2]:<10.2f} "
                      f"{ours[2]:<10.2f} {ours[2]-paper[2]:<+10.2f}")
                print("-" * 80)
    
    print("=" * 80)

def print_diagnostics(time, comp_capacity, comp_switches, valve_states, comp_states_log, scenario):
    """Print diagnostic information about duty cycles and switching behavior
    
    Args:
        comp_states_log: List of (time, comp_on) tuples recorded during simulation
    """
    dt = time[1] - time[0] if len(time) > 1 else 1.0
    duration = time[-1] - time[0] + dt
    n_cases = valve_states.shape[1]
    
    print("\n  Diagnostics:")
    print("  " + "-" * 50)
    
    # Extract number of compressors from comp_states_log
    if comp_states_log:
        n_comps = len(comp_states_log[0][1])
        
        # Calculate duty cycle for each compressor
        comp_on_time = [0.0] * n_comps
        for i in range(len(comp_states_log) - 1):
            t_current, state_current = comp_states_log[i]
            t_next, _ = comp_states_log[i + 1]
            dt_interval = t_next - t_current
            
            for j in range(n_comps):
                if state_current[j] > 0.1:  # Compressor is on
                    comp_on_time[j] += dt_interval
        
        # Last interval to end of simulation
        if len(comp_states_log) > 0:
            t_last, state_last = comp_states_log[-1]
            dt_last = time[-1] - t_last + dt
            for j in range(n_comps):
                if state_last[j] > 0.1:
                    comp_on_time[j] += dt_last
        
        print("  Compressor duty cycles:")
        for i in range(n_comps):
            duty_pct = 100.0 * comp_on_time[i] / duration
            print(f"    Unit {i+1}: {duty_pct:.1f}% on-time")
        
        # Count switches per compressor
        comp_switch_counts = [0] * n_comps
        for i in range(len(comp_states_log) - 1):
            _, state_prev = comp_states_log[i]
            _, state_curr = comp_states_log[i + 1]
            for j in range(n_comps):
                prev_on = state_prev[j] > 0.1
                curr_on = state_curr[j] > 0.1
                if prev_on != curr_on:
                    comp_switch_counts[j] += 1
        
        print("  Compressor switch counts:")
        total_comp_switches = 0
        for i in range(n_comps):
            print(f"    Unit {i+1}: {comp_switch_counts[i]} toggles")
            total_comp_switches += comp_switch_counts[i]
        print(f"    Total: {total_comp_switches} compressor toggles")
    
    # Valve switches
    print("  Valve switch counts:")
    total_valve_switches = 0
    for j in range(n_cases):
        n_switches = int(np.sum(np.abs(np.diff(valve_states[:, j])) > 0.1))
        print(f"    Case {j+1}: {n_switches} toggles")
        total_valve_switches += n_switches
    print(f"    Total: {total_valve_switches} valve toggles")
    
    print("  " + "-" * 50)

# ============================================================================
# BAYESIAN OPTIMIZATION INTERFACE
# ============================================================================

def compute_reference_metrics(scenario='2d-2c', Kp=-75.0, tau_I=50.0, DB=0.20, seed=42):
    """Compute reference baseline metrics for normalization
    
    Args:
        scenario: '2d-2c' or '3d-3c'
        Kp: Proportional gain (negative)
        tau_I: Integral time constant [s]
        DB: Dead band [bar]
        seed: Random seed
        
    Returns:
        dict with keys 'gcon_day', 'gsw_day', 'gpow_day', 'gcon_night', 'gsw_night', 'gpow_night'
    """
    # Run baseline simulation
    time, T_air, P_suc, P_ref, comp_cap, power, valve_states, comp_switches, comp_states_log, n_cases = \
        run_scenario(scenario, duration=14400, seed=seed)
    
    # Split day/night
    day_mask = time < 7200
    night_mask = time >= 7200
    
    # Day metrics (use P_ref=1.7 for constraint violation per paper)
    gcon_day, gsw_day, gpow_day = calculate_performance(
        time[day_mask], T_air[day_mask], P_suc[day_mask], 
        comp_switches[day_mask], power[day_mask], valve_states[day_mask], 
        P_ref=1.7
    )
    
    # Night metrics (use P_ref=1.9 for constraint violation per paper)
    gcon_night, gsw_night, gpow_night = calculate_performance(
        time[night_mask], T_air[night_mask], P_suc[night_mask], 
        comp_switches[night_mask], power[night_mask], valve_states[night_mask], 
        P_ref=1.9
    )
    
    return {
        'gcon_day': gcon_day,
        'gsw_day': gsw_day,
        'gpow_day': gpow_day / 1000,  # Convert to kW
        'gcon_night': gcon_night,
        'gsw_night': gsw_night,
        'gpow_night': gpow_night / 1000,  # Convert to kW
    }

def detect_hard_violations(time, T_air, P_suc, 
                           T_min=1.5, T_max=5.5, P_max=2.05, 
                           violation_duration=60.0):
    """Detect hard constraint violations beyond soft bounds
    
    Args:
        time: Time array [s]
        T_air: Air temperature array [s, n_cases]
        P_suc: Suction pressure array [s]
        T_min: Hard lower bound for temperature [°C]
        T_max: Hard upper bound for temperature [°C]
        P_max: Hard upper bound for pressure [bar]
        violation_duration: Minimum duration for sustained violation [s]
        
    Returns:
        penalty_count: Number of distinct violations
    """
    dt = time[1] - time[0] if len(time) > 1 else 1.0
    n_cases = T_air.shape[1]
    penalty_count = 0
    
    # Temperature violations (per case)
    for j in range(n_cases):
        # Check for too cold
        too_cold = T_air[:, j] < T_min
        if np.any(too_cold):
            # Count sustained violations (> violation_duration)
            cold_duration = np.sum(too_cold) * dt
            if cold_duration > violation_duration:
                penalty_count += 1
        
        # Check for too hot
        too_hot = T_air[:, j] > T_max
        if np.any(too_hot):
            hot_duration = np.sum(too_hot) * dt
            if hot_duration > violation_duration:
                penalty_count += 1
    
    # Pressure violations
    too_high_P = P_suc > P_max
    if np.any(too_high_P):
        high_P_duration = np.sum(too_high_P) * dt
        if high_P_duration > violation_duration:
            penalty_count += 1
    
    return penalty_count

def evaluate_bo_objective(theta, scenario='2d-2c', ref_metrics=None, 
                          alpha=0.5, w_con=1.0, w_pow=0.3, w_sw=0.2,
                          lambda_pen=10.0, seeds=(42,), 
                          enable_alternation=False, enable_hysteresis=False,
                          enable_antiwindup=False, vfd_min_hold=True,
                          verbose=False):
    """Bayesian Optimization objective function
    
    This is the main function for BO. It:
    1. Takes PID parameters (theta)
    2. Runs simulation(s)
    3. Computes normalized metrics
    4. Returns scalar loss with penalties
    
    Args:
        theta: tuple/array of (Kp, tau_I, DB)
        scenario: '2d-2c' or '3d-3c'
        ref_metrics: Dict of reference metrics for normalization
                     If None, will compute from default params
        alpha: Blend weight for day/night (0.5 = equal weight)
        w_con: Weight for constraint violation metric
        w_pow: Weight for power metric
        w_sw: Weight for switching metric
        lambda_pen: Penalty multiplier for hard violations
        seeds: Tuple of random seeds for repeated evaluations
        enable_alternation: Lead/lag unit alternation
        enable_hysteresis: Per-unit hysteresis
        enable_antiwindup: PI anti-windup
        vfd_min_hold: Keep VFD at minimum 10% when on
        verbose: Print detailed info
        
    Returns:
        loss: Scalar objective value (lower is better)
    """
    # Extract and clip parameters
    Kp, tau_I, DB = theta
    Kp = float(np.clip(Kp, -150.0, -10.0))
    tau_I = float(np.clip(tau_I, 10.0, 200.0))
    DB = float(np.clip(DB, 0.05, 0.35))
    
    if verbose:
        print(f"\n  Evaluating: Kp={Kp:.2f}, tau_I={tau_I:.2f}, DB={DB:.3f}")
    
    # Compute reference metrics if not provided
    if ref_metrics is None:
        if verbose:
            print("  Computing reference metrics...")
        ref_metrics = compute_reference_metrics(scenario=scenario, seed=seeds[0])
    
    # Run multiple seeds and average
    losses = []
    for seed_idx, seed in enumerate(seeds):
        if verbose and len(seeds) > 1:
            print(f"    Seed {seed}...")
        
        # Setup system with custom parameters
        if scenario == '2d-2c':
            n_cases = 2
            comp_capacities = [50.0, 50.0]
            V_sl = 0.08
            has_vfd = False
        elif scenario == '3d-3c':
            n_cases = 3
            comp_capacities = [40.0, 30.0, 30.0]
            V_sl = 0.095
            has_vfd = True
        else:
            raise ValueError("Unknown scenario")
        
        # Initialize system
        system = RefrigerationSystem(n_cases, comp_capacities, V_sl, has_vfd=has_vfd)
        
        # Set controller parameters (THIS IS THE KEY PART FOR BO)
        system.controller.cp.K_p = Kp
        system.controller.cp.tau_I = tau_I
        system.controller.cp.DB = DB
        system.controller.cp.enable_alternation = enable_alternation
        system.controller.cp.enable_hysteresis = enable_hysteresis
        system.controller.cp.enable_antiwindup = enable_antiwindup
        system.controller.cp.vfd_min_hold = vfd_min_hold
        
        # Set random seed
        np.random.seed(seed)
        
        # Set initial conditions per Appendix C
        if scenario == '2d-2c':
            system.cases[0].state = np.array([2.0, 0.0, 5.1, 0.0])
            system.cases[1].state = np.array([2.0, 0.0, 0.0, 1.0])
        else:
            system.cases[0].state = np.array([2.0, 0.0, 5.1, 0.0])
            system.cases[1].state = np.array([2.0, 0.0, 0.0, 1.0])
            system.cases[2].state = np.array([2.0, 0.0, 2.5, 0.5])
        system.P_suc = 1.40
        
        # Run simulation
        dt = 1.0
        duration = 14400
        n_steps = int(duration / dt)
        
        time = np.zeros(n_steps)
        T_air = np.zeros((n_steps, n_cases))
        P_suc = np.zeros(n_steps)
        P_ref_profile = np.zeros(n_steps)
        power = np.zeros(n_steps)
        valve_states = np.zeros((n_steps, n_cases))
        comp_switches = np.zeros(n_steps)
        
        system.set_day_mode()
        
        for i in range(n_steps):
            t = i * dt
            time[i] = t
            
            if t >= 7200:
                system.set_night_mode()
            
            valves, comp_on, pwr, comp_sw = system.simulate_step(dt, t)
            
            for j in range(n_cases):
                T_air[i, j] = system.cases[j].state[2]
                valve_states[i, j] = valves[j]
            P_suc[i] = system.P_suc
            P_ref_profile[i] = system.P_ref
            power[i] = pwr
            comp_switches[i] = comp_sw
        
        # Split day/night
        day_mask = time < 7200
        night_mask = time >= 7200
        
        # Calculate metrics
        gcon_day, gsw_day, gpow_day = calculate_performance(
            time[day_mask], T_air[day_mask], P_suc[day_mask],
            comp_switches[day_mask], power[day_mask], valve_states[day_mask],
            P_ref=1.7
        )
        
        gcon_night, gsw_night, gpow_night = calculate_performance(
            time[night_mask], T_air[night_mask], P_suc[night_mask],
            comp_switches[night_mask], power[night_mask], valve_states[night_mask],
            P_ref=1.9
        )
        
        # Convert power to kW
        gpow_day /= 1000
        gpow_night /= 1000
        
        # Detect hard violations
        pen_day = detect_hard_violations(time[day_mask], T_air[day_mask], P_suc[day_mask])
        pen_night = detect_hard_violations(time[night_mask], T_air[night_mask], P_suc[night_mask])
        total_pen = pen_day + pen_night
        
        # Blend day/night metrics
        gcon = alpha * gcon_day + (1 - alpha) * gcon_night
        gsw = alpha * gsw_day + (1 - alpha) * gsw_night
        gpow = alpha * gpow_day + (1 - alpha) * gpow_night
        
        # Normalize by reference metrics
        gcon_norm = gcon / max(ref_metrics['gcon_day'], 0.001)  # Avoid div by zero
        gsw_norm = gsw / max(ref_metrics['gsw_day'], 0.001)
        gpow_norm = gpow / max(ref_metrics['gpow_day'], 0.001)
        
        # Scalar loss
        loss = w_con * gcon_norm + w_sw * gsw_norm + w_pow * gpow_norm + lambda_pen * total_pen
        losses.append(float(loss))
        
        if verbose:
            print(f"      gcon={gcon:.3f} (norm={gcon_norm:.3f}), "
                  f"gsw={gsw:.6f} (norm={gsw_norm:.3f}), "
                  f"gpow={gpow:.2f} (norm={gpow_norm:.3f}), "
                  f"pen={total_pen}, loss={loss:.3f}")
    
    mean_loss = float(np.mean(losses))
    
    if verbose:
        if len(losses) > 1:
            print(f"    Mean loss: {mean_loss:.3f} (std={np.std(losses):.3f})")
        else:
            print(f"    Loss: {mean_loss:.3f}")
    
    return mean_loss

def run_bo_tuning(scenario='2d-2c', n_trials=50, n_seeds=1, verbose=True):
    """Run Bayesian Optimization to tune PID parameters using Ax
    
    This is a complete example showing how to use Ax for optimization.
    
    Args:
        scenario: '2d-2c' or '3d-3c'
        n_trials: Number of BO iterations
        n_seeds: Number of random seeds per evaluation (for robustness)
        verbose: Print progress
        
    Returns:
        best_params: Dict with best Kp, tau_I, DB
        best_loss: Best objective value found
    """
    try:
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties
    except ImportError:
        print("ERROR: Ax is not installed. Install with: pip install ax-platform")
        return None, None
    
    if verbose:
        print("=" * 70)
        print(f"BAYESIAN OPTIMIZATION: PID TUNING FOR {scenario.upper()}")
        print("=" * 70)
    
    # Compute reference metrics once
    if verbose:
        print("\nComputing reference baseline metrics...")
    ref_metrics = compute_reference_metrics(scenario=scenario, seed=42)
    
    if verbose:
        print(f"  Reference day metrics: gcon={ref_metrics['gcon_day']:.3f}, "
              f"gsw={ref_metrics['gsw_day']:.6f}, gpow={ref_metrics['gpow_day']:.2f} kW")
    
    # Create Ax client
    ax_client = AxClient(verbose_logging=False)
    
    # Define search space
    ax_client.create_experiment(
        name=f"pid_tuning_{scenario}",
        parameters=[
            {
                "name": "Kp",
                "type": "range",
                "bounds": [-150.0, -10.0],
                "value_type": "float",
                "log_scale": False,
            },
            {
                "name": "tau_I",
                "type": "range",
                "bounds": [10.0, 200.0],
                "value_type": "float",
                "log_scale": True,  # Log scale often better for time constants
            },
            {
                "name": "DB",
                "type": "range",
                "bounds": [0.05, 0.35],
                "value_type": "float",
                "log_scale": False,
            },
        ],
        objectives={"loss": ObjectiveProperties(minimize=True)},
    )
    
    # Generate random seeds for evaluation
    seed_list = list(range(42, 42 + n_seeds))
    
    # BO loop
    for i in range(n_trials):
        if verbose:
            print(f"\n[Trial {i+1}/{n_trials}]")
        
        # Get next parameter configuration
        parameters, trial_index = ax_client.get_next_trial()
        
        # Evaluate
        theta = (parameters["Kp"], parameters["tau_I"], parameters["DB"])
        loss = evaluate_bo_objective(
            theta, 
            scenario=scenario,
            ref_metrics=ref_metrics,
            seeds=tuple(seed_list),
            verbose=verbose
        )
        
        # Report result
        ax_client.complete_trial(trial_index=trial_index, raw_data={"loss": (loss, 0.0)})
        
        if verbose:
            print(f"  → Loss: {loss:.3f}")
    
    # Get best parameters
    best_parameters, values = ax_client.get_best_parameters()
    best_loss = values[0]["loss"]
    
    if verbose:
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"\nBest parameters found:")
        print(f"  Kp:    {best_parameters['Kp']:.3f}")
        print(f"  tau_I: {best_parameters['tau_I']:.3f} s")
        print(f"  DB:    {best_parameters['DB']:.3f} bar")
        print(f"  Loss:  {best_loss:.3f}")
        print("\n" + "=" * 70)
    
    return best_parameters, best_loss

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SUPERMARKET REFRIGERATION SYSTEM BENCHMARK")
    print("=" * 70)
    
    # Configuration flags for baseline alignment
    # Set these to True to match paper's baseline behavior more closely
    ENABLE_ALTERNATION = False   # Lead/lag unit alternation (increases γ_switch)
    ENABLE_HYSTERESIS = False    # Per-unit hysteresis (increases γ_switch)
    ENABLE_ANTIWINDUP = False    # PI anti-windup
    LOAD_NOISE_STD = 0.0         # Set to ~100 J/s to reproduce valve synchronization
    VFD_MIN_HOLD = True          # Keep VFD at minimum 10% when on
    SEED = 42                    # Random seed for reproducibility
    SHOW_DIAGNOSTICS = True      # Show detailed diagnostics
    
    print(f"\nConfiguration:")
    print(f"  Lead/lag alternation: {ENABLE_ALTERNATION}")
    print(f"  Per-unit hysteresis: {ENABLE_HYSTERESIS}")
    print(f"  PI anti-windup: {ENABLE_ANTIWINDUP}")
    print(f"  Load noise std: {LOAD_NOISE_STD} J/s")
    print(f"  VFD min hold: {VFD_MIN_HOLD}")
    print(f"  Random seed: {SEED}")
    
    # Dictionary to store results for comparison table
    results = {}
    
    # Run Scenario 2d-2c
    print("\n[Scenario 2d-2c: 2 Display Cases, 2 Compressors]")
    print("Running simulation...")
    time, T_air, P_suc, P_ref, comp_cap, power, valve_states, comp_switches, comp_states_log, n_cases = \
        run_scenario('2d-2c', duration=14400, 
                    enable_alternation=ENABLE_ALTERNATION,
                    enable_hysteresis=ENABLE_HYSTERESIS,
                    enable_antiwindup=ENABLE_ANTIWINDUP,
                    load_noise_std=LOAD_NOISE_STD,
                    vfd_min_hold=VFD_MIN_HOLD,
                    seed=SEED)
    
    # Plot results
    fig1 = plot_results(time, T_air, P_suc, P_ref, comp_cap, power, n_cases, '2d-2c')
    
    # Calculate metrics (using constraint upper bounds, not PI setpoints)
    print("\nDay period (0-7200s):")
    day_mask = time < 7200
    metrics_2d2c_day = calculate_performance(time[day_mask], T_air[day_mask], P_suc[day_mask], 
                         comp_switches[day_mask], power[day_mask], valve_states[day_mask], P_ref=1.7)
    results[('2d-2c', 'day')] = (metrics_2d2c_day[0], metrics_2d2c_day[1], metrics_2d2c_day[2]/1000)
    
    if SHOW_DIAGNOSTICS:
        # Filter comp_states_log for day period
        day_comp_log = [(t, s) for t, s in comp_states_log if t < 7200]
        print_diagnostics(time[day_mask], comp_cap[day_mask], comp_switches[day_mask], 
                         valve_states[day_mask], day_comp_log, '2d-2c')
    
    print("\nNight period (7200-14400s):")
    night_mask = time >= 7200
    metrics_2d2c_night = calculate_performance(time[night_mask], T_air[night_mask], P_suc[night_mask], 
                         comp_switches[night_mask], power[night_mask], valve_states[night_mask], P_ref=1.9)
    results[('2d-2c', 'night')] = (metrics_2d2c_night[0], metrics_2d2c_night[1], metrics_2d2c_night[2]/1000)
    
    if SHOW_DIAGNOSTICS:
        # Filter comp_states_log for night period
        night_comp_log = [(t, s) for t, s in comp_states_log if t >= 7200]
        print_diagnostics(time[night_mask], comp_cap[night_mask], comp_switches[night_mask], 
                         valve_states[night_mask], night_comp_log, '2d-2c')
    
    # Run Scenario 3d-3c
    print("\n" + "=" * 70)
    print("\n[Scenario 3d-3c: 3 Display Cases, 3 Compressors]")
    print("Running simulation...")
    time, T_air, P_suc, P_ref, comp_cap, power, valve_states, comp_switches, comp_states_log, n_cases = \
        run_scenario('3d-3c', duration=14400,
                    enable_alternation=ENABLE_ALTERNATION,
                    enable_hysteresis=ENABLE_HYSTERESIS,
                    enable_antiwindup=ENABLE_ANTIWINDUP,
                    load_noise_std=LOAD_NOISE_STD,
                    vfd_min_hold=VFD_MIN_HOLD,
                    seed=SEED)
    
    # Plot results
    fig2 = plot_results(time, T_air, P_suc, P_ref, comp_cap, power, n_cases, '3d-3c')
    
    # Calculate metrics
    print("\nDay period (0-7200s):")
    day_mask = time < 7200
    metrics_3d3c_day = calculate_performance(time[day_mask], T_air[day_mask], P_suc[day_mask], 
                         comp_switches[day_mask], power[day_mask], valve_states[day_mask], P_ref=1.7)
    results[('3d-3c', 'day')] = (metrics_3d3c_day[0], metrics_3d3c_day[1], metrics_3d3c_day[2]/1000)
    
    if SHOW_DIAGNOSTICS:
        # Filter comp_states_log for day period
        day_comp_log = [(t, s) for t, s in comp_states_log if t < 7200]
        print_diagnostics(time[day_mask], comp_cap[day_mask], comp_switches[day_mask], 
                         valve_states[day_mask], day_comp_log, '3d-3c')
    
    print("\nNight period (7200-14400s):")
    night_mask = time >= 7200
    metrics_3d3c_night = calculate_performance(time[night_mask], T_air[night_mask], P_suc[night_mask], 
                         comp_switches[night_mask], power[night_mask], valve_states[night_mask], P_ref=1.9)
    results[('3d-3c', 'night')] = (metrics_3d3c_night[0], metrics_3d3c_night[1], metrics_3d3c_night[2]/1000)
    
    if SHOW_DIAGNOSTICS:
        # Filter comp_states_log for night period
        night_comp_log = [(t, s) for t, s in comp_states_log if t >= 7200]
        print_diagnostics(time[night_mask], comp_cap[night_mask], comp_switches[night_mask], 
                         valve_states[night_mask], night_comp_log, '3d-3c')
    
    # Build comparison table
    build_comparison_table(results)
    
    plt.show()
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)
    
    # ========================================================================
    # BAYESIAN OPTIMIZATION EXAMPLE
    # ========================================================================
    # Uncomment the section below to run Bayesian Optimization for PID tuning
    # 
    # print("\n\n")
    # print("=" * 70)
    # print("BAYESIAN OPTIMIZATION EXAMPLE")
    # print("=" * 70)
    # 
    # # Example 1: Quick manual evaluation of a custom PID configuration
    # print("\n[Example 1: Manual evaluation]")
    # ref_metrics = compute_reference_metrics(scenario='2d-2c', seed=42)
    # theta_test = (-100.0, 75.0, 0.15)  # Custom Kp, tau_I, DB
    # loss = evaluate_bo_objective(theta_test, scenario='2d-2c', 
    #                               ref_metrics=ref_metrics, 
    #                               verbose=True)
    # print(f"\nCustom config {theta_test} → Loss: {loss:.3f}")
    # 
    # # Example 2: Run full BO optimization (requires: pip install ax-platform)
    # print("\n[Example 2: Full BO optimization]")
    # print("Uncomment the lines below to run BO (warning: takes time!)\n")
    # # best_params, best_loss = run_bo_tuning(
    # #     scenario='2d-2c', 
    # #     n_trials=30,      # Start small, increase to 50-100 for better results
    # #     n_seeds=1,        # Use 2-3 for robustness if load noise is enabled
    # #     verbose=True
    # # )
    # # 
    # # if best_params is not None:
    # #     # Validate best parameters found
    # #     print("\n[Validating best parameters with full simulation...]")
    # #     time_val, T_air_val, P_suc_val, P_ref_val, comp_cap_val, power_val, \
    # #         valve_states_val, comp_switches_val, comp_states_log_val, n_cases_val = \
    # #         run_scenario('2d-2c', duration=14400, seed=999)
    # #     
    # #     fig_opt = plot_results(time_val, T_air_val, P_suc_val, P_ref_val, 
    # #                           comp_cap_val, power_val, n_cases_val, '2d-2c (Optimized)')
    # #     plt.show()
    # 
    # print("=" * 70)