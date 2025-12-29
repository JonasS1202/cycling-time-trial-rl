import numpy as np
import math
import config

def get_air_density(elevation):
    """
    Calculates air density based on elevation using International Standard Atmosphere.
    rho = 1.225 * (1 - 2.25577e-5 * h)^4.25588
    """
    if elevation < 0: elevation = 0
    
    # Constants for ISA model (Troposphere)
    base_rho = 1.225 # kg/m^3 at sea level
    temperature_lapse_rate = 0.0065 # K/m
    sea_level_temp = 288.15 # 15C in Kelvin
    
    # Exponent for density calculation derived from: (g*M)/(R*L) - 1
    # g=9.80665, M=0.0289644, R=8.314457, L=0.0065 => ~5.25588
    # Exponent for density is this minus 1 => 4.25588
    exponent = 4.25588
    
    factor = 1.0 - (temperature_lapse_rate * elevation / sea_level_temp)
    if factor < 0: factor = 0
    
    rho = base_rho * (factor ** exponent)
    return rho

def calculate_axial_wind(wind_speed, wind_direction_global, bearing_deg):
    """
    Calculates axial wind component (Headwind > 0, Tailwind < 0).
    wind_direction_global: Direction wind is coming FROM (0=North).
    bearing_deg: Direction rider is moving TO (0=North).
    """
    # Relative angle between where wind is coming from and where rider is going
    # If Wind From North (0) and Rider To North (0), Headwind.
    # Angle Diff = 0. Cos(0) = 1. -> Headwind = Speed * 1.
    rad = math.radians(wind_direction_global - bearing_deg)
    return wind_speed * math.cos(rad)

def update_w_prime(w_prime_balance, w_prime_max, critical_power, power_output, dt):
    """
    Updates W' balance based on power output.
    """
    if power_output > critical_power:
        # Depletion
        w_prime_balance -= (power_output - critical_power) * dt
    else:
        # Recovery
        d_cp = critical_power - power_output
        tau = w_prime_max / max(d_cp, 1e-6)
        w_prime_balance = w_prime_max - (w_prime_max - w_prime_balance) * np.exp(-dt / tau)
    
    # Clip
    w_prime_balance = max(0, min(w_prime_max, w_prime_balance))
    return w_prime_balance

def update_kinematics(speed, power_watts, gradient, mass_kg, cda, crr, rho, wind_speed_axial, dt, g=9.81):
    """
    Updates rider speed based on forces.
    
    Args:
        speed: Current speed (m/s)
        power_watts: Effective power output (W)
        gradient: Slope (rise/run)
        mass_kg: Total mass (kg)
        cda: CdA (m^2)
        crr: Rolling resistance coefficient
        rho: Air density (kg/m^3)
        wind_speed_axial: Headwind component (m/s)
        dt: Time step (s)
        
    Returns:
        new_speed (m/s)
    """
    slope = np.arctan(gradient)
    f_grav = mass_kg * g * np.sin(slope)
    f_roll = mass_kg * g * np.cos(slope) * crr
    
    # Aerodynamic Drag
    # v_air is relative speed of air hitting the rider
    v_air = speed + wind_speed_axial
    f_drag = 0.5 * rho * cda * (v_air**2) * np.sign(v_air)
    
    total_resistance = f_grav + f_roll + f_drag
    
    # Propulsion
    propulsive_force = power_watts / max(speed, 1.0)
    
    accel = (propulsive_force - total_resistance) / mass_kg
    

    new_speed = speed + accel * dt
    new_speed = max(0.1, new_speed) # Minimum speed to prevent divide by zero issues later
    
    return new_speed

class SimulatedRider:
    """
    Stateful rider physics model.
    Encapsulates Speed and W' Balance.
    """
    def __init__(self, mass_kg=80.0, cda=0.32, crr=0.004, w_prime_max=20000.0, critical_power=300.0, start_speed=0.1):
        self.mass_kg = mass_kg
        self.cda = cda
        self.crr = crr
        self.w_prime_max = w_prime_max
        self.critical_power = critical_power
        
        # State
        self.speed = start_speed
        self.w_prime_balance = w_prime_max
        
        # Physics Constants
        self.g = 9.81
        
    def step(self, power_watts, gradient, dt, elevation=0.0, wind_speed_axial=0.0, air_density=None):
        """
        Updates rider state (Speed, W') for one time step.
        
        Args:
            power_watts: Power output (W)
            gradient: Slope (rise/run)
            dt: Time step (s)
            elevation: Current elevation (m), used for air density if not provided.
            wind_speed_axial: Headwind component (m/s)
            air_density: Optional explicit air density (kg/m^3). If None, calculated from elevation.
            
        Returns:
            speed (m/s)
            w_prime_balance (J)
            bonked (bool): True if W' <= 0
        """
        # 1. Determine Air Density
        if air_density is None:
            rho = get_air_density(elevation)
        else:
            rho = air_density
            
        # 2. Update W' Balance
        self.w_prime_balance = update_w_prime(
            self.w_prime_balance, self.w_prime_max, self.critical_power, power_watts, dt
        )
        bonked = (self.w_prime_balance <= 0)
        
        # 3. Update speed (Kinematics)
        self.speed = update_kinematics(
            self.speed, power_watts, gradient, self.mass_kg, self.cda, self.crr, 
            rho, wind_speed_axial, dt, self.g
        )
        
        return self.speed, self.w_prime_balance, bonked
