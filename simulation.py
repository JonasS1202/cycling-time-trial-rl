"""
simulation.py – Simulation verification script
"""

from __future__ import annotations
import math
import datetime as dt
import argparse
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fitparse import FitFile
import gpxpy
import gpxpy.gpx
from scipy.interpolate import interp1d

# Internal imports
import weather
import physics_engine
import utils
import config # Added config import if needed, but Utils handles parsing now

# Constants
R_EARTH = 6371000.0
WIND_HEIGHT_CORRECTION = 0.6 # Correction from 10m to ~1.5m height

# --------------------------------------------------------------------------- #
# 1.  Track helpers                                                           #
# --------------------------------------------------------------------------- #


def load_track(track_path: str | Path) -> pd.DataFrame:
    """Return DataFrame: time, dist, lat, lon, alt, power, speed."""
    try:
        df = utils.load_track(track_path)
        # Compatibility mapping for simulation.py which expects 'alt'
        if 'ele' in df.columns:
             df['alt'] = df['ele']
        return df
    except Exception as e:
        raise ValueError(f"Failed to load track {track_path}: {e}")




# --------------------------------------------------------------------------- #
# 2.  Simulation Logic                                                        #
# --------------------------------------------------------------------------- #

class SimulatedCyclist(physics_engine.SimulatedRider):
    """
    SimulatedCyclist compatibility wrapper for physics_engine.SimulatedRider.
    """
    def __init__(self, mass_kg=80.0, cda=0.35, crr=0.004):
        # Pass through to SimulatedRider
        # Note: SimulatedRider defaults cda=0.32, here we request 0.35.
        super().__init__(mass_kg=mass_kg, cda=cda, crr=crr)
        
    # Inherit step? No, step signature is slightly different (return val)
    # simulation.py expects return new_speed. SimulatedRider returns (speed, w_prime, bonked)
    # We can just override it to match old interface if we want minimal changes to call site
    # BUT user plan said "Unpack return value". 
    # Let's override purely for compatibility if desired, OR change call site.
    # Plan: "Update step call to unpack defaults." -> IMPLIES change call site.
    # So I will NOT override Step here, but I will delete this class if I can just use SimulatedRider alias.
    # However, this class allows me to inject the custom Cda default (0.35) easily without changing verify_ride's instantiation.
    pass


def verify_ride(fit_file_path: str, enable_wind: bool = True, weather_enable: bool = True):
    """
    Loads a ride, simulates it, and compares results.
    """
    print(f"Loading {fit_file_path}...")
    df = load_track(fit_file_path)
    
    if df.empty:
        print("Error: No data loaded.")
        return

    # Create interpolators for Power and Altitude based on Distance
    power_interp = interp1d(df.dist, df.power, kind='linear', bounds_error=False, fill_value=(df.power.iloc[0], df.power.iloc[-1]))
    alt_interp = interp1d(df.dist, df.alt, kind='linear', bounds_error=False, fill_value="extrapolate")
    lat_interp = interp1d(df.dist, df.lat, kind='linear', bounds_error=False, fill_value="extrapolate")
    lon_interp = interp1d(df.dist, df.lon, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    # Real ride duration and distance
    real_duration = (df.time.iloc[-1] - df.time.iloc[0]).total_seconds()
    total_distance = df.dist.iloc[-1]
    start_time = df.time.iloc[0]
    
    # -----------------------------------------------------------
    # Weather Setup
    # -----------------------------------------------------------
    weather_mgr = None
    if weather_enable:
        print("Fetching weather data...")
        # Sample points every ~5km to limit requests, or just use start/mid/end if short.
        # Let's use 10 points along the track.
        indices = np.linspace(0, len(df)-1, 20, dtype=int)
        sample_coords = []
        for i in indices:
            sample_coords.append((df.lat.iloc[i], df.lon.iloc[i]))
            
        try:
            # Determine hours duration
            hours = int(math.ceil(real_duration / 3600)) + 2
            # Fetch weather
            data_map = weather.fetch_multi_weather(sample_coords, start_time, hours)
            weather_mgr = weather.WeatherData(data_map, sample_coords)
            print("Weather fetched successfully.")
        except Exception as e:
            print(f"Weather fetch failed: {e}. Proceeding without weather.")
            weather_mgr = None

    cyclist = SimulatedCyclist()
    
    # Initial state
    current_sim_dist = df.dist.iloc[0]
    current_sim_time = 0.0
    
    # Estimate initial speed
    if len(df) > 1:
        dt_first = (df.iloc[1].time - df.iloc[0].time).total_seconds()
        d_first = df.iloc[1].dist - df.iloc[0].dist
        if dt_first > 0:
            cyclist.speed = d_first / dt_first
    
    sim_log = []
    
    # Simulation parameters
    dt_step = 0.5 
    
    print(f"Simulating (Wind={'ON' if enable_wind else 'OFF'})...")
    
    weather_cache_idx = 0
    
    while current_sim_dist < total_distance:
        # 1. State at current distance
        power = float(power_interp(current_sim_dist))
        lat_curr = float(lat_interp(current_sim_dist))
        lon_curr = float(lon_interp(current_sim_dist))
        
        # 2. Gradient
        lookahead_dist = 10.0
        alt_curr = float(alt_interp(current_sim_dist))
        alt_next = float(alt_interp(min(current_sim_dist + lookahead_dist, total_distance)))
        dx = min(lookahead_dist, total_distance - current_sim_dist)
        gradient = (alt_next - alt_curr) / dx if dx > 0.1 else 0.0
        
        # 3. Weather / Wind
        rho = 1.225
        wind_speed_axial = 0.0
        
        # Calculate ISA density based on altitude if weather not used/available for this step
        # rho = 1.225 * (1 - 2.25577e-5 * h)^4.25588
        base_rho = 1.225
        exponent = 4.25588
        factor = 1.0 - (2.25577e-5 * alt_curr)
        if factor < 0: factor = 0
        rho = base_rho * (factor ** exponent)

        w_E, w_N = 0.0, 0.0 # for logging
        
        if weather_mgr and enable_wind:
            # Find nearest weather point (simple nearest for now)
            # Or assume linear map between sample points.
            # To be efficient: finds nearest index in sample_coords
            # Optimization: could implement KDTree or just scan, but let's just find nearest in simple loop
            # Since we move sequentially, we can track index? No, track is curvy.
            # Simple distance check to all 20 points is fast enough per step? 
            # 20 points * 20000 steps = 400k ops, fine.
            
            # Simple nearest neighbor search
            best_idx = 0
            min_d = 1e9
            for i, (wlat, wlon) in enumerate(weather_mgr.coords):
                # Euclidian approx is fine for selection
                d = (lat_curr - wlat)**2 + (lon_curr - wlon)**2
                if d < min_d:
                    min_d = d
                    best_idx = i
            
            # Get weather at current sim time
            curr_abs_time = start_time + dt.timedelta(seconds=current_sim_time)
            wd = weather_mgr.get_at_index(best_idx, curr_abs_time)
            
            rho = wd.get('rho', 1.225)
            # Apply height correction to wind speed
            w_E = wd.get('w_E', 0.0) * WIND_HEIGHT_CORRECTION
            w_N = wd.get('w_N', 0.0) * WIND_HEIGHT_CORRECTION
            
            # Calculate cyclist bearing
            # Look ahead for bearing
            lat_next = float(lat_interp(min(current_sim_dist + 5.0, total_distance)))
            lon_next = float(lon_interp(min(current_sim_dist + 5.0, total_distance)))
            
            # Bearing of cyclist (Azimuth)
            cyclist_bearing = utils.calculate_bearing(lat_curr, lon_curr, lat_next, lon_next)
            # utils.calculate_bearing returns DEGREES (0-360)
            
            # Convert to radians for the wind vector logic below
            cyclist_bearing = math.radians(cyclist_bearing)
            
            # Wind vector: w_E, w_N
            # We need to project onto cyclist direction.
            # Cyclist vector: (sin(bearing), cos(bearing)) in (E, N) components?
            # N is y, E is x.
            # Azimuth 0 (N) -> x=0, y=1. 
            # Azimuth 90 (E) -> x=1, y=0.
            # x = sin(bearing), y = cos(bearing).
            
            cyc_dir_E = math.sin(cyclist_bearing)
            cyc_dir_N = math.cos(cyclist_bearing)
            
            # Wind vector describes WHERE IT IS BLOWING TO.
            # Headwind = Component of Wind opposing Cyclist.
            # H = - (Wind . Cyclist)
            # If wind blows North (0,1) and cyclist goes North (0,1), dot is 1. H=-1. Tailwind. Correct.
            # If wind blows South (0,-1) and cyclist goes North (0,1), dot is -1. H=1. Headwind. Correct.
            
            dot = w_E * cyc_dir_E + w_N * cyc_dir_N
            wind_speed_axial = -dot
            
        elif not enable_wind:
            wind_speed_axial = 0.0
            
        # 4. Step Physics
        # Unpack return values from SimulatedRider
        new_speed, _, _ = cyclist.step(power, gradient, dt_step, wind_speed_axial=wind_speed_axial, air_density=rho)
        
        # Anti-Stuck: If speed is too low, assuming rider walks or pushes
        # 0.5 m/s ~ 1.8 km/h. High enough to eventually finish.
        if new_speed < 1.0: 
            # Check if we are stuck (power is low/zero but we need to move)
            # We enforce a minimum speed to ensure completion unless we are really at the end
            # Using 3.0 m/s (~10km/h) is generous, but let's use 1.0 m/s (3.6 km/h walking)
             new_speed = max(new_speed, 1.0)

        # 5. Update State
        dist_step = new_speed * dt_step
        current_sim_dist += dist_step
        current_sim_time += dt_step
        
        sim_log.append({
            'time': current_sim_time,
            'dist': current_sim_dist,
            'speed_kmh': new_speed * 3.6,
            'power': power,
            'gradient': gradient * 100,
            'alt': alt_curr,
            'lat': lat_curr,
            'lon': lon_curr,
            'wind_head': wind_speed_axial,
            'w_E': w_E,
            'w_N': w_N
        })
        
        if current_sim_time > real_duration * 2.0:
            print("Simulation took too long, stopping.")
            break

    sim_df = pd.DataFrame(sim_log)
    
    if sim_df.empty:
        print("No simulation data generated.")
        return

    # --- Comparison Stats ---
    sim_duration = sim_df.iloc[-1].time
    
    df['dt'] = df.time.diff().dt.total_seconds().fillna(0)
    df['dx'] = df.dist.diff().fillna(0)
    df['speed'] = (df.dx / df.dt).fillna(0)
    real_moving_time = df[df.speed > 0.5].dt.sum()

    avg_power_real = df['power'].mean()
    zero_power_percent = (df['power'] == 0).mean() * 100
    
    # Wind Stats
    avg_wind_speed = np.sqrt(sim_df['w_E']**2 + sim_df['w_N']**2).mean() if 'w_E' in sim_df else 0.0
    avg_headwind = sim_df['wind_head'].mean() if 'wind_head' in sim_df else 0.0
    max_headwind = sim_df['wind_head'].max() if 'wind_head' in sim_df else 0.0
    avg_rho = sim_df.get('rho', pd.Series([1.225]*len(sim_df))).mean() # handle missing rho if weather off

    print("-" * 40)
    print(f"File:             {fit_file_path}")
    print(f"Date:             {start_time}")
    print(f"Total Distance:   {total_distance/1000:.2f} km")
    print(f"Real Moving Time: {real_moving_time:.1f} s")
    print(f"Sim Duration:     {sim_duration:.1f} s")
    print(f"Diff:             {(sim_duration - real_moving_time):.1f} s")
    print("-" * 40)
    print(f"Avg Power (Real): {avg_power_real:.1f} W")
    print(f"Zero Power %:     {zero_power_percent:.1f}%")
    print(f"Avg Speed (Real): {total_distance/real_moving_time*3.6:.1f} km/h")
    print(f"Avg Speed (Sim):  {total_distance/sim_duration*3.6:.1f} km/h")
    print(f"Avg Wind Spd:     {avg_wind_speed:.1f} m/s")
    print(f"Avg Headwind:     {avg_headwind:.1f} m/s")
    print(f"Max Headwind:     {max_headwind:.1f} m/s")
    print(f"Avg Rho:          {avg_rho:.3f} kg/m3")
    print(f"Elevation:        {df['alt'].min():.1f}m - {df['alt'].max():.1f}m")
    print("-" * 40)

    # --- Plotting ---
    print("Plotting results (including Top View)...")
    
    # Create output directory
    output_dir = Path("runs/simulation_verification_pictures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    ax_spd = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, 0])
    ax_map = fig.add_subplot(gs[1, 1])
    
    # 1. Speed
    real_speed_kmh = (df['speed'] * 3.6).clip(upper=120)
    # Smooth real speed
    real_speed_kmh = real_speed_kmh.rolling(window=5, center=True).mean().fillna(0)
    
    ax_spd.plot(df.dist/1000, real_speed_kmh, label='Real', alpha=0.5, color='blue')
    ax_spd.plot(sim_df.dist/1000, sim_df.speed_kmh, label='Sim', alpha=0.8, color='red', linestyle='--')
    ax_spd.set_ylabel('Speed (km/h)')
    ax_spd.set_title('Speed vs Distance')
    ax_spd.legend()
    
    # 2. Time (Moving Time vs Distance)
    # Filter pauses for the plot to avoid vertical jumps
    # We construct a cumulative moving time axis
    df['is_moving'] = df['speed'] > 0.5
    df['moving_dt'] = df['dt'] * df['is_moving']
    real_moving_time_cum = df['moving_dt'].cumsum()
    
    ax_time.plot(df.dist/1000, real_moving_time_cum, label='Real (Moving)', color='blue')
    ax_time.plot(sim_df.dist/1000, sim_df.time, label='Sim', color='red', linestyle='--')
    ax_time.set_ylabel('Time (s)')
    ax_time.set_title('Moving Time vs Distance')
    ax_time.legend()
    
    # 3. Profile
    ax_prof.fill_between(df.dist/1000, df.alt, alpha=0.3, color='gray', label='Elevation')
    ax_prof.set_ylabel('Elevation (m)')
    ax2 = ax_prof.twinx()
    ax2.plot(sim_df.dist/1000, sim_df.wind_head, label='Headwind (m/s)', color='green', alpha=0.3)
    ax2.set_ylabel('Headwind (m/s)')
    ax_prof.set_xlabel('Distance (km)')
    ax_prof.set_title('Profile & Wind')
    
    # 4. Top View (Map)
    ax_map.plot(df.lon, df.lat, color='black', alpha=0.5, label='Course')
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    ax_map.set_title('Course Map with Wind')
    ax_map.axis('equal')
    
    # Add wind arrows (Quiver)
    # Downsample significantly for arrows
    arrow_step = max(1, len(sim_df) // 30) # ~30 arrows
    subset = sim_df.iloc[::arrow_step]
    
    if enable_wind and weather_mgr:
        # subset has lat, lon, w_E, w_N
        # Quiver takes X, Y, U, V
        # U=w_E (East is X), V=w_N (North is Y)
        ax_map.quiver(subset.lon, subset.lat, subset.w_E, subset.w_N, color='teal', scale=50, width=0.005, label='Wind')
    
    ax_map.legend()
    
    plt.tight_layout()
    
    # Determine output filename
    fit_name = Path(fit_file_path).stem
    output_file = output_dir / f"{fit_name}_verification.png"
    plt.savefig(output_file)
    print(f"Verification plot saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to .fit or .gpx file")
    parser.add_argument("--no-wind", action="store_true", help="Disable wind physics")
    parser.add_argument("--no-weather", action="store_true", help="Disable fetching real weather")
    
    args = parser.parse_args()
    
    verify_ride(
        args.file,
        enable_wind=not args.no_wind,
        weather_enable=not args.no_weather
    )
