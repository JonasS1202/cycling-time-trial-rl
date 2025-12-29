"""
GPX Segmenter
-------------
Segments a GPX file based on gradient changes and visualizes the results.

Usage:
    python gpx_segmenter.py --file path/to/file.gpx
"""

import argparse
import sys
import math
import os
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpxpy
from scipy.optimize import differential_evolution
# from cyclist_env import CyclistITTEnv  <-- Moved to local scope
import config
import utils
import physics_engine


# --------------------------------------------------------------------------- #
# CONFIGURATION PARAMETERS
# --------------------------------------------------------------------------- #

# Path to the default GPX file if none provided
DEFAULT_GPX_FILE = "files/ridermanTT.gpx"

# Smoothing window for gradient calculation (meters)
# Larger window = smoother gradient, less sensitivity to short bumps
SMOOTHING_WINDOW_METERS = config.Simulation.GPX_SMOOTHING_WINDOW

# Minimum length of a segment (meters)
# Segments shorter than this will be merged or ignored to avoid noise
MIN_SEGMENT_LENGTH_METERS = config.Observation.SEGMENT_MIN_LENGTH

# Gradient Change Threshold (%)
# How much the average gradient needs to shift to trigger a NEW segment.
# e.g., 2.0 means if we are at 0% and it goes to 2.5%, that's a split.
# But if it goes to 1.5%, we might treat use a different logic.
# Simple logic: If abs(current_smoothed_grad - current_segment_avg) > THRESHOLD
GRADIENT_CHANGE_THRESHOLD = config.Observation.SEGMENT_GRADIENT_THRESHOLD

# --------------------------------------------------------------------------- #
# UTILS
# --------------------------------------------------------------------------- #




def calculate_gradient(df, smoothing_window=SMOOTHING_WINDOW_METERS):
    """Calculates smoothed gradient."""
    # We want a gradient that represents the "trend", not noise.
    # 1. Resample to regular distance intervals (e.g. every 10m)
    # This makes smoothing window consistent.
    
    max_dist = df.dist.max()
    # Create regular grid
    x_grid = np.arange(0, max_dist, 10.0) # 10m resolution
    
    # Interpolate Elevation
    ele_interp = np.interp(x_grid, df.dist, df.ele)
    lat_interp = np.interp(x_grid, df.dist, df.lat)
    lon_interp = np.interp(x_grid, df.dist, df.lon)
    
    # Create DataFrame first to handle smoothing easily
    if 'bearing' in df.columns:
         # Circular Interpolation for Bearings
         # 1. Convert to radians
         b_rad = np.radians(df.bearing)
         b_sin = np.sin(b_rad)
         b_cos = np.cos(b_rad)
         
         # 2. Interpolate sin and cos components
         sin_interp = np.interp(x_grid, df.dist, b_sin)
         cos_interp = np.interp(x_grid, df.dist, b_cos)
         
         # 3. Reconstruct bearing
         bearing_interp = np.degrees(np.arctan2(sin_interp, cos_interp)) % 360
         
         grid_df = pd.DataFrame({
             'dist': x_grid, 
             'ele': ele_interp, 
             'lat': lat_interp, 
             'lon': lon_interp,
             'bearing': bearing_interp
         })
    else:
         grid_df = pd.DataFrame({
             'dist': x_grid, 
             'ele': ele_interp, 
             'lat': lat_interp, 
             'lon': lon_interp
         })


    # SMOOTHING WINDOW (Lat, Lon, Ele)
    # The user requested to smooth these values using the GPX_SMOOTHING_WINDOW.
    # Note: We do NOT smooth bearing anymore (User Request).
    
    points_window = int(smoothing_window / 10.0)
    if points_window < 1: points_window = 1
    
    # Apply smoothing
    if points_window > 1:
        # Smooth position and elevation
        cols_to_smooth = ['lat', 'lon', 'ele']
        for col in cols_to_smooth:
            grid_df[col] = grid_df[col].rolling(window=points_window, center=True, min_periods=1).mean()
    
    # Calculate Gradient: d_ele / d_dist * 100 (%)
    # Use a rolling window to smooth elevation changes BEFORE deriv or smooth deriv?
    # Better to smooth deriv.
    
    points_window = int(smoothing_window / 10.0)
    if points_window < 1: points_window = 1
    
    # Gradient over the window step
    # simple diff
    grid_df['d_ele'] = grid_df.ele.diff()
    grid_df['d_dist'] = grid_df.dist.diff()
    grid_df['grad_raw'] = (grid_df.d_ele / grid_df.d_dist) * 100
    
    # Rolling mean
    grid_df['grad_smooth'] = grid_df['grad_raw'].rolling(window=points_window, center=True).mean().fillna(0)
    
    return grid_df

def segment_track(grid_df, change_threshold=GRADIENT_CHANGE_THRESHOLD, min_length=MIN_SEGMENT_LENGTH_METERS, debug=False):
    """Segments the track based on gradient changes and wind."""
    
    # Config parameters for wind
    wind_enabled = config.Observation.WIND_SEGMENTATION_ENABLE
    wind_grad_limit = config.Observation.WIND_SEGMENTATION_GRADIENT_THRESHOLD
    hw_angle = config.Observation.WIND_HEADWIND_ANGLE
    tw_angle = config.Observation.WIND_TAILWIND_ANGLE
    
    # Get Wind Params (Fixed)
    if config.Observation.USE_FIXED_WIND:
         w_speed = config.Observation.FIXED_WIND_SPEED
         w_dir = config.Observation.FIXED_WIND_DIRECTION
    else:
         w_speed = 0.0
         w_dir = 0.0
    
    def get_wind_category(bearing):
         rad = math.radians(w_dir - bearing)
         cos_val = math.cos(rad)
         
         hw_thresh = math.cos(math.radians(hw_angle))
         tw_thresh = -math.cos(math.radians(tw_angle))
         
         if cos_val >= hw_thresh:
              return 1 # HEADWIND
         elif cos_val <= tw_thresh:
              return -1 # TAILWIND
         else:
              return 0 # SIDEWIND
              
    segments = [] # List of start_dist indices
    split_types = [] # List of strings: 'start', 'gradient', 'wind'
    
    current_start_idx = 0
    segments.append(current_start_idx)
    split_types.append('start')
    
    seg_sum_grad = 0.0
    seg_count = 0
    
    # We need bearings in grid_df. 
    # parse_gpx adds it to original df. 
    # calculate_gradient interpolates elevation but didn't interpolate bearing.
    # We assume parse_gpx was called. 
    # BUT calculate_gradient creates a NEW grid_df. We need to propagate bearings there.
    # We'll rely on our updated calculate_gradient having it? No, calculate_gradient wasn't updated to interp bearings.
    # I should update calculate_gradient or just grab it here? 
    # Let's just assume simple Nearest Neighbor lookup for bearing or interpolation here if missing.
    # Actually, grid_df is created in calculate_gradient. I need to ensure it has 'bearing'.
    
    if 'bearing' not in grid_df.columns:
         pass

    # Initialize Wind Cat
    if 'bearing' in grid_df.columns:
        current_wind_cat = get_wind_category(grid_df.iloc[0].bearing)
    else:
        current_wind_cat = 0

    if debug:
        with open("segment_debug.txt", "w") as log:
            log.write("Dist(m)\tLat\tLon\tGrad(%)\tBearing\tWindAng\tCosA\tWindCat\tWindStr\n")
    
    for i in range(len(grid_df)):
        new_cat = current_wind_cat # Default to avoid stale data
        grad = grid_df.iloc[i].grad_smooth
        dist = grid_df.iloc[i].dist
        
        if debug:
             with open("segment_debug.txt", "a") as log:
                 b = grid_df.iloc[i].bearing if 'bearing' in grid_df.columns else 0
                 
                 # Detailed Wind Stats
                 w_ang_diff = abs(w_dir - b)
                 rad = math.radians(w_ang_diff)
                 cos_val = math.cos(rad)
                 
                 # Determine cat using internal helper for consistency
                 # But we also want raw stats
                 w_cat_str = {1:'HEADWIND', -1:'TAILWIND', 0:'SIDEWIND'}.get(current_wind_cat, 'UNK')
                 
                 log.write(f"{dist:.1f}\t{grid_df.iloc[i].lat:.5f}\t{grid_df.iloc[i].lon:.5f}\t{grad:.2f}\t{b:.1f}\t{w_ang_diff:.1f}\t{cos_val:.3f}\t{current_wind_cat}\t{w_cat_str}\n")
        
        # Current Segment stats
        if seg_count == 0:
            avg_grad = grad
        else:
            avg_grad = seg_sum_grad / seg_count
            
        # Check divergence
        diff = abs(grad - avg_grad)
        rel_dist = dist - grid_df.iloc[current_start_idx].dist
        
        can_split = (rel_dist > min_length)
        grad_split = (diff > change_threshold)
        
        wind_split = False
        if wind_enabled and abs(grad) < wind_grad_limit and 'bearing' in grid_df.columns:
             b = grid_df.iloc[i].bearing
             new_cat = get_wind_category(b)
             if new_cat != current_wind_cat:
                  wind_split = True
                  # Do not update current_wind_cat here. Only on split.
        
        if can_split and (grad_split or wind_split):
            # Start new segment
            segments.append(i)
            # Determine reason (Prioritize gradient as it's a "harder" feature)
            reason = 'gradient' if grad_split else 'wind'
            split_types.append(reason)
            
            if debug:
                with open("segment_debug.txt", "a") as log:
                     log.write(f"\n{'='*20} SPLIT: {reason.upper()} {'='*20}\n")
                     log.write(f"Dist: {dist:.1f}m | Seg Len: {rel_dist:.1f}m\n")
                     log.write(f"Gradient: Avg {avg_grad:.2f}% -> Curr {grad:.2f}% | Diff {diff:.2f}% (Thresh {change_threshold}%)\n")
                     
                     if wind_enabled and 'bearing' in grid_df.columns:
                        b_prev = grid_df.iloc[i-1].bearing if i > 0 else 0
                        b_curr = grid_df.iloc[i].bearing
                        # Calculate angles manually for log
                        new_cat_name = {1:'HEADWIND', -1:'TAILWIND', 0:'SIDEWIND'}.get(new_cat, 'UNK')
                        old_cat_name = {1:'HEADWIND', -1:'TAILWIND', 0:'SIDEWIND'}.get(current_wind_cat, 'UNK')
                        
                        log.write(f"Wind Change: {old_cat_name} -> {new_cat_name}\n")
                        log.write(f"Bearing: {b_prev:.0f} -> {b_curr:.0f}\n")
                     log.write(f"{'='*50}\n")
            
            current_start_idx = i
            seg_sum_grad = 0.0
            seg_count = 0
            
            # Add current point to new segment
            seg_sum_grad += grad
            seg_count += 1
            
            # Reset Wind Cat (if gradient split caused it)
            if 'bearing' in grid_df.columns:
                 current_wind_cat = get_wind_category(grid_df.iloc[i].bearing)
        else:
            seg_sum_grad += grad
            seg_count += 1
            
    return segments, split_types

def plot_results(original_df, grid_df, segments, split_types, filename_base):
    """Creates a plot of the segments."""
    output_dir = Path("segments")
    output_dir.mkdir(exist_ok=True)
    
    fig, (ax_map, ax_ele) = plt.subplots(2, 1, figsize=(15, 14), gridspec_kw={'height_ratios': [1, 1]})
    
    # Define a color cycle
    colors = plt.cm.tab10.colors # 10 distinct colors
    
    # --- SUBPLOT 1: Top-Down Map View ---
    ax_map.set_title(f"Course Map: {filename_base}")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.axis('equal') 
    
    # Plot Segments with Colors
    for i in range(len(segments)):
        start_idx = segments[i]
        end_idx = segments[i+1] if i < len(segments)-1 else len(grid_df)-1
        
        # Get data for this segment
        # Use simple indexing on grid_df for lat/lon since we interpolated them
        seg_lats = grid_df.iloc[start_idx:end_idx+1].lat
        seg_lons = grid_df.iloc[start_idx:end_idx+1].lon
        
        color = colors[i % len(colors)]
        
        # Plot Line
        ax_map.plot(seg_lons, seg_lats, color=color, linewidth=3, alpha=0.8)
        
        # Mark Start of Segment with a dot
        ax_map.scatter([seg_lons.iloc[0]], [seg_lats.iloc[0]], color=color, s=50, edgecolors='black', zorder=5)

    # Start/Finish Markers
    ax_map.scatter([original_df.iloc[0].lon], [original_df.iloc[0].lat], color='green', s=150, marker='*', label='Start', zorder=10, edgecolors='black')
    ax_map.scatter([original_df.iloc[-1].lon], [original_df.iloc[-1].lat], color='blue', s=150, marker='X', label='Finish', zorder=10, edgecolors='black')
    
    ax_map.legend()

    
    # --- SUBPLOT 2: Elevation Profile ---
    ax = ax_ele 
    
    # Background "Ghost" Elevation
    ax.fill_between(original_df.dist, original_df.ele, color='gray', alpha=0.1, label='Elevation Profile')
    
    # Plot Colored Segments
    for i in range(len(segments)):
        start_idx = segments[i]
        end_idx = segments[i+1] if i < len(segments)-1 else len(grid_df)-1
        
        # Get data
        seg_dist = grid_df.iloc[start_idx:end_idx+1].dist
        seg_ele = grid_df.iloc[start_idx:end_idx+1].ele
        
        color = colors[i % len(colors)]
        
        # Plot Line
        ax.plot(seg_dist, seg_ele, color=color, linewidth=2)
        
        # Vertical Separator Line at start
        if i > 0:
            # Determine style based on split reason
            # split_types[i] corresponds to segments[i]
            reason = split_types[i]
            style = '-' if reason == 'gradient' else ':' # Solid for gradient, Dotted for wind
            linewidth = 1.0 if reason == 'gradient' else 1.5
            
            ax.axvline(x=seg_dist.iloc[0], color='black', linestyle=style, alpha=0.5, linewidth=linewidth)

    ax.set_ylabel("Elevation (m)")
    ax.set_xlabel("Distance (m)")
    ax.set_title(f"Track Segmentation: {filename_base}")
    
    # Fix Y-Axis Scaling
    min_ele = original_df.ele.min()
    max_ele = original_df.ele.max()
    ele_range = max_ele - min_ele
    if ele_range < 50: ele_range = 50 
    margin = ele_range * 0.15
    ax.set_ylim(min_ele - margin, max_ele + margin)

    # Add Gradient Labels
    y_text_pos = max_ele + (ele_range * 0.05)
    
    for i in range(len(segments)):
        start_idx = segments[i]
        end_idx = segments[i+1] if i < len(segments)-1 else len(grid_df)-1
        if start_idx == end_idx: continue
        
        # Stats
        segment_grads = grid_df.iloc[start_idx:end_idx].grad_smooth
        avg_grad = segment_grads.mean()
        
        start_dist = grid_df.iloc[start_idx].dist
        end_dist = grid_df.iloc[end_idx].dist
        mid_dist = (start_dist + end_dist) / 2
        
        color = colors[i % len(colors)]
        
        ax.text(mid_dist, y_text_pos, f"{avg_grad:.1f}%", 
                horizontalalignment='center', verticalalignment='bottom', 
                fontsize=8, color=color, fontweight='bold', rotation=45)

    # Secondary Axis for Gradient (Global view)
    ax2 = ax.twinx()
    ax2.plot(grid_df.dist, grid_df.grad_smooth, color='black', alpha=0.1, linewidth=0.5, label='Gradient Trace')
    ax2.set_ylabel("Gradient (%)")
    ax2.set_ylim(-15, 20)
    
    plt.tight_layout()
    
    out_path = output_dir / f"{filename_base}_segmented.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


class PacingOptimizer:
    """
    Helper class to encapsulate optimization logic and ensure pickleability for multiprocessing.
    """
    def __init__(self, env, segment_ranges, best_cp_power):
        self.env = env
        self.segment_ranges = segment_ranges
        self.best_cp_power = best_cp_power
        self.eval_counter = 0

    def run_simulation_logic(self, power_values, return_trace=False):
        """
        Simulates the course with the given power values.
        Matches CyclistITTEnv.step() logic including Exhaustion State Machine.
        
        Args:
            power_values: Array of target powers per segment
            return_trace: If True, returns (time, penalties, trace_dict)
                          If False, returns (time + penalties)
        """
        dist = 0.0
        speed = 0.1
        w_prime = self.env.w_prime_max
        time_elapsed = 0.0
        
        # State Machine Variables
        is_exhausted = False
        
        total_penalty = 0.0
        
        # Physics Constants from Config
        cp = self.env.critical_power
        w_prime_max = self.env.w_prime_max
        recovery_thresh = config.Physiology.RECOVERY_THRESHOLD_FRACTION * w_prime_max
        exhausted_power_factor = config.Physiology.EXHAUSTED_RECOVERY_POWER_FACTOR
        
        # Trace Data
        trace = {
            'dist': [],
            'speed': [],
            'w_prime': []
        }
        
        current_seg_idx = 0
        if not self.segment_ranges:
             return (float('inf'), 0, {}) if return_trace else float('inf')

        seg_start, seg_end = self.segment_ranges[0]
        target_power = power_values[0]
        
        # We simulate step-by-step until end of course
        while dist < self.env.course_length:
            
            # Determine Segment
            while dist >= seg_end:
                current_seg_idx += 1
                if current_seg_idx >= len(self.segment_ranges):
                     break
                seg_start, seg_end = self.segment_ranges[current_seg_idx]
                target_power = power_values[current_seg_idx]
            
            if dist >= self.env.course_length:
                break

            # 1. Determine Effective Power
            effective_power = target_power
            
            # EXHAUSTION STATE MACHINE OVERRIDE
            if is_exhausted:
                effective_power = exhausted_power_factor * cp
                
            # 2. Update Physics
            prev_speed = speed
            
            # Calculate Wind
            gradient_idx = min(int(dist), len(self.env.gradients) - 1)
            bearing = self.env.bearings[gradient_idx] if hasattr(self.env, 'bearings') else 0.0
            
            axial_wind = physics_engine.calculate_axial_wind(
                self.env.wind_speed, self.env.wind_direction_global, bearing
            )
            
            # env._update_physics returns: speed, w_prime, dist, gradient, rho, bonked
            speed, w_prime, dist, gradient, rho, bonked = self.env._update_physics(
                self.env.time_step, effective_power, speed, w_prime, dist,
                wind_speed_axial=axial_wind
            )
            
            # 3. Update State Machine
            if not is_exhausted and bonked:
                is_exhausted = True
                # Penalties for bonking?
                # For pacing optimization, we remove the hard +1.0 step penalty 
                # to allow the gradient to "feel" the speed loss naturally.
                # However, avoiding bonking is generally good.
                pass 
            elif is_exhausted and w_prime > recovery_thresh:
                is_exhausted = False
                
            time_elapsed += self.env.time_step
            
            if return_trace:
                trace['dist'].append(dist)
                trace['speed'].append(speed * 3.6)
                trace['w_prime'].append(w_prime)
                
            # Timeout Check
            if time_elapsed > self.env.max_time:
                total_penalty += 1000.0
                break
                
        if dist > self.env.course_length:
            overshoot = dist - self.env.course_length
            time_correction = overshoot / max(speed, 0.1)
            time_elapsed -= time_correction
            
        if return_trace:
            return time_elapsed, total_penalty, trace
        else:
            return time_elapsed + total_penalty

    def objective(self, x):
        self.eval_counter += 1
        
        # run_simulation_logic returns (time + penalty) from the env wrapper
        # The env wrapper in this file adds +1000.0 for timeout.
        # We need to explicitly check for bonking to add the HUGE penalty requested.
        
        # Let's call with return_trace=True to check w_prime or bonk status
        time_val, penalty_check, trace = self.run_simulation_logic(x, return_trace=True)
        
        # Huge Penalty for Bonking (User Request)
        # If w_prime hit 0 (or close), MASSIVE PENALTY.
        extra_penalty = 0.0
        
        if trace['w_prime']:
            min_w_prime = min(trace['w_prime'])
            if min_w_prime <= 0.05: # effective 0, loose tolerance
                 extra_penalty += 10000.0
        else:
             extra_penalty += 10000.0
             
        total_val = time_val + penalty_check + extra_penalty
             
        if self.eval_counter % 500 == 0:
            print(f"Eval {self.eval_counter}: {total_val:.2f} (Penalty: {extra_penalty})")
        
        return total_val


def optimize_pacing(env, segments, grid_df):
    """
    Optimizes pacing by assigning a constant power to each segment using scipy.minimize.
    
    Args:
        env (CyclistITTEnv): The physics environment.
        segments (list): List of start indices for segments.
        grid_df (pd.DataFrame): The grid dataframe used for segmentation.
        
    Returns:
        optimized_powers (np.array): Array of power values (Watts) for each segment.
        result (OptimizeResult): The result object from scipy.
        runner (callable): The simulation runner function.
        x0 (np.array): The initial guess.
    """
    print("Optimization started... This may take a minute.")
    
    # identify segment lengths in the environment
    # The environment has its own resolution (likely 1m steps if from GPX).
    # We need to map segments from grid_df (based on GPX) to the environment's distance.
    
    # Helper to simulate a full run with a given set of power values per segment
    # Returns the total time.
    
    # 1. Map segments to distance ranges
    segment_ranges = []
    for i in range(len(segments)):
        start_idx = segments[i]
        end_idx = segments[i+1] if i < len(segments)-1 else len(grid_df)-1
        
        start_dist = grid_df.iloc[start_idx].dist
        end_dist = grid_df.iloc[end_idx].dist
        
        segment_ranges.append((start_dist, end_dist))
        
    print(f" optimizing over {len(segment_ranges)} segments.")
    
    # 0. Find Optimal Constant Power Baseline for this specific segmentation
    # The Env has one, but let's calculate it here to be self-contained and accurate to the "segmentation" model if needed.
    # Actually, env._simulate_constant_power is exactly what we need.
    print("Finding optimal constant power...")
    # Use the env's built-in optimized calculation
    _, best_cp_power = env._calculate_optimal_constant_power_time()
            
    print(f"Optimal Constant Power found: {best_cp_power:.2f} W")

    # Instantiate Optimizer Class
    optimizer = PacingOptimizer(env, segment_ranges, best_cp_power)
    
    # Bounds
    bounds = [(0.0, env.critical_power * 1.5) for _ in range(len(segment_ranges))]
    
    # randomize x0 inside the bounds [0, 1.0 * CP] to match user request for safer start
    # Bounds for optimizer are still [0, 2*CP], but init is safe.
    # UPDATE: Random 0-1.0 CP averages 0.5 CP, which is too slow (triggers Timeout Penalty).
    # We should start at a "Safe but Fast" speed. e.g. 95% CP.
    x0 = np.ones(len(segment_ranges)) * best_cp_power

    # Calculate score for x0
    # Note: run_simulation_logic by default doesn't add the 10000 penalty, but we want to check if it's safe.
    initial_time = optimizer.run_simulation_logic(x0, return_trace=False)
    print(f"Optimizer Initial Score (x0): {initial_time:.2f} s")
    
    # Create Initial Population with x0 injected
    # Ensure x0 is safe under the strict "objective" rules (Huge Penalty for W' < 0.05)
    print("Verifying initial guess safety...")
    safe_x0 = x0.copy()
    safety_factor = 1.0
    
    # Iteratively reduce power if penalized
    for attempt in range(10):
        # Check score with OBJECTIVE function (includes penalties)
        score = optimizer.objective(safe_x0)
        
        if score > 5000: # Threshold indicating Huge Penalty (since ~2000 is normal)
            print(f"Initial guess x0 (factor {safety_factor:.3f}) triggered penalty (Score: {score:.2f}). reducing...")
            safety_factor *= 0.99 # Reduce by 1%
            safe_x0 = x0 * safety_factor
        else:
            print(f"Initial guess x0 validated! Score: {score:.2f}. Using factor {safety_factor:.3f}")
            break
    
    # Population size logic
    pop_factor = 15
    N = len(segment_ranges)
    M = pop_factor * N
    
    # STRATEGY: Cluster population around x0 (User Request)
    # We cannot use identical copies (DE needs diversity).
    # We will use Gaussian noise around safe_x0.
    print("Initializing population clustered around x0...")
    
    # Sigma for noise: 5% of CP seems reasonable to start
    sigma = env.critical_power * 0.05 
    
    # Generate population: centered on safe_x0
    init_pop = np.random.normal(loc=safe_x0, scale=sigma, size=(M, N))
    
    # Inject exact safe_x0 at the first slot to ensure we have the best known point
    init_pop[0] = safe_x0
    
    # Clip to bounds
    # bounds is list of tuples (min, max)
    # assuming all bounds are same (0, 1.5*CP)
    min_b = bounds[0][0]
    max_b = bounds[0][1]
    init_pop = np.clip(init_pop, min_b, max_b)

    # Use differential_evolution
    res = differential_evolution(
        optimizer.objective, 
        bounds=bounds,
        maxiter=1000,
        disp=True,
        polish=True,
        workers=8,
        tol=1e-6,    # Lower tolerance to prevent early stopping
        atol=1e-6,   # Lower absolute tolerance
        init=init_pop # Inject our population
    )
    
    return res.x, res, optimizer.run_simulation_logic, x0

def visualize_optimized_pacing(original_df, grid_df, segments, filename_base, opt_powers, env, runner):
    """
    Plots the optimized pacing strategy over the elevation profile.
    Also plots W' balance simulation result.
    """
    output_dir = Path("segments")
    output_dir.mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # --- SUBPLOT 1: Elevation & Pacing ---
    
    # Plot Elevation (Background)
    ax1.fill_between(original_df.dist, original_df.ele, color='gray', alpha=0.2, label='Elevation')
    ax1.plot(original_df.dist, original_df.ele, color='black', linewidth=1)
    ax1.set_ylabel("Elevation (m)")
    ax1.set_title(f"Optimized Pacing: {filename_base}")

    # Fix Y-Axis Scaling: Don't start at 0 if elevation is high
    min_ele = original_df.ele.min()
    max_ele = original_df.ele.max()
    ele_range = max_ele - min_ele
    if ele_range < 50: ele_range = 50 # Minimum visual range
    
    margin = ele_range * 0.15
    ax1.set_ylim(min_ele - margin, max_ele + margin)
    
    # Plot Power Segments on Twin Axis
    ax1_p = ax1.twinx()
    
    # We need to simulate the run to get the EFFECTIVE power.
    # Because if the rider bonks, actual power << target power.
    __, __, trace = runner(opt_powers, return_trace=True)
    
    # We need to map the trace "effective power" back to distance.
    # The trace gives us speed/dist/w_prime per step.
    # It does not explicitly log effective power in the simple runner unless we add it.
    # BUT, we can infer it or we can just plot "Target Power" vs "W' Balance" which explains it.
    # Actually, showing Effective Power is better.
    # Let's Modify `run_simulation_logic` to include power in trace? 
    # Or just reconstruct it? 
    # Reconstructing is hard because of the state machine.
    # Let's assume the user is happy with Target Power + W' showing the bonk.
    # Wait, the user specifically complained "recovering when on full gas".
    # This implies they see High Target Power but W' goes up.
    # This happens when Actual Power is low (due to bonk).
    # So we MUST show that Actual Power is low.
    
    # Let's plot Target Power as dashed lines, and Effective Power (roughly) or just highlight bonk regions.
    # Actually, if we look at the trace, we can find segments where W' is recovering while Target Power > CP.
    # That proves Bonking.
    
    # Let's Construct arrays for plotting steps of TARGET POWER
    plot_dist_target = []
    plot_power_target = []
    
    for i in range(len(segments)):
        start_idx = segments[i]
        end_idx = segments[i+1] if i < len(segments)-1 else len(grid_df)-1
        
        start_dist = grid_df.iloc[start_idx].dist
        end_dist = grid_df.iloc[end_idx].dist
        
        power = opt_powers[i]
        
        plot_dist_target.extend([start_dist, end_dist])
        plot_power_target.extend([power, power])
        
    ax1_p.plot(plot_dist_target, plot_power_target, color='red', linewidth=2, label='Target Power (W)')
    ax1_p.plot([0, original_df.dist.max()], [env.critical_power, env.critical_power], 'r--', alpha=0.5, label='CP')
    
    ax1_p.set_ylabel("Power (W)", color='red')
    ax1_p.tick_params(axis='y', labelcolor='red')
    
    # Highlight Bonk Areas on Subplot 1?
    # We have trace['w_prime']. Any time it's near zero, that's a bonk risk.
    # Any time it's increasing while Target > CP, that is definite Bonk State.
    
    # Let's rely on Subplot 2 for W'.
    
    # --- SUBPLOT 2: Simulated Speed & W' ---
    
    dists = trace['dist']
    speeds = trace['speed']
    w_primes = trace['w_prime']
    
    ax2.plot(dists, speeds, color='blue', label='Speed (km/h)')
    ax2.set_ylabel("Speed (km/h)", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_xlabel("Distance (m)")
    
    ax2_w = ax2.twinx()
    ax2_w.plot(dists, w_primes, color='green', linestyle='--', label="W' Balance (J)")
    ax2_w.set_ylabel("W' Balance (J)", color='green')
    ax2_w.tick_params(axis='y', labelcolor='green')
    
    plt.tight_layout()
    out_path = output_dir / f"{filename_base}_optimized.png"
    plt.savefig(out_path)
    print(f"Saved optimized plot to {out_path}")

def run_process_course(gpx_path, generate_plot=True):
    """
    Main entry point for processing a course. 
    Can be called by other modules (like CyclistITTEnv or train.py).
    
    Returns:
        grid_df (pd.DataFrame): The interpolated 10m grid with elevation, bearings, etc.
        segments (list): List of start indices.
        original_df (pd.DataFrame): The raw parsed GPX data.
    """
    print(f"Processing course: {gpx_path}")
    if not os.path.exists(gpx_path):
        raise FileNotFoundError(f"File {gpx_path} not found.")
    
    if not os.path.isdir("segments"):
        os.makedirs("segments")

    # 1. Parse
    df = utils.load_track(gpx_path)
    
    # 2. Calculate Gradient (Smoothed) & Interpolate to Grid
    grid_df = calculate_gradient(df, smoothing_window=SMOOTHING_WINDOW_METERS)
    
    # 3. Segment Track
    print(f"Segmenting track (Threshold: {GRADIENT_CHANGE_THRESHOLD}%, Min Len: {MIN_SEGMENT_LENGTH_METERS}m)...")
    segments, split_types = segment_track(grid_df, change_threshold=GRADIENT_CHANGE_THRESHOLD, min_length=MIN_SEGMENT_LENGTH_METERS, debug=True)
    print(f"Found {len(segments)} segments.")
    
    # 4. Plot (Optional)
    if generate_plot:
        filename_base = Path(gpx_path).stem
        plot_results(df, grid_df, segments, split_types, filename_base)
        
    return grid_df, segments, df

def main():
    parser = argparse.ArgumentParser(description="Segment a GPX file and optimize pacing.")
    parser.add_argument("--file", type=str, default=DEFAULT_GPX_FILE, help="Path to GPX file")
    
    # Segmentation Parameters
    parser.add_argument("--smoothing", type=float, default=SMOOTHING_WINDOW_METERS, help="Smoothing window (m)")
    parser.add_argument("--min-length", type=float, default=MIN_SEGMENT_LENGTH_METERS, help="Min segment length (m)")
    parser.add_argument("--threshold", type=float, default=GRADIENT_CHANGE_THRESHOLD, help="Gradient change threshold (%%)")
    
    # Mode
    parser.add_argument("--segment-only", action="store_true", help="Only segment and plot, do not optimize.")
    
    args = parser.parse_args()
    
    gpx_path = args.file
    
    # Use the reusable function
    grid_df, segments, original_df = run_process_course(gpx_path, generate_plot=True)
    
    if args.segment_only:
        print("Mode: Segment Only. Done.")
        return

    # Optimization Mode
    print("\n--- Running Optimization ---")
    
    # Local Import to avoid circular dependency
    from cyclist_env import CyclistITTEnv
    
    env = CyclistITTEnv(gpx_file=gpx_path)
    
    # Run Optimization
    opt_powers, res, runner, x0 = optimize_pacing(env, segments, grid_df)
    
    print(f"\nOptimization Result: {res.message}")
    print(f"Iterations: {res.nit}")
    print(f"Function Evals: {res.nfev}")
    
    # Simulate Final Result
    final_time = runner(opt_powers, return_trace=False)
    print(f"Final Optimized Time: {final_time:.2f} s")
    
    # Visualize Optimization
    filename_base = Path(gpx_path).stem
    visualize_optimized_pacing(original_df, grid_df, segments, filename_base, opt_powers, env, runner)

if __name__ == "__main__":
    main()
