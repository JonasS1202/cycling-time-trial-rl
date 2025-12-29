# evaluate.py

import argparse
import os
import torch
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# Import agents and env setup
from cyclist_env import CyclistITTEnv
from ppo import Agent as PPOAgent, make_env as make_env_ppo
from sac_continous_action import Actor as SACActor, make_env as make_env_sac
import config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False, help="Path to the trained model file")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], default="ppo", help="Algorithm used (ppo or sac)")
    parser.add_argument("--gpx-file", type=str, default=None, help="Path to GPX file")
    parser.add_argument("--baseline-power", type=float, default=None, help="Run a constant power baseline simulation (Watts)")
    parser.add_argument("--optimize-baseline", action="store_true", help="Calculate and run optimal baseline power for the conditions")
    parser.add_argument("--wind-scan", action="store_true", help="Run comparative North/South wind test")
    parser.add_argument("--wind-speed", type=float, help="Specific wind speed (m/s)")
    parser.add_argument("--wind-direction", type=float, help="Specific wind direction (deg)")
    args = parser.parse_args()
    return args


def calculate_normalized_power(power_series, window=30):
    """
    Calculates Normalized Power (NP).
    NP = (mean(rolling_mean(power, 30s)^4))^0.25
    """
    # Use pandas rolling
    if len(power_series) < window:
        return np.mean(power_series)
        
    rolling_p = power_series.rolling(window=window, min_periods=1).mean()
    np_val = np.mean(rolling_p ** 4) ** 0.25
    return np_val

def plot_track_with_speed(ax, lons, lats, speed):
    """
    Plots the track colored by speed.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    
    points = np.array([lons, lats]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a continuous norm
    norm = Normalize(vmin=np.min(speed), vmax=np.max(speed))
    lc = LineCollection(segments, cmap='jet', norm=norm, alpha=1.0)
    
    # Set the values used for colormapping
    lc.set_array(speed)
    lc.set_linewidth(4)
    ax.add_collection(lc)
    
    # Auto-scale
    ax.set_xlim(np.min(lons) - 0.001, np.max(lons) + 0.001)
    ax.set_ylim(np.min(lats) - 0.001, np.max(lats) + 0.001)
    return lc

def plot_top_view(ax, env, wind_vec, speed_series=None):
    """Plots course map and wind arrows."""
    if not hasattr(env, 'lons') or not hasattr(env, 'lats'):
         return

    # Track Heatmap or Black Line
    if speed_series is not None:
        plot_track_with_speed(ax, env.lons, env.lats, speed_series)
    else:
        ax.plot(env.lons, env.lats, color='black', alpha=0.6, label='Course', linewidth=2)

    ax.set_title(f"Course & Wind ({wind_vec[0]} m/s @ {wind_vec[1]}°)", fontsize=13)
    ax.axis('equal')
    
    # Wind Arrows
    rad = math.radians(wind_vec[1])
    u = -math.sin(rad)
    v = -math.cos(rad)
    
    # Grid of arrows
    grid_size = 12 
    idx = np.linspace(0, len(env.lons)-1, grid_size, dtype=int)
    
    for i in idx:
        ax.arrow(env.lons[i], env.lats[i], u*0.003, v*0.003, 
                 head_width=0.003, head_length=0.003, fc='teal', ec='teal', alpha=1.0, width=0.0008, zorder=50)

    # Markers
    ax.scatter(env.lons[0], env.lats[0], color='lime', s=120, label='Start', zorder=60, edgecolors='black')
    ax.scatter(env.lons[-1], env.lats[-1], color='red', s=120, marker='X', label='Finish', zorder=60, edgecolors='black')


def plot_evaluation(df, env, filename, wind_vec=(0,0), race_time=None, training_time=None):
    # Try using a nicer style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.grid(True, linestyle='--', alpha=0.6)

    fig = plt.figure(figsize=(12, 18), dpi=120)
    
    # Metrics
    avg_pwr = df.power_output.mean()
    avg_spd = (df.speed * 3.6).mean()
    np_pwr = calculate_normalized_power(df.power_output)
    
    # Title Header
    title_main = "Agent Pacing Strategy Analysis"
    
    rt_str = "N/A"
    if race_time:
        m = int(race_time // 60)
        s = int(race_time % 60)
        rt_str = f"{m}m {s}s"
        
    tt_str = training_time if training_time else "N/A"
    
    stats_str = f"Avg Power: {avg_pwr:.0f}W  |  Norm Power (NP): {np_pwr:.0f}W  |  Avg Speed: {avg_spd:.1f} km/h"
    fig.suptitle(f"{title_main}\n{stats_str}\nRace Time: {rt_str} | Training Time: {tt_str}", fontsize=16, fontweight='bold', y=0.98)
    
    # Layout
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 1.5], top=0.90, hspace=0.3)
    
    ax_profile = fig.add_subplot(gs[0])
    ax_dynamics = fig.add_subplot(gs[1], sharex=ax_profile)
    ax_map = fig.add_subplot(gs[2])
    
    dist_km = df.distance_covered / 1000.0
    
    # --- ROW 1: Profile & Power ---
    # Elevation (Background)
    ax_ele = ax_profile.twinx()
    
    if hasattr(env, 'elevation_profile'):
        # Fallback to simple mapping if proper x-axis mismatch
        full_dist_km = np.linspace(0, df.distance_covered.iloc[-1]/1000.0, len(env.elevation_profile))
        ele_data = env.elevation_profile
    else:
        full_dist_km = dist_km
        ele_data = np.zeros_like(dist_km) 
    
    min_ele = np.min(ele_data)
    max_ele = np.max(ele_data)
    margin = (max_ele - min_ele) * 0.1 if (max_ele - min_ele) > 0 else 10
    
    # Elevation: Filled Gradient Look
    ax_ele.fill_between(full_dist_km, ele_data, min_ele - 100, color='#90A4AE', alpha=0.3, zorder=0)
    ax_ele.plot(full_dist_km, ele_data, color='#607D8B', lw=1.5, zorder=1)
    ax_ele.set_ylim(min_ele - margin, max_ele + margin * 3) 
    ax_ele.set_ylabel("Elevation (m)", color='#455A64', fontsize=12)
    ax_ele.tick_params(axis='y', colors='#455A64')
    
    # Power (Foreground)
    ax_profile.plot(dist_km, df.power_output, color='#D32F2F', alpha=0.9, lw=2.0, label='Power (W)', zorder=10)
    ax_profile.axhline(avg_pwr, color='#D32F2F', linestyle='--', alpha=0.5, lw=1.5, label=f'Avg: {avg_pwr:.0f}W')
    
    ax_profile.set_ylabel("Power (W)", color='#D32F2F', fontweight='bold', fontsize=12)
    ax_profile.tick_params(axis='y', colors='#D32F2F')
    ax_profile.grid(True, alpha=0.3)
    ax_profile.set_title("Course Profile & Power Strategy", fontsize=14)
    ax_profile.legend(loc='upper right', fontsize=10)
    
    # --- ROW 2: Dynamics ---
    # Speed
    spd_kmh = df.speed * 3.6
    ax_dynamics.fill_between(dist_km, spd_kmh, 0, color='#42A5F5', alpha=0.2)
    ax_dynamics.plot(dist_km, spd_kmh, color='#1976D2', lw=2.0, label='Speed (km/h)', zorder=10)
    ax_dynamics.set_ylabel("Speed (km/h)", color='#1976D2', fontsize=12)
    ax_dynamics.tick_params(axis='y', colors='#1976D2')
    ax_dynamics.set_xlabel("Distance (km)", fontsize=12)
    
    # W' Balance
    ax_wprime = ax_dynamics.twinx()
    w_bal_kj = df.w_prime_balance / 1000.0
    ax_wprime.plot(dist_km, w_bal_kj, color='#388E3C', lw=2.5, label="W' Balance (kJ)", zorder=11)
    ax_wprime.set_ylabel("W' Balance (kJ)", color='#388E3C', fontsize=12)
    ax_wprime.tick_params(axis='y', colors='#388E3C')
    ax_wprime.set_ylim(-1, max(w_bal_kj) * 1.2)
    
    ax_dynamics.set_title("Rider Dynamics: Speed & Anaerobic Capacity", fontsize=14)
    
    # --- ROW 3: Map ---
    # Map speed to course points
    mapped_speeds = None
    if hasattr(env, 'lons') and len(env.lons) > 1:
        course_dists = np.linspace(0, env.course_length, len(env.lons))
        # Ensure agent data covers similar range
        if len(df) > 1:
             agent_dists = df.distance_covered.values
             agent_speeds = spd_kmh.values
             # Interpolate
             mapped_speeds = np.interp(course_dists, agent_dists, agent_speeds)
    
    plot_top_view(ax_map, env, wind_vec, speed_series=mapped_speeds)
    
    # Save
    plt.savefig(filename + ".png", bbox_inches='tight')
    print(f"Plot saved to {filename}.png")



def run_evaluation(env, agent, device, algo="ppo", wind_speed=0.0, wind_dir=0.0):
    """Runs one episode and records the data."""
    
    obs, info = env.reset(options={'wind_speed': wind_speed, 'wind_direction': wind_dir})
    env.unwrapped.internal_logging_enabled = True
    done = False
    
    hist = []
    
    print(f"Running evaluation episode (Algo: {algo}, Wind: {wind_speed}m/s @ {wind_dir}°)...")
    while not done:
        with torch.no_grad():
            if algo == "baseline":
                # Special case for baseline passed as 'agent' (action value)
                env_action = np.array([agent]) # Agent is the float action
            else:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                
                if algo == "ppo":
                    logits = agent.actor(obs_tensor)
                    action = torch.argmax(logits, dim=1)
                    action_np = action.cpu().numpy()
                    env_action = action_np[0]
                elif algo == "sac":
                    # Continuous SAC
                    _, _, mean_action = agent.get_action(obs_tensor)
                    action_np = mean_action.cpu().numpy()
                    env_action = action_np[0]

        # Take a step
        obs, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated

        if 'internal_history' in info:
            hist.extend(info['internal_history'])

    print("Evaluation finished.")
    total_time = 0
    if hist:
         total_time = hist[-1]['time_elapsed']
    
    return {'internal_history': hist}, total_time

def perform_evaluation(model_path, algo="ppo", gpx_file=None, args=None, training_time=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Env
    # Use SAC make_env for continuous if sac, else ppo
    if algo == "ppo":
        env_factory = make_env_ppo("CyclistITT-v0", 0, False, "eval_run", gpx_file=gpx_file)
    else:
        env_factory = make_env_sac("CyclistITT-v0", 0, 0, False, "eval_run", gpx_file=gpx_file)

    # 2. Setup Agent
    temp_vec_env = gym.vector.SyncVectorEnv([env_factory])
    
    if algo == "ppo":
        agent = PPOAgent(temp_vec_env).to(device)
    else:
        agent = SACActor(temp_vec_env).to(device)
        
    # 3. Load Model
    # Handle failure gracefully if model not found or incompatible
    try:
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        agent.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing might fail if weights don't match architecture.")

    temp_vec_env.close()
    
    # 3.5 Use Single Env for Exec
    single_env = env_factory()
    
    # Directory for plots
    run_dir = f"runs/{model_path.split('/')[-2]}"
    os.makedirs(run_dir, exist_ok=True)
    
    # 4. Logic for Wind Scan or Single Run
    if args and args.wind_scan:
        print("--- Running Wind Comparison (North vs South) ---")
        speed = config.Observation.MAX_TRAINING_WIND_SPEED
        
        # Pass 1: North Wind (0 deg)
        info_N, time_N = run_evaluation(single_env, agent, device, algo=algo, wind_speed=speed, wind_dir=0.0)
        plot_evaluation(pd.DataFrame(info_N['internal_history']), single_env.unwrapped, 
                        f"{run_dir}/eval_north_wind", wind_vec=(speed, 0.0), race_time=time_N, training_time="N/A")
        
        # Pass 2: South Wind (180 deg)
        info_S, time_S = run_evaluation(single_env, agent, device, algo=algo, wind_speed=speed, wind_dir=180.0)
        plot_evaluation(pd.DataFrame(info_S['internal_history']), single_env.unwrapped, 
                        f"{run_dir}/eval_south_wind", wind_vec=(speed, 180.0), race_time=time_S, training_time="N/A")
        
        print(f"North Wind Time: {int(time_N//60)}m {int(time_N%60)}s")
        print(f"South Wind Time: {int(time_S//60)}m {int(time_S%60)}s")
        print(f"Difference: {time_N - time_S:.1f}s")
        
    else:
        # Standard Single Run
        wind_spd = args.wind_speed if args and args.wind_speed is not None else 0.0
        wind_dir = args.wind_direction if args and args.wind_direction is not None else 0.0
        
        info, time = run_evaluation(single_env, agent, device, algo=algo, wind_speed=wind_spd, wind_dir=wind_dir)
        print(f"Race finished in {int(time // 60)}m {int(time % 60)}s.")
        print(f"Wind: {wind_spd} m/s @ {wind_dir} deg")
        
        df_hist = pd.DataFrame(info['internal_history'])
        plot_evaluation(df_hist, single_env.unwrapped, f"{run_dir}/evaluation_plot", wind_vec=(wind_spd, wind_dir), race_time=time, training_time=training_time)

    single_env.close()

def perform_baseline_evaluation(target_power, gpx_file=None):
    """Runs a simulation with constant power output."""
    print(f"Running Baseline Evaluation @ {target_power} Watts")
    
    # 1. Setup Env (Continuous for exact power control)
    env = CyclistITTEnv(continuous=True, gpx_file=gpx_file)
    
    cp = env.critical_power
    scale = config.Action.CONTINUOUS_POWER_RANGE_FACTOR * cp
    min_p = 0.0 * cp
    max_p = scale
    
    # Inverse Mapping matches cyclist_env.py:
    # Power = Min + ((Act + 1)/2) * (Max - Min)
    
    target_power = max(min_p, min(max_p, target_power))
    norm_act = (target_power - min_p) / (max_p - min_p)
    action_val = (norm_act * 2.0) - 1.0
    
    action_val = max(-1.0, min(1.0, action_val))
    
    # Reuse run_evaluation logic by passing "baseline" algo
    # We pass action_val as the 'agent'
    info, time = run_evaluation(env, action_val, None, algo="baseline", wind_speed=0.0, wind_dir=0.0)
    
    print(f"Race finished in {int(time // 60)}m {int(time % 60)}s.")
    
    os.makedirs("baselines", exist_ok=True)
    save_path = f"baselines/constant_{int(target_power)}W"
    
    df_hist = pd.DataFrame(info['internal_history'])
    plot_evaluation(df_hist, env, save_path, wind_vec=(0.0, 0.0), race_time=time, training_time="N/A")

def perform_optimized_baseline_evaluation(gpx_file=None, args=None):
    """Calculates optimal power and runs baseline simulation."""
    print("--- Running FIND BEST BASELINE Evaluation ---")
    
    # Setup Env
    env = CyclistITTEnv(continuous=True, gpx_file=gpx_file)
    
    run_dir = "baselines"
    os.makedirs(run_dir, exist_ok=True)
    
    def run_opt_pass(wind_spd, wind_dir, suffix):
        print(f"\nOptimization Pass [{suffix}] (Wind: {wind_spd}m/s @ {wind_dir}°)")
        
        # 1. Reset Env to set Wind
        # We need to force wind, but reset only sets it randomly unless we override.
        # CyclistITTEnv.reset has options to override.
        # However, _calculate_optimal_constant_power_time uses self.wind_speed
        # We must ensure self.wind_speed is set BEFORE calculating optimal power.
        
        # Reset to apply wind to internal state
        env.reset(options={'wind_speed': wind_spd, 'wind_direction': wind_dir})
        
        # 2. Find Optimal Power for CURRENT Env conditions
        opt_time, opt_watts = env._calculate_optimal_constant_power_time()
        print(f"--> Found Optimal Power: {opt_watts:.2f} W (Est. Time: {opt_time:.2f} s)")
        
        # Apply safety margin to prevent precision-error bonking
        # If we run exactly at the limit, tiny differences in stepping/float math
        # can cause a bonk. Reduce by 0.5% or 1W
        opt_watts = opt_watts * 0.995 
        print(f"    Applying safety margin -> {opt_watts:.2f} W")
        
        # 3. Scale Action
        # Re-calc action value from watts
        cp = env.critical_power
        scale = config.Action.CONTINUOUS_POWER_RANGE_FACTOR * cp
        min_p = 0.0 * cp
        max_p = scale
        
        # Inverse Mapping to match cyclist_env.py
        target_val = max(min_p, min(max_p, opt_watts))
        norm_act = (target_val - min_p) / (max_p - min_p)
        action_val = (norm_act * 2.0) - 1.0

        action_val = max(-1.0, min(1.0, action_val))
        
        # 4. Run Simulation
        # Important: pass the SAME wind to run_evaluation so it doesn't re-randomize or change
        info, time = run_evaluation(env, action_val, None, algo="baseline", wind_speed=wind_spd, wind_dir=wind_dir)
        
        save_path = f"{run_dir}/baseline_{suffix}_{int(opt_watts)}W"
        df_hist = pd.DataFrame(info['internal_history'])
        plot_evaluation(df_hist, env, save_path, wind_vec=(wind_spd, wind_dir), race_time=time, training_time="N/A")
        return time, opt_watts

    if args and args.wind_scan:
        w_spd = args.wind_speed if args.wind_speed is not None else config.Observation.MAX_TRAINING_WIND_SPEED
        
        time_N, watt_N = run_opt_pass(w_spd, 0.0, "north")
        time_S, watt_S = run_opt_pass(w_spd, 180.0, "south")
        
        print(f"\nComparison:")
        print(f"North (Power {watt_N:.1f}W): {int(time_N//60)}m {int(time_N%60)}s")
        print(f"South (Power {watt_S:.1f}W): {int(time_S//60)}m {int(time_S%60)}s")
        
    else:
        # Single run
        w_spd = args.wind_speed if args.wind_speed is not None else 0.0
        w_dir = args.wind_direction if args.wind_direction is not None else 0.0
        run_opt_pass(w_spd, w_dir, "optimized")

if __name__ == "__main__":
    args = parse_args()
    
    if args.optimize_baseline:
        perform_optimized_baseline_evaluation(args.gpx_file, args)
    elif args.baseline_power is not None:
        perform_baseline_evaluation(args.baseline_power, args.gpx_file)
    else:
        if args.model_path is None:
             print("Error: --model-path is required unless --baseline-power or --optimize-baseline is specified.")
             exit(1)
        perform_evaluation(args.model_path, args.algo, args.gpx_file, args, training_time=None)
