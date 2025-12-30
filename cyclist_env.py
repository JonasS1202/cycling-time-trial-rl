import gymnasium as gym
import numpy as np
import math

import config
import physics_engine
import utils
import gpx_segmenter

class CyclistITTEnv(gym.Env):
    """
    A custom Gymnasium environment for a cyclist in an Individual Time Trial (ITT).
    
    Configuration is loaded from config.py.
    """
    metadata = {"render_modes": []}

    def __init__(self, course_length_km=None, time_step_s=None, continuous=False, gpx_file=None):
        super().__init__()

        # === Simulation Parameters ===
        # Use config defaults if not overridden
        self.time_step = time_step_s if time_step_s else config.Simulation.PHYSICS_TIME_STEP

        
        # Course Length logic
        if course_length_km is None:
            course_length_km = config.Simulation.DEFAULT_COURSE_LENGTH_KM

        # === State Variables ===
        self.start_elevation = 0.0

        if gpx_file:
            print(f"Loading course from GPX: {gpx_file}")
            self._load_course_from_gpx(gpx_file)
        else:
            # Default Synthetic Course
            self.course_length = course_length_km * 1000  # meters
            
            # Simple course profile (synthetic)
            self.gradients = np.array(
                [0.0] * int(self.course_length * 0.60) +
                [0.03] * int(self.course_length * 0.10) +
                [0.0] * int(self.course_length * 0.05) +
                [-0.03] * int(self.course_length * 0.10) +
                [0.0] * int(self.course_length * 0.15)
            )
            
            # Calculate elevation profile from gradients for synthetic course
            self.elevation_profile = np.zeros(len(self.gradients))
            current_ele = 0.0 # Synthetic always starts at 0
            for i, grad in enumerate(self.gradients):
                current_ele += grad # 1m step
                self.elevation_profile[i] = current_ele
                
            # Compute Segments and Ascent for Synthetic too
            self._compute_internal_segments()
            self._compute_cumulative_ascent()

            # Initialize bearings for synthetic course (default North 0.0)
            self.bearings = np.zeros(len(self.gradients))
        
        # === Dynamic Scalability & Limits ===
        course_km = self.course_length / 1000.0
        self.max_time = course_km * config.Simulation.TIME_LIMIT_FACTOR 
        self.baseline_time = course_km * config.Simulation.BASELINE_TIME_FACTOR
        
        # === Rewards Scaling ===
        self.time_bonus_const = config.Rewards.TIME_BONUS_FACTOR_PER_KM * course_km
        self.no_bonk_bonus_const = config.Rewards.NO_BONK_BONUS_FACTOR_PER_KM * course_km
        self.avg_power_bonus_const = config.Rewards.AVERAGE_POWER_BONUS_FACTOR_PER_KM * course_km
        self.w_prime_penalty_const = config.Rewards.W_PRIME_LEFTOVER_PENALTY_CONST 
        
        # === Cyclist Physiology ===
        self.critical_power = config.Physiology.CRITICAL_POWER
        self.w_prime_max = config.Physiology.W_PRIME_MAX
        self.w_prime_balance = self.w_prime_max
        self.recovery_threshold = config.Physiology.RECOVERY_THRESHOLD_FRACTION * self.w_prime_max

        # === Cyclist Physics ===
        self.mass_kg = config.Physics.TOTAL_MASS
        self.cda = config.Physics.CDA
        self.crr = config.Physics.CRR
        self.g = config.Physics.GRAVITY
        self.rho = config.Physics.AIR_DENSITY

        # === State Variables ===
        self.distance_covered = 0.0
        self.speed = 0.0
        self.power_output = 0.0
        self.time_elapsed = 0
        self.is_exhausted = False
        self.has_bonked_this_episode = False
        self.internal_logging_enabled = False
        
        
        # Wind State (Randomized per episode)
        self.wind_speed = 0.0
        self.wind_direction_global = 0.0 # Degrees 0-360
        
        # Initialize defaults immediately if Fixed Wind is enabled 
        # (so that optimal power calculation in __init__ is accurate)
        if config.Observation.USE_FIXED_WIND:
            self.wind_speed = config.Observation.FIXED_WIND_SPEED
            self.wind_direction_global = config.Observation.FIXED_WIND_DIRECTION
        
        # === Observation Features ===
        # 1. Dist remaining, 2. W', 3. Exhausted, 4. Action, 5. Speed, 6. Current Grad
        # 7. Current Axial Wind (NEW)
        # 8. Ascent
        # 9-16. Lookahead
        # 17+. Segments (current + 4 future) * (Dist, Grad, AvgAxialWind)
        
        n_base = 7 # Added Wind
        n_ascent = 1
        n_lookahead = len(config.Observation.LOOKAHEAD_DISTANCES)
        n_segments = config.Observation.N_SEGMENTS_FORESIGHT * 3 # Added Wind
        
        self.n_features = n_base + n_ascent + n_lookahead + n_segments
        self.n_stack = config.Observation.N_STACK 
        
        self.obs_queue = np.zeros((self.n_stack, self.n_features), dtype=np.float32)

        # === Action Space ===
        self.use_fixed_segments = config.Observation.USE_FIXED_SEGMENTS
        self.power_multipliers = config.Action.POWER_MULTIPLIERS
        self.continuous = continuous
        
        if self.continuous:
            if self.use_fixed_segments:
                # 1D Action: [Power]
                # Map to [0.5*CP, 1.5*CP]
                print(f"Using Fixed Segments mode (from gpx_segmenter). Action Space is 1D (Power) [0.0*CP, {config.Action.CONTINUOUS_POWER_RANGE_FACTOR}*CP].")
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            else:
                # 2D Action: [Power, Distance]
                # [-1, 1] -> [MIN, MAX] for Distance
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(len(self.power_multipliers))

        # === Observation Space ===
        total_obs_dim = self.n_stack * self.n_features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )
        
        # Calculate optimal constant power time
        self.optimal_cp_time, self.optimal_cp_watts = self._calculate_optimal_constant_power_time()
        print(f"Optimal Constant Power Time: {self.optimal_cp_time:.2f}s @ {self.optimal_cp_watts:.1f}W")

    def _load_course_from_gpx(self, gpx_path):
        """
        Parses GPX file using gpx_segmenter and populates environment state.
        Ensures consistency with the segmenter tool.
        """
        # Call the shared segmentation logic
        # This will also generate the plot if generate_plot=True!
        # We probably want to generate the plot once on init to satisfy the user request:
        # "everytime i start a train run... segmented.png is generated"
        print(f"Using gpx_segmenter to load and segment course: {gpx_path}")
        grid_df, segments_indices, original_df = gpx_segmenter.run_process_course(gpx_path, generate_plot=True)
        
        # 1. Basic Course Properties
        # grid_df uses a regular 10m grid (by default in gpx_segmenter). 
        # BUT CyclistITTEnv operates on a 1m grid (implied by self.gradients array where index ~ meters?)
        # OLD LOGIC: self.gradients was np.gradient(ele_new, 1.0), so 1 index = 1 meter.
        # gpx_segmenter grid_df has 'dist' column, and resolution is ~10m.
        
        # PROBLEM: gpx_segmenter uses a 10m smoothing grid. 
        # The Gym Env loop updates distance continuously, but looks up gradient based on index=int(dist).
        # If we use 10m grid from segmenter, we need to interpolate it back to 1m for the Env to work as designed?
        # Or we adapt Env to use grid_df interpolation?
        # The Env code: gradient_idx = min(int(distance), len(self.gradients) - 1)
        # This STRICTLY assumes 1 index = 1 meter.
        
        # So we MUST interpolate grid_df back to 1m resolution for the Environment arrays.
        
        max_dist = grid_df.dist.max()
        self.course_length = max_dist
        self.start_elevation = original_df.iloc[0].ele
        
        # Create 1m grid
        x_new = np.arange(0, int(self.course_length) + 1, 1.0)
        
        # Interpolate from grid_df (which is already smoothed)
        # We rely on grid_df's data as the "Truth" for segments.
        self.elevation_profile = np.interp(x_new, grid_df.dist, grid_df.ele)
        self.lats = np.interp(x_new, grid_df.dist, grid_df.lat)
        self.lons = np.interp(x_new, grid_df.dist, grid_df.lon)
        
        if 'bearing' in grid_df.columns:
            # Interpolate bearings carefully (cyclic)
            b_rad = np.radians(grid_df.bearing)
            b_sin = np.sin(b_rad)
            b_cos = np.cos(b_rad)
            
            sin_interp = np.interp(x_new, grid_df.dist, b_sin)
            cos_interp = np.interp(x_new, grid_df.dist, b_cos)
            
            self.bearings = np.degrees(np.arctan2(sin_interp, cos_interp)) % 360
        else:
            self.bearings = np.zeros(len(x_new))
            
        # Calculate Gradients on this 1m grid
        # Note: gpx_segmenter has 'grad_smooth'. 
        # We should use that interpolated? Or re-calculate?
        # If segments were defined based on 'grad_smooth', we should use that to remain consistent.
        # Interpolating 'grad_smooth' is safer than re-deriving elevation which might verify slightly.
        self.gradients = np.interp(x_new, grid_df.dist, grid_df.grad_smooth) / 100.0 # Convert % to slope (0.05)
        
        # Clip gradients to physics limits
        self.gradients = np.clip(self.gradients, -0.30, 0.30)

        # 2. Segments
        # gpx_segmenter returns 'segments' as INDICES into grid_df.
        # We need to map these to our new 1m-based reality and store them formatted for the Env.
        
        self.segments = []
        
        # grid_df indices -> Distance -> Env Indices (which are == Distance in meters)
        
        for i in range(len(segments_indices)):
            idx_start = segments_indices[i]
            idx_end = segments_indices[i+1] if i < len(segments_indices)-1 else len(grid_df)-1
            
            dist_start = grid_df.iloc[idx_start].dist
            dist_end = grid_df.iloc[idx_end].dist
            
            # Convert to Env Indices (int meters)
            env_start = int(dist_start)
            env_end = int(dist_end)
            
            # Calculate average gradient for this segment as perceived by the Segmenter
            # (To be consistent with what the plot showed)
            # grid_df uses 'grad_smooth'.
            avg_grad_pct = grid_df.iloc[idx_start:idx_end].grad_smooth.mean()
            
            # Calculate Average Bearing
            if 'bearing' in grid_df.columns:
                segment_bearings = grid_df.iloc[idx_start:idx_end].bearing
                rads = np.radians(segment_bearings)
                avg_sin = np.mean(np.sin(rads))
                avg_cos = np.mean(np.cos(rads))
                avg_bearing = (math.degrees(math.atan2(avg_sin, avg_cos)) + 360) % 360
            else:
                avg_bearing = 0.0
            
            self.segments.append({
                'start': env_start,
                'end': env_end,
                'avg_grad': avg_grad_pct, # Keep as percentage for features? or slope? 
                                          # _get_obs uses: g_val = float(seg['avg_grad'])
                                          # And normalizes: g = segment_foresight[i+1] * GD_SCALE
                                          # If GD_SCALE expects slope, this is wrong.
                                          # Config check: GD_SCALE=10.0 (usually). 
                                          # Previous code: self.gradients is slope. 
                                          # avg_grad was "final_avg" of self.gradients.
                                          # So avg_grad should be SLOPE (0.05 for 5%).
                'length': env_end - env_start,
                'avg_bearing': avg_bearing
            })
            
        # Correction: avg_grad in previous structure was from slope-scaled gradients.
        # So I must divide percentage by 100.
        for s in self.segments:
            s['avg_grad'] = s['avg_grad'] / 100.0

        # Optimization: Map every index to a segment ID for fast lookup
        self.segment_map = np.zeros(len(self.gradients), dtype=np.int32)
        for seg_id, seg in enumerate(self.segments):
            # Clip end to avoid out of bounds
            s = seg['start']
            e = min(seg['end'], len(self.segment_map))
            self.segment_map[s:e] = seg_id
               
        # Compute Ascent
        self._compute_cumulative_ascent()



    def _compute_cumulative_ascent(self):
        """
        Pre-calculates the cumulative ascent remaining from each point.
        self.cumulative_ascent_remaining[i] = Sum of all positive gradients from i to end.
        """
        # Gradients are effectively "meters climbed per meter forward" (since update step is 1m)
        # So positive gradient value directly equals meters climbed in that step.
        pos_grads = np.maximum(self.gradients, 0.0)
        # We want cumulative sum from right to left (remaining)
        # cumsum from right: flip, cumsum, flip back
        self.cumulative_ascent_remaining = np.flip(np.cumsum(np.flip(pos_grads)))

    def _compute_internal_segments(self):
        """
        Segments the track based on gradient changes AND wind conditions (if enabled).
        Populates self.segments as list of (start_idx, end_idx, avg_grad).
        """
        # Uses config parameters
        grad_threshold = config.Observation.SEGMENT_GRADIENT_THRESHOLD
        min_len = config.Observation.SEGMENT_MIN_LENGTH 
          
        wind_enabled = config.Observation.WIND_SEGMENTATION_ENABLE
        wind_grad_limit = config.Observation.WIND_SEGMENTATION_GRADIENT_THRESHOLD
        hw_angle = config.Observation.WIND_HEADWIND_ANGLE
        tw_angle = config.Observation.WIND_TAILWIND_ANGLE
          
        # Determine Wind Conditions (Fixed or Current Instance State?)
        # Since this method is usually called ONCE at init (unless dynamic), 
        # we must use the wind stored in self.
        # But wait, self.wind_speed is randomized in reset().
        # If we want STATIC segmentation for the RL (so obs size is consistent?), 
        # using dynamic wind is problematic unless we re-segment every episode.
        # BUT, the user requested "FIXED WIND ONLY" for training.
        # So we can safely use the config fixed values OR self values if initialized.
          
        # If FIXED WIND is on, we use it. 
        if config.Observation.USE_FIXED_WIND:
            w_speed = config.Observation.FIXED_WIND_SPEED
            w_dir = config.Observation.FIXED_WIND_DIRECTION
        else:
            # Fallback to current instance wind if set, or 0
            w_speed = getattr(self, 'wind_speed', 0.0)
            w_dir = getattr(self, 'wind_direction_global', 0.0)
               
        # Helper to classify wind
        def get_wind_category(bearing):
            # Relative angle
            # Wind Dir is FROM. Bearing is TO.
            # Headwind if diff is ~0 (Wind N, Rider N) -> cos(0)=1 (Headwind in physics engine)
            # Wait, physics_engine says:
            # rad = math.radians(wind_direction_global - bearing_deg)
            # return wind_speed * math.cos(rad)
            # So cos=1 is Headwind. cos=-1 is Tailwind.
               
            rad = math.radians(w_dir - bearing)
            cos_val = math.cos(rad)
               
            # Headwind Cone: cos(0) to cos(45) -> 1.0 to 0.707
            hw_thresh = math.cos(math.radians(hw_angle))
               
            # Tailwind Cone: cos(180) to cos(180-45) -> -1.0 to -0.707
            # i.e. cos_val < -0.707
            tw_thresh = -math.cos(math.radians(tw_angle)) # e.g. -0.707
               
            if cos_val >= hw_thresh:
                return 1 # HEADWIND
            elif cos_val <= tw_thresh:
                return -1 # TAILWIND
            else:
                return 0 # SIDEWIND / OTHER
          
        segments = [] 
        current_start_idx = 0
        seg_sum_grad = 0.0
        seg_count = 0
          
        # We need a list of indices where segments start
        segment_indices = [0]
          
        # For Wind Logic, we need to track the current "Wind Category" of the segment
        # Initialize with first point
        first_bearing = self.bearings[0] if hasattr(self, 'bearings') else 0.0
        current_wind_cat = get_wind_category(first_bearing)
          
        for i in range(len(self.gradients)):
            grad = self.gradients[i] * 100.0 # Convert to percentage
            dist_from_start = i - current_start_idx 
               
            # --- GRADIENT LOGIC ---
            if seg_count == 0:
                avg_grad = grad
            else:
                avg_grad = seg_sum_grad / seg_count
               
            grad_diff = abs(grad - avg_grad)
               
            # --- WIND LOGIC ---
            is_wind_split = False
            if wind_enabled and abs(grad) < wind_grad_limit and hasattr(self, 'bearings'):
                # Check if wind category changed
                bearing = self.bearings[i]
                new_cat = get_wind_category(bearing)
                    
                if new_cat != current_wind_cat:
                    # Wind condition changed significantly!
                    is_wind_split = True
                    current_wind_cat = new_cat # Update tracker
               
               
            # --- SPLIT DECISION ---
            # Split if:
            # 1. Gradient deviant > threshold AND min_len met
            # 2. Wind category switched AND min_len met (and we are on flat ground)
               
            can_split = (dist_from_start > min_len)
               
            grad_split = (grad_diff > grad_threshold)
               
            if can_split and (grad_split or is_wind_split):
                # Trigger Split
                segment_indices.append(i)
                current_start_idx = i
                seg_sum_grad = 0.0
                seg_count = 0
                    
                # Update loop state
                seg_sum_grad += grad
                seg_count += 1
                    
                # Reset Wind Cat for new segment (already done if it was a wind split, 
                # but if it was a gradient split, we should also re-evaluate wind cat for the new segment start)
                if hasattr(self, 'bearings'):
                    current_wind_cat = get_wind_category(self.bearings[i])
                    
            else:
                seg_sum_grad += grad
                seg_count += 1
                    
        # Now convert indices to (start, end, avg) tuples
        # Actually for O(1) lookup during step, we might want an array that maps index -> segment_id
        # But iterating 5 segments ahead is fast enough.
          
        self.segments = []
        for k in range(len(segment_indices)):
            start = segment_indices[k]
            end = segment_indices[k+1] if k < len(segment_indices)-1 else len(self.gradients)
               
            # Calculate actual average for this finalized segment
            # self.gradients is 0-1 scalled (slope). 
            segment_slice = self.gradients[start:end]
            final_avg = np.mean(segment_slice) # Slope scale
               
            self.segments.append({
                'start': start,
                'end': end,
                'avg_grad': final_avg,
                'length': end - start,
                'avg_bearing': 0.0 # Placeholder, computed below
            })

            # Compute Bearing for segment
            if hasattr(self, 'bearings'):
                segment_bearings = self.bearings[start:end]
                # Circular mean for angles
                rads = np.radians(segment_bearings)
                avg_sin = np.mean(np.sin(rads))
                avg_cos = np.mean(np.cos(rads))
                avg_bearing_rad = np.atan2(avg_sin, avg_cos)
                self.segments[-1]['avg_bearing'] = (math.degrees(avg_bearing_rad) + 360) % 360
               
        # Optimization: Map every index to a segment ID for fast lookup
        self.segment_map = np.zeros(len(self.gradients), dtype=np.int32)
        for seg_id, seg in enumerate(self.segments):
            self.segment_map[seg['start']:seg['end']] = seg_id


    def _simulate_constant_power(self, power_watts):
        """Simulate course with constant power. Returns (time, bonked)."""
        # Use SimulatedRider for cleaner physics loop
        rider = physics_engine.SimulatedRider(
            mass_kg=self.mass_kg,
            cda=self.cda,
            crr=self.crr,
            w_prime_max=self.w_prime_max,
            critical_power=self.critical_power,
            start_speed=0.1
        )
          
        dist = 0.0
        time_elapsed = 0.0
          
        while dist < self.course_length:
            # Calculate Environment State
            idx = min(int(dist), len(self.gradients)-1)
               
            # Gradient
            grad = self.gradients[idx]
               
            # Elevation (for air density)
            ele = self.elevation_profile[idx] if hasattr(self, 'elevation_profile') else 0.0
               
            # Wind
            bearing = self.bearings[idx] if hasattr(self, 'bearings') else 0.0
            axial_wind = physics_engine.calculate_axial_wind(self.wind_speed, self.wind_direction_global, bearing)
               
            # Step Physics
            # SimulatedRider returns (speed, w_prime, bonked)
            # We don't need to manually update state variables, the rider object does it.
            # But we do need to update 'dist' and check limits.
               
            speed, w_prime, bonked = rider.step(
                power_watts, grad, self.time_step, 
                elevation=ele, wind_speed_axial=axial_wind
            )
               
            if bonked:
                # If we bonk, return strictly as a failure for the optimizer
                # (Or return time so far?) 
                # Original code: if bonked: return time_elapsed, True
                return time_elapsed, True
               
            time_elapsed += self.time_step
            dist += speed * self.time_step
               
            if time_elapsed > self.max_time:
                return time_elapsed, False
                    
        return time_elapsed, False

    def _calculate_optimal_constant_power_time(self):
        """Binary search for best constant power time."""
        low = 0.0
        high = 1000.0 # Enough for cycling
        best_time = float('inf')
          
        for _ in range(15):
            mid = (low + high) / 2
            time, bonked = self._simulate_constant_power(mid)
            if bonked:
                high = mid
            else:
                if time < best_time:
                    best_time = time
                low = mid
        return best_time, low # low is the highest viable power that didn't bonk

    def _update_physics(self, dt, power_watts, speed, w_prime, distance, wind_speed_axial=0.0):
        """Shared physics update logic."""
        gradient_idx = min(int(distance), len(self.gradients) - 1)
        gradient = self.gradients[gradient_idx]

        # Air Density
        ele = self.elevation_profile[gradient_idx] if hasattr(self, 'elevation_profile') else 0.0
        rho = physics_engine.get_air_density(ele)

        # W' Balance
        w_prime = physics_engine.update_w_prime(
            w_prime, self.w_prime_max, self.critical_power, power_watts, dt
        )
        bonked = (w_prime <= 0)

        # Physics
        speed = physics_engine.update_kinematics(
            speed, power_watts, gradient, self.mass_kg, self.cda, self.crr, 
            rho, wind_speed_axial, dt, self.g
        )
          
        distance += speed * dt
          
        return speed, w_prime, distance, gradient, rho, bonked




    def _get_obs(self, action):
        gradient_idx = min(int(self.distance_covered), len(self.gradients) - 1)
        current_gradient = self.gradients[gradient_idx]
        current_ele = self.elevation_profile[gradient_idx]
          
        if gradient_idx < len(self.gradients):
            current_bearing = self.bearings[gradient_idx] if hasattr(self, 'bearings') else 0.0
        else:
            current_bearing = 0.0
               
        current_axial_wind = physics_engine.calculate_axial_wind(
            self.wind_speed, self.wind_direction_global, current_bearing
        )
          
        # === 1. Ascent Remaining ===
        if hasattr(self, 'cumulative_ascent_remaining'):
            ascent_remote = self.cumulative_ascent_remaining[gradient_idx]
        else:
            ascent_remote = 0.0
               
        # === 2. Elevation Lookahead ===
        lookahead_values = []
        for dist_add in config.Observation.LOOKAHEAD_DISTANCES:
            target_idx = min(gradient_idx + int(dist_add), len(self.gradients) - 1)
            future_ele = self.elevation_profile[target_idx]
            # Delta relative to current
            delta = future_ele - current_ele
            lookahead_values.append(delta)
               
        # === 3. Segment Foresight ===
        # Get current segment ID
        if hasattr(self, 'segment_map'):
            cur_seg_id = self.segment_map[gradient_idx]
        else:
            cur_seg_id = 0
               
        segment_foresight = []
          
        for k in range(config.Observation.N_SEGMENTS_FORESIGHT):
            target_seg_id = cur_seg_id + k
               
            if target_seg_id < len(self.segments):
                seg = self.segments[target_seg_id]
                    
                # Dist
                if k == 0:
                    d_val = float(seg['end'] - gradient_idx)
                else:
                    d_val = float(seg['length'])
                         
                g_val = float(seg['avg_grad'])
                    
                # Wind for Segment
                seg_avg_bearing = seg.get('avg_bearing', 0.0)
                w_val = physics_engine.calculate_axial_wind(
                    self.wind_speed, self.wind_direction_global, seg_avg_bearing
                )
            else:
                # Out of bounds
                d_val = 0.0
                g_val = 0.0
                w_val = 0.0
                    
            segment_foresight.append(d_val)
            segment_foresight.append(g_val)
            segment_foresight.append(w_val)

        # Normalization
        norm_dist_rem = (self.course_length - self.distance_covered) / self.course_length
        norm_w_prime = self.w_prime_balance / self.w_prime_max
        norm_speed = self.speed / 30.0 
          
        GD_SCALE = config.Observation.GRADIENT_SCALE
        ELE_SCALE = config.Observation.ELEVATION_SCALE
        WIND_SCALE = config.Observation.WIND_SCALE
          
        norm_grad = current_gradient * GD_SCALE
        norm_current_wind = current_axial_wind * WIND_SCALE
        norm_ascent = ascent_remote * ELE_SCALE
          
        norm_lookahead = [val * ELE_SCALE for val in lookahead_values]
          
        # Segment Normalization [Dist, Grad, Wind]
        norm_segments = []
        for i in range(0, len(segment_foresight), 3):
            d = segment_foresight[i] / 1000.0
            g = segment_foresight[i+1] * GD_SCALE
            w = segment_foresight[i+2] * WIND_SCALE
            norm_segments.append(d)
            norm_segments.append(g)
            norm_segments.append(w)

        try:
            # Handle 2D action [Power, Distance] - optimize to show Power
            if hasattr(action, 'shape') and len(action.shape) > 0:
                action_val = float(action[0])
            elif isinstance(action, (list, tuple)):
                action_val = float(action[0])
            else:
                action_val = float(action)
        except:
            action_val = 0.0

        features = [
            norm_dist_rem,          
            norm_w_prime,           
            float(self.is_exhausted), 
            action_val,             
            norm_speed,             
            norm_grad,     
            norm_current_wind, 
            norm_ascent
        ] + norm_lookahead + norm_segments
          
        return np.array(features, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.distance_covered = 0.0
        self.speed = 0.1 
        self.power_output = 0.0
        self.time_elapsed = 0

        self.obs_queue = np.zeros((self.n_stack, self.n_features), dtype=np.float32)

        self.w_prime_balance = self.w_prime_max
        self.is_exhausted = False
        self.has_bonked_this_episode = False
        self.prev_power_output = 0.0 
        self.prev_power_output = 0.0 
        self.prev_target_power = 0.0 

        # Logging Accumulators
        self.total_energy_joules = 0.0


        if config.Observation.USE_FIXED_WIND:
            self.wind_speed = config.Observation.FIXED_WIND_SPEED
            self.wind_direction_global = config.Observation.FIXED_WIND_DIRECTION
        else:
            # Randomize Wind
            max_wind = config.Observation.MAX_TRAINING_WIND_SPEED
            self.wind_speed = np.random.uniform(0.0, max_wind)
            self.wind_direction_global = np.random.uniform(0.0, 360.0)

        # Or options override?
        if options and 'wind_speed' in options:
            self.wind_speed = options['wind_speed']
        if options and 'wind_direction' in options:
            self.wind_direction_global = options['wind_direction']
               
        # print(f"Episode Wind: {self.wind_speed:.1f} m/s @ {self.wind_direction_global:.0f} deg")

        obs = self._get_obs(action=0.0) 
        for _ in range(self.n_stack):
            self.obs_queue = np.concatenate((self.obs_queue[1:], [obs]), axis=0)

        return self.obs_queue.flatten(), {}

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        
        internal_history = []
        
        # 1. PARSE ACTION
        if self.continuous:
            # Expecting shape (2,)
            # [0] -> Power
            # [1] -> Distance
             
            # Extract Power & Distance
            if hasattr(action, 'shape') and action.size > 1:
                act_power = action[0]
                act_dist = action[1]
            elif isinstance(action, (list, tuple)) and len(action) > 1:
                act_power = action[0]
                act_dist = action[1]
            else:
                # Fallback for scalar/1D-1-element
                if hasattr(action, 'item'):
                    act_power = action.item()
                else:
                    act_power = float(action)
                act_dist = 0.0
             
            # Power Mapping
            # [-1, 1] mapped to [0, FACTOR*CP]
            # Formula: Power = ((action + 1) / 2) * (FACTOR * CP)
            scale = config.Action.CONTINUOUS_POWER_RANGE_FACTOR * self.critical_power
            # Power Mapping
            # [-1, 1] mapped to [0.5*CP, 1.5*CP]
            # OLD: [0, 1.5*CP] -> Mean was 0.75 CP (Too slow)
            # NEW: [0.5*CP, 1.5*CP] -> Mean is 1.0 CP (Optimal)

            # Normalized action [-1, 1] -> [0, 1]
            norm_act = (float(act_power) + 1.0) / 2.0
            norm_act = max(0.0, min(1.0, norm_act)) # Clip safety

            # Center the action space around Critical Power
            # Center the action space around Critical Power
            min_power = 0.0 * self.critical_power
            max_power = 1.5 * self.critical_power

            self.target_power = min_power + (norm_act * (max_power - min_power))
             
            # Distance Mapping
            # [-1, 1] mapped to [MIN, MAX]
            min_d = config.Action.MIN_SEGMENT_DISTANCE
            max_d = config.Action.MAX_SEGMENT_DISTANCE
            norm_d = (float(act_dist) + 1.0) / 2.0 # [0, 1]
            # Clip just in case
            norm_d = max(0.0, min(1.0, norm_d))
            target_segment_dist = min_d + (max_d - min_d) * norm_d
        else:
            # Discrete Action
            power_multiplier = self.power_multipliers[action]
            self.target_power = power_multiplier * self.critical_power
            # Default distance if not fixed segments
            target_segment_dist = config.Action.MIN_SEGMENT_DISTANCE 

        # FIXED SEGMENT OVERRIDE
        if self.use_fixed_segments:
            # Find current segment
            # We can use self.segment_map if available for O(1)
            current_idx = min(int(self.distance_covered), len(self.gradients)-1)
            if hasattr(self, 'segment_map'):
                seg_id = self.segment_map[current_idx]
                # Get that segment's end
                if seg_id < len(self.segments):
                    seg = self.segments[seg_id]
                    # Target is the end of this segment relative to current
                    # self.segments uses Env Indices (Meters).
                    seg_end_dist = float(seg['end'])
                    dist_remaining_in_segment = seg_end_dist - self.distance_covered
                       
                    target_segment_dist = max(1.0, dist_remaining_in_segment)
            else:
                # Fallback if map not built (should not happen)
                target_segment_dist = config.Action.MIN_SEGMENT_DISTANCE

        # Locked Power Logic: Instant Application
        current_segment_power = self.target_power
        if self.is_exhausted:
            current_segment_power = config.Physiology.EXHAUSTED_RECOVERY_POWER_FACTOR * self.critical_power
            
        self.power_output = current_segment_power # Instant application

        # Calculate Smoothness Penalty (REMOVED)
        # self.delta_target_power = abs(self.target_power - self.prev_target_power)
        # smoothness_penalty = self.delta_target_power * config.Rewards.SMOOTHNESS_PENALTY_FACTOR
        # total_reward -= smoothness_penalty

        # 2. SEGMENT EXECUTION LOOP
        segment_start_dist = self.distance_covered
        segment_dist_covered = 0.0
        
        # Safety for very small steps
        if target_segment_dist < 1.0: target_segment_dist = 1.0
        
        while segment_dist_covered < target_segment_dist:
            
            # Physics Update
            # Note: W' might change state to Exhausted inside this loop?
            # If so, we should arguably update power?
            # "Locked Power" implies we TRY to hold it. 
            # But if we exhaust, we physically fail.
            # self._update_physics checks power vs CP for W' drain.
            # But updating "bonked" status...
            
            # We need to respect Exhaustion *State* updates during the segment?
            # If we bonk MID-segment, power should drop?
            # Re-evaluating "Locked Power": 
            # If I bonk, I CANNOT hold the power.
            # So I should check bonk status every tick.
            
            effective_power = self.power_output
            if self.is_exhausted:
                effective_power = config.Physiology.EXHAUSTED_RECOVERY_POWER_FACTOR * self.critical_power
                # Update self.power_output for history/consistency? 
                # Or just use effective_power for physics?
                self.power_output = effective_power 

            # 2. Physics Update
            
            # Axial Wind logic inside loop
            # Need bearing at current dist
            current_grad_idx = min(int(self.distance_covered), len(self.gradients)-1)
            bearing_at_step = self.bearings[current_grad_idx] if hasattr(self, 'bearings') else 0.0
            
            axial_wind_step = physics_engine.calculate_axial_wind(
                self.wind_speed, self.wind_direction_global, bearing_at_step
            )

            (self.speed, self.w_prime_balance, self.distance_covered, 
             gradient, self.rho, bonked) = self._update_physics(
                self.time_step, effective_power, self.speed, 
                self.w_prime_balance, self.distance_covered, 
                wind_speed_axial=axial_wind_step
            )

            # Handle Exhaustion Logic (State Machine)
            if not self.is_exhausted and bonked:
                self.is_exhausted = True
                self.has_bonked_this_episode = True
            elif self.is_exhausted and self.w_prime_balance > self.recovery_threshold:
                self.is_exhausted = False
            
            # Update Time
            self.time_elapsed += self.time_step
            
            # Step Rewards
            reward = config.Rewards.TIME_PENALTY_PER_STEP
            
            if self.is_exhausted:
                reward -= config.Rewards.EXHAUSTION_PENALTY 
                
            if self.w_prime_balance >= self.w_prime_max and effective_power < self.critical_power:
                reward -= config.Rewards.FULL_BATTERY_PENALTY 
            
            total_reward += reward

            # Track Energy for Average Power Calculation
            self.total_energy_joules += effective_power * self.time_step


            if self.internal_logging_enabled:
                internal_history.append({
                    'speed': self.speed,
                    'power_output': effective_power,
                    'w_prime_balance': self.w_prime_balance,
                    'distance_covered': self.distance_covered,
                    'gradient': gradient,
                    'rho': self.rho,
                    'time_elapsed': self.time_elapsed
                })

            # Check Termination
            if self.distance_covered >= self.course_length:
                terminated = True
                break
            if self.time_elapsed >= self.max_time:
                truncated = True
                break
            
            # Update loop var
            segment_dist_covered = self.distance_covered - segment_start_dist

        self.prev_power_output = self.power_output
        self.prev_target_power = self.target_power

        # 3. TERMINAL BONUSES
        if terminated:
            if self.time_elapsed > 0: 
                time_bonus = self.time_bonus_const * (self.baseline_time / self.time_elapsed)
                total_reward += time_bonus
            
            if not self.has_bonked_this_episode:
                total_reward += self.no_bonk_bonus_const

            if self.w_prime_balance > 0:
                percent_left = self.w_prime_balance / self.w_prime_max
                leftover_penalty = self.w_prime_penalty_const * percent_left
                total_reward -= leftover_penalty
               
            # 3b. Baseline Proximity Bonus
            # Reward for being close to the optimal time (within tolerance)
        #   baseline_threshold = self.optimal_cp_time * config.Rewards.BEAT_BASELINE_TOLERANCE
               
        #   if self.time_elapsed < baseline_threshold:
        #        # 1. Base Hurdle Bonus
        #        total_reward += config.Rewards.BEAT_BASELINE_HURDLE_BONUS
                    
        #        # 2. Shaped Bonus based on how much faster than the threshold we are
        #        # Determine "margin"
        #        margin_sec = baseline_threshold - self.time_elapsed
                    
        #        # Apply exponent for shaping
        #        margin_shaped = margin_sec ** config.Rewards.BEAT_BASELINE_EXPONENT
                    
        #        total_reward += margin_shaped * config.Rewards.BEAT_BASELINE_BONUS_PER_SEC

            # 3b. Average Power Bonus
            if self.time_elapsed > 0:
                avg_pwr = self.total_energy_joules / self.time_elapsed
                total_reward += avg_pwr * self.avg_power_bonus_const


        # GLOBAL SCALING
        total_reward *= config.Rewards.REWARD_SCALING

        current_observation = self._get_obs(action)
        self.obs_queue = np.concatenate((self.obs_queue[1:], [current_observation]), axis=0)
        
        info = {
            'internal_history': internal_history
        }

        if terminated or truncated:
            avg_pwr = 0.0
            if self.time_elapsed > 0:
                avg_pwr = self.total_energy_joules / self.time_elapsed
               
            info['episode_stats'] = {
                'time_s': self.time_elapsed,
                'bonked': self.has_bonked_this_episode,
                'avg_power_w': avg_pwr,
                'distance_km': self.distance_covered / 1000.0
            }


        return self.obs_queue.flatten(), total_reward, terminated, truncated, info