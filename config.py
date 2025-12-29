
"""
Configuration file for Cycling RL Environment.
All physics, physiology, and reward parameters are defined here.
"""

class Physics:
    """Physical constants and rider parameters."""
    GRAVITY = 9.81          # m/s^2
    AIR_DENSITY = 1.225     # kg/m^3 (rho) at sea level
    RIDER_MASS = 68.0       # kg
    BIKE_MASS = 10.0        # kg
    TOTAL_MASS = RIDER_MASS + BIKE_MASS # kg
    CDA = 0.42              # Drag Coefficient * Area (m^2)
    CRR = 0.004             # Coefficient of Rolling Resistance

class Physiology:
    """Cyclist physiological capabilities."""
    CRITICAL_POWER = 287.5  # Watts (CP / FTP)
    W_PRIME_MAX = 24600.0   # Joules (Anaerobic Work Capacity)
    
    # Recovery Logic
    # Power below this threshold allows recovery (simplified model often uses CP, 
    # but here we might have a specific recovery zone if needed).
    # In the current env, recovery happens whenever P < CP.
    
    # Bonk / Exhaustion
    RECOVERY_THRESHOLD_FRACTION = 0.8 # Fraction of W' needed to exit "Exhausted" state based on max W'
    EXHAUSTED_RECOVERY_POWER_FACTOR = 0.25 # Factor of CP to use when exhausted (recovery mode)

class Simulation:
    """Time and resolution settings."""
    PHYSICS_TIME_STEP = 1.0 # seconds (Internal physics update freq)

    
    # Default Course Generation (if no GPX provided)
    DEFAULT_COURSE_LENGTH_KM = 10.0

    # GPX Processing
    GPX_SMOOTHING_WINDOW = 100 # Window size (meters) for smoothing elevation data
    
    # Time Limits
    # Multipliers to calculate dynamic time limits based on course length
    # Limit = distance_km * TIME_LIMIT_FACTOR (e.g., 10km * 360 = 3600s = 1h. Implies 10km/h min speed)
    TIME_LIMIT_FACTOR = 360.0 
    
    # Baseline for Pacing Bonus
    # Target = distance_km * BASELINE_FACTOR (e.g., 10km * 120 = 1200s = 20min. Implies 30km/h avg)
    BASELINE_TIME_FACTOR = 120.0

class Rewards:
    """
    Reward function coefficients.
    Some rewards are scaled by course length (km) to maintain magnitude consistency.
    """
    # 1. Time Penalty (Base "Existence" Penalty)
    # 1. Time Penalty (Base "Existence" Penalty)
    TIME_PENALTY_PER_STEP = -0.01
    
    # 2. Terminal Speed Bonus
    # Multiplier for the time bonus: CONST * (Baseline / Time)
    # This is scaled by distance in the env: val * course_km
    TIME_BONUS_FACTOR_PER_KM = 5.0
    
    # 3. Bonk Avoidance Bonus (Terminal)
    # Bonus for finishing without hitting 0 W'
    # This is scaled by distance in the env: val * course_km
    NO_BONK_BONUS_FACTOR_PER_KM = 0 #2.0
    
    # 3b. Average Power Bonus (Terminal)
    # Small bonus for high average power to guide the agent
    # Scaled by course length (km)
    AVERAGE_POWER_BONUS_FACTOR_PER_KM = 0 #0.02

    # 3b. Beat Baseline Bonus
    # Reward for getting close to or beating the baseline time.
    # We use a tolerance factor (e.g., 1.10 = within 10% of baseline) to start rewarding early.
    BEAT_BASELINE_TOLERANCE = 1.0 
    
    # Bonus for simply being within the tolerance window (Binary or Base)
    BEAT_BASELINE_HURDLE_BONUS = 0.0
    
    # Scaled bonus: ((Baseline * Tolerance) - Time) * BONUS_PER_SEC
    # This creates a gradient pulling the agent towards 0 time.
    BEAT_BASELINE_BONUS_PER_SEC = 0.0 
    
    # Exponent for the time difference to emphasize being faster
    # If 1.0, linear. If > 1.0, faster times get exponentially more reward.
    BEAT_BASELINE_EXPONENT = 1.0

    
    # 4. Energy Efficiency (Terminal)
    # Penalty for leaving too much energy in the tank at the finish
    W_PRIME_LEFTOVER_PENALTY_CONST = 0 #20.0
    
    # 5. Smoothness (REMOVED)
    # SMOOTHNESS_PENALTY_FACTOR = 0
    
    # 6. Exhaustion Pain (Step)
    # Penalty applied every step while in "Exhausted" state
    EXHAUSTION_PENALTY = 0 #0.05

    # 7. Full Battery Underutilization (Step)
    # Penalty for power < CP while W' is full
    FULL_BATTERY_PENALTY = 0
    
    # 8. Global Scaling
    # The final total reward is scaled by this before being returned to the agent
    REWARD_SCALING = 1.0

class Observation:
    """Normalization and feature settings."""
    # Scaling factor for gradients (e.g. 20% slope -> 1.0 normalized)
    GRADIENT_SCALE = 5.0 
    
    # Wind/Air Config
    WIND_SCALE = 0.1 # 10m/s -> 1.0
    MAX_TRAINING_WIND_SPEED = 5.0 # m/s
    
    # Foresight Configuration
    # Lookahead points for elevation preview (meters ahead of current position)
    # Minimal Mode: No lookahead
    LOOKAHEAD_DISTANCES = [] 
    
    # Number of future segments to include in observation (including current)
    # Minimal Mode: 0 (Only rely on current state)
    N_SEGMENTS_FORESIGHT = 0
    
    # Observation Stacking
    N_STACK = 1

    # Scaling factor for elevation difference (e.g. 100m climb -> 1.0)
    ELEVATION_SCALE = 0.01 

    # Internal Segmentation Logic (for Environment consistency)
    SEGMENT_MIN_LENGTH = 100.0   # Meters
    SEGMENT_GRADIENT_THRESHOLD = 2.5 # Percent Change

    # Smart Segmentation (Wind)
    WIND_SEGMENTATION_ENABLE = True
    WIND_HEADWIND_ANGLE = 60.0 # Degrees +/- from direct headwind
    WIND_TAILWIND_ANGLE = 60.0 # Degrees +/- from direct tailwind
    WIND_SEGMENTATION_GRADIENT_THRESHOLD = 5.0 # % gradient. If abs(grad) > this, strictly use gradient logic.

    # Fixed Weather (Training)
    USE_FIXED_WIND = True
    FIXED_WIND_SPEED = 5.0 # m/s
    FIXED_WIND_DIRECTION = 180.0

    # Fixed Segments (Agent only controls Power)
    USE_FIXED_SEGMENTS = True

class Action:
    """Discrete Action Space settings."""
    # Continuous Action Range (Fraction of CP)
    CONTINUOUS_POWER_RANGE_FACTOR = 1.5 # Range: 0 to 1.5 * CP

    # Segment Distance Constraints (in meters)
    MIN_SEGMENT_DISTANCE = 50.0   # Minimum segment length agent can choose
    MAX_SEGMENT_DISTANCE = 1000.0 # Maximum segment length agent can choose

    # Discrete power multipliers relative to CP
    POWER_MULTIPLIERS = [
        0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 
        1.0, 
        1.02, 1.05, 1.10, 1.20, 1.3, 1.4, 1.5
    ]
