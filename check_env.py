
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import cyclist_env  # Registers the environment if using registration or imports class

def main():
    print("Checking CyclistITTEnv consistency with Gymnasium API...")
    
    # Initialize the environment with default parameters
    env = cyclist_env.CyclistITTEnv()
    
    # Run the check
    # warn=True will print warnings about missing features or incorrect spaces
    check_env(env, warn=True)
    
    print("Environment check complete. If no warnings above, the environment is compliant.")

if __name__ == "__main__":
    main()
