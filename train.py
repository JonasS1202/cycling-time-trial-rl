
import argparse
import sys
import subprocess
import gpx_segmenter
import config

def main():
    parser = argparse.ArgumentParser(description="Train cycling RL agent")
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True, help="Algorithm to use (ppo or sac)")
    parser.add_argument("--only-segment", action="store_true", help="Only run segmentation and generate plot, then exit.")
    
    # parse_known_args returns (namespace, list_of_unknown_args)
    # This allows --total-timesteps etc to be in list_of_unknown_args
    args, unknown_args = parser.parse_known_args()
    
    # Check for Segmentation only mode
    if args.only_segment:
        print("Running Segmentation Only...")
        # Get default file from config or guess
        # Actually, CyclistITTEnv uses a kwarg or specific path. 
        # Typically the scripts hardcode 'files/ridermanTT.gpx' or read from argument.
        # If --file was passed in unknown_args, we should grab it.
        
        gpx_file = "files/ridermanTT.gpx" # Default
        
        # Quick hack to find --file in unknown_args without complex parsing
        if "--file" in unknown_args:
             idx = unknown_args.index("--file")
             if idx + 1 < len(unknown_args):
                 gpx_file = unknown_args[idx+1]
                 
        print(f"Segmenting {gpx_file}...")
        gpx_segmenter.run_process_course(gpx_file, generate_plot=True)
        print("Segmentation complete. Plot saved.")
        return

    # We want the sub-script to see the unknown args as its own sys.argv
    # We prepend the script name (ppo.py or sac...py) so it looks like a normal invocation
    
    script_name = "ppo.py" if args.algo == "ppo" else "sac_continous_action.py"
    print(f"Starting {args.algo.upper()} training...")
    
    p = subprocess.Popen([sys.executable, script_name] + unknown_args)
    
    try:
        p.wait()
    except KeyboardInterrupt:
        print("\nMain script received interrupt. Waiting for training script to shut down gracefully...")
        try:
            p.wait()
        except KeyboardInterrupt:
            # If user presses Ctrl+C again, force kill
            print("\nForce killing training script...")
            p.kill()

if __name__ == "__main__":
    main()
