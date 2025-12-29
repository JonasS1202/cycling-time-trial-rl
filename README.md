
# Cycling RL: Wind-Aware Pacing Optimizer

A reinforcement-learning toolkit for optimizing pacing strategies in Individual Time Trials (ITT) with support for **complex terrain**, **wind physics**, and **energy management (W' balance)**.

## Features
- **Physics Engine**: Calculates speed based on power, gradient, air resistance (CdA), rolling resistance (Crr), and gravity.
- **Physiology Model**: Tracks W' Balance (Anaerobic Work Capacity) to simulate exhaustion and recovery.
- **Environment**: Gymnasium-compatible `CyclistITT-v0` environment.
- **Algorithms**: Training support for PPO (Discrete) and SAC (Continuous).
- **GPX Support**: Train/Evaluate on real-world course files.
- **Observation Space**: Robust, sector-based lookahead system for "smart" course foresight.

---

## 🚀 Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JonasS1202/cycling_rl.git
    cd cycling_rl
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

---

## 🏋️ Training the Agent

Use the `train.py` wrapper to start training. You can choose between PPO (Discrete Action Space) and SAC (Continuous Action Space).

### 1. Train with PPO (Recommended)
PPO uses discrete power levels (e.g., 0.5x CP, 1.0x CP, 1.5x CP).
```bash
python train.py --algo ppo --total-timesteps 2000000
```

### 2. Train with SAC
SAC allows the agent to essentially "choose any wattage" (Continuous).
```bash
python train.py --algo sac --total-timesteps 1000000
```

### Advanced Training Options
All extra arguments are passed to the underlying scripts (`ppo.py` / `sac_continous_action.py`).
- **Track with Tensorboard**:
    ```bash
    tensorboard --logdir runs
    ```
    *Note: Training logs are saved to `runs/ExperimentName_...`*

---

## 🗺️ Course Segmentation

You can pre-segment a GPX course to verify the gradient "chunks" the agent will see.

### Run Segmentation & Plot
```bash
python gpx_segmenter.py --file files/ridermanTT.gpx
```
* **Output**: `segments/ridermanTT_segmented.png`

### Run Segmentation via Training Script
This is useful if you want to quickly check the segmentation logic used by the environment without training.
```bash
python train.py --algo ppo --only-segment
```

---

## 🔬 Simulation & Verification

Verify the physics engine against real-world ride data (.fit or .gpx files). This compares standard physics models against recorded speed/power.

### Run Verification Simulation
```bash
python simulation.py files/ridermanTT.gpx --no-weather
```
* **Output**: Terminal stats (`Avg Speed`, `Diff` in time) + Plot in `runs/simulation_verification_pictures/`.

---

## 📈 Evaluation & Visualization

Once trained, use `evaluate.py` to watch your agent race.

### Basic Evaluation
```bash
python evaluate.py --model-path runs/YourExperimentName/ppo.cleanrl_model
```
* **Result**: Generates a `pacing_strategy.png` in the run folder showing Speed, Power, W' Balance, and Elevation.

### Evaluate on a Specific GPX Course
Test how your agent handles a real-world file.
```bash
python evaluate.py --model-path runs/YourRun/model.pt --gpx-file files/my_course.gpx
```

### Compare with Constant Power Baseline
Generate a pacing strategy for a constant power output coverage (e.g., 300W).
```bash
python evaluate.py --baseline-power 300 --gpx-file files/ridermanTT.gpx
```
* **Result**: Saves plot to `baselines/`.

### Optimize Baseline (Best Constant Power)
Finds the theoretical best constant power for a course and simulates it.
```bash
python evaluate.py --optimize-baseline --gpx-file files/ridermanTT.gpx
```

### Wind sensitivity test (North vs South)
Runs evaluation twice with opposing winds to see strategy adaptation.
```bash
python evaluate.py --model-path runs/YourModel.pt --gpx-file files/course.gpx --wind-scan --wind-speed 5.0
```

---

## 🛠️ Development & Debugging

### Check Environment Compliance
Run this to ensure the Gymnasium environment is valid and the observation space is correct.
```bash
python check_env.py
```

### Debugging GPX Segmentation Optimization
To debug the "Optimization" mode (Differential Evolution) of segmentation pacing:
```bash
python gpx_segmenter.py --file files/ridermanTT.gpx
```
*(Without `--segment-only`, it runs the optimizer)*

---

## 📂 Project Structure

- `cyclist_env.py`: The core Gymnasium environment with Update Physics and Reward Logic.
- `train.py`: Unified entry point for training.
- `ppo.py` / `sac_continous_action.py`: RL algorithms (CleanRL based).
- `evaluate.py`: Evaluation script with Matplotlib visualization.
- `physics_engine.py`: Shared core physics logic (`SimulatedRider`).
- `utils.py`: Shared file loading (`load_track`) and geometry helpers.
- `gpx_segmenter.py`: Tool for segmenting courses and optimizing "perfect knowledge" baselines.
- `simulation.py`: Verification tool comparing physics model vs real ride data.

---

## ⚠️ Common Issues

- **`tensorboard: command not found`**: 
  Ensure you have activated your virtual environment: `source .venv/bin/activate`.
- **`ModuleNotFoundError`**: 
  Always run scripts from the root directory: `python3 scripts/script.name` (or install as package).