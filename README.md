# Deep Reinforcement Learning for Time Trial Pacing Strategy Optimization

This repository contains the codebase for the Bachelor Thesis "Deep Reinforcement Learning for Pacing Strategy Optimization in Individual Time Trials". It implements a Soft Actor-Critic (SAC) agent capable of optimizing power output for cyclists on complex terrains, considering physical constraints and physiological fatigue (W' balance).

## Features

- **Physics Engine**: Simulation of cycling dynamics including gradient, air density, rolling resistance, and gravity.
- **Physiology Model**: Critical Power (CP) and W' Balance (Anaerobic Work Capacity) model to simulate exhaustion and recovery.
- **Environment**: Custom Gymnasium-compatible `CyclistITT-v0` environment.
- **Algorithm**: Soft Actor-Critic (SAC) implementation for continuous power control.
- **Course Processing**: Support for GPX files with automated segmentation and gradient smoothing.
- **Lookahead System**: Environment observation includes foresight of upcoming terrain features.

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/JonasS1202/cycling-time-trial-rl.git
cd cycling-time-trial-rl
```

### 2. Install dependencies
It is recommended to use a virtual environment (Python 3.8+).

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Training

The training process uses the Soft Actor-Critic (SAC) algorithm. The agent learns to map state observations (velocity, fatigue, upcoming gradient) to optimal power output.

To start training:
```bash
python train.py --total-timesteps 1000000
```

### Optional Arguments
- `--total-timesteps`: Number of interaction steps (default: 1,000,000).
- `--learning-rate`: Learning rate for the optimizer.
- `--buffer-size`: Replay buffer size.
- `--batch-size`: Batch size for updates.

Training logs are saved to the `runs/` directory and can be visualized using TensorBoard:
```bash
tensorboard --logdir runs
```

---

## Evaluation

Use `evaluate.py` to test the trained model on specific courses.

### Basic Evaluation
```bash
python evaluate.py --model-path runs/YourExperimentName/sac_model.pt
```
This generates a `pacing_strategy.png` visualizing Speed, Power, W' Balance, and Elevation over the course.

### Evaluation on GPX Courses
To test on a real-world track:
```bash
python evaluate.py --model-path runs/YourExperimentName/sac_model.pt --gpx-file files/ridermanTT.gpx
```

### Comparison with Baseline
Compare the agent's strategy against an Optimal Constant Power (OCP) baseline. The OCP is the highest constant power the cyclist can sustain without exhaustion for the specific course duration.
```bash
python evaluate.py --optimize-baseline --gpx-file files/ridermanTT.gpx
```

---

## Project Structure

- `cyclist_env.py`: Custom Gymnasium environment implementing the MDP (Markov Decision Process).
- `sac_continous_action.py`: Implementation of the Soft Actor-Critic algorithm (based on CleanRL).
- `train.py`: Entry point for training experiments.
- `evaluate.py`: Scripts for model evaluation and visualization.
- `physics_engine.py`: Core physics calculations (Newtonian mechanics for cycling).
- `gpx_segmenter.py`: Utility for processing GPX files and segmenting courses.
- `check_env.py`: Utility to verify Gymnasium environment compliance.

## Environment Verification

To ensure the environment adheres to the Gymnasium API standards:
```bash
python check_env.py
```