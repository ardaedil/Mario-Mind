# MarioMind: An Experimental Study of Deep Reinforcement Learning in World 1-1

MarioMind is a reproducible reinforcement learning lab for experimenting with agents on a Gym-compatible World 1-1 environment. Instead of a single one-off model, this project emphasizes controlled comparisons across reward shaping, frame stacking, action spaces, exploration schedules, replay sizes, and failure modes.

## Why this project is different
- It is designed as an **experiment framework**, not just a training script.
- It includes **baseline agents** (random/always-right/reflex) alongside DQN/DDQN.
- It includes **analysis modules** for plots and failure summaries.
- It includes a **classical search toy world** to compare model-based planning and RL.

## Course concepts demonstrated
- MDP components (states, actions, transitions, rewards, policy, episodes)
- Q-learning and deep Q-learning
- Replay buffers and target networks
- Epsilon-greedy exploration and exploration schedules
- Reward shaping and ablation studies
- Evaluation protocol and failure analysis

## Legal note
This repository does **not** include ROMs, sprites, audio, save states, or Nintendo assets. You must provide a legal local setup exposing a Gym/Gymnasium-compatible Mario World 1-1 environment id.

## Setup
```bash
cd mariomind
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Train
```bash
python -m mariomind.training.train --agent-type dqn --episodes 10 --output-dir results/runs/smoke
# or
python scripts/train_dqn.py
python scripts/train_ddqn.py
```

## Evaluate
```bash
python scripts/evaluate_checkpoint.py results/runs/dqn/final_model.pt --episodes 10
```

## Run experiments
```bash
python scripts/run_experiments.py
```

## Visualize
```bash
python scripts/make_plots.py results/runs/smoke/metrics.csv --output-dir results/plots/smoke
```

## Core ideas (quick explanations)
- **DQN vs DDQN**: DDQN uses online network for action selection and target network for value evaluation to reduce overestimation bias.
- **Replay buffer**: random mini-batches decorrelate updates and improve stability.
- **Target network**: delayed targets stabilize bootstrapped learning.
- **Epsilon-greedy**: explore early with higher epsilon, then exploit learned policy as epsilon decays.

## Experiment results (placeholder)
- Add aggregate tables and figures from `results/experiments`.

## Failure analysis (placeholder)
- Add failure mode counts and qualitative notes from `results/failures`.

## Real Mario Environment Smoke Test

Use this milestone to verify real environment wiring (import/create/reset/step/info inspection) **without training**.

### Install Mario environment dependencies
```bash
pip install gymnasium gym-super-mario-bros nes-py opencv-python
```

### Run smoke test
```bash
python scripts/smoke_test_mario_env.py --env-id SuperMarioBros-1-1-v0 --action-space right_only
```

Action space options:
- `--action-space right_only`
- `--action-space simple`
- `--action-space complex`

### Successful output should include
- raw observation type/shape
- step-by-step reward and terminated/truncated flags
- info keys
- checks for `x_pos`, `flag_get`, `time`, `score`, `coins`, `life`, `status`
- wrapper initialization confirmation for `MarioEnv`

### Troubleshooting
- If `gym_super_mario_bros` or `nes_py` is missing, the smoke script exits gracefully and prints exact install guidance.
- If env creation fails, verify your local legal Mario setup and env id.

### Legal reminder
This repo includes **no ROMs, sprites, music, or Nintendo assets**. You must provide your own legal local environment.
