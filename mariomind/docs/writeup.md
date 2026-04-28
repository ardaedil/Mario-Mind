# MarioMind Write-up (Professor-Facing)

## 1) MDP formulation
- **Agent**: baseline controller or deep Q agent.
- **Environment**: Gym-compatible World 1-1 simulator.
- **State**: stack of preprocessed 84x84 grayscale frames.
- **Action space**: configurable presets (RIGHT_ONLY, SIMPLE_MOVEMENT, CUSTOM_SMALL_ACTION_SPACE).
- **Transition dynamics**: unknown game physics + enemy/object interactions.
- **Reward**: configurable reward functions (default, progress-only, sparse, etc.).
- **Episode**: reset until death, timeout, or flag completion.
- **Policy**: epsilon-greedy over Q-network outputs (for DQN-family).

## 2) Why frame stacking matters
Single images hide velocity and temporal context. Frame stacks allow the model to infer motion, jumps, and enemy trajectories.

## 3) Why reward shaping matters
Sparse rewards can be hard to optimize. Progress/death/time shaping can speed learning but may introduce unintended local optima.

## 4) Why replay buffers help
Experience replay breaks temporal correlations and improves sample reuse, making optimization more data-efficient and stable.

## 5) Why target networks help
A slowly updated target network reduces non-stationarity in TD targets, preventing oscillations and divergence.

## 6) DQN vs DDQN
- DQN uses max over target Q-values directly.
- DDQN decouples action selection (online net) and evaluation (target net), reducing optimistic bias.

## 7) Experiments and hypotheses
- Agent comparison: DDQN should outperform DQN and baselines on stability.
- Frame stack ablation: stack=4 should beat stack=1 on consistency.
- Reward shaping: progress+death penalty should improve early learning.
- Exploration schedule: slow decay should help robustness.
- Replay size: medium/large buffers should improve stability at cost of memory.

## 8) Results tables (placeholder)
- Insert summary tables from `results/experiments`.

## 9) Failure analysis
Common patterns: early deaths, stall near obstacles, non-progress jumping loops, reward gaming without completion.

## 10) Lessons learned (placeholder)
- TODO

## 11) Limitations
- Requires external legal local Mario environment package.
- Pixel RL remains compute-heavy and sensitive to hyperparameters.

## 12) Future work
- Prioritized replay
- Full dueling DDQN experiments
- Better obstacle detectors for reflex baseline
- Interactive Streamlit dashboard
