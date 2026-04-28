# Real Mario Local Setup Guide

This guide helps verify your local environment can create and step through **SuperMarioBros-1-1-v0** before expensive RL training.

## 1) Install dependencies

```bash
pip install gymnasium gym-super-mario-bros nes-py opencv-python
```

## 2) Run setup checker

```bash
python scripts/check_mario_setup.py
```

This prints Python/platform details, dependency availability, env creation status, observation shape, reward, termination flags, and key info fields (`x_pos`, `flag_get`, `time`, `score`, `coins`, `life`, `status`).

## 3) Run full Mario smoke test

```bash
python scripts/smoke_test_mario_env.py --env-id SuperMarioBros-1-1-v0 --action-space right_only
```

## 4) Run non-learning baselines on Mario

```bash
python scripts/run_baselines.py --env mario --env-id SuperMarioBros-1-1-v0 --episodes 5 --action-space right_only --output-dir results/baselines/mario_right_only
```

## Troubleshooting (Linux/WSL)

- If `gym_super_mario_bros` import fails, reinstall with the command above and ensure your active Python/venv matches your shell.
- If `nes_py` is missing, install it explicitly and re-run setup checker.
- If graphical rendering is unstable on WSL/headless systems, run without `--render`.
- If env creation fails with id errors, verify the package version exposes `SuperMarioBros-1-1-v0`.

## Legal note

This repository contains **no ROMs, sprites, music, save states, or Nintendo assets**. You must use your own legal local environment setup.
