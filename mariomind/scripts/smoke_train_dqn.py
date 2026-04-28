#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mariomind.training.train import train
from mariomind.utils.config import TrainConfig


def verify_outputs(output_dir: Path, epsilon_min: float) -> None:
    metrics_path = output_dir / "metrics.csv"
    config_path = output_dir / "config.json"
    final_ckpt = output_dir / "checkpoints" / "final.pt"

    if not metrics_path.exists():
        raise RuntimeError("metrics.csv was not created")
    if not config_path.exists():
        raise RuntimeError("config.json was not created")
    if not final_ckpt.exists():
        raise RuntimeError("final checkpoint was not created")

    with open(metrics_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("metrics.csv has no rows")

    losses = [float(r.get("loss", 0.0) or 0.0) for r in rows]
    eps = [float(r.get("epsilon", 1.0) or 1.0) for r in rows]

    if all(loss == 0.0 for loss in losses):
        raise RuntimeError("No learning updates detected: all loss values are zero")
    if min(eps) < epsilon_min - 1e-9:
        raise RuntimeError(f"Epsilon fell below minimum floor: min={min(eps)} < {epsilon_min}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("agent_type") not in {"dqn", "ddqn"}:
        raise RuntimeError("Config does not contain expected agent_type")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny neural training smoke test on dummy env")
    parser.add_argument("--agent-type", choices=["dqn", "ddqn"], required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = TrainConfig(
        agent_type=args.agent_type,
        episodes=args.episodes,
        output_dir=args.output_dir,
        max_steps_per_episode=120,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=50,
        replay_buffer_size=500,
        batch_size=16,
        checkpoint_frequency=2,
    )
    train(cfg, env_kind="dummy")
    verify_outputs(Path(args.output_dir), epsilon_min=cfg.epsilon_end)
    print(f"Smoke training completed and verified for {args.agent_type}: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
