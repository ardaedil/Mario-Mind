from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mariomind.agents.ddqn_agent import DoubleDQNAgent
from mariomind.agents.dqn_agent import DQNAgent, DQNConfig
from mariomind.envs.mario_env import MarioEnv, MarioEnvConfig
from mariomind.utils.device import get_device


def evaluate(checkpoint: str, episodes: int = 10, env_id: str = "SuperMarioBros-1-1-v0") -> dict:
    env = MarioEnv(MarioEnvConfig(env_id=env_id))
    agent = DQNAgent(env.action_space_n, env.observation_space.shape, get_device(), DQNConfig())
    try:
        agent.load(checkpoint)
    except Exception:
        agent = DoubleDQNAgent(env.action_space_n, env.observation_space.shape, get_device(), DQNConfig())
        agent.load(checkpoint)
    agent.epsilon = 0.0

    rows = []
    for ep in range(episodes):
        state, info = env.reset(seed=1234 + ep)
        while True:
            action = agent.select_action(state, info)
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        rows.append(
            {
                "episode": ep + 1,
                "reward": info.get("episode_reward", 0.0),
                "max_x": info.get("max_x", 0.0),
                "length": info.get("time_alive", 0),
                "success": int(info.get("success", False)),
                "dead": int(info.get("dead", False)),
            }
        )

    summary = {
        "episodes": episodes,
        "success_rate": sum(r["success"] for r in rows) / episodes,
        "avg_reward": sum(r["reward"] for r in rows) / episodes,
        "avg_max_x": sum(r["max_x"] for r in rows) / episodes,
        "avg_length": sum(r["length"] for r in rows) / episodes,
        "deaths": sum(r["dead"] for r in rows),
        "best_run": max(rows, key=lambda r: r["reward"]),
        "worst_run": min(rows, key=lambda r: r["reward"]),
    }

    out_dir = Path("results/evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "evaluation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    env.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MarioMind checkpoint")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--env-id", type=str, default="SuperMarioBros-1-1-v0")
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.episodes, args.env_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
