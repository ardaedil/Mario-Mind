from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mariomind.agents.ddqn_agent import DoubleDQNAgent
from mariomind.agents.dqn_agent import DQNAgent, DQNConfig
from mariomind.envs.dummy_env import DummyEnvConfig, DummyPlatformerEnv


def evaluate(checkpoint: str, episodes: int = 10, output_dir: str = "results/evaluations", env_kind: str = "dummy") -> dict:
    if env_kind != "dummy":
        raise RuntimeError("This milestone supports evaluation smoke tests on dummy env only.")

    env = DummyPlatformerEnv(DummyEnvConfig())
    agent = DQNAgent(env.action_space_n, env.observation_space.shape, "cpu", DQNConfig())
    agent.load(checkpoint)
    if agent.agent_type == "ddqn":
        agent = DoubleDQNAgent(env.action_space_n, env.observation_space.shape, "cpu", DQNConfig())
        agent.load(checkpoint)
    agent.epsilon = 0.0

    rows = []
    for ep in range(episodes):
        state, info = env.reset(seed=1234 + ep)
        steps = 0
        while True:
            action = agent.select_action(state, info)
            state, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        rows.append(
            {
                "episode": ep + 1,
                "reward": info.get("episode_reward", 0.0),
                "max_x": info.get("max_x", 0.0),
                "length": steps,
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

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "evaluation_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    env.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MarioMind checkpoint")
    parser.add_argument("--env", choices=["dummy", "mario"], default="dummy")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/evaluations")
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.episodes, args.output_dir, args.env)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
