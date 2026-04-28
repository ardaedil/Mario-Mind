#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

from mariomind.agents.always_right_agent import AlwaysRightAgent
from mariomind.agents.random_agent import RandomAgent
from mariomind.agents.reflex_agent import ReflexAgent
from mariomind.envs.dummy_env import DummyEnvConfig, DummyPlatformerEnv
from mariomind.envs.mario_dependency import INSTALL_GUIDANCE, missing_mario_modules
from mariomind.utils.seeding import set_seed

def _map_action_space(name: str) -> str:
    mapping = {
        "right_only": "RIGHT_ONLY",
        "simple": "SIMPLE_MOVEMENT",
        "complex": "SIMPLE_MOVEMENT",
        "custom_small": "CUSTOM_SMALL_ACTION_SPACE",
    }
    return mapping[name]


def build_env(args):
    action_space = _map_action_space(args.action_space)
    if args.env == "dummy":
        return DummyPlatformerEnv(
            DummyEnvConfig(
                frame_skip=1,
                frame_stack=4,
                action_space=action_space,
                reward_function="default",
                max_steps=args.max_steps,
            )
        )

    missing = missing_mario_modules()
    if missing:
        raise RuntimeError(f"Missing modules: {', '.join(missing)}\n{INSTALL_GUIDANCE}")

    from mariomind.envs.mario_env import MarioEnv, MarioEnvConfig

    return MarioEnv(
        MarioEnvConfig(
            env_id=args.env_id,
            frame_skip=1,
            frame_stack=4,
            action_space=action_space,
            reward_function="default",
        )
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def run_agent(agent_name: str, agent, env, episodes: int, max_steps: int, render: bool):
    metrics = []
    step_rows = []
    for ep in range(1, episodes + 1):
        state, info = env.reset(seed=ep)
        total_reward = 0.0
        max_x = float(info.get("x_pos", 0))
        action_counts: dict[int, int] = {}

        for step in range(1, max_steps + 1):
            action = agent.select_action(state, info)
            action_counts[action] = action_counts.get(action, 0) + 1
            state, reward, terminated, truncated, info = env.step(action)
            if render:
                env.render()
            total_reward += reward
            x_pos = float(info.get("x_pos", 0.0))
            max_x = max(max_x, x_pos)
            step_rows.append(
                {
                    "agent_name": agent_name,
                    "episode": ep,
                    "step": step,
                    "reward": reward,
                    "x_pos": x_pos,
                    "action": action,
                    "done": int(terminated or truncated),
                    "success": int(info.get("success", False)),
                    "dead": int(info.get("dead", False)),
                }
            )
            if terminated or truncated:
                break

        episode_len = step
        success = bool(info.get("success", False))
        dead = bool(info.get("dead", False))
        timeout = (not success) and (not dead) and episode_len >= max_steps
        metrics.append(
            {
                "agent_name": agent_name,
                "episode": ep,
                "total_reward": total_reward,
                "max_x": max_x,
                "final_x": float(info.get("x_pos", 0.0)),
                "episode_length": episode_len,
                "success": int(success),
                "dead": int(dead),
                "timeout": int(timeout),
                "average_reward_per_step": total_reward / max(episode_len, 1),
                "action_counts": json.dumps(action_counts, sort_keys=True),
                "final_info": json.dumps(info, sort_keys=True),
            }
        )
    return metrics, step_rows


def summarize(metrics: list[dict]) -> list[dict]:
    by_agent: dict[str, list[dict]] = {}
    for row in metrics:
        by_agent.setdefault(row["agent_name"], []).append(row)
    summary = []
    for agent_name, rows in by_agent.items():
        rewards = [r["total_reward"] for r in rows]
        max_xs = [r["max_x"] for r in rows]
        summary.append(
            {
                "agent_name": agent_name,
                "mean_reward": _mean(rewards),
                "std_reward": _std(rewards),
                "mean_max_x": _mean(max_xs),
                "best_max_x": max(max_xs) if max_xs else 0,
                "success_rate": _mean([r["success"] for r in rows]),
                "death_rate": _mean([r["dead"] for r in rows]),
                "timeout_rate": _mean([r["timeout"] for r in rows]),
                "mean_episode_length": _mean([r["episode_length"] for r in rows]),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_report(path: Path, summary: list[dict], cfg: dict) -> None:
    lines = ["# Baseline Report", "", "## Config", "```json", json.dumps(cfg, indent=2), "```", "", "## Summary"]
    for row in summary:
        lines.append(
            f"- {row['agent_name']}: mean_reward={row['mean_reward']:.2f}, mean_max_x={row['mean_max_x']:.2f}, "
            f"success_rate={row['success_rate']:.2%}, death_rate={row['death_rate']:.2%}, timeout_rate={row['timeout_rate']:.2%}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline non-learning agents on dummy or Mario env")
    parser.add_argument("--env", choices=["dummy", "mario"], required=True)
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--action-space", choices=["right_only", "simple", "complex", "custom_small"], default="simple")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)

    try:
        env = build_env(args)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    agents = {
        "random": RandomAgent(env.action_space_n),
        "always_right": AlwaysRightAgent(env.action_space_n, right_action_idx=env.action_cfg.right_action_index),
        "reflex": ReflexAgent(
            env.action_space_n,
            right_idx=env.action_cfg.right_action_index,
            jump_idx=env.action_cfg.jump_action_index,
        ),
    }

    all_metrics = []
    all_steps = []
    for name, agent in agents.items():
        metrics, steps = run_agent(name, agent, env, args.episodes, args.max_steps, args.render)
        all_metrics.extend(metrics)
        all_steps.extend(steps)

    env.close()

    summary = summarize(all_metrics)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "metrics.csv", all_metrics)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "steps.csv", all_steps)
    cfg = vars(args)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_report(out_dir / "baseline_report.md", summary, cfg)

    print(f"Baseline run completed. Outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
