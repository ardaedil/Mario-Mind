from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from mariomind.agents.always_right_agent import AlwaysRightAgent
from mariomind.agents.ddqn_agent import DoubleDQNAgent
from mariomind.agents.dqn_agent import DQNAgent, DQNConfig
from mariomind.agents.random_agent import RandomAgent
from mariomind.agents.reflex_agent import ReflexAgent
from mariomind.envs.dummy_env import DummyEnvConfig, DummyPlatformerEnv
from mariomind.utils.config import TrainConfig, to_dict
from mariomind.utils.device import get_device
from mariomind.utils.seeding import set_seed


def build_env(cfg: TrainConfig, env_kind: str):
    if env_kind == "dummy":
        return DummyPlatformerEnv(
            DummyEnvConfig(
                frame_skip=cfg.frame_skip,
                frame_stack=cfg.frame_stack,
                action_space=cfg.action_space,
                reward_function=cfg.reward_function,
                max_steps=cfg.max_steps_per_episode,
            )
        )
    from mariomind.envs.mario_env import MarioEnv, MarioEnvConfig

    return MarioEnv(
        MarioEnvConfig(
            env_id=cfg.env_id,
            frame_skip=cfg.frame_skip,
            frame_stack=cfg.frame_stack,
            action_space=cfg.action_space,
            reward_function=cfg.reward_function,
        )
    )


def build_agent(cfg: TrainConfig, action_dim: int, state_shape: tuple[int, ...], right_idx: int, jump_idx: int):
    if cfg.agent_type == "random":
        return RandomAgent(action_dim)
    if cfg.agent_type == "always_right":
        return AlwaysRightAgent(action_dim, right_action_idx=right_idx)
    if cfg.agent_type == "reflex":
        return ReflexAgent(action_dim, right_idx=right_idx, jump_idx=jump_idx)

    dcfg = DQNConfig(
        lr=cfg.learning_rate,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        buffer_size=cfg.replay_buffer_size,
        target_sync_freq=cfg.target_sync_freq,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
    )
    device = get_device()
    if cfg.agent_type in ("dqn", "dueling_ddqn"):
        return DQNAgent(action_dim, state_shape, device, dcfg, dueling=(cfg.agent_type == "dueling_ddqn"))
    if cfg.agent_type == "ddqn":
        return DoubleDQNAgent(action_dim, state_shape, device, dcfg)
    raise ValueError(f"Unknown agent type: {cfg.agent_type}")


def train(cfg: TrainConfig, env_kind: str = "mario") -> Path:
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = build_env(cfg, env_kind=env_kind)
    agent = build_agent(cfg, env.action_space_n, env.observation_space.shape, env.action_cfg.right_action_index, env.action_cfg.jump_action_index)

    checkpoints_dir = out / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "reward", "max_x", "length", "success", "dead", "epsilon", "loss", "train_seconds"],
        )
        writer.writeheader()
        for ep in range(1, cfg.episodes + 1):
            start = time.time()
            state, info = env.reset(seed=cfg.seed + ep)
            ep_loss = []
            for step in range(cfg.max_steps_per_episode):
                action = agent.select_action(state, info)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.observe(state, action, reward, next_state, done)
                logs = agent.train_step()
                if "loss" in logs:
                    ep_loss.append(logs["loss"])
                state = next_state
                if done:
                    break
            if hasattr(agent, "save") and ep % cfg.checkpoint_frequency == 0:
                agent.save(checkpoints_dir / f"ep_{ep}.pt")
            writer.writerow(
                {
                    "episode": ep,
                    "reward": info.get("episode_reward", 0.0),
                    "max_x": info.get("max_x", 0.0),
                    "length": info.get("time_alive", step + 1),
                    "success": int(info.get("success", False)),
                    "dead": int(info.get("dead", False)),
                    "epsilon": getattr(agent, "epsilon", 0.0),
                    "loss": sum(ep_loss) / max(len(ep_loss), 1),
                    "train_seconds": time.time() - start,
                }
            )
    if hasattr(agent, "save"):
        agent.save(checkpoints_dir / "final.pt")
    with open(out / "config.json", "w", encoding="utf-8") as f:
        payload = to_dict(cfg)
        payload["env"] = env_kind
        json.dump(payload, f, indent=2)
    env.close()
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MarioMind agents.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--env", type=str, choices=["mario", "dummy"], default="mario")
    parser.add_argument("--agent-type", type=str, default="dqn")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/runs/smoke")
    args = parser.parse_args()

    cfg = TrainConfig(agent_type=args.agent_type, episodes=args.episodes, output_dir=args.output_dir)
    train(cfg, env_kind=args.env)


if __name__ == "__main__":
    main()
