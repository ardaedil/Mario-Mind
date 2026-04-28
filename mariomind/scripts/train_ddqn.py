import argparse

from mariomind.training.train import train
from mariomind.utils.config import TrainConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["mario", "dummy"], default="mario")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="results/runs/ddqn")
    args = parser.parse_args()
    train(TrainConfig(agent_type="ddqn", episodes=args.episodes, output_dir=args.output_dir), env_kind=args.env)
