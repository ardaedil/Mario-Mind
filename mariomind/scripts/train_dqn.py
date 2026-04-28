from mariomind.training.train import train
from mariomind.utils.config import TrainConfig

if __name__ == "__main__":
    train(TrainConfig(agent_type="dqn", output_dir="results/runs/dqn"))
