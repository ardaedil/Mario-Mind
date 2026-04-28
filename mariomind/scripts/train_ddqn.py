from mariomind.training.train import train
from mariomind.utils.config import TrainConfig

if __name__ == "__main__":
    train(TrainConfig(agent_type="ddqn", output_dir="results/runs/ddqn"))
