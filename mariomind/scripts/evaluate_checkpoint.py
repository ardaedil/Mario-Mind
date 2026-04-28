import argparse

from mariomind.training.evaluate import evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()
    print(evaluate(args.checkpoint, episodes=args.episodes))
