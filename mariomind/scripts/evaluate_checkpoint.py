import argparse

from mariomind.training.evaluate import evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["dummy", "mario"], default="dummy")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--output-dir", type=str, default="results/evaluations")
    args = p.parse_args()
    print(evaluate(args.checkpoint, episodes=args.episodes, output_dir=args.output_dir, env_kind=args.env))
