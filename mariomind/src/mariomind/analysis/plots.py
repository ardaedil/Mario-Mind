from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_plots(metrics_csv: str, output_dir: str = "results/plots") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)

    plots = {
        "reward_curve.png": ("episode", "reward", "Episode Reward"),
        "max_x_curve.png": ("episode", "max_x", "Max X Position"),
        "loss_curve.png": ("episode", "loss", "Training Loss"),
        "epsilon_curve.png": ("episode", "epsilon", "Epsilon"),
    }
    for filename, (x, y, title) in plots.items():
        plt.figure(figsize=(8, 4))
        plt.plot(df[x], df[y])
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.savefig(out / filename)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=str)
    parser.add_argument("--output-dir", type=str, default="results/plots")
    args = parser.parse_args()
    make_plots(args.metrics_csv, args.output_dir)


if __name__ == "__main__":
    main()
