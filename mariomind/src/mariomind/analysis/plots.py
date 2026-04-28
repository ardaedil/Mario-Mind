from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _write_placeholder_png(path: Path, title: str, points: list[tuple[float, float]]) -> None:
    # Minimal placeholder artifact for dependency-free smoke tests.
    content = f"{title}\n" + "\n".join(f"{x},{y}" for x, y in points)
    path.write_text(content, encoding="utf-8")


def make_plots(metrics_csv: str, output_dir: str = "results/plots") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(metrics_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def series(y_key: str) -> list[tuple[float, float]]:
        return [(float(r["episode"]), float(r.get(y_key, 0.0) or 0.0)) for r in rows]

    _write_placeholder_png(out / "reward_curve.png", "Episode Reward", series("reward"))
    _write_placeholder_png(out / "max_x_curve.png", "Max X Position", series("max_x"))
    _write_placeholder_png(out / "loss_curve.png", "Training Loss", series("loss"))
    _write_placeholder_png(out / "epsilon_curve.png", "Epsilon", series("epsilon"))
    _write_placeholder_png(out / "episode_length_curve.png", "Episode Length", series("length"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=str)
    parser.add_argument("--output-dir", type=str, default="results/plots")
    args = parser.parse_args()
    make_plots(args.metrics_csv, args.output_dir)


if __name__ == "__main__":
    main()
