from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_line(df: pd.DataFrame, x_col: str, y_col: str, output_dir: Path, filename: str, title: str) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        return

    plt.figure()
    plt.plot(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def _save_bar(series: pd.Series, output_dir: Path, filename: str, title: str, ylabel: str) -> None:
    plt.figure()
    series.plot(kind="bar")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def _make_baseline_plots(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = df.groupby("agent_name")

    summary = grouped.agg(
        mean_total_reward=("total_reward", "mean"),
        std_total_reward=("total_reward", "std"),
        mean_max_x=("max_x", "mean"),
        best_max_x=("max_x", "max"),
        mean_episode_length=("episode_length", "mean"),
        success_rate=("success", "mean"),
        death_rate=("dead", "mean"),
        timeout_rate=("timeout", "mean"),
    )

    summary.to_csv(output_dir / "baseline_plot_summary.csv")

    _save_bar(summary["mean_total_reward"], output_dir, "baseline_mean_total_reward.png", "Mean Total Reward by Agent", "mean total reward")
    _save_bar(summary["mean_max_x"], output_dir, "baseline_mean_max_x.png", "Mean Max X by Agent", "mean max x")
    _save_bar(summary["best_max_x"], output_dir, "baseline_best_max_x.png", "Best Max X by Agent", "best max x")
    _save_bar(summary["success_rate"], output_dir, "baseline_success_rate.png", "Success Rate by Agent", "success rate")
    _save_bar(summary["death_rate"], output_dir, "baseline_death_rate.png", "Death Rate by Agent", "death rate")
    _save_bar(summary["timeout_rate"], output_dir, "baseline_timeout_rate.png", "Timeout Rate by Agent", "timeout rate")


def _make_training_plots(df: pd.DataFrame, output_dir: Path) -> None:
    x_col = "episode" if "episode" in df.columns else df.columns[0]

    reward_col = None
    for candidate in ["reward", "episode_reward", "total_reward"]:
        if candidate in df.columns:
            reward_col = candidate
            break

    if reward_col:
        _save_line(df, x_col, reward_col, output_dir, "training_reward.png", "Training Reward")

    for col, filename, title in [
        ("max_x", "training_max_x.png", "Max X Over Training"),
        ("loss", "training_loss.png", "Loss Over Training"),
        ("epsilon", "training_epsilon.png", "Epsilon Over Training"),
        ("episode_length", "training_episode_length.png", "Episode Length Over Training"),
    ]:
        _save_line(df, x_col, col, output_dir, filename, title)


def make_plots(metrics_csv: str | Path, output_dir: str | Path) -> None:
    metrics_csv = Path(metrics_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv)

    if df.empty:
        raise ValueError(f"No rows found in {metrics_csv}")

    # Baseline metrics have one row per agent/episode.
    if {"agent_name", "total_reward"}.issubset(df.columns):
        _make_baseline_plots(df, output_dir)
        return

    # Training metrics usually have one row per episode.
    _make_training_plots(df, output_dir)