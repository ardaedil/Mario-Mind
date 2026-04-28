from __future__ import annotations

import csv
from pathlib import Path


def _write_placeholder(path: Path, title: str, rows: list[dict]) -> None:
    content = [title]
    if rows:
        keys = list(rows[0].keys())
        content.append(",".join(keys))
        for r in rows:
            content.append(",".join(str(r[k]) for k in keys))
    path.write_text("\n".join(content), encoding="utf-8")


def make_baseline_plots(summary_csv: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    mean_max_x = [{"agent_name": r["agent_name"], "mean_max_x": r.get("mean_max_x", 0)} for r in rows]
    mean_reward = [{"agent_name": r["agent_name"], "mean_reward": r.get("mean_reward", 0)} for r in rows]
    rates = [
        {
            "agent_name": r["agent_name"],
            "success_rate": r.get("success_rate", 0),
            "death_rate": r.get("death_rate", 0),
            "timeout_rate": r.get("timeout_rate", 0),
        }
        for r in rows
    ]

    _write_placeholder(out / "bar_mean_max_x.png", "Mean max_x by agent", mean_max_x)
    _write_placeholder(out / "bar_mean_reward.png", "Mean reward by agent", mean_reward)
    _write_placeholder(out / "bar_rates.png", "Success/death/timeout rates by agent", rates)
    _write_placeholder(out / "line_best_episode_x.png", "Best episode x_pos over time (if steps.csv exists)", [])
