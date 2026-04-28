from __future__ import annotations

from pathlib import Path

import pandas as pd


def classify_failure(row: pd.Series) -> str:
    if row["success"] == 1:
        return "success"
    if row["length"] < 80 and row["max_x"] < 150:
        return "died_early"
    if row["max_x"] < 100:
        return "stuck_low_progress"
    if row["reward"] > 20 and row["success"] == 0:
        return "high_reward_no_completion"
    return "other_failure"


def analyze_failures(metrics_csv: str, out_dir: str = "results/failures") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    df["failure_mode"] = df.apply(classify_failure, axis=1)
    df.to_csv(out / "failure_summary.csv", index=False)
    notes = [
        "# Failure Notes",
        "- died_early: likely poor exploration or jump timing.",
        "- stuck_low_progress: may indicate sparse observations/reward mismatch.",
        "- high_reward_no_completion: reward shaping may overfit intermediate behavior.",
    ]
    (out / "failure_notes.md").write_text("\n".join(notes), encoding="utf-8")
