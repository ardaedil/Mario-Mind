import argparse
import csv
from pathlib import Path

from mariomind.analysis.baseline_plots import make_baseline_plots
from mariomind.analysis.plots import make_plots

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metrics_csv", type=str)
    p.add_argument("--output-dir", type=str, default="results/plots")
    args = p.parse_args()

    with open(args.metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])

    if "agent_name" in header and "total_reward" in header:
        summary_csv = str(Path(args.metrics_csv).with_name("summary.csv"))
        make_baseline_plots(summary_csv=summary_csv, output_dir=args.output_dir)
    else:
        make_plots(args.metrics_csv, args.output_dir)
