import argparse

from mariomind.analysis.plots import make_plots

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metrics_csv", type=str)
    p.add_argument("--output-dir", type=str, default="results/plots")
    args = p.parse_args()
    make_plots(args.metrics_csv, args.output_dir)
