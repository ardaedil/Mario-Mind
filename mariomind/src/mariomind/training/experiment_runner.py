from __future__ import annotations

import itertools
from dataclasses import replace

from mariomind.training.train import train
from mariomind.utils.config import TrainConfig


def run_experiments(base: TrainConfig, grid: dict[str, list]):
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = base
        tag_parts = []
        for k, v in zip(keys, values):
            cfg = replace(cfg, **{k: v})
            tag_parts.append(f"{k}-{v}")
        cfg = replace(cfg, output_dir=f"results/experiments/{'_'.join(tag_parts)}")
        train(cfg)
