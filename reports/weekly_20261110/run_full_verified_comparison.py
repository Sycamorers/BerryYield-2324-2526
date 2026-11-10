"""Run the 12 predictive models on identical manual or YOLO samples.

Use `--count-source manual` and `--count-source yolo` in separate processes.
The healthy image/date set and yield labels are identical in both runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.weekly_20261110 import run_analysis as benchmark

BASE = Path(__file__).resolve().parent
VERIFIED = BASE / "analysis/verified_counts"


def load_verified(source: str):
    healthy_path = VERIFIED / "healthy_plot_dates.csv"
    yield_path = BASE / "analysis/yields.csv"
    healthy = pd.read_csv(healthy_path, dtype={"season": str}, parse_dates=["date"])
    healthy["plot"] = healthy["plot"].str.upper()
    fields = ["canopy_size_pixels", "mean_roundness", "plant_count"]
    for code, name in (("FL", "flower"), ("G", "green"),
                       ("W", "white"), ("P", "pink")):
        healthy[name] = healthy[f"{source}_{code}"]
        fields.append(name)
    valid = np.isfinite(healthy[fields]).all(axis=1)
    valid &= healthy.canopy_size_pixels.gt(0)
    valid &= healthy.mean_roundness.between(0, 1)
    valid &= healthy[["flower", "green", "white", "pink"]].ge(0).all(axis=1)
    observations = healthy.loc[valid, ["season", "date", "plot", *fields]].copy()
    observations = observations.sort_values(["season", "plot", "date"])
    excluded = healthy.loc[~valid, ["season", "date", "plot", *fields]].copy()
    yields = pd.read_csv(yield_path, dtype={"season": str}, parse_dates=["date"])
    yields["plot"] = yields["plot"].str.upper()
    assert not observations.duplicated(["season", "date", "plot"]).any()
    assert set(observations.season) == {"2324", "2526"}
    assert not yields.duplicated(["season", "date", "plot"]).any()
    return observations, yields, excluded, [healthy_path, yield_path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-source", choices=("manual", "yolo"), required=True)
    args = parser.parse_args()
    output = VERIFIED / "full_models" / args.count_source
    output.mkdir(parents=True, exist_ok=True)
    benchmark.MODEL_INFO = {name: spec for name, spec in benchmark.MODEL_INFO.items()
                            if name not in {"Zero", "Train mean", "Train median"}}
    assert set(benchmark.ALL) == {"canopy_size_pixels", "mean_roundness", "flower", "green", "white", "pink"}
    benchmark.OUT = output
    benchmark.load_inputs = lambda: load_verified(args.count_source)
    benchmark.main()
    path = output / "summary.json"
    summary = json.loads(path.read_text())
    summary["count_source"] = args.count_source
    summary["input_dataset"] = str((VERIFIED / "healthy_plot_dates.csv").relative_to(ROOT))
    summary["comparison_design"] = "Same healthy plot-dates, yields, forecast IDs, hyperparameters and fitting protocol for manual and YOLO counts."
    path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
