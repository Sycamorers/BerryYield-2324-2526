"""Check local original image coverage against the published manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "raw_rgb_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = []
    wrong_size = []
    for row in rows:
        path = root / row["package_path"]
        if not path.is_file():
            missing.append(row["package_path"])
        elif path.stat().st_size != int(row["bytes"]):
            wrong_size.append(row["package_path"])
    weights = [root / "tmp/25_yield/best.pt",
               root / "tmp/25_yield/models_wts/sb_new.pt",
               root / "tmp/25_yield/models_wts/sam_b.pt"]
    absent_weights = [str(path.relative_to(root)) for path in weights if not path.is_file()]
    summary = {
        "raw_rgb_manifest_rows": len(rows),
        "raw_rgb_by_season": {season: sum(row["season"] == season for row in rows)
                              for season in ("2324", "2526")},
        "missing_images": missing[:10],
        "wrong_size_images": wrong_size[:10],
        "absent_weights": absent_weights,
        "canopy_plot_records": sum(1 for season in ("2324", "2526")
                                   for _ in (root / "canopy_pipeline/final_results/per_plot" / season).glob("*/*.json")),
        "yolo_2526_plot_records": len(list((root / "analysis/yolo_recount_2526/per_plot").glob("*.json"))),
    }
    summary["passed"] = (len(rows) == 30681 and not missing and not wrong_size
                         and not absent_weights and summary["canopy_plot_records"] == 862
                         and summary["yolo_2526_plot_records"] == 504)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
