"""Stage the 2324/2526 code, measurements and original RGB images for handoff.

The staging directory is deliberately not uploaded. Source images remain intact
on the external disk; each copied RGB file is recorded with SHA-256 provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = Path("/mnt/SB2324/data/GNV")
DEST = ROOT / "BerryYield_2324_2526_handoff_20261110"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def copy_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        with destination.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    temp = destination.with_name(destination.name + ".copying")
    with source.open("rb") as inp, temp.open("wb") as out:
        for chunk in iter(lambda: inp.read(4 << 20), b""):
            digest.update(chunk)
            out.write(chunk)
    shutil.copystat(source, temp)
    temp.replace(destination)
    return digest.hexdigest()


def copy_tree(source: Path, relative: str) -> None:
    destination = DEST / relative
    shutil.copytree(source, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))


def main() -> None:
    DEST.mkdir(exist_ok=True)
    for relative in (
        "Jing-2026-1110-yield prediction new data-revised.docx",
        "Jing-2026-1110-yield prediction new data-revised.pdf",
        "check-in & data-col 2526.xlsx",
        "1-s2.0-S0168169926001213-main.pdf",
        "README.md",
        "canopy_pipeline/run_canopy_pipeline.py",
        "canopy_pipeline/count_berries_from_canopy_tracks.py",
        "canopy_pipeline/README.md",
        "reports/__init__.py",
        "reports/build_canopy_report.py",
        "reports/build_weekly_20261027.py",
        "canopy_temporal_benchmark/__init__.py",
        "canopy_temporal_benchmark/load_2526_workbook.py",
        "canopy_temporal_benchmark/run_benchmark.py",
        "canopy_temporal_benchmark/neural.py",
        "tmp/25_yield/best.pt",
        "tmp/25_yield/models_wts/sb_new.pt",
        "tmp/25_yield/models_wts/sam_b.pt",
    ):
        source = ROOT / relative
        if source.is_file():
            target = DEST / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    copy_tree(ROOT / "reports/weekly_20261110", "reports/weekly_20261110")
    copy_tree(ROOT / "analysis/yolo_recount_2526", "analysis/yolo_recount_2526")
    copy_tree(ROOT / "data/real_data/2324_GNV_processed",
              "data/real_data/2324_GNV_processed")
    for season in ("2324", "2526"):
        copy_tree(ROOT / "canopy_pipeline/final_results/per_plot" / season,
                  f"canopy_pipeline/final_results/per_plot/{season}")
    for relative in ("canopy_pipeline/final_results/gnv_canopy_traits_3years.csv",):
        source = ROOT / relative
        if source.is_file():
            target = DEST / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    mask_root = EXTERNAL / "2324_GNV_data/canopy_tracking_sam_masks_3years"
    for season in ("2324", "2526"):
        copy_tree(mask_root / season,
                  f"data/GNV/2324_GNV_data/canopy_tracking_sam_masks_3years/{season}")

    manifest_path = DEST / "raw_rgb_manifest.csv"
    rows = []
    for season, relative in (
        ("2324", Path("2324_GNV_data/2324_GNV_processed")),
        ("2526", Path("2526_GNV_data/2D_data")),
    ):
        source_root = EXTERNAL / relative
        for day in sorted(source_root.iterdir()):
            if not day.is_dir() or len(day.name) != 6 or not day.name.isdigit():
                continue
            if season == "2526":
                images = sorted(p for p in day.rglob("*") if p.is_file() and
                                p.suffix.lower() in IMAGE_SUFFIXES)
            else:
                images = sorted(p for plot in day.iterdir()
                                if plot.is_dir() and plot.name.startswith("RGB_")
                                for p in plot.iterdir()
                                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            for source in images:
                packed_relative = Path("data/GNV") / relative / source.relative_to(source_root)
                destination = DEST / packed_relative
                digest = copy_file(source, destination)
                rows.append({"season": season, "source": str(source),
                             "package_path": str(packed_relative),
                             "bytes": source.stat().st_size, "sha256": digest})
        print(f"Copied {season} RGB images", flush=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=["season", "source", "package_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {season: {"images": sum(row["season"] == season for row in rows),
                        "bytes": sum(row["bytes"] for row in rows if row["season"] == season)}
               for season in ("2324", "2526")}
    (DEST / "raw_rgb_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = """# BerryYield 2324/2526 handoff

This directory contains the final boss report, original RGB frame sequences for
the two analyzed seasons, retained canopy masks and plant records, manual
counting, the original 2324 YOLO summaries, the newly rerun 2526 YOLO detections,
model results, figures, code and the earlier PheMuT paper. No 2425 data is used.

Start with `Jing-2026-1110-yield prediction new data-revised.pdf`. The current
quality-controlled dataset is
`reports/weekly_20261110/analysis/verified_counts/healthy_plot_dates.csv`.
Read `date_qc.csv` and `README.json` in the same directory for exclusion rules.
The full manual-versus-YOLO model matrix is in
`reports/weekly_20261110/analysis/verified_counts/full_models/full_comparison.xlsx`.
The `raw_rgb_manifest.csv` lists every copied original RGB image and its SHA-256.
It covers the original frame sequences, including 2526 camera subdirectories.
Run `verify_handoff_directory.py` to confirm that every saved representative
image and mask is present.

To rerun 2526 YOLO counting from packaged images:

```
python canopy_pipeline/count_berries_from_canopy_tracks.py \\
  --image-root data/GNV --device cpu --force
python reports/weekly_20261110/build_verified_counts_dataset.py
python reports/weekly_20261110/run_simple_verified_model.py
python reports/weekly_20261110/analyze_verified_counts.py
python reports/weekly_20261110/run_full_verified_comparison.py --count-source manual
python reports/weekly_20261110/run_full_verified_comparison.py --count-source yolo
python reports/weekly_20261110/analyze_full_verified_comparison.py
```

The count script records each detection's source image, class, box, confidence
and matched plant IDs. It uses the packaged five-class `sb_new.pt` model.
Canopy masks and original frame sequences are provided for image-pipeline review.
Set `BERRYYIELD_GNV_ROOT=$PWD/data/GNV` when rerunning the canopy pipeline.
The saved per-plot JSON retains original `/mnt/SB2324` paths for provenance;
use the local `data/GNV` tree after transferring this directory.

The earlier comparison in `reports/weekly_20261110/analysis` used manual counts
on the initial complete-case set. The `verified_counts/full_models` directory
contains the current complete date-QC manual-versus-YOLO model comparison.
No files have been uploaded.
"""
    (DEST / "HANDOFF_README.md").write_text(readme)
    print(json.dumps({"directory": str(DEST), "images": len(rows),
                      "rgb_gb": round(sum(row["bytes"] for row in rows) / 1e9, 2),
                      "seasons": summary}, indent=2))


if __name__ == "__main__":
    main()
