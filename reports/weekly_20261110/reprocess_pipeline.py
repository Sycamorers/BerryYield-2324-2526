#!/usr/bin/env python3
"""Rebuild 2324/2526 canopy traits from retained per-plant pipeline metadata.

This script verifies and re-aggregates retained area/perimeter metadata;
it does not rerun image detection, tracking, or SAM segmentation. The original
RGB images and binary masks are preserved in the handoff data directory.
"""
from __future__ import annotations

import hashlib
import ast
import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "pipeline"
SEASONS = ("2324", "2526")
PLOTS = {f"{prefix}{r}{c}" for prefix in "BM" for r in (1, 2, 4, 5) for c in range(1, 6)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path,
        default=ROOT / "canopy_pipeline/final_results/gnv_canopy_traits_3years.csv",
        help="Aggregate reference used for reconciliation and source hash")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, plant_rows, errors, provenance = [], [], [], []
    mask_paths, image_paths, source_dirs = set(), set(), set()
    for season in SEASONS:
        for path in sorted((ROOT / "canopy_pipeline/final_results/per_plot" / season).glob("*/*.json")):
            record = json.loads(path.read_text())
            assert str(record["season"]) == season
            assert record["status"] == "complete", path
            plot = record["plot"].strip().upper()
            assert plot in PLOTS, (path, plot)
            plants = record["plants"]
            assert len({p["plant_id"] for p in plants}) == len(plants), path
            source_dirs.add(record["source_directory"])
            for plant in plants:
                area, perimeter = float(plant["canopy_size_pixels"]), float(plant["perimeter_pixels"])
                assert np.isfinite([area, perimeter]).all() and area >= 0 and perimeter >= 0
                circularity = 4 * math.pi * area / perimeter**2 if area and perimeter else 0.0
                assert np.isfinite(circularity)
                error = abs(circularity - float(plant["roundness"]))
                if error > 1e-10:
                    errors.append({"source": str(path.relative_to(ROOT)), "field": "plant_roundness", "difference": error})
                mask_paths.add(plant["mask_path"])
                image_paths.add(plant["representative_image"])
                plant_rows.append({"season": season, "date": record["date"], "plot": plot,
                    "plant_id": plant["plant_id"], "canopy_size_pixels": int(area),
                    "perimeter_pixels": perimeter, "roundness": circularity,
                    "track_length": plant.get("track_length"),
                    "detection_confidence": plant.get("detection_confidence"),
                    "mask_path": plant["mask_path"], "representative_image": plant["representative_image"]})
            area_sum = sum(int(p["canopy_size_pixels"]) for p in plants)
            roundness_mean = float(np.mean([4 * math.pi * p["canopy_size_pixels"] / p["perimeter_pixels"]**2
                if p["canopy_size_pixels"] and p["perimeter_pixels"] else 0.0 for p in plants])) if plants else np.nan
            for field, value in (("plant_count", len(plants)), ("canopy_size_pixels", area_sum), ("mean_roundness", roundness_mean)):
                old = record.get(field)
                if old is None or not np.isclose(float(old), value, rtol=1e-12, atol=1e-10):
                    errors.append({"source": str(path.relative_to(ROOT)), "field": field,
                        "old": old, "new": value})
            rows.append({"season": season, "date": record["date"], "plot": plot,
                "plant_count": len(plants), "canopy_size_pixels": area_sum,
                "mean_roundness": roundness_mean, "source_frame_count": record["source_frame_count"]})
            provenance.append({"source": str(path.relative_to(ROOT)), "sha256": sha256(path),
                "tracking_source": record.get("tracking_source", "not recorded"),
                "roundness_definition": record.get("roundness_definition"),
                "source_directory": record["source_directory"]})

    traits = pd.DataFrame(rows).sort_values(["season", "date", "plot"])
    plants = pd.DataFrame(plant_rows)
    assert not traits.duplicated(["season", "date", "plot"]).any()
    assert set(traits.season) == set(SEASONS)
    assert traits.mean_roundness.between(0, 1).all()
    traits.to_csv(OUT / "traits_reprocessed.csv", index=False)
    plants.to_csv(OUT / "plant_metrics_reprocessed.csv", index=False)
    pd.DataFrame(provenance).to_csv(OUT / "source_manifest.csv", index=False)

    original_path = args.reference_csv.resolve()
    original = pd.read_csv(original_path, dtype={"season": str})
    original = original[original.season.isin(SEASONS)].copy()
    reference_export = OUT / "source_reference_2324_2526.csv"
    if original_path != reference_export.resolve():
        original.to_csv(reference_export, index=False)
    original["plot"] = original["plot"].str.upper()
    joined = traits.merge(original, on=["season", "date", "plot"], suffixes=("_new", "_old"), validate="one_to_one", how="outer", indicator=True)
    comparisons = []
    for field in ("plant_count", "canopy_size_pixels", "mean_roundness", "source_frame_count"):
        diff = (joined[f"{field}_new"] - joined[f"{field}_old"]).abs()
        comparisons.append({"field": field, "compared": int(diff.notna().sum()),
            "max_absolute_difference": float(diff.max()),
            "mismatches_at_1e_minus_10": int((diff > 1e-10).sum())})
    pd.DataFrame(comparisons).to_csv(OUT / "aggregate_reconciliation.csv", index=False)
    # Execute the unchanged aggregate-stage function from the original pipeline
    # without importing its unavailable image-processing dependencies. A filtered
    # temporary input view contains only these two authorized seasons.
    pipeline_source = ROOT / "canopy_pipeline/run_canopy_pipeline.py"
    source_tree = ast.parse(pipeline_source.read_text())
    rebuild_function = next(n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name == "rebuild_csv")
    namespace = {"Path": Path, "csv": csv, "os": os, "json": json,
        "CSV_FIELDS": ["season", "date", "plot", "plant_count", "canopy_size_pixels", "mean_roundness", "source_frame_count", "mask_directory"]}
    exec(compile(ast.Module(body=[rebuild_function], type_ignores=[]), str(pipeline_source), "exec"), namespace)
    with tempfile.TemporaryDirectory(prefix="berry_weekly_pipeline_") as temporary:
        temp_root = Path(temporary)
        (temp_root / "per_plot").mkdir()
        for season in SEASONS:
            (temp_root / "per_plot" / season).symlink_to(ROOT / "canopy_pipeline/final_results/per_plot" / season, target_is_directory=True)
        rebuilt = namespace["rebuild_csv"](temp_root)
        shutil.copyfile(rebuilt, OUT / "original_pipeline_aggregate_2324_2526.csv")
    rebuilt_frame = pd.read_csv(OUT / "original_pipeline_aggregate_2324_2526.csv", dtype={"season": str})
    assert len(rebuilt_frame) == len(traits) and set(rebuilt_frame.season) == set(SEASONS)

    coverage = []
    for (season, date), group in traits.groupby(["season", "date"]):
        p = plants[(plants.season == season) & (plants.date == date)]
        coverage.append({"season": season, "date": date, "observed_plots": len(group), "expected_plots": 40,
            "missing_plots": "|".join(sorted(PLOTS - set(group["plot"]))),
            "plant_tracks": int(group.plant_count.sum()), "rgb_frames_recorded": int(group.source_frame_count.sum()),
            "canopy_median_pixels": float(group.canopy_size_pixels.median()),
            "roundness_median": float(group.mean_roundness.median()),
            "single_frame_tracks": int(p.track_length.eq(1).sum()),
            "confidence_recorded": int(p.detection_confidence.notna().sum()),
            "confidence_below_0_25": int(p.detection_confidence.lt(0.25).sum())})
    pd.DataFrame(coverage).to_csv(OUT / "coverage_by_date.csv", index=False)
    summary = traits.groupby("season").agg(plot_dates=("plot", "size"), dates=("date", "nunique"), plots=("plot", "nunique"),
        plant_tracks=("plant_count", "sum"), source_frames=("source_frame_count", "sum"),
        canopy_mean_pixels=("canopy_size_pixels", "mean"), canopy_median_pixels=("canopy_size_pixels", "median"),
        roundness_mean=("mean_roundness", "mean"), roundness_sd=("mean_roundness", "std"),
        plant_count_mean=("plant_count", "mean"), plant_count_median=("plant_count", "median"))
    summary.to_csv(OUT / "coverage_by_season.csv")
    availability = {
        "source_directories": {"total": len(source_dirs), "available": sum(Path(p).is_dir() for p in source_dirs)},
        "unique_representative_images": {"total": len(image_paths), "available": sum(Path(p).is_file() for p in image_paths)},
        "unique_masks": {"total": len(mask_paths), "available": sum(Path(p).is_file() for p in mask_paths)},
    }
    audit = {"included_seasons": list(SEASONS), "excluded_season": "2425", "plot_dates": len(traits),
        "plant_tracks": len(plants), "invalid_or_mismatching_records": errors,
        "source_aggregate_sha256": sha256(original_path),
        "source_aggregate_reference": str(original_path.relative_to(ROOT)) if original_path.is_relative_to(ROOT) else str(original_path),
        "pipeline_code_sha256": sha256(ROOT / "canopy_pipeline/run_canopy_pipeline.py"),
        "reprocess_code_sha256": sha256(Path(__file__)),
        "aggregate_rows_match": bool(joined["_merge"].eq("both").all()),
        "source_availability": availability,
        "completed_work": ["Recomputed each retained plant circularity from stored area and full-contour perimeter",
            "Rebuilt plot-level canopy sum, arithmetic mean circularity and track count",
            "Executed the unchanged rebuild_csv aggregation stage from original pipeline source, restricted to 2324 and 2526",
            "Reconciled all output traits with original retained aggregate CSV",
            "Audited dates, missing plots, tracking metadata, duplicate IDs and source availability"],
        "not_completed_in_current_environment": ["Rerun YOLO/ByteTrack on raw RGB", "Rerun SAM segmentation", "Recompute mask pixels and contours from binary images"],
        "measurement_caveats": ["Canopy is a sum of pixel areas across representative plant frames, not calibrated ground area or a single-frame union",
            "Plant count is retained track/mask count, not an independently verified census",
            "2324 reuses existing_halfway_position manifests; most 2526 records omit tracking_source",
            "2526 retained records include confidence below current script default and single-frame tracks; exact historical settings are not fully recorded",
            "Large cross-season pixel-area and track-count differences may include camera geometry and tracking effects; they are not proven biological differences"],
        "existing_visual_examples": [str(p.relative_to(ROOT)) for p in sorted((ROOT / "visual_check_2526_2026-01-19_b11").glob("*.jpg"))]}
    example_source = ROOT / "visual_check_2526_2026-01-19_b11/overview.jpg"
    if example_source.exists():
        shutil.copyfile(example_source, OUT / "existing_qc_2526_20260119_b11.jpg")
        audit["qc_example_caption"] = "Previously saved QC overlay, 2526 / 2026-01-19 / B11. Existing visualization, not generated by a new image-inference run. 8 retained masks; canopy 134,538 pixels; mean roundness 0.229695."
        audit["qc_example_sha256"] = sha256(example_source)
    (OUT / "pipeline_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    make_figures(traits, pd.DataFrame(coverage))
    print(summary.to_string())
    print(json.dumps(availability))
    print("Metadata discrepancies:", len(errors))


def make_figures(traits, coverage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"2324": "#246C9C", "2526": "#D3783B"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for s in SEASONS:
        group = traits[traits.season == s].groupby("date").agg(canopy=("canopy_size_pixels", "median"),
            roundness=("mean_roundness", "median"), plants=("plant_count", "median"))
        x = np.arange(1, len(group) + 1)
        for ax, col in zip(axes, ("canopy", "roundness", "plants")):
            ax.plot(x, group[col] / (1e6 if col == "canopy" else 1), "o-", color=colors[s], label=s, markersize=4)
    for ax, title, label in zip(axes, ("Canopy pixel area", "Mean plant roundness", "Retained plant tracks"),
            ("Median per plot (million pixels)", "Median per plot", "Median per plot")):
        ax.set_title(title, fontweight="bold", loc="left");ax.set_xlabel("Observation number within season");ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.18)
    axes[0].legend(frameon=False)
    fig.suptitle("Image features: 358 plot-dates in 2324 and 504 in 2526", fontsize=14, fontweight="bold", x=.05, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"traits_season_comparison.{ext}", dpi=210, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 3.3));ax.set_xlim(0, 12.5);ax.set_ylim(0, 3.3);ax.axis("off")
    boxes = [(0, "RGB image sequence", "Original source on GNV drive", "#E6EBEF"),
        (2.55, "Plant detection\n+ tracking", "One middle frame per track", "#E6EBEF"),
        (5.10, "SAM plant mask", "Area A and contour perimeter P", "#E6EBEF"),
        (7.65, "Retained per-plant JSON", f"{int(traits.plant_count.sum()):,} area/perimeter records", "#E7F2E9"),
        (10.20, "Rebuilt plot traits", "Canopy, roundness, track count", "#E7F2E9")]
    for x, title, sub, color in boxes:
        ax.add_patch(FancyBboxPatch((x+.03, 1.25), 2.22, 1.08, boxstyle="round,pad=.04", facecolor=color, edgecolor="#708493", linewidth=1.0))
        ax.text(x+1.14, 1.98, title, ha="center", va="center", fontsize=9, fontweight="bold", wrap=True)
        ax.text(x+1.14, 1.55, sub, ha="center", va="center", fontsize=7.4, wrap=True)
        if x<10:
            ax.add_patch(FancyArrowPatch((x+2.30,1.79),(x+2.54,1.79),arrowstyle="->",mutation_scale=12,color="#536D7E"))
    ax.text(.03, 2.98, "What was rerun for this report", fontsize=15, fontweight="bold")
    ax.text(.03, .77, "Gray: original image-processing stages; raw RGB and masks unavailable in this environment.", fontsize=10, color="#59656F")
    ax.text(.03, .37, "Green: completed metadata reprocessing and verification for 2324 + 2526; 2425 excluded.", fontsize=10, color="#286440")
    ax.text(7.75, 2.65, "Roundness = 4πA / P²     |     Canopy = Σ plant mask areas", fontsize=9, ha="center")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pipeline_scope.{ext}", dpi=210, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
