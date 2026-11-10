"""Join manual and image-derived counts, then apply whole-date quality control."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / "analysis" / "verified_counts"
STAGES = {"FL": "flower", "G": "green", "W": "white", "P": "pink", "R": "red"}
EXPECTED_PLANTS = {"2324": 20, "2526": 10}


def load_yolo_2324() -> pd.DataFrame:
    source = ROOT / "data/real_data/2324_GNV_processed"
    rows = []
    for path in sorted(source.glob("*/consolidated_summary_with_yield.csv")):
        date = pd.to_datetime(path.parent.name, format="%y%m%d").strftime("%Y-%m-%d")
        frame = pd.read_csv(path, index_col=0)
        for plot in frame.columns:
            row = {"season": "2324", "date": date, "plot": str(plot).lower(),
                   "yolo_source": str(path.relative_to(ROOT))}
            for code, name in STAGES.items():
                row["yolo_" + code] = pd.to_numeric(
                    frame.loc["strawberry_" + name, plot], errors="coerce")
            rows.append(row)
    return pd.DataFrame(rows)


def load_yolo_2526() -> pd.DataFrame:
    path = ROOT / "analysis/yolo_recount_2526/yolo_counts_2526.csv"
    frame = pd.read_csv(path, dtype={"season": str})
    frame["plot"] = frame["plot"].str.lower()
    frame = frame.rename(columns={code: "yolo_" + code for code in STAGES})
    frame = frame.rename(columns={"source_json": "yolo_source", "status": "yolo_status"})
    return frame[["season", "date", "plot", "yolo_status", "yolo_source",
                  *["yolo_" + code for code in STAGES]]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manual = pd.read_csv(BASE / "correlations/counts_harmonized.csv", dtype={"season": str})
    manual["plot"] = manual["plot"].str.lower()
    manual = manual.rename(columns={name: "manual_" + code for code, name in STAGES.items()})
    canopy = pd.read_csv(BASE / "pipeline/traits_reprocessed.csv", dtype={"season": str})
    canopy["plot"] = canopy["plot"].str.lower()
    yolo = pd.concat([load_yolo_2324(), load_yolo_2526()], ignore_index=True)
    if yolo.duplicated(["season", "date", "plot"]).any():
        raise ValueError("Duplicate YOLO plot/date records")
    rows = canopy.merge(manual, on=["season", "date", "plot"], how="outer", validate="one_to_one")
    rows = rows.merge(yolo, on=["season", "date", "plot"], how="outer", validate="one_to_one")
    rows = rows.sort_values(["season", "date", "plot"])
    rows.to_csv(OUT / "all_plot_dates.csv", index=False)

    audit = []
    for (season, date), group in rows.groupby(["season", "date"]):
        image_n = int(group.canopy_size_pixels.notna().sum())
        manual_n = int(group[["manual_" + c for c in STAGES]].notna().all(axis=1).sum())
        yolo_n = int(group[["yolo_" + c for c in STAGES]].notna().all(axis=1).sum())
        median_tracks = group.plant_count.median()
        reasons = []
        if image_n < 32:
            reasons.append("image features available for fewer than 32/40 plots")
        if manual_n < 32:
            reasons.append("manual counts available for fewer than 32/40 plots")
        if yolo_n < 32:
            reasons.append("YOLO counts available for fewer than 32/40 plots")
        if pd.notna(median_tracks) and median_tracks > 2 * EXPECTED_PLANTS[season]:
            reasons.append("median retained tracks exceeds twice planted count")
        audit.append({"season": season, "date": date, "image_n": image_n,
                      "manual_complete_n": manual_n, "yolo_complete_n": yolo_n,
                      "median_plant_tracks": median_tracks,
                      "keep": not reasons, "reason": "; ".join(reasons)})
    date_qc = pd.DataFrame(audit)
    date_qc.to_csv(OUT / "date_qc.csv", index=False)
    healthy = rows.merge(date_qc[["season", "date", "keep"]],
                         on=["season", "date"], validate="many_to_one")
    healthy = healthy[healthy["keep"]].drop(columns="keep")
    healthy = healthy.dropna(subset=["canopy_size_pixels", "mean_roundness",
                                     *["manual_" + c for c in STAGES],
                                     *["yolo_" + c for c in STAGES]])
    healthy.to_csv(OUT / "healthy_plot_dates.csv", index=False)

    pairs = []
    for code in STAGES:
        part = healthy[["season", "date", "plot", "manual_" + code,
                        "yolo_" + code]].copy()
        part.columns = ["season", "date", "plot", "manual", "yolo"]
        part["feature"] = code
        pairs.append(part)
    pairs = pd.concat(pairs, ignore_index=True)
    pairs.to_csv(OUT / "manual_yolo_pairs.csv", index=False)
    summaries = []
    for (season, feature), group in pairs.groupby(["season", "feature"]):
        delta = group.yolo - group.manual
        summaries.append({"season": season, "feature": feature, "n": len(group),
                          "manual_mean": group.manual.mean(), "yolo_mean": group.yolo.mean(),
                          "yolo_minus_manual": delta.mean(), "mae": delta.abs().mean(),
                          "pearson_r": group.manual.corr(group.yolo)})
    pd.DataFrame(summaries).to_csv(OUT / "manual_yolo_summary.csv", index=False)
    metadata = {
        "seasons": ["2324", "2526"],
        "date_rule": "Keep whole dates only when image, manual and YOLO each cover >=32/40 plots and median plant tracks <=2x planted count; drop incomplete plot rows thereafter.",
        "source_2324_yolo": "data/real_data/2324_GNV_processed/*/consolidated_summary_with_yield.csv",
        "source_2526_yolo": "analysis/yolo_recount_2526/yolo_counts_2526.csv; derived from raw RGB images and canopy representative plant boxes",
        "source_manual": "2324 counting_yield CSV; 2526 check-in & data-col 2526.xlsx",
        "prediction_features": ["FL", "G", "W", "P", "canopy_size_pixels", "mean_roundness"],
        "descriptive_only": ["R (mature red fruit)", "plant_count (retained tracks)"],
        "healthy_rows": healthy.groupby("season").size().to_dict(),
        "excluded_dates": date_qc.loc[~date_qc.keep, ["season", "date", "reason"]].to_dict("records"),
    }
    (OUT / "README.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
