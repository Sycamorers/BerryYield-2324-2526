"""Download original images, masks, weights and complete analysis to this checkout."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


REPO = "Sycamorers/BerryYield-2324-2526"
ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = [
    "data/**",
    "analysis/**",
    "tmp/25_yield/**",
    "canopy_pipeline/final_results/**",
    "reports/weekly_20261110/analysis/**",
    "check-in & data-col 2526.xlsx",
    "raw_rgb_manifest.csv",
    "raw_rgb_summary.json",
    "HANDOFF_QA.json",
    "REPRO_QA.json",
]


def main() -> None:
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=ROOT,
        allow_patterns=DATA_PATHS,
        max_workers=8,
    )
    print(f"Downloaded dataset {REPO} into {ROOT}")


if __name__ == "__main__":
    main()
