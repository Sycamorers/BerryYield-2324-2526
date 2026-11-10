# BerryYield 2324/2526

This repository contains the complete canopy, YOLO counting, data quality, yield-model, and report code for the 2023–24 and 2025–26 strawberry seasons. The [companion Hugging Face dataset](https://huggingface.co/datasets/Sycamorers/BerryYield-2324-2526) contains the original RGB images, plant masks, detector weights, source spreadsheets, complete saved results, and manifests. The two repositories use the same relative paths.

The prediction inputs are **FL, G, W, P, canopy size, and roundness**. Mature red fruit R and retained plant-track count are descriptive or quality-control data. The final report selects lag random forest with all six inputs by average MAE rank across the evaluated model, input, horizon, and count-source combinations. Individual evaluation winners differ.

## Get a usable local copy

```bash
git clone https://github.com/Sycamorers/BerryYield-2324-2526.git
cd BerryYield-2324-2526
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/verify_handoff.py
```

The download script places the original images, masks, weights, and complete derived results into the checkout. Hugging Face access is required if the dataset is private. The manifest check expects **30,681 original RGB images**, **862 canopy plot records**, **504 image-derived 2526 YOLO plot records**, and all three detector/segmentation weights.

## Reproduce the current analysis

The current processed dataset is `reports/weekly_20261110/analysis/verified_counts/healthy_plot_dates.csv`. It contains 318 matched plot-dates from 2324 and 439 from 2526. The complete model matrix is `reports/weekly_20261110/analysis/verified_counts/full_models/full_comparison.xlsx`. These are included in the GitHub checkout for immediate inspection.

Run the following from the repository root after downloading the dataset:

```bash
python reports/weekly_20261110/build_verified_counts_dataset.py
python reports/weekly_20261110/run_simple_verified_model.py
python reports/weekly_20261110/analyze_verified_counts.py
python reports/weekly_20261110/run_full_verified_comparison.py --count-source manual
python reports/weekly_20261110/run_full_verified_comparison.py --count-source yolo
python reports/weekly_20261110/analyze_full_verified_comparison.py
python reports/weekly_20261110/build_verified_boss_report.py
```

For an image-derived recount from the raw 2526 RGB frames, run:

```bash
python canopy_pipeline/count_berries_from_canopy_tracks.py --image-root data/GNV --device cpu --force
```

The canopy pipeline uses the relocated image tree when `BERRYYIELD_GNV_ROOT=$PWD/data/GNV` is set. It can be run for either analyzed season with `--season 2324` or `--season 2526`. The saved plant records retain their original `/mnt/SB2324` paths for provenance. The recount command uses `--image-root data/GNV` to resolve those paths locally.

## Data and results

- `canopy_pipeline/`: plant detection, tracking, SAM masks, roundness, and image-derived fruit recount.
- `data/real_data/2324_GNV_processed/`: original 2324 count and yield CSV sources.
- `check-in & data-col 2526.xlsx`: 2526 manual count and yield source.
- `analysis/yolo_recount_2526/`: 2526 YOLO detections linked to source RGB frames.
- `reports/weekly_20261110/`: date-level quality control, manual-versus-YOLO analyses, model runs, and report generation.
- `raw_rgb_manifest.csv`: source path, packaged path, size, and SHA-256 for every original RGB image.
- `Jing-2026-1110-yield prediction new data-revised.pdf`: final report.

The GitHub checkout includes code and compact processed results. Large original files and fitted model bundles live in the Hugging Face dataset. The 2024–25 season is excluded from this study.
