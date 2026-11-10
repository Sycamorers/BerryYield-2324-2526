# GNV plant canopy pipeline

This pipeline processes GNV plot image sequences in three stages:

1. plant-only YOLO detection and ByteTrack tracking;
2. one representative (temporal-middle) frame per plant track;
3. box-prompted SAM segmentation, followed by per-plot aggregation.

It never loads the fruit detector. Each plant mask is saved independently as an
8-bit PNG under
`/mnt/SB2324/data/GNV/2324_GNV_data/canopy_tracking_sam_masks_3years`. The
extra level is necessary because the GNV mount root belongs to a legacy UID and
is read-only to the current account; the masks still remain on the requested
GNV disk. Numeric and tracking metadata are stored in
`canopy_pipeline/final_results/per_plot`. The final combined table is
`gnv_canopy_traits_3years.csv` at the repository root, with an identical copy
in `canopy_pipeline/final_results`.

Metrics:

- `canopy_size_pixels`: sum of all per-plant mask pixel areas in a plot;
- `mean_roundness`: mean plant circularity, where circularity is
  `4*pi*area/perimeter^2`.

For 2023-24 and the already-processed 2024-25 dates, the program reuses the
current consolidated pipeline's `halfway_position` tracking manifests and runs
new, plant-box-only SAM masks. Dates without those manifests and 2025-26 are
tracked from RGB sequences with the plant detector.

The program writes every plot atomically and skips completed plots, so rerunning
the same command safely resumes an interrupted job. Zero-byte raw image files
are ignored automatically.

```bash
conda run -n yolo8 python canopy_pipeline/run_canopy_pipeline.py
```

Useful checks:

```bash
conda run -n yolo8 python canopy_pipeline/run_canopy_pipeline.py --discover-only
conda run -n yolo8 python canopy_pipeline/run_canopy_pipeline.py --season 2526 --date 260119 --plot b11
```

Only directories carrying one of the 40 experimental plot IDs are included.
The 2023 row-level scans (`row1` through `row8`) cannot be assigned to individual
plots from their directory names and are intentionally excluded.

## 2025–26 flower and fruit YOLO recount

The separate `count_berries_from_canopy_tracks.py` uses the saved representative
plant frame and box for each track, runs the five-class `sb_new.pt` detector on
the original RGB image, and counts FL/G/W/P/R detections overlapping each plant
box. The 504 plot outputs retain image paths, detection boxes, confidences and
matched plant IDs in `analysis/yolo_recount_2526/`. The source images are copied
into the handoff directory. For a relocated copy, use `--image-root data/GNV`;
use `--force` to recompute existing complete results.

The canopy pipeline accepts `BERRYYIELD_GNV_ROOT` for a relocated GNV image
tree. Plant masks in the handoff directory mirror their original relative paths.
