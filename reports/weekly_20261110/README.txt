BerryYield 2324/2526 report and analysis, 10 November 2026

Final boss report:
  Jing-2026-1110-yield prediction new data-revised.docx
  Jing-2026-1110-yield prediction new data-revised.pdf

Current verified analysis:
  analysis/verified_counts/healthy_plot_dates.csv: matched image features,
    manual FL/G/W/P/R and image-derived YOLO FL/G/W/P/R after whole-date QC.
  analysis/verified_counts/date_qc.csv: complete date-level exclusion reasons.
  analysis/verified_counts/manual_yolo_summary.csv: paired counting comparison.
  analysis/verified_counts/verified_feature_distributions.csv: per-season feature distributions.
  analysis/verified_counts/verified_feature_yield_correlations.csv: 1/2/3-week correlations.
  analysis/verified_counts/simple_model_metrics.csv: simplified model evaluation.
  analysis/verified_counts/full_models/full_comparison.xlsx: complete paired
    manual/YOLO comparison for 12 predictive models, six feature sets, three
    horizons, both transfer directions, both chronological within-season
    settings and all available early/middle/late harvest periods. The workbook
    also contains matched phase-specific versus overall-fitted results.
  analysis/verified_counts/full_models/comparison_audit.json: equal-sample QA.

Reproduction order:
  python reports/weekly_20261110/build_verified_counts_dataset.py
  python reports/weekly_20261110/run_simple_verified_model.py
  python reports/weekly_20261110/analyze_verified_counts.py
  python reports/weekly_20261110/run_full_verified_comparison.py --count-source manual
  python reports/weekly_20261110/run_full_verified_comparison.py --count-source yolo
  python reports/weekly_20261110/analyze_full_verified_comparison.py
  python reports/weekly_20261110/build_verified_boss_report.py

The 2526 YOLO source is analysis/yolo_recount_2526, generated from actual RGB
images with canopy_pipeline/count_berries_from_canopy_tracks.py. The 2324 YOLO
source is data/real_data/2324_GNV_processed/*/consolidated_summary_with_yield.csv.
The manual sources are the 2324 counting_yield CSV files and the 2526 check-in
workbook. The 2425 season is excluded. Only FL, G, W, P, canopy size and
roundness are predictive inputs. R and retained plant-track count are not.

The earlier exploratory comparison remains under analysis/ and correlations/
for audit. The verified_counts/full_models directory contains the current
12-model, six-predictor quality-controlled comparison. The separate handoff
directory also includes original RGB images, canopy masks, detector weights
and the source manifest.
Obsolete fitted artifacts from the previous plant-track input definition are
isolated under analysis/_legacy_previous_model_artifacts for provenance.
