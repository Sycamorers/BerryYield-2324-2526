"""Recount FL/G/W/P/R from RGB frames selected by the canopy plant tracker.

The legacy 2324 summary counts detections that fall inside or overlap each
representative plant box.  This script preserves that rule for comparability.
Each selected RGB frame is inferred only once, then its detections are matched
to the tracked plant boxes.  Outputs are resumable and retain image provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


FEATURES = ("FL", "G", "W", "P", "R")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlaps(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def count_plot(source: Path, model: YOLO, device: str, model_sha256: str,
               image_root: Path | None) -> dict:
    canopy = json.loads(source.read_text())
    per_image: dict[str, list[dict]] = defaultdict(list)
    for plant in canopy.get("plants", []):
        per_image[plant["representative_image"]].append(plant)

    totals = {feature: 0 for feature in FEATURES}
    detections = []
    missing_images = []
    for name, plants in sorted(per_image.items()):
        image = Path(name)
        if image_root is not None:
            old_root = Path("/mnt/SB2324/data/GNV")
            try:
                image = image_root / image.relative_to(old_root)
            except ValueError:
                pass
        if not image.is_file():
            missing_images.append(name)
            continue
        result = model.predict(str(image), device=device, conf=0.25, imgsz=640,
                               verbose=False, save=False)[0]
        for index, box in enumerate(result.boxes):
            cls = int(box.cls.item())
            if cls not in range(5):
                raise ValueError(f"Unexpected YOLO class {cls} in {image}")
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            matched = [plant["plant_id"] for plant in plants
                       if overlaps(xyxy, plant["bbox_xyxy"])]
            totals[FEATURES[cls]] += len(matched)
            detections.append({
                "image": str(image),
                "detection_index": index,
                "feature": FEATURES[cls],
                "confidence": float(box.conf.item()),
                "bbox_xyxy": xyxy,
                "matched_plant_ids": matched,
            })

    return {
        "season": canopy["season"],
        "date": canopy["date"],
        "plot": canopy["plot"],
        "status": "complete" if not missing_images else "missing_image",
        "canopy_record": str(source.resolve()),
        "rgb_source_directory": canopy["source_directory"],
        "model_sha256": model_sha256,
        "model_classes": dict(model.names),
        "confidence_threshold": 0.25,
        "image_size": 640,
        "count_rule": "one count per detection overlapping each representative plant box",
        "plant_count": len(canopy.get("plants", [])),
        "selected_image_count": len(per_image),
        "missing_images": missing_images,
        "counts": totals,
        "detections": detections,
    }


def consolidate(output: Path) -> None:
    rows = []
    for path in sorted((output / "per_plot").glob("*.json")):
        record = json.loads(path.read_text())
        rows.append({
            "season": record["season"], "date": record["date"],
            "plot": record["plot"], "status": record["status"],
            "plant_count": record["plant_count"],
            "selected_image_count": record["selected_image_count"],
            **record["counts"],
            "source_json": str(path.resolve()),
        })
    path = output / "yolo_counts_2526.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                ["season", "date", "plot", "status", "plant_count",
                                 "selected_image_count", *FEATURES, "source_json"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Consolidated {len(rows)} plots to {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("canopy_pipeline/final_results/per_plot/2526"))
    parser.add_argument("--model", type=Path,
                        default=Path("tmp/25_yield/models_wts/sb_new.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("analysis/yolo_recount_2526"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-root", type=Path,
                        help="Local replacement for /mnt/SB2324/data/GNV in saved canopy records")
    parser.add_argument("--max-plots", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Recompute complete plot outputs instead of resuming")
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    sources = sorted(args.input.glob("*/*.json"))
    if args.max_plots:
        sources = sources[:args.max_plots]
    args.output.mkdir(parents=True, exist_ok=True)
    plot_output = args.output / "per_plot"
    plot_output.mkdir(exist_ok=True)
    model_sha256 = sha256(args.model)
    model = YOLO(str(args.model))
    if tuple(model.names.values()) != FEATURES:
        raise ValueError(f"Unexpected model classes: {model.names}")
    for i, source in enumerate(sources, 1):
        record = json.loads(source.read_text())
        target = plot_output / f"{record['date']}_{record['plot']}.json"
        if target.is_file() and not args.force:
            old = json.loads(target.read_text())
            if old.get("status") == "complete" and old.get("model_sha256") == model_sha256:
                continue
        result = count_plot(source, model, args.device, model_sha256, args.image_root)
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(result, indent=2))
        temp.replace(target)
        print(f"{i}/{len(sources)} {result['date']} {result['plot']} "
              f"{result['counts']} {result['status']}", flush=True)
    consolidate(args.output)


if __name__ == "__main__":
    main()
