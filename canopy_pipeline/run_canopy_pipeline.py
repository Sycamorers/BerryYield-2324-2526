#!/usr/bin/env python3
"""Track GNV strawberry plants, segment each plant with SAM, and aggregate traits.

The unit of work and output is one (season, date, plot).  Binary masks are kept
under the GNV data mount; numeric results and resumable metadata stay in this
repository.  The script deliberately does not load or run a fruit detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GNV_ROOT = Path(os.environ.get("BERRYYIELD_GNV_ROOT", "/mnt/SB2324/data/GNV"))
# The GNV root itself is owned by a legacy UID and is not writable by the
# current account.  The 2324 season directory is intentionally group/world
# writable, so the GNV-wide mask collection lives there while still remaining
# on the requested /mnt/SB2324/data/GNV disk.
DEFAULT_MASK_ROOT = GNV_ROOT / "2324_GNV_data" / "canopy_tracking_sam_masks_3years"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "canopy_pipeline" / "results"
DEFAULT_PLANT_MODEL = PROJECT_ROOT / "tmp" / "25_yield" / "best.pt"
DEFAULT_SAM_CHECKPOINT = (
    PROJECT_ROOT / "tmp" / "25_yield" / "models_wts" / "sam_b.pt"
)
DEFAULT_TRACKER = "bytetrack.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VALID_PLOTS = {
    f"{prefix}{row}{column}"
    for prefix in ("b", "m")
    for row in (1, 2, 4, 5)
    for column in range(1, 6)
}
SEASON_ROOTS = {
    "2324": GNV_ROOT / "2324_GNV_data" / "2324_GNV_processed",
    "2425": GNV_ROOT / "2425_GNV_data" / "raw_data",
    "2526": GNV_ROOT / "2526_GNV_data" / "2D_data",
}
PROCESSED_OVERRIDES = {
    "2425": (
        GNV_ROOT
        / "2425_GNV_data"
        / "processed_data"
        / "2425_GNV_processed"
    )
}
CSV_FIELDS = [
    "season",
    "date",
    "plot",
    "plant_count",
    "canopy_size_pixels",
    "mean_roundness",
    "source_frame_count",
    "mask_directory",
]


@dataclass(frozen=True)
class PlotJob:
    season: str
    date: str
    plot: str
    source_dir: Path
    segments: tuple[tuple[Path, ...], ...]


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def direct_images(directory: Path) -> tuple[Path, ...]:
    images = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stat().st_size > 0
        and "depth" not in path.name.lower()
    ]
    return tuple(sorted(images, key=natural_key))


def image_segments(plot_dir: Path) -> tuple[tuple[Path, ...], ...]:
    """Return one sequence for a normal plot, or one per camera subdirectory."""
    images = direct_images(plot_dir)
    if images:
        return (images,)
    segments = []
    for child in sorted((p for p in plot_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        child_images = direct_images(child)
        if child_images:
            segments.append(child_images)
    return tuple(segments)


def extract_plot(name: str) -> str | None:
    matches = re.findall(r"(?<![A-Za-z0-9])([BMbm](?:1|2|4|5)[1-5])(?!\d)", name)
    if not matches:
        # Directory names normally end in the plot ID, so allow underscore adjacency.
        match = re.search(r"([BMbm](?:1|2|4|5)[1-5])$", name)
        return match.group(1).lower() if match else None
    plot = matches[-1].lower()
    return plot if plot in VALID_PLOTS else None


def normalize_date(name: str) -> str | None:
    match = re.search(r"(20\d{6}|\d{6})", name)
    if not match:
        return None
    raw = match.group(1)
    fmt = "%Y%m%d" if len(raw) == 8 else "%y%m%d"
    try:
        return datetime.strptime(raw, fmt).date().isoformat()
    except ValueError:
        return None


def discover_jobs(
    seasons: set[str] | None = None,
    dates: set[str] | None = None,
    plots: set[str] | None = None,
) -> tuple[list[PlotJob], list[str]]:
    jobs: list[PlotJob] = []
    notes: list[str] = []
    for season, root in SEASON_ROOTS.items():
        if seasons and season not in seasons:
            continue
        if not root.is_dir():
            notes.append(f"missing season root: {root}")
            continue
        for date_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
            date = normalize_date(date_dir.name)
            if date is None:
                continue
            compact_date = date.replace("-", "")
            if dates and date not in dates and compact_date not in dates:
                continue
            override_date_dir = None
            override_root = PROCESSED_OVERRIDES.get(season)
            if override_root:
                candidate = override_root / date[2:].replace("-", "")
                if candidate.is_dir():
                    override_date_dir = candidate
            override_plots = {}
            if override_date_dir:
                override_plots = {
                    plot: path
                    for path in override_date_dir.iterdir()
                    if path.is_dir() and (plot := extract_plot(path.name)) is not None
                }
            for plot_dir in sorted((p for p in date_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
                plot = extract_plot(plot_dir.name)
                if plot is None:
                    continue
                if plots and plot not in plots:
                    continue
                plot_dir = override_plots.get(plot, plot_dir)
                segments = image_segments(plot_dir)
                if not segments:
                    notes.append(f"no RGB images: {plot_dir}")
                    continue
                jobs.append(PlotJob(season, date, plot, plot_dir, segments))
    jobs.sort(key=lambda job: (job.season, job.date, job.plot))
    return jobs, notes


def reset_yolo_tracker(model: YOLO) -> None:
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) or []:
        reset = getattr(tracker, "reset", None)
        if reset:
            reset()


def track_segment(
    images: tuple[Path, ...],
    model: YOLO,
    device: str,
    confidence: float,
    image_size: int,
    tracker: str,
    augment: bool,
) -> dict[int, list[dict[str, object]]]:
    tracks: dict[int, list[dict[str, object]]] = defaultdict(list)
    reset_yolo_tracker(model)
    for frame_index, image_path in enumerate(images):
        result = model.track(
            source=str(image_path),
            persist=True,
            tracker=tracker,
            classes=[0],
            conf=confidence,
            imgsz=image_size,
            device=device,
            save=False,
            verbose=False,
            augment=augment,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.detach().cpu().numpy().astype(int)
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        for track_id, box, score in zip(ids, xyxy, confidences):
            tracks[int(track_id)].append(
                {
                    "frame_index": frame_index,
                    "image_path": str(image_path),
                    "bbox": [float(value) for value in box],
                    "confidence": float(score),
                }
            )
    return dict(tracks)


def representative(observations: list[dict[str, object]]) -> dict[str, object]:
    """Use the temporal middle observation, matching the established pipeline."""
    return observations[len(observations) // 2]


def halfway_manifest(source_dir: Path) -> Path | None:
    manifests = sorted(source_dir.glob("halfway_position_*.json"))
    return manifests[0] if manifests else None


def load_halfway_tracks(job: PlotJob) -> list[dict[str, object]] | None:
    """Load representative plant boxes produced by the established tracker."""
    manifest = halfway_manifest(job.source_dir)
    if manifest is None:
        return None
    with manifest.open(encoding="utf-8") as handle:
        data = json.load(handle)
    tracks = []
    seen = set()
    for image_name, entries in data.items():
        image_path = job.source_dir / image_name
        if not image_path.is_file():
            continue
        for ordinal, entry in enumerate(entries):
            raw_id = str(entry.get("track_id", ordinal))
            unique_id = raw_id
            if unique_id in seen:
                unique_id = f"{raw_id}_{ordinal}"
            seen.add(unique_id)
            bbox = entry.get("position", [])
            if len(bbox) != 4:
                continue
            safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", unique_id)
            tracks.append(
                {
                    "plant_id": f"existing_t{safe_id}",
                    "segment_index": 0,
                    "raw_track_id": raw_id,
                    "track_length": None,
                    "frame_index": 0,
                    "image_path": str(image_path),
                    "bbox": [float(value) for value in bbox],
                    "confidence": None,
                }
            )
    return tracks


def circularity(mask: np.ndarray) -> tuple[float, float]:
    """Return 4*pi*A/P^2 and the full contour perimeter for a binary mask."""
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    area = float(np.count_nonzero(binary))
    if area == 0.0 or perimeter == 0.0:
        return 0.0, perimeter
    return float(4.0 * math.pi * area / (perimeter * perimeter)), perimeter


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    if not cv2.imwrite(str(temporary), mask.astype(np.uint8) * 255):
        raise OSError(f"failed to write mask: {temporary}")
    os.replace(temporary, path)


def prompt_sam(
    predictor: SamPredictor,
    image_path: Path,
    plants: list[dict[str, object]],
    batch_size: int,
    autocast_context,
) -> Iterator[tuple[dict[str, object], np.ndarray]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with autocast_context():
        predictor.set_image(rgb)
        for start in range(0, len(plants), batch_size):
            chunk = plants[start : start + batch_size]
            boxes = np.asarray([plant["bbox"] for plant in chunk], dtype=np.float32)
            transformed = predictor.transform.apply_boxes_torch(
                torch.as_tensor(boxes, device=predictor.device), rgb.shape[:2]
            )
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=False,
            )
            for plant, mask in zip(chunk, masks[:, 0].detach().cpu().numpy()):
                yield plant, mask.astype(bool)
    predictor.reset_image()


def process_job(
    job: PlotJob,
    detector: YOLO | None,
    predictor: SamPredictor,
    args: argparse.Namespace,
) -> dict[str, object]:
    started = time.monotonic()
    mask_dir = args.mask_root / job.season / job.date / job.plot
    record_path = args.result_root / "per_plot" / job.season / job.date / f"{job.plot}.json"
    if record_path.exists() and not args.force:
        with record_path.open(encoding="utf-8") as handle:
            old = json.load(handle)
        if old.get("status") == "complete" and all(
            Path(plant["mask_path"]).exists() for plant in old.get("plants", [])
        ):
            print(f"SKIP complete {job.season} {job.date} {job.plot}", flush=True)
            return old
    if args.force and mask_dir.exists():
        # Remove only files created by this pipeline so a changed track set
        # cannot leave stale plant masks behind after recalibration.
        for old_mask in mask_dir.glob("plant_*.png"):
            old_mask.unlink()

    tracked = load_halfway_tracks(job)
    tracking_source = "existing_halfway_position" if tracked is not None else "new_bytetrack"
    if tracked is None:
        if detector is None:
            raise RuntimeError("plant detector was not loaded for a raw sequence")
        tracked = []
        for segment_index, images in enumerate(job.segments):
            segment_tracks = track_segment(
                images,
                detector,
                args.device,
                args.confidence,
                args.image_size,
                args.tracker,
                args.augment,
            )
            for raw_track_id, observations in sorted(segment_tracks.items()):
                if len(observations) < args.min_track_frames:
                    continue
                chosen = representative(observations)
                tracked.append(
                    {
                        "plant_id": f"s{segment_index:02d}_t{raw_track_id:05d}",
                        "segment_index": segment_index,
                        "raw_track_id": raw_track_id,
                        "track_length": len(observations),
                        **chosen,
                    }
                )

    by_image: dict[Path, list[dict[str, object]]] = defaultdict(list)
    for plant in tracked:
        by_image[Path(str(plant["image_path"]))].append(plant)

    metrics = []
    for image_path, plants in sorted(by_image.items(), key=lambda item: str(item[0])):
        for plant, mask in prompt_sam(
            predictor, image_path, plants, args.sam_batch_size, args.autocast_context
        ):
            area = int(np.count_nonzero(mask))
            roundness, perimeter = circularity(mask)
            mask_path = mask_dir / f"plant_{plant['plant_id']}_frame_{plant['frame_index']:06d}.png"
            save_mask(mask_path, mask)
            metrics.append(
                {
                    "plant_id": plant["plant_id"],
                    "segment_index": plant["segment_index"],
                    "raw_track_id": plant["raw_track_id"],
                    "track_length": plant["track_length"],
                    "representative_image": str(image_path),
                    "representative_frame_index": plant["frame_index"],
                    "detection_confidence": plant["confidence"],
                    "bbox_xyxy": plant["bbox"],
                    "mask_path": str(mask_path),
                    "canopy_size_pixels": area,
                    "perimeter_pixels": perimeter,
                    "roundness": roundness,
                }
            )

    roundness_values = [float(item["roundness"]) for item in metrics]
    output: dict[str, object] = {
        "status": "complete",
        "season": job.season,
        "date": job.date,
        "plot": job.plot,
        "source_directory": str(job.source_dir),
        "source_frame_count": sum(len(segment) for segment in job.segments),
        "source_segment_count": len(job.segments),
        "tracking_source": tracking_source,
        "plant_count": len(metrics),
        "canopy_size_pixels": sum(int(item["canopy_size_pixels"]) for item in metrics),
        "mean_roundness": (sum(roundness_values) / len(roundness_values)) if roundness_values else None,
        "roundness_definition": "4*pi*mask_area/(full_contour_perimeter^2)",
        "mask_directory": str(mask_dir),
        "elapsed_seconds": time.monotonic() - started,
        "plants": metrics,
    }
    atomic_json(record_path, output)
    print(
        f"DONE {job.season} {job.date} {job.plot}: "
        f"plants={len(metrics)} area={output['canopy_size_pixels']} "
        f"roundness={output['mean_roundness']} time={output['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return output


def rebuild_csv(result_root: Path) -> Path:
    rows = []
    per_plot = result_root / "per_plot"
    if per_plot.exists():
        for path in per_plot.glob("*/*/*.json"):
            with path.open(encoding="utf-8") as handle:
                record = json.load(handle)
            if record.get("status") != "complete":
                continue
            rows.append({field: record.get(field) for field in CSV_FIELDS})
    rows.sort(key=lambda row: (row["season"], row["date"], row["plot"]))
    destination = result_root / "gnv_canopy_traits_3years.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", action="append", choices=sorted(SEASON_ROOTS))
    parser.add_argument("--date", action="append", help="YYYY-MM-DD or YYYYMMDD; repeatable")
    parser.add_argument("--plot", action="append", help="Plot such as b11; repeatable")
    parser.add_argument("--plant-model", type=Path, default=DEFAULT_PLANT_MODEL)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-model-type", default="vit_b", choices=sorted(sam_model_registry))
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--tracker", default=DEFAULT_TRACKER)
    parser.add_argument("--augment", action="store_true", help="Use YOLO test-time augmentation")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--min-track-frames", type=int, default=2)
    parser.add_argument("--sam-batch-size", type=int, default=32)
    parser.add_argument("--max-plots", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.plot = {value.lower() for value in args.plot} if args.plot else None
    if args.date:
        normalized_dates = set()
        for value in args.date:
            parsed = normalize_date(value)
            if parsed is None:
                raise ValueError(f"invalid --date value: {value}")
            normalized_dates.update({parsed, parsed.replace("-", "")})
        args.date = normalized_dates
    else:
        args.date = None
    args.season = set(args.season) if args.season else None
    jobs, notes = discover_jobs(args.season, args.date, args.plot)
    if args.max_plots is not None:
        jobs = jobs[: args.max_plots]
    frames = sum(sum(len(segment) for segment in job.segments) for job in jobs)
    print(f"Discovered {len(jobs)} plot jobs and {frames} RGB frames", flush=True)
    for note in notes[:20]:
        print(f"NOTE {note}", flush=True)
    if len(notes) > 20:
        print(f"NOTE ... and {len(notes) - 20} more", flush=True)
    if args.discover_only:
        return 0
    if not jobs:
        print("No matching plot jobs", file=sys.stderr)
        return 2
    for path in (args.plant_model, args.sam_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    use_cuda_amp = not args.no_amp and args.device != "cpu" and torch.cuda.is_available()
    args.autocast_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.float16))
        if use_cuda_amp
        else nullcontext
    )
    needs_detector = any(halfway_manifest(job.source_dir) is None for job in jobs)
    detector = None
    if needs_detector:
        print(f"Loading plant detector: {args.plant_model}", flush=True)
        detector = YOLO(str(args.plant_model))
    else:
        print("Reusing existing tracking manifests; plant detector not needed", flush=True)
    print(f"Loading SAM {args.sam_model_type}: {args.sam_checkpoint}", flush=True)
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint))
    sam_device = torch.device("cuda:0" if args.device != "cpu" else "cpu")
    sam.to(device=sam_device).eval()
    predictor = SamPredictor(sam)

    failures = []
    with torch.inference_mode():
        for index, job in enumerate(jobs, start=1):
            print(
                f"[{index}/{len(jobs)}] {job.season} {job.date} {job.plot} "
                f"frames={sum(len(s) for s in job.segments)} segments={len(job.segments)}",
                flush=True,
            )
            try:
                process_job(job, detector, predictor, args)
            except Exception as exc:  # Continue the multi-day run and retain an audit trail.
                failures.append(
                    {
                        "season": job.season,
                        "date": job.date,
                        "plot": job.plot,
                        "source_directory": str(job.source_dir),
                        "error": repr(exc),
                    }
                )
                print(f"ERROR {job.season} {job.date} {job.plot}: {exc!r}", file=sys.stderr, flush=True)
            rebuild_csv(args.result_root)

    csv_path = rebuild_csv(args.result_root)
    atomic_json(args.result_root / "failures.json", failures)
    print(f"CSV: {csv_path}", flush=True)
    print(f"Failures: {len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
