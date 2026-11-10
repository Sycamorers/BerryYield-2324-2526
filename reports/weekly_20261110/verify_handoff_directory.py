"""Check that the staged handoff contains all source frames and plant masks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "BerryYield_2324_2526_handoff_20261110"
OLD_GNV = Path("/mnt/SB2324/data/GNV")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    with (PACKAGE / "raw_rgb_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = []
    size_mismatch = []
    for row in rows:
        path = PACKAGE / row["package_path"]
        if not path.is_file():
            missing.append(row["package_path"])
        elif path.stat().st_size != int(row["bytes"]):
            size_mismatch.append(row["package_path"])
    sha_mismatch = []
    for row in rows[::max(1, len(rows) // 100)]:
        path = PACKAGE / row["package_path"]
        if path.is_file() and digest(path) != row["sha256"]:
            sha_mismatch.append(row["package_path"])
    used_images = set()
    used_masks = set()
    source_records = 0
    plants = 0
    for season in ("2324", "2526"):
        for source in (ROOT / "canopy_pipeline/final_results/per_plot" / season).glob("*/*.json"):
            record = json.loads(source.read_text())
            source_records += 1
            for plant in record["plants"]:
                plants += 1
                used_images.add(plant["representative_image"])
                used_masks.add(plant["mask_path"])

    def check_saved(paths):
        absent = []
        for path in paths:
            try:
                relative = Path(path).relative_to(OLD_GNV)
            except ValueError:
                absent.append(path)
                continue
            if not (PACKAGE / "data/GNV" / relative).is_file():
                absent.append(path)
        return absent

    image_missing = check_saved(used_images)
    mask_missing = check_saved(used_masks)
    yolo_files = list((PACKAGE / "analysis/yolo_recount_2526/per_plot").glob("*.json"))
    result = {
        "raw_images": len(rows),
        "raw_images_by_season": {season: sum(row["season"] == season for row in rows)
                                 for season in ("2324", "2526")},
        "source_plot_records": source_records,
        "source_plant_records": plants,
        "representative_images": len(used_images),
        "mask_files_referenced": len(used_masks),
        "yolo_2526_plot_records": len(yolo_files),
        "missing_raw_files": missing[:10],
        "size_mismatch": size_mismatch[:10],
        "sampled_hash_mismatch": sha_mismatch[:10],
        "missing_representative_images": image_missing[:10],
        "missing_referenced_masks": mask_missing[:10],
        "passed": not any((missing, size_mismatch, sha_mismatch,
                           image_missing, mask_missing)) and source_records == 862
                  and len(yolo_files) == 504,
    }
    (PACKAGE / "HANDOFF_QA.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
