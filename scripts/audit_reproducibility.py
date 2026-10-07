#!/usr/bin/env python3
"""Read-only input audit; writes private reports without loading models or images."""
import argparse
import ast
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def source_inventory():
    sources = list(ROOT.glob("*.py")) + list((ROOT / "deprecated_and_archived").rglob("*.py")) + list((ROOT / "usflc_xai").glob("*.py"))
    inventory = {}
    for path in sources:
        text = path.read_text()
        tree = ast.parse(text)
        inventory[str(path.relative_to(ROOT))] = {
            "imports": sorted({name for node in ast.walk(tree) for name in (
                [alias.name for alias in node.names] if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])}),
            "absolute_literals": [{"line": node.lineno, "value": node.value} for node in ast.walk(tree)
                                  if isinstance(node, ast.Constant) and isinstance(node.value, str)
                                  and node.value.startswith(("/home/", "/mnt/", "/Users/", "/data/"))],
        }
    for path in list(ROOT.glob("*.ipynb")) + list((ROOT / "deprecated_and_archived").rglob("*.ipynb")):
        notebook = json.loads(path.read_text())
        cells = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            lines = "".join(cell.get("source", [])).splitlines()
            cells.append({"cell": index, "references": [line for line in lines if any(token in line for token in
                         ("import ", "/home/", "/mnt/", "/Users/", ".csv", ".ckpt", ".pt", ".json", ".npy"))]})
        inventory[str(path.relative_to(ROOT))] = {"cells": cells, "cells_with_outputs": sum(bool(c.get("outputs")) for c in notebook["cells"])}
    return inventory


def audit(args):
    # Defaults work without third-party packages, including before uv sync.
    metadata_file = Path(args.metadata).resolve()
    split_file = Path(args.split).resolve()
    image_directory = Path(args.images).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(metadata_file)
    test = read_csv(split_file)
    required = {"MI_ID", "liver_fatty", "IMG_ID_LIST"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Metadata lacks required columns")
    if not test or "MI_ID" not in test[0]:
        raise ValueError("Split lacks MI_ID")
    counts = Counter(row["MI_ID"] for row in rows)
    metadata = {row["MI_ID"]: row for row in rows}
    image_names = set(os.listdir(image_directory))
    summary = {"metadata_rows": len(rows), "duplicate_metadata_ids": sum(n - 1 for n in counts.values()),
               "test_rows": len(test), "duplicate_test_ids": len(test) - len({r['MI_ID'] for r in test}),
               "uploaded_jpg_files": sum(name.endswith('.jpg') for name in image_names),
               "test_ids_missing_metadata": 0, "test_image_references": 0, "missing_test_image_references": 0,
               "eligible_positive_patients": 0, "eligible_patients_with_all_images": 0,
               "patients_with_all_images": 0, "malformed_image_lists": 0}
    candidate_rows = []
    with (report_dir / "missing_test_images.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MI_ID", "IMG_ID", "expected_path"])
        for subject in test:
            record = metadata.get(subject["MI_ID"])
            if record is None:
                summary["test_ids_missing_metadata"] += 1
                continue
            try:
                ids = ast.literal_eval(record["IMG_ID_LIST"])
                if not isinstance(ids, (list, tuple)):
                    raise ValueError("Not an image list")
            except (ValueError, SyntaxError):
                summary["malformed_image_lists"] += 1
                continue
            missing = [img for img in ids if f"{record['MI_ID']}_{img}.jpg" not in image_names]
            summary["test_image_references"] += len(ids)
            summary["missing_test_image_references"] += len(missing)
            summary["patients_with_all_images"] += bool(ids) and not missing
            eligible = float(record["liver_fatty"]) > 0 and len(ids) >= 20
            summary["eligible_positive_patients"] += eligible
            summary["eligible_patients_with_all_images"] += eligible and not missing
            for img in missing:
                writer.writerow([record["MI_ID"], img, image_directory / f"{record['MI_ID']}_{img}.jpg"])
            if eligible and not missing:
                candidate_rows.append({"MI_ID": record["MI_ID"], "image_count": len(ids)})
    with (report_dir / "candidate_patients.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["MI_ID", "image_count"])
        writer.writeheader()
        writer.writerows(sorted(candidate_rows, key=lambda r: (r["image_count"], r["MI_ID"])))
    manifest = []
    files = list((ROOT / "usflc_xai").glob("*.py")) + list((ROOT / "data/meta_data").glob("*.csv")) + list((ROOT / "data/fattyliver_2_class_certained_0_123_4_20_40_dataset_lists").rglob("*.csv")) + list((ROOT / "checkpoints").rglob("*.ckpt"))
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    (report_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (report_dir / "source_inventory.json").write_text(json.dumps(source_inventory(), indent=2) + "\n")
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Private detail reports written under", report_dir)
    if args.strict and (summary["missing_test_image_references"] or summary["test_ids_missing_metadata"] or summary["malformed_image_lists"] or summary["duplicate_metadata_ids"] or summary["duplicate_test_ids"]):
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=ROOT / "data/meta_data/TWB_ABD_expand_modified_gasex_21072022.csv")
    parser.add_argument("--split", default=ROOT / "data/fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/dataset09/test_dataset09.csv")
    parser.add_argument("--images", default=ROOT / "data/cropped_images")
    parser.add_argument("--report-dir", default=ROOT / "outputs/reproducibility")
    parser.add_argument("--strict", action="store_true", help="Return nonzero for incomplete or ambiguous inputs")
    raise SystemExit(audit(parser.parse_args()))
