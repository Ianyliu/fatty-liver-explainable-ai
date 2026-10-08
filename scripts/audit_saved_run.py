#!/usr/bin/env python3
"""Validate saved inclusion/prediction tables without inference or fitting.

Patient-level findings are written only to the specified local report directory.
"""
import argparse
import ast
import csv
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table(path):
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        fields = next(reader, [])
        return fields, list(reader)


def binary(value):
    number = float(value)
    if number not in (0.0, 1.0):
        raise ValueError("Nonbinary value")
    return int(number)


def inspect_patient(path, metadata, split_ids, images):
    patient = path.parent.name
    item = {"patient": patient, "errors": []}
    if patient not in metadata or patient not in split_ids:
        item["errors"].append("Subject absent from metadata or test split")
        return item
    record = metadata[patient]
    image_ids = ast.literal_eval(record["IMG_ID_LIST"])
    fields, raw_rows = table(path)
    if len(fields) != len(set(fields)) or not {"y", "yhat"}.issubset(fields):
        item["errors"].append("Duplicate headers or missing y/yhat columns")
        return item
    if not raw_rows or any(len(row) != len(fields) for row in raw_rows):
        item["errors"].append("Empty/ragged prediction table")
        return item
    feature_indices = [i for i, name in enumerate(fields) if name not in {"y", "yhat"}]
    features = [fields[i] for i in feature_indices]
    if set(features) != set(image_ids):
        item["errors"].append("Feature IDs differ from metadata image list")
    try:
        rows = [tuple(binary(value) for value in row) for row in raw_rows]
    except (ValueError, TypeError):
        item["errors"].append("Invalid/nonbinary inclusion or label value")
        return item
    y_index, yhat_index = fields.index("y"), fields.index("yhat")
    if {row[y_index] for row in rows} != {int(float(record["liver_fatty"]) > 0)}:
        item["errors"].append("Saved y differs from binary metadata label")
    masks = [tuple(row[i] for i in feature_indices) for row in rows]
    labels_by_mask = {}
    for mask, row in zip(masks, rows):
        labels_by_mask.setdefault(mask, set()).add(row[yhat_index])
    inconsistent = sum(len(labels) > 1 for labels in labels_by_mask.values())
    if inconsistent:
        item["errors"].append("Identical inclusion masks have conflicting predictions")
    matrix_file = path.with_name("design_matrix.csv")
    item["design_matrix_present"] = matrix_file.is_file()
    if matrix_file.is_file():
        matrix_fields, matrix_rows = table(matrix_file)
        try:
            matched = matrix_fields == features and [tuple(binary(v) for v in row) for row in matrix_rows] == masks
        except (ValueError, TypeError):
            matched = False
        item["design_matrix_matches"] = matched
        if not matched:
            item["errors"].append("Design matrix differs from saved prediction inclusion columns")
    unique_rows = set(rows)
    class_counts = Counter(row[yhat_index] for row in unique_rows)
    missing = [img for img in image_ids if f"{patient}_{img}.jpg" not in images]
    item.update({"rows": len(rows), "unique_rows": len(unique_rows), "features": len(features),
                 "class_counts": dict(Counter(row[yhat_index] for row in rows)),
                 "unique_class_counts": dict(class_counts), "conflicting_masks": inconsistent,
                 "missing_images": len(missing), "complete_images": not missing,
                 "eligible_positive": float(record["liver_fatty"]) > 0 and len(image_ids) >= 20,
                 "minimum_cv_support": min(class_counts.get(cls, 0) for cls in (0, 1)) >= 10})
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/meta_data/TWB_ABD_expand_modified_gasex_21072022.csv")
    parser.add_argument("--split", type=Path, default=ROOT / "data/fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/dataset09/test_dataset09.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/cropped_images")
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    with args.metadata.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if len({row["MI_ID"] for row in records}) != len(records):
        parser.error("Duplicate metadata IDs")
    metadata = {row["MI_ID"]: row for row in records}
    with args.split.open(newline="") as handle:
        split_ids = {row["MI_ID"] for row in csv.DictReader(handle)}
    paths = sorted(args.run_dir.glob("*/pred_results.csv"))
    if not paths:
        parser.error("No patient pred_results.csv files found")
    with os.scandir(args.images) as entries:
        images = {entry.name for entry in entries if entry.is_file()}
    findings = [inspect_patient(path, metadata, split_ids, images) for path in paths]
    summary = {"patients": len(findings), "test_patients_without_predictions": len(split_ids - {path.parent.name for path in paths}),
               "patients_with_errors": sum(bool(item["errors"]) for item in findings),
               "design_matrices_present": sum(item.get("design_matrix_present", False) for item in findings),
               "design_matrices_matching": sum(item.get("design_matrix_matches", False) for item in findings),
               "prediction_rows": sum(item.get("rows", 0) for item in findings),
               "patients_with_both_classes": sum(len(item.get("class_counts", {})) == 2 for item in findings),
               "patients_with_conflicting_masks": sum(item.get("conflicting_masks", 0) > 0 for item in findings),
               "patients_with_complete_images": sum(item.get("complete_images", False) for item in findings),
               "eligible_patients_with_complete_images_and_cv_support": sum(item.get("eligible_positive", False) and item.get("complete_images", False) and item.get("minimum_cv_support", False) and not item["errors"] for item in findings)}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "patients.json").write_text(json.dumps(findings, indent=2) + "\n")
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    raise SystemExit(bool(summary["patients_with_errors"]))


if __name__ == "__main__":
    main()
