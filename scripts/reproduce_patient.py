#!/usr/bin/env python3
"""Validate one saved patient table; optionally replay the existing RidgeRun."""
import argparse
import ast
import csv
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--predictions", required=True, type=Path, help="Existing patient pred_results.csv")
    parser.add_argument("--output", required=True, type=Path, help="New, absent output directory")
    parser.add_argument("--execute", action="store_true", help="Fit the original ridge classifier (10,000 bootstraps)")
    args = parser.parse_args()
    if Path(args.patient).name != args.patient or args.patient in {".", ".."}:
        parser.error("Patient must be a single directory name")
    import project_paths as paths
    predictions = args.predictions.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; choose a new directory to preserve reference results")
    with predictions.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not {"y", "yhat"}.issubset(fields) or not rows:
        parser.error("Predictions must contain nonempty image columns plus y and yhat")
    images = [name for name in fields if name not in {"y", "yhat"}]
    if not images or any(row[c] not in {"0", "1", "0.0", "1.0"} for row in rows for c in images):
        parser.error("Image columns must be binary inclusion indicators")
    if any(row["yhat"] not in {"0", "1", "0.0", "1.0"} for row in rows):
        parser.error("Expected binary yhat values")
    # The existing runner drops duplicate rows before choosing CV folds.
    unique_rows = {tuple(row[c] for c in fields) for row in rows}
    classes = [row[fields.index("yhat")] for row in unique_rows]
    class_counts = [sum(float(value) == cls for value in classes) for cls in (0, 1)]
    if min(class_counts) < 3:
        parser.error("Too few unique rows in one prediction class for the existing CV implementation")
    with open(paths.metadata_path(), newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row["MI_ID"] == args.patient]
    if len(matches) != 1:
        parser.error("Expected exactly one metadata record")
    record = matches[0]
    metadata_images = ast.literal_eval(record["IMG_ID_LIST"])
    if float(record["liver_fatty"]) <= 0 or len(metadata_images) < 20:
        parser.error("Patient does not meet the existing positive / >=20-image cohort rule")
    if any(name not in set(metadata_images) for name in images):
        parser.error("Prediction columns do not match metadata image IDs")
    if {float(row["y"]) for row in rows} != {1.0}:
        parser.error("Saved ground truth does not match the binary positive cohort")
    with open(paths.split_path(), newline="") as handle:
        if args.patient not in {row["MI_ID"] for row in csv.DictReader(handle)}:
            parser.error("Patient is absent from the configured test split")
    missing = [name for name in metadata_images if not (Path(paths.image_dir()) / f"{args.patient}_{name}.jpg").is_file()]
    if missing:
        parser.error(f"Missing {len(missing)} patient images; see the private audit report")
    print(f"Validated {len(rows)} saved rows; {len(unique_rows)} unique rows; class counts {class_counts}.")
    if not args.execute:
        print("Validation only. Add --execute after the audit blockers and original-run provenance are resolved.")
        return
    os.environ["XAI_RIDGE_OUTPUT"] = str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Expose only the requested patient's original file to the unmodified runner.
    with tempfile.TemporaryDirectory(prefix="ridge-input-", dir=output.parent) as directory:
        subject_dir = Path(directory) / args.patient
        subject_dir.mkdir()
        (subject_dir / "pred_results.csv").symlink_to(predictions)
        from ridge_run import RidgeRun
        runner = RidgeRun(result_dir=directory)
        runner.selected_mi_ids = {args.patient}
        runner.run()
    expected = output / args.patient / "ridge_coefficients.csv"
    if not expected.is_file():
        raise SystemExit("Existing runner skipped the patient; inspect the job log")


if __name__ == "__main__":
    main()
