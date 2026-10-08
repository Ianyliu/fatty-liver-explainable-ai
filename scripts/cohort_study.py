#!/usr/bin/env python3
"""Orchestrate a ten-patient P0 study using the unchanged patient-study runner."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

import patient_study as study
from summarize_patient_study import validate_run

ROOT = Path(__file__).resolve().parents[1]


def cohort_rows(manifest):
    with Path(manifest).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10 or len({row["MI_ID"] for row in rows}) != 10:
        raise ValueError("This bounded study requires exactly ten distinct patients")
    return rows


def pilot_gate(rows, directory):
    """Require successful smoke evidence for each of the same ten patients."""
    reports = []
    for index, row in enumerate(rows):
        path = Path(directory) / f"task-{index}" / "report.json"
        report = json.loads(path.read_text())
        if (report["status"] != "passed" or report["patient"] != row["MI_ID"]
                or report["image_count"] != int(row["image_count"])
                or report["settings"]["samples"] != 20
                or report["model_prediction_calls_before_references"] != 21):
            raise ValueError("Pilot evidence is unsuccessful or differs from the selected cohort")
        reports.append(path)
    return reports


def prepare(args):
    import project_paths as paths
    output = args.output.resolve()
    allowed = (Path(paths.output_root()) / "complete_patient").resolve()
    if allowed not in output.parents:
        raise ValueError("Cohort output must be below outputs/complete_patient/")
    rows = cohort_rows(args.manifest)
    pilot_reports = pilot_gate(rows, args.pilot)
    # Validate every patient before creating any child study.
    for row in rows:
        record, images = study.patient_record(row["MI_ID"], paths)
        if float(record["liver_fatty"]) <= 0 or len(images) != int(row["image_count"]) or len(images) < 20:
            raise ValueError("Manifest patient fails the complete eligible-cohort gate")
    output.mkdir(parents=True, exist_ok=False)
    files = [args.manifest.resolve(), *pilot_reports, Path(__file__),
             ROOT / "scripts/summarize_patient_study.py", ROOT / "slurm/cohort_study.sbatch",
             ROOT / "slurm/cohort_summary.sbatch"]
    tasks, studies = [], []
    for index, row in enumerate(rows):
        child = output / f"patient-{index:02}"
        study.prepare(SimpleNamespace(output=child, patient=row["MI_ID"],
                                      train_samples=1000, eval_samples=200, seeds=[0, 1, 2]))
        plan_path = child / "plan.json"
        plan = json.loads(plan_path.read_text())
        files.append(plan_path)
        steps = len([n for n in study.deletion_counts(len(plan["images"]), plan["deletion_fractions"], 3) if n])
        calls = 2000 + 200 + 2 + len(plan["images"]) + 5 * steps
        studies.append({"patient_index": index, "patient": row["MI_ID"], "plan": str(plan_path),
                        "expected_queries_per_seed": calls})
        for seed in plan["seeds"]:
            tasks.append({"index": len(tasks), "patient_index": index,
                          "seed": seed, "plan": str(plan_path)})
    cohort = {"schema": 1, "scope": "ten-patient convenience cohort; prospective P0 comparison",
        "selection": "Frozen image-recovery pilot manifest: image count then patient ID; not representative",
        "train_samples_per_arm": 1000, "evaluation_samples": 200, "seeds": [0, 1, 2],
        "patients": studies, "tasks": tasks, "planned_model_queries": sum(
            item["expected_queries_per_seed"] * 3 for item in studies),
        "aggregation": "Mean paired seed differences within patient, then equal weight across patients",
        "files": [{"path": str(path.resolve()), "sha256": study.sha256(path)} for path in files]}
    for source in (Path(__file__), ROOT / "scripts/summarize_patient_study.py", ROOT / "slurm/cohort_study.sbatch",
                   ROOT / "slurm/cohort_summary.sbatch"):
        destination = output / "cohort_source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    study.write_json(output / "cohort_plan.json", cohort)
    print("Prepared ten-patient cohort, 30 patient/seed tasks:", output / "cohort_plan.json")


def verified_cohort(path):
    cohort = json.loads(Path(path).read_text())
    for entry in cohort["files"]:
        if study.sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("Cohort input/source changed after preparation: " + entry["path"])
    tasks = cohort["tasks"]
    if len(tasks) != 30 or [task["index"] for task in tasks] != list(range(30)):
        raise ValueError("Invalid cohort task mapping")
    if len({(task["patient_index"], task["seed"]) for task in tasks}) != 30:
        raise ValueError("Duplicate patient/seed task")
    for task in tasks:
        if (not 0 <= task["patient_index"] < 10 or task["seed"] not in (0, 1, 2)
                or task["plan"] != cohort["patients"][task["patient_index"]]["plan"]):
            raise ValueError("Cohort task differs from its patient plan")
    return cohort


def aggregate(fidelity, expected_patients, seeds):
    """Keep missing novel-mask results explicit and use patients as the unit."""
    expected_patients, seeds = tuple(expected_patients), tuple(seeds)
    indexed = {(row["patient_index"], row["seed"], row["arm"]): row for row in fidelity}
    expected = {(p, s, arm) for p in expected_patients for s in seeds for arm in ("random", "adaptive")}
    if len(indexed) != len(fidelity) or set(indexed) != expected:
        raise ValueError("Missing or duplicate patient/seed/arm evidence")
    patients = []
    for patient in expected_patients:
        pairs = [(indexed[patient, seed, "random"], indexed[patient, seed, "adaptive"]) for seed in seeds]
        available = [(r, a) for r, a in pairs if r["novel_mae"] is not None and a["novel_mae"] is not None]
        complete = len(available) == len(seeds)
        patients.append({"patient_index": patient, "available_seed_pairs": len(available),
            "planned_seed_pairs": len(seeds), "primary_available": complete,
            "random_mean_novel_mae": float(np.mean([r["novel_mae"] for r, _ in available])) if complete else None,
            "adaptive_mean_novel_mae": float(np.mean([a["novel_mae"] for _, a in available])) if complete else None,
            "paired_adaptive_minus_random_mae": float(np.mean([a["novel_mae"] - r["novel_mae"] for r, a in available])) if complete else None})
    usable = [row["paired_adaptive_minus_random_mae"] for row in patients if row["primary_available"]]
    return patients, {"planned_patients": len(patients), "patients_with_all_seed_pairs": len(usable),
        "mean_patient_paired_mae_difference": float(np.mean(usable)) if len(usable) == len(patients) else None,
        "available_patient_mean_paired_mae_difference": float(np.mean(usable)) if usable else None,
        "missing_primary_patients": [row["patient_index"] for row in patients if not row["primary_available"]]}


def summarize(args):
    cohort = verified_cohort(args.cohort)
    fidelity, deletion, evidence, diagnostics = [], [], [], []
    total = 0
    # Require every planned patient/seed, independently validating raw CSVs.
    for item in cohort["patients"]:
        plan_path = Path(item["plan"])
        plan = study.verified_plan(plan_path)
        for seed in cohort["seeds"]:
            directory = plan_path.parent / "runs" / f"seed-{seed}"
            result = validate_run(plan, study.sha256(plan_path), directory, seed)
            total += result["report"]["total_model_queries"]
            diagnostics.append({"patient_index": item["patient_index"], "seed": seed,
                "adaptive_balance_reached": result["report"]["sampling_summary"]["class_balance_reached"],
                "gpu_name": result["report"]["gpu_name"], "elapsed_seconds": result["report"]["elapsed_seconds"],
                "model_queries": result["report"]["total_model_queries"]})
            fidelity.extend(dict(patient_index=item["patient_index"], **row) for row in result["fidelity"])
            deletion.extend(dict(patient_index=item["patient_index"], **row) for row in result["deletion_summary"])
            evidence.extend({"path": str(path), "sha256": study.sha256(path)} for path in sorted(directory.glob("*.csv")))
            evidence.append({"path": str(directory / "report.json"), "sha256": study.sha256(directory / "report.json")})
    if total != cohort["planned_model_queries"]:
        raise ValueError("Total cohort query budget differs from plan")
    patient_rows, analysis = aggregate(fidelity, range(10), cohort["seeds"])
    output = args.cohort.resolve().parent / "cohort_summary"
    output.mkdir(exist_ok=False)
    study.write_csv(output / "fidelity_by_seed.csv", fidelity)
    study.write_csv(output / "fidelity_by_patient.csv", patient_rows)
    study.write_csv(output / "deletion_by_seed.csv", deletion)
    study.write_csv(output / "run_diagnostics.csv", diagnostics)
    deletion_patients = []
    for patient in range(10):
        for arm in ("random", "adaptive"):
            means = {control: float(np.mean([row["area_under_curve"] for row in deletion
                if row["patient_index"] == patient and row["arm"] == arm and row["control"] == control]))
                for control in ("descending", "ascending", "random")}
            deletion_patients.append({"patient_index": patient, "arm": arm,
                **{control + "_mean_auc": value for control, value in means.items()},
                "descending_minus_random_auc": means["descending"] - means["random"],
                "descending_minus_ascending_auc": means["descending"] - means["ascending"]})
    study.write_csv(output / "deletion_by_patient.csv", deletion_patients)
    analysis.update(scope=cohort["scope"], validation="All 30 planned runs and raw CSV calculations passed",
                    total_model_queries=total, cohort_plan_sha256=study.sha256(args.cohort),
                    adaptive_balance_reached_runs=sum(row["adaptive_balance_reached"] for row in diagnostics),
                    mean_patient_descending_minus_random_auc={arm: float(np.mean([
                        row["descending_minus_random_auc"] for row in deletion_patients if row["arm"] == arm]))
                        for arm in ("random", "adaptive")},
                    limitations=["Convenience sample of ten small-image-count patients; no population inference.",
                                 "Three seeds are repeated measurements within patients.",
                                 "Equal training rows, unequal sampler overhead; no efficiency claim.",
                                 "Novel-mask filtering and weak class support must be inspected per patient/seed."],
                    evidence=evidence)
    study.write_json(output / "analysis.json", analysis)
    lines = ["# Ten-patient prospective P0 results", "", analysis["validation"] + ".",
             f"Fresh model calls: {total:,}. Patient-level paired MAE difference (adaptive minus random): "
             f"{analysis['mean_patient_paired_mae_difference']}.", "", *analysis["limitations"], "",
             f"Adaptive achieved its class-balance target in {analysis['adaptive_balance_reached_runs']}/30 runs.",
             f"Mean patient descending-minus-random deletion AUC: {analysis['mean_patient_descending_minus_random_auc']}.",
             "Inspect fidelity_by_seed.csv for class support, baselines and overlap; fidelity_by_patient.csv for patient-level paired means; deletion_by_seed.csv and deletion_by_patient.csv for ranked and control AUCs."]
    (output / "report.md").write_text("\n\n".join(lines) + "\n")
    print("Validated cohort summary:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--manifest", required=True, type=Path)
    prep.add_argument("--pilot", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--cohort", required=True, type=Path)
    run.add_argument("--task", required=True, type=int)
    run.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    run.add_argument("--threads", type=int, default=2)
    summary = commands.add_parser("summarize")
    summary.add_argument("--cohort", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        cohort = verified_cohort(args.cohort)
        if not 0 <= args.task < len(cohort["tasks"]):
            raise ValueError("Task index outside the frozen cohort")
        task = cohort["tasks"][args.task]
        study.run(SimpleNamespace(plan=Path(task["plan"]), seed=task["seed"], output=None,
                                  device=args.device, threads=args.threads))


if __name__ == "__main__":
    main()
