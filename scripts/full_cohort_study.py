#!/usr/bin/env python3
"""Extend the frozen P0 comparison to all eligible patients, reusing completed runs."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

import cohort_study
import patient_study as study
from import_image_bundle import expected_cohort
from summarize_patient_study import validate_run

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"train_samples": 1000, "evaluation_samples": 200, "seeds": [0, 1, 2],
            "minimum_images": 3, "ridge_alpha": 1.0, "adaptive_target_class1": 0.5,
            "deletion_fractions": [0, .1, .2, .3, .4, .5]}


def eligible_rows(manifest, metadata, split):
    with Path(manifest).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = expected_cohort(metadata, split)
    ids = [row["MI_ID"] for row in rows]
    if len(rows) != 135 or len(set(ids)) != 135 or set(ids) != set(expected):
        raise ValueError("Manifest must contain exactly the 135 eligible local test patients")
    if any(int(row["image_count"]) != len(expected[row["MI_ID"]]["images"]) for row in rows):
        raise ValueError("Manifest image counts differ from metadata")
    return sorted(rows, key=lambda row: (int(row["image_count"]), row["MI_ID"]))


def check_settings(plan):
    if any(plan[key] != value for key, value in SETTINGS.items()):
        raise ValueError("Child scientific settings differ from the fixed P0 comparison")
    if plan["streams"] != {str(seed): study.stream_seeds(seed) for seed in SETTINGS["seeds"]}:
        raise ValueError("Child RNG streams differ from the fixed P0 comparison")


def queries_per_seed(plan):
    steps = len([n for n in study.deletion_counts(len(plan["images"]), plan["deletion_fractions"], 3) if n])
    return 2 * plan["train_samples"] + plan["evaluation_samples"] + 2 + len(plan["images"]) + 5 * steps


def task_mapping(patients):
    tasks = []
    for item in patients:
        if item["reused"]:
            continue
        for seed in SETTINGS["seeds"]:
            tasks.append({"index": len(tasks), "patient_index": item["patient_index"],
                          "seed": seed, "plan": item["plan"]})
    return tasks


def prepare(args):
    import project_paths as paths
    output = args.output.resolve()
    allowed = (Path(paths.output_root()) / "complete_patient").resolve()
    if allowed not in output.parents:
        raise ValueError("Output must be below outputs/complete_patient/")
    rows = eligible_rows(args.manifest, paths.metadata_path(), paths.split_path())
    pilot = cohort_study.verified_cohort(args.reuse)
    reuse = {item["patient"]: item for item in pilot["patients"]}
    if len(reuse) != 10 or not set(reuse) <= {row["MI_ID"] for row in rows}:
        raise ValueError("Prior ten-patient study does not belong to the full cohort")
    # Validate old evidence and all current patients before creating the new plan.
    old_evidence = []
    for item in pilot["patients"]:
        plan_path = Path(item["plan"])
        child = study.verified_plan(plan_path)
        check_settings(child)
        for seed in SETTINGS["seeds"]:
            directory = plan_path.parent / "runs" / f"seed-{seed}"
            validate_run(child, study.sha256(plan_path), directory, seed)
            old_evidence.extend(sorted(directory.glob("*.csv")))
            old_evidence.append(directory / "report.json")
    for row in rows:
        _, images = study.patient_record(row["MI_ID"], paths)
        if len(images) != int(row["image_count"]):
            raise ValueError("Patient image coverage changed")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "scripts/cohort_study.py", ROOT / "scripts/import_image_bundle.py",
               ROOT / "scripts/summarize_patient_study.py", ROOT / "slurm/full_cohort.sbatch",
               ROOT / "slurm/full_cohort_summary.sbatch"]
    files = [args.manifest.resolve(), args.reuse.resolve(), *sources, *old_evidence]
    patients = []
    for index, row in enumerate(rows):
        reused = row["MI_ID"] in reuse
        if reused:
            plan_path = Path(reuse[row["MI_ID"]]["plan"])
        else:
            child_output = output / f"patient-{index:03}"
            study.prepare(SimpleNamespace(output=child_output, patient=row["MI_ID"],
                                          train_samples=1000, eval_samples=200, seeds=[0, 1, 2]))
            plan_path = child_output / "plan.json"
        child = study.verified_plan(plan_path)
        check_settings(child)
        files.append(plan_path)
        patients.append({"patient_index": index, "patient": row["MI_ID"], "plan": str(plan_path),
                         "reused": reused, "expected_queries_per_seed": queries_per_seed(child)})
    tasks = task_mapping(patients)
    if len(tasks) != 375:
        raise ValueError("Expected 375 new tasks after reusing ten completed patients")
    plan = {"schema": 1, "scope": "All 135 eligible positive test09 patients; extension after ten-patient review",
        "prepared_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "selection": "All test09 patients with liver_fatty > 0 and >=20 metadata images; no outcome-based exclusion",
        "settings": SETTINGS, "patients": patients, "tasks": tasks,
        "reused_patients": 10, "new_patients": 125,
        "planned_total_model_queries": sum(p["expected_queries_per_seed"] * 3 for p in patients),
        "planned_new_model_queries": sum(p["expected_queries_per_seed"] * 3 for p in patients if not p["reused"]),
        "review": "Ten-patient methods gate passed; adaptive worse mean MAE in 9/10 patients and balance missed in 30/30; retain fixed protocol and all outcomes",
        "files": [{"path": str(p.resolve()), "sha256": study.sha256(p)} for p in files]}
    for source in sources:
        destination = output / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    study.write_json(output / "full_cohort_plan.json", plan)
    print("Prepared full cohort: 135 patients, ten reused, 375 new tasks;", plan["planned_new_model_queries"], "new model calls")


def verified(path):
    plan = json.loads(Path(path).read_text())
    for entry in plan["files"]:
        if study.sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen full-cohort source/input changed: " + entry["path"])
    patients = plan["patients"]
    if (len(patients) != 135 or [p["patient_index"] for p in patients] != list(range(135))
            or len({p["patient"] for p in patients}) != 135
            or sum(p["reused"] for p in patients) != 10 or plan["settings"] != SETTINGS):
        raise ValueError("Invalid full-cohort patient mapping/settings")
    if plan["tasks"] != task_mapping(patients) or len(plan["tasks"]) != 375:
        raise ValueError("Invalid full-cohort task mapping")
    return plan


def figures(patient_rows, deletion_rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [row for row in patient_rows if row["primary_available"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    x = [row["random_mean_novel_mae"] for row in valid]
    y = [row["adaptive_mean_novel_mae"] for row in valid]
    high = max([.01, *x, *y]) * 1.08
    axes[0].scatter(x, y, color="#0072B2", alpha=.7, s=28)
    axes[0].plot([0, high], [0, high], linestyle="--", color="#666666")
    axes[0].set(xlabel="Random sampling: mean novel-mask MAE", ylabel="Adaptive sampling: mean novel-mask MAE",
                title="Each point is one patient", xlim=(0, high), ylim=(0, high))
    for arm, color in [("random", "#0072B2"), ("adaptive", "#D55E00")]:
        values = [row["descending_minus_random_auc"] for row in deletion_rows if row["arm"] == arm]
        axes[1].scatter(np.arange(len(values)), values, label=arm.capitalize(), s=15, alpha=.7, color=color)
    axes[1].axhline(0, color="#666666", linestyle="--")
    axes[1].set(xlabel="Patient index (manifest order)", ylabel="Descending minus random deletion AUC",
                title="Negative values favor coefficient-ranked deletion")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(Path(output) / "patient_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(Path(output) / "patient_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def summarize(args):
    plan = verified(args.plan)
    fidelity, deletion, evidence, diagnostics = [], [], [], []
    total = 0
    for item in plan["patients"]:
        child_path = Path(item["plan"])
        child = study.verified_plan(child_path)
        check_settings(child)
        if child["patient"] != item["patient"] or queries_per_seed(child) != item["expected_queries_per_seed"]:
            raise ValueError("Child identity or query budget differs from full plan")
        for seed in SETTINGS["seeds"]:
            directory = child_path.parent / "runs" / f"seed-{seed}"
            checked = validate_run(child, study.sha256(child_path), directory, seed)
            report = checked["report"]
            total += report["total_model_queries"]
            fidelity.extend(dict(patient_index=item["patient_index"], **row) for row in checked["fidelity"])
            deletion.extend(dict(patient_index=item["patient_index"], **row) for row in checked["deletion_summary"])
            diagnostics.append({"patient_index": item["patient_index"], "seed": seed, "reused": item["reused"],
                "adaptive_balance_reached": report["sampling_summary"]["class_balance_reached"],
                "gpu_name": report["gpu_name"], "elapsed_seconds": report["elapsed_seconds"],
                "model_queries": report["total_model_queries"]})
            evidence.extend({"path": str(p), "sha256": study.sha256(p)} for p in [*sorted(directory.glob("*.csv")), directory / "report.json"])
    if total != plan["planned_total_model_queries"]:
        raise ValueError("Full-cohort total query budget differs from plan")
    patients, analysis = cohort_study.aggregate(fidelity, range(135), SETTINGS["seeds"])
    deletion_patients = []
    for index in range(135):
        for arm in ("random", "adaptive"):
            means = {control: float(np.mean([row["area_under_curve"] for row in deletion
                if row["patient_index"] == index and row["arm"] == arm and row["control"] == control]))
                for control in ("descending", "ascending", "random")}
            deletion_patients.append({"patient_index": index, "arm": arm,
                **{control + "_mean_auc": value for control, value in means.items()},
                "descending_minus_random_auc": means["descending"] - means["random"],
                "descending_minus_ascending_auc": means["descending"] - means["ascending"]})
    output = args.plan.resolve().parent / "summary"
    output.mkdir(exist_ok=False)
    for name, rows in [("fidelity_by_seed", fidelity), ("fidelity_by_patient", patients),
                       ("deletion_by_seed", deletion), ("deletion_by_patient", deletion_patients), ("run_diagnostics", diagnostics)]:
        study.write_csv(output / (name + ".csv"), rows)
    analysis.update(status="passed", scope=plan["scope"], total_model_queries=total,
        new_model_queries=plan["planned_new_model_queries"], reused_patients=10, validated_patient_seed_runs=405,
        adaptive_balance_reached_runs=sum(row["adaptive_balance_reached"] for row in diagnostics),
        plan_sha256=study.sha256(args.plan), evidence=evidence,
        limitations=["Eligible positive test09 patients only; cohort extension follows inspection of ten-patient results.",
                     "Seeds are repeated measurements; patient means receive equal weight.",
                     "Equal training rows with unequal inference overhead; no equal-compute claim.",
                     "Novel-mask filtering, class support and training-mean baselines require inspection.",
                     "Descriptive results; no clinical, population-significance or historical reproduction claim."])
    study.write_json(output / "analysis.json", analysis)
    figures(patients, deletion_patients, output)
    (output / "report.md").write_text("# Full eligible-cohort P0 comparison\n\n"
        f"All 405 patient/seed runs validated across 135 patients; {total:,} total model calls, including the reused ten-patient results.\n\n"
        f"Mean patient paired adaptive-minus-random novel-mask MAE: {analysis['mean_patient_paired_mae_difference']}. "
        f"Adaptive reached its balance target in {analysis['adaptive_balance_reached_runs']}/405 runs.\n\n"
        + "\n\n".join(analysis["limitations"]) + "\n")
    print("Validated full-cohort report:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--manifest", required=True, type=Path)
    prep.add_argument("--reuse", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--task", required=True, type=int)
    run.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    run.add_argument("--threads", type=int, default=2)
    summary = commands.add_parser("summarize")
    summary.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        plan = verified(args.plan)
        if not 0 <= args.task < len(plan["tasks"]):
            raise ValueError("Task index outside the full cohort")
        task = plan["tasks"][args.task]
        child = study.verified_plan(Path(task["plan"]))
        check_settings(child)
        if child["patient"] != plan["patients"][task["patient_index"]]["patient"]:
            raise ValueError("Task patient identity mismatch")
        study.run(SimpleNamespace(plan=Path(task["plan"]), seed=task["seed"], output=None,
                                  device=args.device, threads=args.threads))


if __name__ == "__main__":
    main()
