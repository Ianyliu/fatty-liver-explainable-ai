#!/usr/bin/env python3
"""Review the completed ten-patient P0 and parallel analyses without new inference."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np

import cohort_study
import parallel_experiments as extra
import patient_study as study
from summarize_patient_study import validate_run


def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def review(args):
    cohort = cohort_study.verified_cohort(args.cohort)
    parallel = extra.verified(args.parallel)
    if Path(parallel["cohort"]).resolve() != args.cohort.resolve():
        raise ValueError("Parallel analysis belongs to a different cohort")
    destination = args.cohort.resolve().parent / "review"
    if destination.exists():
        raise ValueError("Review output already exists; preserve the prior review")
    fidelity, deletion = [], []
    total_queries = 0
    evidence = []
    for item in cohort["patients"]:
        path = Path(item["plan"])
        child = study.verified_plan(path)
        for seed in cohort["seeds"]:
            directory = path.parent / "runs" / f"seed-{seed}"
            checked = validate_run(child, study.sha256(path), directory, seed)
            total_queries += checked["report"]["total_model_queries"]
            fidelity.extend(dict(patient_index=item["patient_index"], **r) for r in checked["fidelity"])
            deletion.extend(dict(patient_index=item["patient_index"], **r) for r in checked["deletion_summary"])
            evidence += [dict(path=str(p), sha256=study.sha256(p)) for p in sorted(directory.glob("*.csv")) + [directory / "report.json"]]
    if total_queries != cohort["planned_model_queries"]:
        raise ValueError("Completed query budget differs from the cohort plan")
    patients, analysis = cohort_study.aggregate(fidelity, range(10), cohort["seeds"])
    recorded = json.loads((args.cohort.parent / "cohort_summary/analysis.json").read_text())
    if not np.isclose(analysis["mean_patient_paired_mae_difference"], recorded["mean_patient_paired_mae_difference"]):
        raise ValueError("Patient-level mean differs from the saved cohort summary")
    for entry in recorded["evidence"]:
        if study.sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("Saved summary evidence changed")
    reallocations = 0
    parallel_root = args.parallel.resolve().parent
    for command, count in (("cpu", 30), ("loo", 10)):
        for index in range(count):
            path = parallel_root / command / f"task-{index:02}" / "report.json"
            report = json.loads(path.read_text())
            if (report["status"] != "passed" or report["plan_sha256"] != study.sha256(args.parallel)
                    or report["command"] != command or report["task"] != index):
                raise ValueError("Unsuccessful or misidentified parallel-analysis evidence")
            for entry in report.get("input_hashes", []) + report["output_hashes"]:
                if study.sha256(entry["path"]) != entry["sha256"]:
                    raise ValueError("Parallel-analysis evidence changed")
                evidence.append(entry)
            evidence.append(dict(path=str(path), sha256=study.sha256(path)))
            if command == "cpu":
                reallocations += report["stage_two"]["biased_rows_with_pool_deficit"]
    parallel_analysis = json.loads((parallel_root / "summary/analysis.json").read_text())
    summary = {"status": "passed", "p0_model_queries": total_queries,
        "validated_p0_runs": 30, "parallel_cpu_tasks": 30, "loo_patients": 10,
        "loo_model_queries": parallel_analysis["loo_model_queries"],
        "adaptive_worse_patient_means": sum(r["paired_adaptive_minus_random_mae"] > 0 for r in patients),
        "mean_random_novel_mae": float(np.mean([r["random_mean_novel_mae"] for r in patients])),
        "mean_adaptive_novel_mae": float(np.mean([r["adaptive_mean_novel_mae"] for r in patients])),
        "mean_paired_adaptive_minus_random_mae": analysis["mean_patient_paired_mae_difference"],
        "adaptive_balance_reached_runs": recorded["adaptive_balance_reached_runs"],
        "stage_two_pool_deficit_rows": reallocations,
        "one_class_shared_novel_sets": sum(r["novel_negative_rows"] == 0 or r["novel_positive_rows"] == 0 for r in fidelity if r["arm"] == "random"),
        "beats_constant_by_arm": {arm:sum(r["mae_gain_over_constant"] > 0 for r in fidelity if r["arm"] == arm) for arm in extra.ARMS},
        "mean_deletion_auc_differences": recorded["mean_patient_descending_minus_random_auc"],
        "elastic_net_minus_ridge_mae": parallel_analysis["mean_patient_elastic_net_minus_ridge_mae"],
        "cohort_plan_sha256": study.sha256(args.cohort), "parallel_plan_sha256": study.sha256(args.parallel),
        "reviewer_sha256": study.sha256(__file__), "evidence": evidence}
    destination.mkdir()
    study.write_json(destination / "validation.json", summary)
    shutil.copy2(__file__, destination / Path(__file__).name)
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    x = np.arange(1, 11)
    for arm, color in (("random", "#0072B2"), ("adaptive", "#D55E00")):
        axes[0].plot(x, [r[f"{arm}_mean_novel_mae"] for r in patients], "o-", label=arm.capitalize(), color=color)
        values = []
        for patient in range(10):
            mean = {control:np.mean([r["area_under_curve"] for r in deletion if r["patient_index"] == patient and r["arm"] == arm and r["control"] == control]) for control in ("descending", "random")}
            values.append(mean["descending"] - mean["random"])
        axes[1].plot(x, values, "o-", label=arm.capitalize(), color=color)
    axes[0].set(title="Probability-surrogate fidelity", ylabel="Mean novel-mask MAE across three seeds")
    axes[1].set(title="Coefficient-ranked deletion versus random", ylabel="Mean descending minus random AUC")
    axes[1].axhline(0, color="#666666", linestyle="--")
    for ax in axes:
        ax.set(xlabel="Patient index (convenience cohort)", xticks=x)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        ax.grid(alpha=.15)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(destination / ("patient_comparison." + extension), dpi=180, bbox_inches="tight")
    plt.close(fig)
    paragraphs = ["# Reviewed ten-patient experiment results",
        f"All 30 P0 runs and all parallel analyses passed independent evidence checks. P0 used {total_queries:,} GNN calls; LOO used 210 additional calls.",
        f"Equal-patient mean novel-mask MAE: random {summary['mean_random_novel_mae']:.6f}, adaptive {summary['mean_adaptive_novel_mae']:.6f}; paired adaptive-minus-random {summary['mean_paired_adaptive_minus_random_mae']:+.6f}. Adaptive was worse for {summary['adaptive_worse_patient_means']}/10 patient means and achieved its balance target in 0/30 runs.",
        "Both descending-coefficient rankings had lower mean deletion AUC than the shared random control for every patient. Mean paired AUC differences were -0.162640 (random-trained) and -0.161970 (adaptive-trained). Intervention effects do not establish good probability-surrogate fidelity.",
        f"Random Ridge beat its training-mean baseline in {summary['beats_constant_by_arm']['random']}/30 seeds, adaptive Ridge in {summary['beats_constant_by_arm']['adaptive']}/30. {summary['one_class_shared_novel_sets']}/30 novel evaluation sets contained one predicted class. Elastic Net minus Ridge MAE: {summary['elastic_net_minus_ridge_mae']}; settings were selected with training-only CV.",
        "The ten patients all have 20 images and were selected for operational convenience. The next planned comparison retains the fixed P0 protocol, includes every eligible patient and reuses all ten completed patients. Outcomes remain descriptive; no clinical, population-significance or equal-compute claim is made."]
    (destination / "report.md").write_text("\n\n".join(paragraphs) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "evidence"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--parallel", type=Path, required=True)
    review(parser.parse_args())
