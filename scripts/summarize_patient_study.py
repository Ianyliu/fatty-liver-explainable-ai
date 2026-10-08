#!/usr/bin/env python3
"""Validate n=1 study evidence tables, then export descriptive results and figures."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patient_study import sha256, verified_plan, write_csv, write_json

ARMS = ("random", "adaptive")
CONTROLS = ("descending", "ascending", "random")


def read_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty evidence table: " + str(path))
    return rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(bool(np.allclose(actual, expected, rtol=1e-9, atol=1e-11)), message)


def matrix(rows, images):
    values = np.array([[int(row[image]) for image in images] for row in rows])
    require(bool(np.isin(values, [0, 1]).all()), "Nonbinary mask in evidence table")
    return values


def column(rows, key, dtype=float):
    return np.array([dtype(row[key]) for row in rows])


def recompute_metrics(target, prediction, labels):
    if not len(target):
        return {"status": "unavailable_no_rows", "rows": 0}
    errors = prediction - target
    return {"status": "available", "rows": len(target),
        "mae": float(np.abs(errors).sum() / len(errors)),
        "rmse": float(np.sqrt(np.dot(errors, errors) / len(errors))),
        "class_agreement": float(np.count_nonzero((prediction >= .5) == labels) / len(labels)),
        "class_counts": {str(c): int(np.count_nonzero(labels == c)) for c in (0, 1)},
        "one_class": bool(len(set(labels)) < 2),
        "outside_probability_range": int(np.count_nonzero((prediction < 0) | (prediction > 1)))}


def compare_metrics(actual, expected):
    require(actual.keys() == expected.keys(), "Metric fields disagree")
    for key in actual:
        if isinstance(actual[key], float):
            close(actual[key], expected[key], "Recomputed metric disagrees: " + key)
        else:
            require(actual[key] == expected[key], "Recomputed metric disagrees: " + key)


def rank_similarity(first, second):
    from scipy.stats import rankdata
    first, second = np.asarray(first), np.asarray(second)
    require(first.shape == second.shape and first.ndim == 1, "Rank vectors must align")
    require(bool(np.isfinite(first).all() and np.isfinite(second).all()), "Rank vectors must be finite")
    a, b = rankdata(first), rankdata(second)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def coefficient_stability(plan, runs):
    rows = []
    k = min(5, len(plan["images"]))
    for arm in ARMS:
        for i, first in enumerate(runs):
            for j in range(i + 1, len(runs)):
                second = runs[j]
                def top(values):
                    return set(sorted(range(len(values)), key=lambda n: (-values[n], plan["images"][n]))[:k])
                a, b = first["coefficients"][arm], second["coefficients"][arm]
                rows.append({"arm": arm, "seed_a": plan["seeds"][i], "seed_b": plan["seeds"][j],
                    "spearman_rank_correlation": rank_similarity(a, b), "top_k": k,
                    "top_k_overlap_count": len(top(a) & top(b))})
    return rows


def validate_run(plan, plan_hash, directory, seed):
    report = json.loads((directory / "report.json").read_text())
    require(report["status"] == "passed", "Missing or unsuccessful seed " + str(seed))
    require(report["seed"] == seed and report["patient"] == plan["patient"], "Run identity mismatch")
    require(report["plan_sha256"] == plan_hash, "Run belongs to a different plan")
    images = plan["images"]
    ledger = read_rows(directory / "queries.csv")
    require(column(ledger, "query", int).tolist() == list(range(1, len(ledger) + 1)), "Broken query sequence")
    query_stages = Counter(row["stage"] for row in ledger)
    require(dict(query_stages) == report["query_counts"], "Query ledger differs from report")
    counts = sorted({min(len(images) - plan["minimum_images"], int(len(images) * f + 1e-12))
                     for f in plan["deletion_fractions"]})
    steps = len([count for count in counts if count])
    expected_stages = {"original": 1, "random_training": plan["train_samples"],
        "adaptive_validation": 1, "adaptive_singletons": len(images),
        "adaptive_training": plan["train_samples"], "shared_evaluation": plan["evaluation_samples"],
        "shared_random_deletion": steps}
    expected_stages.update({f"{arm}_{control}_deletion": steps for arm in ARMS for control in CONTROLS[:2]})
    require(dict(query_stages) == expected_stages, "Unexpected query budget")
    require(len(ledger) == report["total_model_queries"], "Incorrect total model queries")
    require(bool(np.isfinite(column(ledger, "p_class1")).all()), "Nonfinite probability")
    require(bool(((column(ledger, "p_class1") >= 0) & (column(ledger, "p_class1") <= 1)).all()), "Invalid GNN probability")
    require(bool(np.isin(column(ledger, "yhat", int), [0, 1]).all()), "Invalid predicted class")
    ledger_by_stage = {stage: [row for row in ledger if row["stage"] == stage] for stage in query_stages}

    def match_queries(rows, stage):
        expected = ledger_by_stage[stage]
        require(len(rows) == len(expected), "Evidence row count differs from ledger: " + stage)
        require(bool(np.array_equal(matrix(rows, images), matrix(expected, images))), "Mask differs from ledger: " + stage)
        for key in ("p_class1", "yhat", "logit0", "logit1", "edges"):
            close(column(rows, key), column(expected, key), "Response differs from ledger: " + stage)

    evaluation = read_rows(directory / "evaluation.csv")
    match_queries(evaluation, "shared_evaluation")
    eval_matrix = matrix(evaluation, images)
    require(bool((eval_matrix.sum(axis=1) >= plan["minimum_images"]).all()), "Too-small evaluation subset")
    eval_keys = [tuple(row) for row in eval_matrix]
    training, coefficients, overlaps, targets = {}, {}, {}, {}
    for arm in ARMS:
        training[arm] = read_rows(directory / f"{arm}_training.csv")
        match_queries(training[arm], arm + "_training")
        train_matrix = matrix(training[arm], images)
        require(bool((train_matrix.sum(axis=1) >= plan["minimum_images"]).all()), "Too-small training subset")
        train_keys = {tuple(row) for row in train_matrix}
        require(len(train_keys) == report["arms"][arm]["unique_training_masks"], "Incorrect unique training mask count")
        overlaps[arm] = np.array([key in train_keys for key in eval_keys])
        require(bool(np.array_equal(overlaps[arm], column(evaluation, "in_" + arm + "_training", int))), "Incorrect overlap flags")
        targets[arm] = column(training[arm], "p_class1")
        coefficient_rows = read_rows(directory / f"{arm}_coefficients.csv")
        require([row["image_id"] for row in coefficient_rows] == images, "Coefficient column ordering changed")
        coefficients[arm] = column(coefficient_rows, "coefficient")
        predicted = eval_matrix @ coefficients[arm] + report["arms"][arm]["intercept"]
        close(predicted, column(evaluation, arm + "_surrogate"), "Surrogate predictions differ from exported coefficients")
    novel = ~(overlaps["random"] | overlaps["adaptive"])
    require(bool(np.array_equal(novel, column(evaluation, "shared_novel", int))), "Incorrect common novel-mask flags")
    require(int(novel.sum()) == report["shared_novel_evaluation_rows"], "Incorrect novel row count")
    require(len(set(eval_keys)) == report["unique_evaluation_masks"], "Incorrect unique evaluation count")
    for key, selected in (("evaluation_subset_sizes", np.ones(len(novel), dtype=bool)), ("shared_novel_subset_sizes", novel)):
        require({str(k): v for k, v in Counter(map(int, eval_matrix[selected].sum(axis=1))).items()} == report[key], "Incorrect subset-size histogram")
    target, labels = column(evaluation, "p_class1"), column(evaluation, "yhat", int)
    fidelity_rows = []
    for arm in ARMS:
        predicted = column(evaluation, arm + "_surrogate")
        all_metrics = recompute_metrics(target, predicted, labels)
        novel_metrics = recompute_metrics(target[novel], predicted[novel], labels[novel])
        baseline = recompute_metrics(target[novel], np.full(novel.sum(), targets[arm].mean()), labels[novel])
        for key, value in (("all_evaluation_draws", all_metrics), ("shared_novel_evaluation", novel_metrics),
                           ("constant_baseline_shared_novel", baseline)):
            compare_metrics(value, report["arms"][arm][key])
        train_labels = column(training[arm], "yhat", int)
        require({str(c): int(np.sum(train_labels == c)) for c in (0, 1)} == report["arms"][arm]["training_class_counts"], "Incorrect training class counts")
        fidelity_rows.append({"seed": seed, "arm": arm, "novel_rows": int(novel.sum()),
            "novel_negative_rows": int(np.sum(labels[novel] == 0)), "novel_positive_rows": int(np.sum(labels[novel] == 1)),
            "novel_mae": novel_metrics.get("mae"), "novel_rmse": novel_metrics.get("rmse"),
            "constant_baseline_novel_mae": baseline.get("mae"),
            "mae_gain_over_constant": baseline["mae"] - novel_metrics["mae"] if novel.sum() else None,
            "all_draws_mae": all_metrics["mae"], "novel_class_agreement": novel_metrics.get("class_agreement"),
            "novel_out_of_range_scores": novel_metrics.get("outside_probability_range"),
            "overlapping_evaluation_rows": int(overlaps[arm].sum()), "unique_training_masks": len({tuple(row) for row in matrix(training[arm], images)}),
            "training_negative_rows": int(np.sum(train_labels == 0)), "training_positive_rows": int(np.sum(train_labels == 1))})

    deletion = read_rows(directory / "deletion.csv")
    require(len(deletion) == 6 * len(counts), "Unexpected deletion table size")
    deletion_rows = []
    original = ledger_by_stage["original"][0]
    random_order = np.random.RandomState(plan["streams"][str(seed)]["random_deletion"]).permutation(len(images)).tolist()
    for arm in ARMS:
        for control in CONTROLS:
            points = [row for row in deletion if row["arm"] == arm and row["control"] == control]
            require(column(points, "deleted_count", int).tolist() == counts, "Incorrect deletion steps")
            order = random_order if control == "random" else sorted(range(len(images)), key=lambda i: (
                -coefficients[arm][i] if control == "descending" else coefficients[arm][i], images[i]))
            expected_masks = [[int(i not in set(order[:count])) for i in range(len(images))] for count in counts]
            require(bool(np.array_equal(matrix(points, images), expected_masks)), "Deletion masks differ from coefficient ranking")
            close(column(points, "deleted_fraction"), np.array(counts) / len(images), "Incorrect deletion fraction")
            close(column(points, "retained_count"), len(images) - np.array(counts), "Incorrect retained count")
            close(column(points, "delta_from_original"), float(original["p_class1"]) - column(points, "p_class1"), "Incorrect probability change")
            close(float(points[0]["p_class1"]), float(original["p_class1"]), "Wrong zero-deletion baseline")
            stage = "shared_random_deletion" if control == "random" else f"{arm}_{control}_deletion"
            match_queries(points[1:], stage)
            x, y = column(points, "deleted_fraction"), column(points, "p_class1")
            area = float(sum((x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2 for i in range(len(x) - 1)))
            recorded = report["deletion"][arm][control]
            close(area, recorded["area_under_curve"], "Deletion AUC differs from table")
            close(area / (x[-1] - x[0]), recorded["normalized_area"], "Normalized deletion AUC differs")
            close(y[-1], recorded["final_probability"], "Final deletion probability differs")
            deletion_rows.append({"seed": seed, "arm": arm, "control": control,
                "area_under_curve": area, "normalized_area": area / (x[-1] - x[0]),
                "final_probability": y[-1], "final_delta_from_original": float(original["p_class1"]) - y[-1]})
    return {"report": report, "evaluation": evaluation, "deletion": deletion, "coefficients": coefficients,
            "fidelity": fidelity_rows, "deletion_summary": deletion_rows}


def figures(plan, runs, output):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
    colors = {"random": "#0072B2", "adaptive": "#D55E00"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.dpi": 180})
    seeds = plan["seeds"]
    captions = []

    def save(fig, name, caption):
        fig.savefig(output / (name + ".png"), bbox_inches="tight")
        fig.savefig(output / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)
        captions.append({"figure": name, "caption": caption})

    fig, axes = plt.subplots(1, len(seeds), figsize=(4 * len(seeds), 4), sharey=True, squeeze=False)
    max_mae = max((row["constant_baseline_novel_mae"] or 0) for run in runs for row in run["fidelity"])
    max_mae = max(max_mae, max((row["novel_mae"] or 0) for run in runs for row in run["fidelity"]))
    for ax, seed, run in zip(axes[0], seeds, runs):
        for i, row in enumerate(run["fidelity"]):
            if row["novel_mae"] is not None:
                ax.bar(i, row["novel_mae"], color=colors[row["arm"]], width=.6)
                ax.scatter(i, row["constant_baseline_novel_mae"], marker="D", color="#333333", zorder=3)
        ax.set_xticks([0, 1], ARMS)
        ax.set_title(f"Seed {seed} · {run['report']['shared_novel_evaluation_rows']} novel draws")
        ax.set_ylim(0, max(.01, max_mae * 1.15))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Probability MAE on shared novel masks ↓")
    fig.suptitle(f"One complete patient · {plan['train_samples']:,} training draws per arm")
    fig.legend(handles=[Patch(color=colors[a], label=a.capitalize() + " Ridge") for a in ARMS] + [
        Line2D([], [], color="#333333", marker="D", linestyle="none", label="Arm's training-mean baseline")],
        loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .09, 1, .92))
    save(fig, "fidelity", "One patient, repeated seeds. Both arms use exactly the same evaluation draws unseen by either training set; repeated masks retain their draw multiplicity. Diamonds use each arm's training mean. Lower MAE is better. No population confidence interval is inferred.")

    fig, axes = plt.subplots(2, len(seeds), figsize=(4 * len(seeds), 6), sharex=True, sharey=True, squeeze=False)
    control_colors = {"descending": "#0072B2", "ascending": "#D55E00", "random": "#666666"}
    for col, (seed, run) in enumerate(zip(seeds, runs)):
        for row, arm in enumerate(ARMS):
            ax = axes[row, col]
            for control in CONTROLS:
                points = [p for p in run["deletion"] if p["arm"] == arm and p["control"] == control]
                ax.plot(column(points, "deleted_fraction"), column(points, "p_class1"), marker="o", markersize=3,
                        color=control_colors[control], linestyle="--" if control == "random" else "-", label=control.capitalize())
            ax.set_ylim(0, 1.02)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.2)
            ax.set_title(f"{arm.capitalize()} · seed {seed}")
            if col == 0:
                ax.set_ylabel("GNN class-1 probability")
            if row == 1:
                ax.set_xlabel("Image nodes deleted")
    fig.suptitle("One complete patient · deletion by signed Ridge coefficient")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, .94))
    save(fig, "deletion", "Each step rebuilds the graph with retained images. Descending deletes the largest signed class-1 coefficients first; ascending reverses that ranking. The same seeded random control is shown for both arms. The original prediction is a shared baseline. Probabilities and nonmonotonic changes are preserved; lower descending AUC is the intended direction, not an assumed outcome.")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
    raw_scores = [float(row[a + "_surrogate"]) for run in runs for row in run["evaluation"] if row["shared_novel"] == "1" for a in ARMS]
    low, high = min([0] + raw_scores) - .03, max([1] + raw_scores) + .03
    seed_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for ax, arm in zip(axes, ARMS):
        for i, (seed, run) in enumerate(zip(seeds, runs)):
            points = [row for row in run["evaluation"] if row["shared_novel"] == "1"]
            ax.scatter(column(points, "p_class1"), column(points, arm + "_surrogate"), s=12, alpha=.55,
                       color=seed_colors[i % len(seed_colors)], label=f"Seed {seed} (n={len(points)})")
        ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1)
        ax.set(xlim=(low, high), ylim=(low, high), xlabel="GNN class-1 probability", title=arm.capitalize())
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Ridge score (unclipped)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Shared novel-mask fidelity · one patient, repeated seeds")
    fig.tight_layout(rect=(0, 0, 1, .93))
    save(fig, "predictions", "Each point is a shared novel evaluation draw. Identical masks can recur, so points are not independent patients. Dashed line denotes perfect probability fidelity. Both axes use the same scale across arms; Ridge scores outside [0,1] remain visible.")

    fig, axes = plt.subplots(2, 1, figsize=(11, 4.8))
    limit = max(float(np.max(np.abs(run["coefficients"][arm]))) for run in runs for arm in ARMS) or 1e-6
    for ax, arm in zip(axes, ARMS):
        values = np.array([run["coefficients"][arm] for run in runs])
        heatmap = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_yticks(range(len(seeds)), [f"Seed {seed}" for seed in seeds])
        ax.set_xticks(range(len(plan["images"])), range(1, len(plan["images"]) + 1), fontsize=8)
        ax.set_title(arm.capitalize())
    axes[1].set_xlabel("Image index (same sorted image columns across all fits)")
    fig.suptitle("One patient · exploratory coefficient variation across seeds")
    fig.subplots_adjust(left=.08, right=.87, bottom=.12, top=.87, hspace=.6)
    color_axis = fig.add_axes([.9, .2, .018, .6])
    fig.colorbar(heatmap, cax=color_axis, label="Signed class-1 Ridge coefficient")
    save(fig, "coefficients", "Exploratory descriptive view, added after the primary protocol was frozen. Columns align across all seeds/arms and map to private image IDs in image_columns.csv. A common symmetric zero-centered color scale preserves coefficient sign and magnitude. These coefficients describe the fitted surrogate and are not clinical image-level ground truth.")
    write_json(output / "figure_captions.json", captions)


def summarize(plan_path, output=None):
    plan_path = plan_path.resolve()
    plan = verified_plan(plan_path)
    plan_hash = sha256(plan_path)
    runs = [validate_run(plan, plan_hash, plan_path.parent / "runs" / f"seed-{seed}", seed) for seed in plan["seeds"]]
    output = (output or plan_path.parent / "summary").resolve()
    require(plan_path.parent in output.parents and (plan_path.parent / "runs") not in output.parents,
            "Summary must be a new directory under the study, outside runs/")
    output.mkdir(parents=True, exist_ok=False)
    fidelity = [row for run in runs for row in run["fidelity"]]
    deletion = [row for run in runs for row in run["deletion_summary"]]
    stability = coefficient_stability(plan, runs)
    write_csv(output / "fidelity.csv", fidelity)
    write_csv(output / "deletion_summary.csv", deletion)
    if stability:
        write_csv(output / "coefficient_stability.csv", stability)
    write_csv(output / "image_columns.csv", [{"image_index": i + 1, "image_id": image} for i, image in enumerate(plan["images"])])
    shutil.copy2(__file__, output / "summarize_patient_study.py")
    differences = [run["fidelity"][1]["novel_mae"] - run["fidelity"][0]["novel_mae"] for run in runs
                   if all(row["novel_mae"] is not None for row in run["fidelity"])]
    analysis = {"scope": plan["scope"], "plan_sha256": plan_hash,
        "validation": "All planned seeds passed; masks, ledger budgets, overlap, raw-score predictions, fidelity and deletion AUC independently recomputed from CSVs.",
        "paired_adaptive_minus_random_novel_mae": differences,
        "mean_paired_difference": float(np.mean(differences)) if differences else None,
        "total_model_queries": sum(run["report"]["total_model_queries"] for run in runs),
        "seeds": plan["seeds"], "train_samples_per_arm": plan["train_samples"],
        "evaluation_draws_per_seed": plan["evaluation_samples"],
        "summary_source_sha256": sha256(__file__),
        "exploratory_coefficient_stability": stability,
        "runs": [{"seed": seed, "slurm_job_id": run["report"]["slurm_job_id"],
            "gpu_name": run["report"]["gpu_name"], "elapsed_seconds": run["report"]["elapsed_seconds"],
            "adaptive_class_balance_reached": run["report"]["sampling_summary"]["class_balance_reached"],
            "report_sha256": sha256(plan_path.parent / "runs" / f"seed-{seed}" / "report.json")}
            for seed, run in zip(plan["seeds"], runs)]}
    write_json(output / "analysis.json", analysis)
    figures(plan, runs, output)
    lines = ["# Complete-patient methods study (n=1)", "",
        f"Completed seeds {', '.join(map(str, plan['seeds']))}; {plan['train_samples']:,} training draws per strategy and {plan['evaluation_samples']} shared evaluation draws per seed. "
        f"Total fresh GNN calls: {analysis['total_model_queries']:,}. Both strategies fit Ridge(alpha={plan['ridge_alpha']}) to class-1 probability. "
        "This is a prospective methods comparison, with no historical coefficient replay or patient-population inference.", "",
        "## Shared novel-mask fidelity", "",
        "Primary rows exclude any mask seen in either training arm; the same remaining rows score both surrogates. "
        "Multiplicity is retained. This conditional evaluation can exclude large subset sizes. Lower MAE is better; the constant baseline is each arm's training mean.", "",
        "| Seed | Strategy | Novel draws (negative/positive) | Ridge MAE | Constant MAE | Ridge RMSE | Training negative/positive | Unique training masks |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    def formatted(value):
        return "unavailable" if value is None else f"{value:.6f}"
    for row in fidelity:
        lines.append(f"| {row['seed']} | {row['arm']} | {row['novel_rows']} ({row['novel_negative_rows']}/{row['novel_positive_rows']}) | "
            f"{formatted(row['novel_mae'])} | {formatted(row['constant_baseline_novel_mae'])} | {formatted(row['novel_rmse'])} | "
            f"{row['training_negative_rows']}/{row['training_positive_rows']} | {row['unique_training_masks']} |")
    lines += ["", "Paired adaptive minus random MAE by seed: " + ", ".join(formatted(value) for value in differences) + ". "
              "Negative values favor adaptive on this diagnostic. These are repeated seeds for one patient, with no inferential confidence interval.", "",
              "![Novel-mask fidelity](fidelity.png)", "", "![Unclipped surrogate predictions](predictions.png)", "",
              "## Deletion", "",
              "The GNN re-encodes retained images and rebuilds the graph at each step. Signed descending and ascending coefficient orders have a shared seeded random control. "
              "AUC uses actual deletion fractions; normalized AUC divides by the observed span. Lower descending probability/AUC is the intended direction. Curves need not be monotonic.", "",
              "| Seed | Strategy | Order | AUC | Normalized AUC | Probability after 50% deletion | Change from original |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    if differences:
        position = lines.index("## Shared novel-mask fidelity")
        lines[position:position] = [
            f"On the shared novel-mask diagnostic, adaptive had lower MAE in {sum(value < 0 for value in differences)}/{len(differences)} seeds "
            f"and higher MAE in {sum(value > 0 for value in differences)}/{len(differences)} seeds. "
            f"Mean paired adaptive minus random MAE was {analysis['mean_paired_difference']:+.6f}. "
            "This describes the specified uniform-mask evaluation for one patient; it does not select a strategy for a population.", ""]
        position = lines.index("![Novel-mask fidelity](fidelity.png)")
        for arm in ARMS:
            available = [row for row in fidelity if row["arm"] == arm and row["mae_gain_over_constant"] is not None]
            lower = sum(row["mae_gain_over_constant"] > 0 for row in available)
            lines[position:position] = [f"{arm.capitalize()} Ridge had lower novel-mask MAE than its own training-mean constant baseline in {lower}/{len(available)} seeds. "
                "The baseline comparison is necessary when GNN predictions concentrate in one class.", ""]
            position += 2
    for row in deletion:
        lines.append(f"| {row['seed']} | {row['arm']} | {row['control']} | {row['area_under_curve']:.6f} | {row['normalized_area']:.6f} | {row['final_probability']:.6f} | {row['final_delta_from_original']:.6f} |")
    lines += [""]
    for arm in ARMS:
        lower = sum(run["report"]["deletion"][arm]["descending"]["area_under_curve"] <
                    run["report"]["deletion"][arm]["random"]["area_under_curve"] for run in runs)
        lines += [f"{arm.capitalize()} descending-coefficient deletion had lower AUC than the shared random control in {lower}/{len(runs)} seeds. "
            "This is a within-patient descriptive control comparison.", ""]
    lines += ["", "![Deletion controls](deletion.png)", "", "## Exploratory coefficient stability", "",
        "This descriptive diagnostic was added after freezing the primary fidelity/deletion protocol. "
        "It uses the existing fits, with no extra GNN calls or significance tests. Spearman correlation compares the image-coefficient rankings, "
        "using average ranks for ties; top-five overlap uses the same deterministic image-ID tie-breaking as deletion. Constant rankings have unavailable correlation.", "",
        "| Strategy | Seeds | Spearman correlation | Top-five overlap |", "| --- | --- | --- | --- |"]
    for row in stability:
        lines.append(f"| {row['arm']} | {row['seed_a']}/{row['seed_b']} | {formatted(row['spearman_rank_correlation'])} | {row['top_k_overlap_count']}/{row['top_k']} |")
    lines += ["", "![Coefficient variation](coefficients.png)", "", "## Execution provenance", "",
        "| Seed | Slurm task | GPU | Runner seconds | GNN calls |", "| --- | --- | --- | --- | --- |"]
    for seed, run in zip(plan["seeds"], runs):
        record = run["report"]
        lines.append(f"| {seed} | {record['slurm_job_id']} | {record['gpu_name']} | {record['elapsed_seconds']:.1f} | {record['total_model_queries']} |")
    lines += ["", "Each seed fits both strategies in the same allocation. GPU types can differ between seeds; "
        "these timings are execution records rather than a strategy speed comparison. Seeded masks do not promise bitwise-identical GPU aggregation across devices.", "",
        "## Validation and limits", "",
        analysis["validation"], "",
        "Adaptive achieves its 0.5 class target in seeds: " + ", ".join(str(row["seed"]) for row in analysis["runs"] if row["adaptive_class_balance_reached"]) +
        " (an empty list means none). Failure to achieve that target remains part of the result.", "",
        "The comparison fixes training-row count, with unequal sampler query overhead. Every requested duplicate consumes a new inference call. "
        "Shared evaluation, original zero-deletion baseline and random deletion controls are accounted for once. No class balancing, clipping or favorable-seed selection is applied after seeing outcomes. "
        "Repeated masks, seeds and image nodes do not increase the number of independent patients. Class agreement is uninformative when GNN evaluation labels occupy one class. "
        "Missing images still gate cohort expansion; these results do not support a clinical or cohort-level performance claim.", "",
        "Inputs and executed scientific sources are SHA-256 locked in `../plan.json` and `../source_snapshot/`. "
        "Per-seed raw evidence is in `../runs/seed-*/`; `queries.csv` contains every model call. This summary preserves all planned seeds. "
        "Fidelity/deletion tables, PNG/SVG figures and captions are exported together. All study artifacts remain in ignored private storage.", ""]
    (output / "report.md").write_text("\n".join(lines))
    artifact_files = [path for path in output.iterdir() if path.is_file()]
    write_json(output / "artifact_manifest.json", [{"file": path.name, "sha256": sha256(path)} for path in sorted(artifact_files)])
    print("Validated and summarized n=1 study:", output / "report.md")
    print("Paired adaptive minus random novel MAE:", differences)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summarize(args.plan, args.output)


if __name__ == "__main__":
    main()
