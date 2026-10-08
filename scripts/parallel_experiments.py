#!/usr/bin/env python3
"""Independent design diagnostics, training-only Elastic Net CV, and LOO inference."""
import argparse
from collections import Counter
from datetime import datetime
from itertools import combinations
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import shutil
import sys
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

import cohort_study
import patient_study as study
from summarize_patient_study import column, matrix, rank_similarity, read_rows, validate_run

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("random", "adaptive")
METHODS = ("ridge", "elastic_net", "marginal_correlation")


def prepare(args):
    import project_paths as paths
    output = args.output.resolve()
    allowed = (Path(paths.output_root()) / "parallel_experiments").resolve()
    if allowed not in output.parents:
        raise ValueError("Output must be below outputs/parallel_experiments/")
    cohort = cohort_study.verified_cohort(args.cohort)
    for item in cohort["patients"]:
        study.verified_plan(Path(item["plan"]))
    sources = [Path(__file__), ROOT / "slurm/parallel_cpu.sbatch", ROOT / "slurm/parallel_loo.sbatch",
               ROOT / "slurm/parallel_summary.sbatch"]
    plan = {"schema": 1, "prepared_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "scope": "Additional exploratory analyses on the frozen ten-patient P0 convenience cohort",
        "cohort": str(args.cohort.resolve()), "tasks": cohort["tasks"], "patients": cohort["patients"],
        "elastic_net": {"alphas": np.logspace(-4, 0, 25).tolist(), "l1_ratios": [.1, .5, .9, 1.],
            "folds": 5, "fold_seed": "patient-study training seed", "tol": 1e-5, "max_iter": 20000,
            "target": "class-1 probability", "features": "unscaled binary inclusion; intercept fitted",
            "selection": "Minimum training-only CV MSE; evaluation responses never used for selection"},
        "stability": {"top_k": 5, "sign_zero_tolerance": 1e-10, "complete_seeds": [0, 1, 2]},
        "loo": {"seed": 0, "calls_per_patient": "1 + image count", "planned_model_queries": sum(
            1 + len(json.loads(Path(item["plan"]).read_text())["images"]) for item in cohort["patients"]),
            "rankings": ["drop in class-1 probability", "drop in original predicted-class probability"],
            "comparison": "Full minus omitted-node probability; graph rebuilt for every query"},
        "files": [{"path": str(path.resolve()), "sha256": study.sha256(path)} for path in [args.cohort, *sources]],
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}}
    output.mkdir(parents=True, exist_ok=False)
    for source in sources:
        target = output / "source_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    study.write_json(output / "plan.json", plan)
    print("Prepared independent analyses:", output / "plan.json")


def verified(path):
    plan = json.loads(Path(path).read_text())
    for entry in plan["files"]:
        if study.sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("Parallel-analysis input/source changed: " + entry["path"])
    for name, version in plan["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Parallel-analysis package version changed: " + name)
    cohort_study.verified_cohort(Path(plan["cohort"]))
    return plan


def design_diagnostics(x, labels, images):
    x, labels = np.asarray(x), np.asarray(labels)
    if x.ndim != 2 or x.shape[1] != len(images) or not np.isin(x, [0, 1]).all():
        raise ValueError("Design must have binary image columns")
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.linalg.matrix_rank(centered))
    summary = {"rows": len(x), "negative_rows": int(np.sum(labels == 0)),
        "positive_rows": int(np.sum(labels == 1)), "positive_fraction": float(np.mean(labels)),
        "minority_fraction": float(min(np.mean(labels), 1 - np.mean(labels))),
        "unique_masks": len(np.unique(x, axis=0)), "duplicate_rows": len(x) - len(np.unique(x, axis=0)),
        "subset_size_counts": {str(k): v for k, v in Counter(map(int, x.sum(axis=1))).items()},
        "centered_design_rank": rank, "image_columns": len(images),
        "centered_design_singular": rank < len(images),
        "centered_design_condition_number": float(singular[0] / singular[-1]) if rank == len(images) else None}
    rows = [{"image_id": image, "included_rows": int(x[:, index].sum()),
             "excluded_rows": int(len(x) - x[:, index].sum()), "inclusion_fraction": float(x[:, index].mean())}
            for index, image in enumerate(images)]
    return summary, rows


def stage_two_audit(training, ledger, images, summary):
    singleton_rows = [row for row in ledger if row["stage"] == "adaptive_singletons"]
    positive = set()
    if len(singleton_rows) != len(images):
        raise ValueError("Missing singleton pool evidence")
    seen = set()
    for row in singleton_rows:
        selected = [image for image in images if int(row[image])]
        if len(selected) != 1 or selected[0] in seen:
            raise ValueError("Invalid singleton pool evidence")
        seen.add(selected[0])
        if int(row["yhat"]) == 1:
            positive.add(selected[0])
    one_pool = len(positive) in (0, len(images))
    counts, stages, audits = Counter({0: 0, 1: 0}), Counter({"random": 0, "biased": 0}), []
    target = {int(key): value for key, value in summary["requested_class_counts"].items()}
    for index, row in enumerate(training):
        biased = not one_pool and not (counts[0] < target[0] and counts[1] < target[1])
        stages["biased" if biased else "random"] += 1
        if biased:
            size = sum(int(row[image]) for image in images)
            proportion = .85 if counts[1] < target[1] else .15
            requested = int(size * proportion)
            realized = min(max(requested, size - (len(images) - len(positive))), len(positive))
            actual = sum(int(row[image]) for image in positive)
            if actual != realized:
                raise ValueError("Stage-II pool composition differs from the documented clamping rule")
            audits.append({"training_row": index, "subset_size": size, "requested_positive_proportion": proportion,
                "requested_positive_count": requested, "realized_positive_count": actual,
                "realized_positive_fraction": actual / size, "pool_deficit_reallocated": requested != realized})
        counts[int(row["yhat"])] += 1
    if dict(stages) != summary["stages"] or one_pool != summary["one_pool_fallback"]:
        raise ValueError("Reconstructed sampling stages disagree with the report")
    return {"positive_singleton_pool": len(positive), "negative_singleton_pool": len(images) - len(positive),
            "one_pool_fallback": one_pool, "stages": dict(stages),
            "biased_rows_with_pool_deficit": sum(row["pool_deficit_reallocated"] for row in audits)}, audits


def marginal_correlations(x, y):
    xc, yc = x - x.mean(axis=0), y - y.mean()
    denominator = np.sqrt((xc * xc).sum(axis=0) * np.dot(yc, yc))
    return [float(np.dot(xc[:, i], yc) / d) if d > 0 else None for i, d in enumerate(denominator)]


def elastic_fit(x, y, seed, settings):
    model = ElasticNetCV(alphas=settings["alphas"], l1_ratio=settings["l1_ratios"],
        cv=KFold(n_splits=settings["folds"], shuffle=True, random_state=seed),
        tol=settings["tol"], max_iter=settings["max_iter"], n_jobs=1, selection="cyclic")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return model


def cpu_analysis(plan, task, child_plan, directory, output):
    checked = validate_run(child_plan, study.sha256(Path(task["plan"])), directory, task["seed"])
    images = child_plan["images"]
    evaluation = checked["evaluation"]
    ex, target, labels = matrix(evaluation, images), column(evaluation, "p_class1"), column(evaluation, "yhat", int)
    novel = column(evaluation, "shared_novel", int).astype(bool)
    design, coefficients, metrics, inclusion = {}, [], [], []
    for arm in ARMS:
        rows = read_rows(directory / f"{arm}_training.csv")
        x, y = matrix(rows, images), column(rows, "p_class1")
        design[arm], image_rows = design_diagnostics(x, column(rows, "yhat", int), images)
        inclusion.extend(dict(arm=arm, **row) for row in image_rows)
        model = elastic_fit(x, y, task["seed"], plan["elastic_net"])
        scores = model.predict(ex)
        if not np.isfinite(scores).all() or not np.isfinite(model.coef_).all():
            raise ValueError("Nonfinite Elastic Net fit")
        coefficients.extend({"arm": arm, "method": "elastic_net", "image_id": image, "value": float(value)}
                            for image, value in zip(images, model.coef_))
        coefficients.extend({"arm": arm, "method": "ridge", "image_id": image, "value": float(value)}
                            for image, value in zip(images, checked["coefficients"][arm]))
        coefficients.extend({"arm": arm, "method": "marginal_correlation", "image_id": image, "value": value}
                            for image, value in zip(images, marginal_correlations(x, y)))
        for method, predicted in (("ridge", column(evaluation, arm + "_surrogate")), ("elastic_net", scores)):
            novel_metrics = study.metrics(target, predicted, labels, novel)
            all_metrics = study.metrics(target, predicted, labels)
            metrics.append({"arm": arm, "method": method, "novel_rows": int(novel.sum()),
                "novel_mae": novel_metrics.get("mae"), "novel_rmse": novel_metrics.get("rmse"),
                "novel_negative_rows": int(np.sum(labels[novel] == 0)), "novel_positive_rows": int(np.sum(labels[novel] == 1)),
                "novel_outside_probability_range": novel_metrics.get("outside_probability_range"),
                "all_draws_mae": all_metrics["mae"], "constant_baseline_novel_mae": study.metrics(
                    target, np.full(len(target), y.mean()), labels, novel).get("mae"),
                "alpha": float(model.alpha_) if method == "elastic_net" else child_plan["ridge_alpha"],
                "l1_ratio": float(model.l1_ratio_) if method == "elastic_net" else None,
                "nonzero_coefficients": int(np.sum(np.abs(model.coef_) > 1e-10)) if method == "elastic_net" else None})
        fitted_evidence = [{"evaluation_row": i, "shared_novel": bool(novel[i]), "gnn_p_class1": float(target[i]),
            "ridge_score": float(evaluation[i][arm + "_surrogate"]), "elastic_net_score": float(scores[i])} for i in range(len(target))]
        study.write_csv(output / f"{arm}_evaluation.csv", fitted_evidence)
        study.write_json(output / f"{arm}_elastic_net.json", {"alpha": float(model.alpha_),
            "l1_ratio": float(model.l1_ratio_), "intercept": float(model.intercept_),
            "cv_mean_mse": np.mean(model.mse_path_, axis=-1).tolist(), "iterations": int(model.n_iter_),
            "alpha_grid": model.alphas_.tolist(), "training_rows": len(x), "evaluation_rows_used_for_fit": 0})
    stage_summary, stage_rows = stage_two_audit(read_rows(directory / "adaptive_training.csv"),
        read_rows(directory / "queries.csv"), images, checked["report"]["sampling_summary"])
    if stage_rows:
        study.write_csv(output / "stage_two_pool_composition.csv", stage_rows)
    study.write_csv(output / "image_inclusion.csv", inclusion)
    study.write_csv(output / "rankings.csv", coefficients)
    study.write_csv(output / "fidelity.csv", metrics)
    return {"design": design, "stage_two": stage_summary, "model_queries": 0,
            "shared_novel_rows": int(novel.sum()), "class_balance_reached": checked["report"]["sampling_summary"]["class_balance_reached"]}


def loo_analysis(child, output, predict):
    images = child["images"]
    original = predict(images)
    cls = original["yhat"]
    original_class_p = original["p_class1"] if cls else 1 - original["p_class1"]
    rows = []
    for image in images:
        response = predict([other for other in images if other != image])
        predicted_class_p = response["p_class1"] if cls else 1 - response["p_class1"]
        rows.append({"omitted_image": image, "retained_count": len(images) - 1, **response,
            "delta_class1": original["p_class1"] - response["p_class1"],
            "original_predicted_class": cls, "p_original_class": predicted_class_p,
            "delta_original_class": original_class_p - predicted_class_p})
    study.write_csv(output / "leave_one_out.csv", rows)
    return {"original_prediction": original, "model_queries": 1 + len(images)}


def run(args):
    plan = verified(args.plan)
    root = args.plan.resolve().parent
    items = plan["tasks"] if args.command == "cpu" else plan["patients"]
    if not 0 <= args.task < len(items):
        raise ValueError("Task index outside plan")
    item = items[args.task]
    child = study.verified_plan(Path(item["plan"]))
    output = root / ("cpu" if args.command == "cpu" else "loo") / f"task-{args.task:02}"
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "plan_sha256": study.sha256(args.plan), "task": args.task,
              "patient_index": item["patient_index"], "command": args.command,
              "slurm_job_id": os.getenv("SLURM_JOB_ID"), "host": socket.gethostname(),
              "seed": item.get("seed", plan["loo"]["seed"])}
    started = time.monotonic()
    study.write_json(output / "report.json", report)
    try:
        if args.command == "cpu":
            directory = Path(item["plan"]).parent / "runs" / f"seed-{item['seed']}"
            inputs = sorted(directory.glob("*.csv")) + [directory / "report.json"]
            report["input_hashes"] = [{"path": str(p), "sha256": study.sha256(p)} for p in inputs]
            report.update(cpu_analysis(plan, item, child, directory, output))
        else:
            predictor = study.PatientPredictor(child, args.device, args.threads, plan["loo"]["seed"])
            report.update(loo_analysis(child, output, predictor))
            report["device"] = args.device
            report["gpu_name"] = predictor.torch.cuda.get_device_name(predictor.device) if predictor.device.type == "cuda" else None
        report["status"] = "passed"
        report["output_hashes"] = [{"path": str(p), "sha256": study.sha256(p)} for p in sorted(output.iterdir()) if p.name != "report.json"]
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        study.write_json(output / "report.json", report)
    print("Independent analysis passed:", output)


def stability_tables(rankings, patient, seeds, tolerance):
    pairs, frequency = [], []
    for arm in ARMS:
        for method in METHODS:
            by_seed = {}
            for seed in seeds:
                rows = [r for r in rankings if r["seed"] == seed and r["arm"] == arm and r["method"] == method]
                by_seed[seed] = {r["image_id"]: r["value"] for r in rows}
            images = sorted(by_seed[seeds[0]])
            if not images or any(set(by_seed[s]) != set(images) for s in seeds):
                raise ValueError("Stability image columns do not align")
            # Constant-response marginal correlations are undefined, not zero.
            defined = all(all(by_seed[s][image] is not None for image in images) for s in seeds)
            defined = defined and all(np.ptp([by_seed[s][image] for image in images]) > tolerance for s in seeds)
            top = {s: set(sorted(images, key=lambda image: (-by_seed[s][image], image))[:5]) for s in seeds} if defined else {}
            for a, b in combinations(seeds, 2):
                va, vb = [by_seed[a][i] for i in images], [by_seed[b][i] for i in images]
                sign = lambda value: 0 if abs(value) <= tolerance else (1 if value > 0 else -1)
                intersection = len(top[a] & top[b]) if defined else None
                pairs.append({"patient_index": patient, "arm": arm, "method": method, "seed_a": a, "seed_b": b,
                    "defined": defined, "spearman": rank_similarity(va, vb) if defined else None,
                    "top_five_jaccard": intersection / len(top[a] | top[b]) if defined else None,
                    "sign_agreement": float(np.mean([sign(x) == sign(y) for x, y in zip(va, vb)])) if defined else None})
            frequency.extend({"patient_index": patient, "arm": arm, "method": method, "image_id": image,
                "top_five_seeds": sum(image in top[s] for s in seeds) if defined else None,
                "planned_seeds": len(seeds), "defined": defined} for image in images)
    return pairs, frequency


def summarize(args):
    plan = verified(args.plan)
    root = args.plan.resolve().parent
    output = root / "summary"
    reports, fidelity, rankings, inclusion, audits, loo_rows = [], [], [], [], [], []
    plan_hash = study.sha256(args.plan)
    for command, count in (("cpu", 30), ("loo", 10)):
        for index in range(count):
            directory = root / command / f"task-{index:02}"
            report = json.loads((directory / "report.json").read_text())
            if report["status"] != "passed" or report["plan_sha256"] != plan_hash or report["task"] != index or report["command"] != command:
                raise ValueError("Missing, failed or misidentified parallel analysis")
            for entry in report.get("input_hashes", []) + report["output_hashes"]:
                if study.sha256(entry["path"]) != entry["sha256"]:
                    raise ValueError("Analysis evidence changed")
            reports.append(report)
            if command == "cpu":
                identity = {"patient_index": report["patient_index"], "seed": report["seed"]}
                fidelity.extend(dict(**identity, **row) for row in read_rows(directory / "fidelity.csv"))
                inclusion.extend(dict(**identity, **row) for row in read_rows(directory / "image_inclusion.csv"))
                for row in read_rows(directory / "rankings.csv"):
                    rankings.append(dict(**identity, **row, parsed_value=float(row["value"]) if row["value"] else None))
                audits.extend({**identity, "arm": arm, **{key: value for key, value in values.items() if key != "subset_size_counts"},
                    "subset_size_counts": json.dumps(values["subset_size_counts"], sort_keys=True)} for arm, values in report["design"].items())
            else:
                rows = read_rows(directory / "leave_one_out.csv")
                child = study.verified_plan(Path(plan["patients"][index]["plan"]))
                if len(rows) != len(child["images"]) or {row["omitted_image"] for row in rows} != set(child["images"]):
                    raise ValueError("Incomplete LOO evidence")
                if report["model_queries"] != 1 + len(rows):
                    raise ValueError("LOO query budget mismatch")
                for row in rows:
                    p = float(row["p_class1"])
                    original = report["original_prediction"]
                    expected_delta = original["p_class1"] - p
                    original_class_delta = expected_delta if original["yhat"] else -expected_delta
                    if not np.allclose([float(row["delta_class1"]), float(row["delta_original_class"])], [expected_delta, original_class_delta]):
                        raise ValueError("LOO probability changes disagree")
                loo_rows.extend(dict(patient_index=index, **row) for row in rows)
    stability, frequency = [], []
    for patient in range(10):
        rows = [dict(row, value=row["parsed_value"]) for row in rankings if row["patient_index"] == patient]
        pairs, counts = stability_tables(rows, patient, [0, 1, 2], plan["stability"]["sign_zero_tolerance"])
        stability.extend(pairs); frequency.extend(counts)
    patient_fidelity = []
    for patient in range(10):
        for arm in ARMS:
            per_method = {}
            for method in ("ridge", "elastic_net"):
                values = [float(row["novel_mae"]) if row["novel_mae"] else None for row in fidelity
                    if row["patient_index"] == patient and row["arm"] == arm and row["method"] == method]
                per_method[method] = float(np.mean(values)) if len(values) == 3 and all(value is not None for value in values) else None
            patient_fidelity.append({"patient_index": patient, "arm": arm, "ridge_mean_novel_mae": per_method["ridge"],
                "elastic_net_mean_novel_mae": per_method["elastic_net"], "elastic_net_minus_ridge_mae":
                per_method["elastic_net"] - per_method["ridge"] if all(v is not None for v in per_method.values()) else None})
    loo_queries = sum(r["model_queries"] for r in reports if r["command"] == "loo")
    if loo_queries != plan["loo"]["planned_model_queries"]:
        raise ValueError("Total LOO query budget differs from plan")
    paired_means = {}
    for arm in ARMS:
        values = [row["elastic_net_minus_ridge_mae"] for row in patient_fidelity if row["arm"] == arm]
        paired_means[arm] = float(np.mean(values)) if len(values) == 10 and all(v is not None for v in values) else None
    output.mkdir(exist_ok=False)
    for name, rows in (("design_diagnostics", audits), ("image_inclusion", inclusion), ("fidelity_by_seed", fidelity),
        ("fidelity_by_patient", patient_fidelity), ("stability", stability), ("top_five_frequency", frequency), ("leave_one_out", loo_rows)):
        study.write_csv(output / (name + ".csv"), rows)
    study.write_json(output / "analysis.json", {"status": "passed", "scope": plan["scope"],
        "validated_cpu_tasks": 30, "validated_loo_patients": 10, "loo_model_queries": loo_queries,
        "mean_patient_elastic_net_minus_ridge_mae": paired_means,
        "plan_sha256": plan_hash, "limitations": ["Exploratory ten-patient convenience cohort; three seeds, not five.",
            "Elastic Net CV uses training responses only; the common novel evaluation set is reused across models.",
            "Marginal correlations are rankings, not a fitted multivariable probability surrogate.",
            "Constant or undefined ranking vectors have unavailable stability metrics; no artificial perfect top-five agreement.",
            "LOO measures single-node removal from the full graph; ranked deletion for new methods is separate.",
            "No equal-query-budget or sampling-ratio ablation is performed here."]})
    (output / "report.md").write_text("# Parallel experiment results\n\nAll 30 CPU analyses and ten leave-one-out baselines passed; 210 additional GNN calls.\n\nInspect design_diagnostics.csv, fidelity_by_patient.csv, stability.csv, top_five_frequency.csv and leave_one_out.csv. Elastic Net selects hyperparameters using training-only five-fold CV. Marginal correlations provide rankings only. These are exploratory results on ten convenience-selected patients and three seeds.\n")
    print("Parallel experiment summary:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--cohort", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    for command in ("cpu", "loo"):
        sub = commands.add_parser(command)
        sub.add_argument("--plan", required=True, type=Path)
        sub.add_argument("--task", required=True, type=int)
        sub.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
        sub.add_argument("--threads", type=int, default=2)
    sub = commands.add_parser("summarize")
    sub.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
