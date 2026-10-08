"""Read and independently validate proceedings evidence; never perform inference.

Private identifiers are used only to join preserved artifacts. Exports use ordinal
patient indices and aggregate numerical summaries. No source artifact is changed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cohort_study
import full_cohort_study
import parallel_experiments as extra
import patient_study as study
from summarize_patient_study import column, matrix, read_rows, validate_run

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/complete_patient/cohort-20261007-v1/cohort_plan.json"
PARALLEL = ROOT / "outputs/parallel_experiments/ten-patient-20261007-v1/plan.json"
REVIEW = ROOT / "outputs/reports/ten-patient-completion-20261007-v2/review.json"
FULL = ROOT / "outputs/complete_patient/full-135-20261007-v2/full_cohort_plan.json"
ARMS = ("random", "adaptive")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_entry(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": study.sha256(path)}


def verify_entries(entries):
    for entry in entries:
        require(study.sha256(entry["path"]) == entry["sha256"],
                "Evidence hash changed: " + entry["path"])


def full_gate(path=FULL):
    """No output is created unless the complete summary and frozen plan pass."""
    path = Path(path)
    summary = path.parent / "summary/analysis.json"
    require(summary.exists(), "Full-cohort validated summary is absent; no partial fallback is allowed")
    plan = full_cohort_study.verified(path)
    analysis = json.loads(summary.read_text())
    require(analysis.get("status") == "passed", "Full summary has not passed validation")
    require(analysis.get("validated_patient_seed_runs") == 405, "Full summary must validate 405 runs")
    require(analysis.get("planned_patients") == 135, "Full summary must contain 135 patients")
    require(analysis.get("plan_sha256") == study.sha256(path), "Full summary plan hash differs")
    require(analysis.get("total_model_queries") == plan["planned_total_model_queries"], "Full query budget differs")
    verify_entries(analysis["evidence"])
    return plan, analysis


def patient_average(frame, columns, keys=("patient_index", "arm")):
    """Only complete three-seed values contribute to a patient's metric."""
    result = []
    for key, group in frame.groupby(list(keys), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        require(len(group) == 3 and set(group.seed) == {0, 1, 2}, "Missing/duplicate seed in patient mean")
        for name in columns:
            values = pd.to_numeric(group[name], errors="raise")
            row[name] = float(values.mean()) if values.notna().all() else np.nan
        result.append(row)
    return pd.DataFrame(result)


def finite_correlation(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b).correlation)


def collect(plan, plan_path, population):
    fidelity, deletion, trajectories, design, stages, pool_rows, sizes, evidence = [], [], [], [], [], [], [], []
    total = 0
    for item in plan["patients"]:
        path = Path(item["plan"])
        child = study.verified_plan(path)
        full_cohort_study.check_settings(child)
        evidence.append(hash_entry(path))
        sizes.append(len(child["images"]))
        for seed in (0, 1, 2):
            directory = path.parent / "runs" / f"seed-{seed}"
            checked = validate_run(child, study.sha256(path), directory, seed)
            index = item["patient_index"]
            report = checked["report"]
            total += report["total_model_queries"]
            fidelity.extend(dict(patient_index=index, **row) for row in checked["fidelity"])
            deletion.extend(dict(patient_index=index, **row) for row in checked["deletion_summary"])
            # Export no masks or image IDs to plotting tables.
            trajectories.extend({"patient_index": index, "seed": seed, "arm": row["arm"],
                "control": row["control"], "deleted_fraction": float(row["deleted_fraction"]),
                "deleted_count": int(row["deleted_count"]), "p_class1": float(row["p_class1"])}
                for row in checked["deletion"])
            ledger = read_rows(directory / "queries.csv")
            for arm in ARMS:
                training = read_rows(directory / f"{arm}_training.csv")
                x = matrix(training, child["images"])
                summary, _ = extra.design_diagnostics(x, column(training, "yhat", int), child["images"])
                # Histograms remain nested in the evidence report; scalar diagnostics form the figure table.
                summary.pop("subset_size_counts")
                design.append(dict(patient_index=index, seed=seed, arm=arm, **summary))
                if arm == "adaptive":
                    audit, rows = extra.stage_two_audit(training, ledger, child["images"], report["sampling_summary"])
                    stages.append(dict(patient_index=index, seed=seed,
                        biased_rows=audit["stages"]["biased"], random_rows=audit["stages"]["random"],
                        deficit_rows=audit["biased_rows_with_pool_deficit"],
                        positive_pool=audit["positive_singleton_pool"], negative_pool=audit["negative_singleton_pool"],
                        one_pool_fallback=audit["one_pool_fallback"],
                        balance_reached=report["sampling_summary"]["class_balance_reached"]))
                    pool_rows.extend(dict(patient_index=index, seed=seed, **row) for row in rows)
            evidence.extend(hash_entry(p) for p in [*sorted(directory.glob("*.csv")), directory / "report.json"])
    patients, aggregate = cohort_study.aggregate(fidelity, range(len(sizes)), (0, 1, 2))
    expected = 67410 if population == "pilot" else plan["planned_total_model_queries"]
    require(total == expected, "Unexpected aggregate GNN-call budget")
    return {"population": population, "n": len(sizes), "image_counts": sizes, "queries": total,
        "fidelity": pd.DataFrame(fidelity), "patients": pd.DataFrame(patients),
        "deletion": pd.DataFrame(deletion), "trajectories": pd.DataFrame(trajectories),
        "design": pd.DataFrame(design), "stages": pd.DataFrame(stages), "pool_rows": pd.DataFrame(pool_rows),
        "aggregate": aggregate, "evidence": [hash_entry(plan_path), *evidence]}


def load_pilot():
    review = json.loads(REVIEW.read_text())
    require(review["status"] == "validated" and review["patients"] == 10, "Pilot review is not validated")
    verify_entries(review["evidence"])
    # The independently reviewed exporter refitted all 60 Ridge/60 Elastic Net models.
    require(review["independent_refits"] == 120, "Pilot independent-refit review is incomplete")
    require(study.sha256(ROOT / "scripts/report_ten_patient_completion.py") == review["report_source_sha256"],
            "Pilot review exporter source changed")
    cohort = cohort_study.verified_cohort(PILOT)
    parallel = extra.verified(PARALLEL)
    data = collect(cohort, PILOT, "pilot")
    data["evidence"].extend([hash_entry(REVIEW), hash_entry(PARALLEL)])
    # Eligible-cohort size is a verified input fact even before its inference completes.
    expansion = full_cohort_study.verified(FULL)
    image_counts = []
    for item in expansion['patients']:
        child_path = Path(item['plan'])
        child = json.loads(child_path.read_text())
        require(child['patient'] == item['patient'] and child['ground_truth'] == 1,
                'Expansion eligibility identity/label differs')
        image_counts.append(len(child['images']))
        data['evidence'].append(hash_entry(child_path))
    require(len(image_counts) == 135 and sum(image_counts) == 3072 and min(image_counts) == 20
            and max(image_counts) == 35, 'Eligible expansion image counts differ')
    data['evidence'].append(hash_entry(FULL))
    summary = PARALLEL.parent / "summary"
    for p in sorted(summary.glob("*.csv")):
        data["evidence"].append(hash_entry(p))
    data["enet"] = pd.read_csv(summary / "fidelity_by_seed.csv")
    data["stability"] = pd.read_csv(summary / "stability.csv")
    data["loo"] = pd.read_csv(summary / "leave_one_out.csv")
    require(len(data["enet"]) == 120 and len(data["stability"]) == 180 and len(data["loo"]) == 200,
            "Pilot secondary table coverage differs")
    agreements = []
    for index in range(10):
        loo = data["loo"][data["loo"].patient_index == index].set_index("omitted_image")
        for seed in (0, 1, 2):
            p = PARALLEL.parent / "cpu" / f"task-{3 * index + seed:02}/rankings.csv"
            rankings = pd.read_csv(p)
            data["evidence"].append(hash_entry(p))
            for (arm, method), group in rankings.groupby(["arm", "method"]):
                agreements.append({"patient_index": index, "seed": seed, "arm": arm, "method": method,
                    "spearman": finite_correlation(group.value, loo.loc[group.image_id, "delta_class1"])})
    data["loo_agreement"] = patient_average(pd.DataFrame(agreements), ["spearman"],
                                           ("patient_index", "arm", "method"))
    paired = data["aggregate"]["mean_patient_paired_mae_difference"]
    require(np.isclose(paired, review["mean_patient_paired_mae_difference"]), "Pilot reviewed MAE differs")
    require(int(data["stages"].deficit_rows.sum()) == review["pool_deficit_draws"], "Pilot reallocation audit differs")
    return data


def load_full(path=FULL):
    plan, analysis = full_gate(path)
    data = collect(plan, Path(path), "full")
    data["evidence"].append(hash_entry(Path(path).parent / "summary/analysis.json"))
    require(data["aggregate"] == {key: analysis[key] for key in data["aggregate"]}, "Raw full-cohort aggregate differs")
    return data


def strict_mean(values):
    values = pd.to_numeric(pd.Series(values), errors="raise")
    return float(values.mean()) if len(values) and values.notna().all() else None


def dispersion(values):
    values = np.asarray(pd.to_numeric(pd.Series(values), errors="raise").dropna(), dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "sd": None, "median": None, "q1": None, "q3": None}
    return {"n": len(values), "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "median": float(np.median(values)), "q1": float(np.quantile(values, .25, method="linear")),
        "q3": float(np.quantile(values, .75, method="linear"))}
