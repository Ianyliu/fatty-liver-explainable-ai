#!/usr/bin/env python3
"""Prospective n=1 sampling/fidelity/deletion study, separate from saved-run replay."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke_patient import patient_record, sha256


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export an empty evidence table")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mask_key(mask):
    return tuple(int(value) for value in mask)


def random_masks(count, n_images, minimum, seed):
    rng = np.random.RandomState(seed)
    masks = np.zeros((count, n_images), dtype=np.int64)
    for row in masks:
        size = rng.randint(minimum, n_images + 1)
        row[rng.choice(n_images, size, replace=False)] = 1
    return masks


def metrics(target, predicted, labels, selected=None):
    target, predicted, labels = map(np.asarray, (target, predicted, labels))
    if target.shape != predicted.shape or target.shape != labels.shape:
        raise ValueError("Metric inputs must have identical shapes")
    if not np.isfinite(target).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric inputs must be finite")
    if selected is not None:
        target, predicted, labels = target[selected], predicted[selected], labels[selected]
    if not len(target):
        return {"status": "unavailable_no_rows", "rows": 0}
    return {
        "status": "available", "rows": len(target),
        "mae": float(np.mean(np.abs(predicted - target))),
        "rmse": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "class_agreement": float(np.mean((predicted >= 0.5) == labels)),
        "class_counts": {str(cls): int(np.sum(labels == cls)) for cls in (0, 1)},
        "one_class": bool(len(np.unique(labels)) < 2),
        "outside_probability_range": int(np.sum((predicted < 0) | (predicted > 1))),
    }


def deletion_order(coefficients, images, descending):
    if len(coefficients) != len(images) or not np.isfinite(coefficients).all():
        raise ValueError("Invalid deletion coefficients")
    return sorted(range(len(images)), key=lambda i: (
        -float(coefficients[i]) if descending else float(coefficients[i]), images[i]))


def deletion_counts(n_images, fractions, minimum):
    # Floor deletion counts; deduplicate rounded counts and retain >=minimum nodes.
    return sorted({min(n_images - minimum, int(np.floor(n_images * f + 1e-12))) for f in fractions})


def stream_seeds(seed):
    def derived(tag):
        return int(np.random.SeedSequence([seed, 20261007, tag]).generate_state(1)[0])
    return {"training": seed, "evaluation": derived(1), "random_deletion": derived(2)}


def prepare(args):
    import project_paths as paths
    output = args.output.resolve()
    allowed = (Path(paths.output_root()) / "complete_patient").resolve()
    if allowed not in output.parents:
        raise ValueError("Study directory must be below the configured outputs/complete_patient/")
    if not 10 <= args.train_samples <= 1000 or not 10 <= args.eval_samples <= 1000:
        raise ValueError("Use 10–1000 training/evaluation masks for the n=1 workflow")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        raise ValueError("Seeds must be distinct integers in [0, 2**32)")
    record, images = patient_record(args.patient, paths)
    if float(record["liver_fatty"]) <= 0 or len(images) < 20:
        raise ValueError("This study requires the complete positive / >=20-image patient")
    encoder_weights = Path(os.environ["TORCH_HOME"]) / "hub/checkpoints/densenet121-a639ec97.pth"
    if not encoder_weights.is_file() or not sha256(encoder_weights).startswith("a639ec97"):
        raise ValueError("Verified local DenseNet121 weights are required")
    source_names = [
        "scripts/patient_study.py", "scripts/smoke_patient.py", "project_paths.py",
        "sampling_marginal_relation_pipeline.py", "usflc_xai/__init__.py", "usflc_xai/models.py",
        "usflc_xai/datasets.py", "pyproject.toml", "uv.lock", "docs/P0_EXPERIMENT_PROTOCOL.md",
        "slurm/patient_study.sbatch",
    ]
    files = [ROOT / name for name in source_names]
    files += [Path(paths.metadata_path()), Path(paths.split_path()), Path(paths.checkpoint_path()), encoder_weights]
    files += [Path(paths.image_dir()) / f"{record['MI_ID']}_{image}.jpg" for image in images]
    manifest = [{"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size} for p in files]
    plan = {
        "schema": 1, "scope": "prospective complete-patient methods study; n=1",
        "prepared_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "patient": record["MI_ID"], "images": images, "ground_truth": 1,
        "paths": {"images": str(Path(paths.image_dir()).resolve()),
                  "checkpoint": str(Path(paths.checkpoint_path()).resolve()),
                  "torch_home": str(Path(os.environ["TORCH_HOME"]).resolve())},
        "train_samples": args.train_samples, "evaluation_samples": args.eval_samples,
        "seeds": args.seeds, "streams": {str(seed): stream_seeds(seed) for seed in args.seeds},
        "minimum_images": 3, "ridge_alpha": 1.0, "target": "GNN class-1 softmax probability",
        "adaptive_target_class1": 0.5, "deletion_fractions": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "primary_fidelity": "MAE on shared evaluation draws unseen by either training arm",
        "evaluation_policy": "200 by default; independent RNG draws, overlap flagged, shared novel subset primary",
        "constant_baseline": "each arm's training-mean probability; never fitted to evaluation outcomes",
        "budget": "equal training rows; report singleton/validation overhead; no equal-compute claim",
        "duplicate_policy": "retain training/evaluation multiplicity; every prediction consumes a query",
        "deletion_policy": "signed descending/ascending coefficients and one shared seeded random order; floor counts",
        "prediction_policy": "uploaded loader on every subset, no feature/prediction cache; original deletion baseline reused",
        "files": manifest,
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "torch-geometric", "numpy", "scipy", "scikit-learn", "pillow")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name in source_names:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    write_json(output / "plan.json", plan)
    print("Prepared immutable-input n=1 plan:", output / "plan.json", flush=True)


def verified_plan(path):
    plan = json.loads(Path(path).read_text())
    for entry in plan["files"]:
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("An input/source changed after planning: " + entry["path"])
    for name, expected in plan["versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Runtime version changed after planning: " + name)
    return plan


class PatientPredictor:
    """Real graph inference through the uploaded loader for every requested mask."""
    def __init__(self, plan, device, threads, seed):
        import torch
        import torchvision.transforms as transforms
        from usflc_xai import datasets, models
        self.torch, self.datasets, self.plan = torch, datasets, plan
        # Use the exact manifested cache and input paths, regardless of later .env changes.
        torch.hub.set_dir(str(Path(plan["paths"]["torch_home"]) / "hub"))
        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available inside this allocation")
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), transforms.Resize([224, 224]),
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        _, self.image_encoder = models.image_encoder_model("densenet121", True, 2, self.device)
        self.graph_encoder, _ = models.encoder_model("SETNET_GAT", 1024, 1, 2, self.device)
        payload = torch.load(plan["paths"]["checkpoint"], map_location=self.device)
        self.graph_encoder.load_state_dict(payload["model_state_dict"], strict=True)
        self.image_encoder.eval()
        self.graph_encoder.eval()

    def __call__(self, selected):
        torch = self.torch
        with torch.inference_mode():
            graph = self.datasets.single_data_loader(
                self.plan["patient"], selected, self.transform, self.image_encoder, 1, 2,
                self.device, crop_image_dir=self.plan["paths"]["images"] + os.sep)
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=self.device)
            logits = self.graph_encoder(graph.x.to(self.device), graph.edge_index_corr.to(self.device), batch, 1)
            if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid graph-model output")
            return {"p_class1": float(logits.softmax(dim=1)[0, 1].item()),
                    "yhat": int(logits.argmax(dim=1).item()), "logit0": float(logits[0, 0].item()),
                    "logit1": float(logits[0, 1].item()), "edges": int(graph.edge_index_corr.shape[1])}


def experiment(plan, seed, output, predict):
    from sampling_marginal_relation_pipeline import LIME_subj_pipeline
    images = plan["images"]
    n_images, count = len(images), plan["train_samples"]
    streams = plan["streams"][str(seed)]
    query_counts, stage_seconds = Counter(), Counter()
    query_log = []

    def query(selected, stage):
        started = time.monotonic()
        result = predict(selected)
        stage_seconds[stage] += time.monotonic() - started
        query_counts[stage] += 1
        if not np.isfinite(result["p_class1"]) or not 0 <= result["p_class1"] <= 1:
            raise RuntimeError("Invalid GNN probability")
        query_log.append({"query": len(query_log) + 1, "stage": stage,
            **{image: int(image in selected) for image in images}, **result})
        if sum(query_counts.values()) % 250 == 0:
            print("Completed model queries:", sum(query_counts.values()), flush=True)
        return result

    evaluation_design = random_masks(plan["evaluation_samples"], n_images, plan["minimum_images"], streams["evaluation"])
    original = query(images, "original")
    design = {"random": random_masks(count, n_images, plan["minimum_images"], streams["training"])}
    records = {"random": []}
    for mask in design["random"]:
        records["random"].append(query([image for image, flag in zip(images, mask) if flag], "random_training"))

    adaptive_calls = []

    def classify(selected):
        stage = "adaptive_validation" if not adaptive_calls else (
            "adaptive_singletons" if len(selected) == 1 else "adaptive_training")
        result = query(list(selected), stage)
        adaptive_calls.append((tuple(selected), result))
        return result["yhat"]

    sampler_root = output / "sampler"
    sampler_root.mkdir()
    # The class creates a patient directory and verifies real image existence.
    sampler = LIME_subj_pipeline(
        test_data_id="09", img_list=images, y=1, mi_id=plan["patient"], img_dir=plan["paths"]["images"],
        pred_func=classify, result_parent_dir=str(sampler_root), random_state=streams["training"])
    sampler.predict_on_random_samples_until_convergence(
        n_samples=count, target_positive_proportion=plan["adaptive_target_class1"],
        min_sample_size=plan["minimum_images"], max_sample_size=n_images,
        max_iter=count, append_to_original=False)
    sampler.generate_pred_results_matrix()
    design["adaptive"] = sampler.pred_results_x
    if len(adaptive_calls) != count + n_images + 1 or sampler.img_list != images:
        raise RuntimeError("Unexpected adaptive call accounting or column ordering")
    training_calls = adaptive_calls[-count:]
    if any(tuple(sample) != selected or prediction["yhat"] != label for sample, (selected, prediction), label in
           zip(sampler.samples, training_calls, sampler.sample_pred_results)):
        raise RuntimeError("Adaptive probabilities are not aligned with their masks")
    records["adaptive"] = [record for _, record in training_calls]
    models, targets, train_keys = {}, {}, {}
    for arm in ("random", "adaptive"):
        targets[arm] = np.array([row["p_class1"] for row in records[arm]])
        models[arm] = Ridge(alpha=plan["ridge_alpha"]).fit(design[arm], targets[arm])
        train_keys[arm] = {mask_key(mask) for mask in design[arm]}
        rows = [{**dict(zip(images, map(int, mask))), **record} for mask, record in zip(design[arm], records[arm])]
        write_csv(output / f"{arm}_training.csv", rows)
        write_csv(output / f"{arm}_coefficients.csv", [
            {"image_id": image, "coefficient": float(coef)} for image, coef in zip(images, models[arm].coef_)])

    evaluation = [query([image for image, flag in zip(images, mask) if flag], "shared_evaluation")
                  for mask in evaluation_design]
    evaluation_target = np.array([row["p_class1"] for row in evaluation])
    evaluation_labels = np.array([row["yhat"] for row in evaluation])
    overlap = {arm: np.array([mask_key(mask) in train_keys[arm] for mask in evaluation_design])
               for arm in models}
    novel = ~(overlap["random"] | overlap["adaptive"])
    fitted = {arm: model.predict(evaluation_design) for arm, model in models.items()}
    evaluation_rows = []
    for index, (mask, record) in enumerate(zip(evaluation_design, evaluation)):
        evaluation_rows.append({**dict(zip(images, map(int, mask))), **record,
            "in_random_training": int(overlap["random"][index]),
            "in_adaptive_training": int(overlap["adaptive"][index]), "shared_novel": int(novel[index]),
            "random_surrogate": float(fitted["random"][index]), "adaptive_surrogate": float(fitted["adaptive"][index])})
    write_csv(output / "evaluation.csv", evaluation_rows)
    arm_metrics = {}
    for arm, model in models.items():
        labels = np.array([row["yhat"] for row in records[arm]])
        baseline = np.full(len(evaluation), targets[arm].mean())
        arm_metrics[arm] = {
            "training_rows": count, "unique_training_masks": len(train_keys[arm]),
            "training_class_counts": {str(cls): int(np.sum(labels == cls)) for cls in (0, 1)},
            "training_probability_range": [float(targets[arm].min()), float(targets[arm].max())],
            "intercept": float(model.intercept_), "alpha": plan["ridge_alpha"],
            "all_evaluation_draws": metrics(evaluation_target, fitted[arm], evaluation_labels),
            "shared_novel_evaluation": metrics(evaluation_target, fitted[arm], evaluation_labels, novel),
            "constant_baseline_shared_novel": metrics(evaluation_target, baseline, evaluation_labels, novel),
            "evaluation_rows_overlapping_training": int(overlap[arm].sum()),
        }

    counts = deletion_counts(n_images, plan["deletion_fractions"], plan["minimum_images"])
    random_order = np.random.RandomState(streams["random_deletion"]).permutation(n_images).tolist()

    def curve(order, stage):
        result = []
        for deleted in counts:
            removed = set(order[:deleted])
            mask = [int(i not in removed) for i in range(n_images)]
            prediction = original if deleted == 0 else query(
                [image for image, flag in zip(images, mask) if flag], stage)
            result.append({"deleted_count": deleted, "deleted_fraction": deleted / n_images,
                "retained_count": n_images - deleted, **dict(zip(images, mask)), **prediction,
                "delta_from_original": original["p_class1"] - prediction["p_class1"]})
        return result

    random_curve = curve(random_order, "shared_random_deletion")
    deletion_rows, deletion_summary = [], {}
    for arm, model in models.items():
        curves = {
            "descending": curve(deletion_order(model.coef_, images, True), arm + "_descending_deletion"),
            "ascending": curve(deletion_order(model.coef_, images, False), arm + "_ascending_deletion"),
            "random": random_curve,
        }
        deletion_summary[arm] = {}
        for control, points in curves.items():
            deletion_rows += [{"arm": arm, "control": control, **point} for point in points]
            fractions = np.array([point["deleted_fraction"] for point in points])
            probabilities = np.array([point["p_class1"] for point in points])
            area = float(np.trapz(probabilities, fractions))
            span = float(fractions[-1] - fractions[0])
            deletion_summary[arm][control] = {
                "area_under_curve": area, "normalized_area": area / span if span else None,
                "max_deleted_fraction": float(fractions[-1]),
                "final_probability": float(probabilities[-1]),
                "final_delta_from_original": float(original["p_class1"] - probabilities[-1]),
            }
    write_csv(output / "deletion.csv", deletion_rows)
    write_csv(output / "queries.csv", query_log)
    return {
        "original_prediction": original, "arms": arm_metrics, "deletion": deletion_summary,
        "sampling_summary": sampler.sampling_summary, "query_counts": dict(query_counts),
        "total_model_queries": sum(query_counts.values()), "stage_seconds": dict(stage_seconds),
        "evaluation_rows": len(evaluation), "shared_novel_evaluation_rows": int(novel.sum()),
        "unique_evaluation_masks": len({mask_key(mask) for mask in evaluation_design}),
        "evaluation_subset_sizes": dict(Counter(map(int, evaluation_design.sum(axis=1)))),
        "shared_novel_subset_sizes": dict(Counter(map(int, evaluation_design[novel].sum(axis=1)))),
        "streams": streams,
    }


def run(args):
    plan_path = args.plan.resolve()
    plan = verified_plan(plan_path)
    if args.seed not in plan["seeds"] or args.threads < 1:
        raise ValueError("Seed must be in the plan; threads must be positive")
    output = (args.output or plan_path.parent / "runs" / f"seed-{args.seed}").resolve()
    if (plan_path.parent / "runs").resolve() not in output.parents:
        raise ValueError("Run output must be a new directory under this study's runs/")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": plan["scope"], "patient": plan["patient"],
        "seed": args.seed, "plan_sha256": sha256(plan_path), "host": socket.gethostname(),
        "device": args.device, "python": platform.python_version(), "python_executable": sys.executable,
        "python_base_prefix": sys.base_prefix, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "limitations": ["One patient; seeds and masks are not independent patients.",
            "Prospective probability Ridge, not historical bootstrap/classifier reproduction.",
            "Equal training rows with unequal sampler query overhead; no equal-compute comparison.",
            "Novel-mask fidelity is conditional on the observed training masks and may exclude large subset sizes.",
            "No prediction clipping, significance testing, or patient-population inference."]}
    started = time.monotonic()
    write_json(output / "report.json", report)
    try:
        predictor = PatientPredictor(plan, args.device, args.threads, args.seed)
        report["cuda_build"] = predictor.torch.version.cuda
        report["gpu_name"] = predictor.torch.cuda.get_device_name(predictor.device) if predictor.device.type == "cuda" else None
        report.update(experiment(plan, args.seed, output, predictor))
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
    print("Complete-patient study passed:", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Freeze patient, settings, input hashes and source snapshots")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--patient")
    prepare_parser.add_argument("--train-samples", type=int, default=1000)
    prepare_parser.add_argument("--eval-samples", type=int, default=200)
    prepare_parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    run_parser = commands.add_parser("run", help="Execute one seed from a verified n=1 plan")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    run_parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    main()
