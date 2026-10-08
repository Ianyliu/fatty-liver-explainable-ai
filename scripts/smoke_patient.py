#!/usr/bin/env python3
"""Bounded inference and diagnostic Ridge fit; separate from historical replay."""
import argparse
import ast
import csv
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_record(patient, paths):
    """Select the smallest complete eligible patient, or validate an explicit ID."""
    with open(paths.split_path(), newline="") as handle:
        test_ids = {row["MI_ID"] for row in csv.DictReader(handle)}
    with open(paths.metadata_path(), newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["MI_ID"] for row in rows if row["MI_ID"] in test_ids)
    if any(n != 1 for name, n in counts.items() if not patient or name == patient):
        raise ValueError("Ambiguous duplicate test-patient metadata records")
    candidates = []
    for row in rows:
        if row["MI_ID"] not in test_ids or (patient and row["MI_ID"] != patient):
            continue
        images = ast.literal_eval(row["IMG_ID_LIST"])
        if not isinstance(images, list) or not images or not all(isinstance(img, str) for img in images) or len(set(images)) != len(images):
            raise ValueError("Invalid or duplicate metadata image IDs")
        if not patient and (float(row["liver_fatty"]) <= 0 or len(images) < 20):
            continue
        if all((Path(paths.image_dir()) / f"{row['MI_ID']}_{img}.jpg").is_file() for img in images):
            candidates.append((len(images), row["MI_ID"], row, images))
    if patient and len(candidates) != 1:
        raise ValueError("Requested patient must have one test-split metadata record and all images")
    if not candidates:
        raise ValueError("No complete eligible test patient; see the input audit")
    _, _, row, images = min(candidates, key=lambda item: item[:2])
    return row, sorted(images)


def run(args, output, report):
    import numpy as np
    import torch
    import torchvision.transforms as transforms
    from sklearn.linear_model import Ridge
    import project_paths as paths
    from usflc_xai import datasets, models

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; request a GPU allocation or use --device cpu")
    record, images = patient_record(args.patient, paths)
    patient = record["MI_ID"]
    if args.min_images > len(images):
        raise ValueError("--min-images exceeds this patient's image count")
    checkpoint = Path(paths.checkpoint_path())
    weights = Path(torch.hub.get_dir()) / "checkpoints/densenet121-a639ec97.pth"
    if not checkpoint.is_file() or not weights.is_file():
        raise FileNotFoundError("Graph checkpoint and cached DenseNet121 weights are required; no downloads attempted")
    if not sha256(weights).startswith("a639ec97"):
        raise ValueError("DenseNet121 cache hash does not match the expected weight identifier")
    report.update({
        "patient": patient, "image_count": len(images), "ground_truth": int(float(record["liver_fatty"]) > 0),
        "device": str(device), "cuda_build": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "torch-geometric", "numpy", "scipy", "scikit-learn", "pillow")},
        "settings": {"seed": args.seed, "threads": args.threads, "samples": args.samples,
                     "min_images": args.min_images, "sampling": args.sampling,
                     "encoder": "DenseNet121 IMAGENET1K_V1", "graph_encoder": "SETNET_GAT",
                     "input_dim": 1024, "hidden_dim": 512, "num_layers": 1, "num_classes": 2,
                     "edges": "feature correlation > 0.95, excluding self edges; PyG GAT adds self loops",
                     "surrogate": "Ridge(alpha=1.0), target=GNN class-1 softmax probability",
                     "holdout_rows": args.holdout, "bootstrap_fits": 0},
    })
    artifacts = [Path(paths.metadata_path()), Path(paths.split_path()), checkpoint, weights,
                 ROOT / "pyproject.toml", ROOT / "uv.lock", Path(__file__), ROOT / "project_paths.py",
                 ROOT / "usflc_xai/datasets.py", ROOT / "usflc_xai/models.py"]
    artifacts += [Path(paths.image_dir()) / f"{patient}_{img}.jpg" for img in images]
    if args.sampling == "adaptive":
        artifacts.append(ROOT / "sampling_marginal_relation_pipeline.py")
    report["inputs"] = [{"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size} for p in artifacts]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), transforms.Resize([224, 224]),
        transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    print("Loading cached DenseNet121 and graph checkpoint on", device, flush=True)
    _, image_encoder = models.image_encoder_model("densenet121", True, 2, device)
    graph_encoder, _ = models.encoder_model("SETNET_GAT", 1024, 1, 2, device)
    # The supplied checkpoint is local and already inventoried. Preserve its
    # original state dictionary format while allowing CUDA-saved tensors on CPU.
    payload = torch.load(checkpoint, map_location=device)
    graph_encoder.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    image_encoder.eval()
    graph_encoder.eval()

    prediction_calls = 0

    def predict(selected):
        nonlocal prediction_calls
        prediction_calls += 1
        # Reuse the uploaded loader for every graph; no approximation using
        # cached full-patient features or altered graph construction.
        with torch.inference_mode():
            graph = datasets.single_data_loader(
                patient, selected, transform, image_encoder, float(record["liver_fatty"]),
                2, device, crop_image_dir=paths.image_dir())
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)
            logits = graph_encoder(graph.x.to(device), graph.edge_index_corr.to(device), batch, 1)
            if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid GNN output")
            return {"logits": logits[0].cpu().tolist(), "yhat": int(logits.argmax(dim=1).item()),
                    "p_class1": float(logits.softmax(dim=1)[0, 1].item()),
                    "nodes": int(graph.x.shape[0]), "edges": int(graph.edge_index_corr.shape[1])}

    started = time.monotonic()
    report["original_prediction"] = predict(images)
    print("Original patient prediction complete; generating perturbations", flush=True)
    design = np.zeros((args.samples, len(images)), dtype=np.int64)
    predictions = []
    if args.sampling == "random":
        for index in range(args.samples):
            size = rng.randint(args.min_images, len(images) + 1)
            indices = np.sort(rng.choice(len(images), size=size, replace=False))
            design[index, indices] = 1
            predictions.append(predict([images[i] for i in indices]))
            print(f"Perturbation {index + 1}/{args.samples} complete", flush=True)
    else:
        from sampling_marginal_relation_pipeline import LIME_subj_pipeline
        recorded = {}

        def classify(selected):
            result = predict(list(selected))
            recorded[tuple(sorted(selected))] = result
            return result["yhat"]

        sampler = LIME_subj_pipeline(
            test_data_id="09", img_list=images, y=report["ground_truth"], mi_id=patient,
            img_dir=paths.image_dir(), pred_func=classify, result_parent_dir=str(output),
            random_state=args.seed)
        sampler.predict_on_random_samples_until_convergence(
            n_samples=args.samples, target_positive_proportion=0.5,
            min_sample_size=args.min_images, max_sample_size=len(images), max_iter=args.samples,
            append_to_original=False)
        sampler.generate_pred_results_matrix()
        design = sampler.pred_results_x
        predictions = [recorded[tuple(sorted(sample))] for sample in sampler.samples]
        if sampler.img_list != images or len(predictions) != args.samples:
            raise RuntimeError("Adaptive masks are misaligned or incomplete")
        report["sampling_summary"] = sampler.sampling_summary
        print("Adaptive perturbations complete:", sampler.sampling_summary, flush=True)
    report["model_prediction_calls_before_references"] = prediction_calls
    report["inference_seconds"] = time.monotonic() - started
    targets = np.array([p["p_class1"] for p in predictions])
    labels = np.array([p["yhat"] for p in predictions])
    split = args.samples - args.holdout
    surrogate = Ridge(alpha=1.0).fit(design[:split], targets[:split])
    fitted = surrogate.predict(design)
    if not np.isfinite(surrogate.coef_).all() or not np.isfinite(fitted).all():
        raise RuntimeError("Nonfinite surrogate result")
    report["surrogate"] = {
        "train_rows": split, "holdout_rows": args.holdout, "intercept": float(surrogate.intercept_),
        "train_mae": float(np.abs(fitted[:split] - targets[:split]).mean()),
        "holdout_mae": float(np.abs(fitted[split:] - targets[split:]).mean()),
        "holdout_class_agreement": float(((fitted[split:] >= 0.5) == labels[split:]).mean()),
        "original_p_class1_estimate": float(surrogate.predict(np.ones((1, len(images))))[0]),
        "unique_masks": int(len(np.unique(design, axis=0))),
        "prediction_class_counts": {str(cls): int((labels == cls).sum()) for cls in (0, 1)},
        "target_probability_range": [float(targets.min()), float(targets.max())],
    }
    report["limitations"] = [
        "Small diagnostic fit with fixed alpha and a tiny holdout; no scientific fidelity claim.",
        "Probability Ridge surrogate differs from historical RidgeClassifierCV with hard labels and bootstrap intervals.",
        "Bounded sampling diagnostic; cohort-scale adaptive performance and correlation/uncertainty paths are not validated.",
        "A fixed seed documents this run; it cannot recreate historical RNG states or guarantee cross-device equality.",
    ]
    if len(np.unique(labels)) < 2:
        report["limitations"].append("All perturbations predict one class; classification agreement alone is uninformative.")
    # Optional bounded check of known masks, without refitting historical outputs.
    # Take the first four rows in each saved class, rather than selecting on agreement.
    report["reference_comparisons"] = []
    for reference in args.reference_predictions:
        with reference.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if set(fields) != set(images + ["y", "yhat"]) or len(fields) != len(images) + 2:
                raise ValueError("Reference columns must be the same patient image IDs plus y and yhat")
            selected = []
            counts = {0: 0, 1: 0}
            for row_index, row in enumerate(reader):
                if any(row[name] not in {"0", "1", "0.0", "1.0"} for name in fields):
                    raise ValueError("Reference must contain only binary inclusion indicators and labels")
                label = int(float(row["yhat"]))
                if counts[label] >= 4:
                    continue
                if int(float(row["y"])) != report["ground_truth"]:
                    raise ValueError("Reference ground truth differs from the configured patient")
                included = [name for name in images if int(float(row[name]))]
                if len(included) < 2:
                    raise ValueError("Reference mask needs at least two images for correlation graph construction")
                selected.append({"saved_row": row_index, "saved_yhat": label, **predict(included)})
                counts[label] += 1
                if sum(counts.values()) == 8:
                    break
        if not selected:
            raise ValueError("Reference prediction table is empty")
        mismatches = sum(row["saved_yhat"] != row["yhat"] for row in selected)
        report["reference_comparisons"].append({
            "path": str(reference.resolve()), "sha256": sha256(reference), "rows_checked": len(selected),
            "mismatches": mismatches, "rows": selected,
        })
        print(f"Saved-mask comparison: {len(selected)} checked, {mismatches} mismatches", flush=True)
        if mismatches:
            report["limitations"].append("Saved GNN labels disagree with current inference; historical model/run provenance needs reconciliation.")
    with (output / "pred_results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(images + ["yhat", "y", "p_class1", "logit0", "logit1", "edges", "split", "ridge_p_class1"])
        for i, prediction in enumerate(predictions):
            writer.writerow(design[i].tolist() + [prediction["yhat"], report["ground_truth"], prediction["p_class1"],
                *prediction["logits"], prediction["edges"], "train" if i < split else "holdout", float(fitted[i])])
    with (output / "ridge_coefficients.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["IMG_ID", "coefficient"])
        writer.writerows(zip(images, surrogate.coef_))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", help="Test-split ID; default selects smallest complete positive patient with >=20 images")
    parser.add_argument("--output", required=True, type=Path, help="New private output directory; never an original/reference directory")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--samples", type=int, default=20, help="Bounded to 10–20; production sampling uses a separate workflow")
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--min-images", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sampling", choices=("random", "adaptive"), default="random")
    parser.add_argument("--reference-predictions", type=Path, action="append", default=[],
                        help="Optionally compare up to 8 saved masks per table; repeat for separate historical runs")
    args = parser.parse_args()
    if not 10 <= args.samples <= 20 or not 1 <= args.holdout < args.samples:
        parser.error("Use 10–20 samples and a nonempty train/holdout split")
    if args.threads < 1 or args.min_images < 2 or not 0 <= args.seed < 2**32:
        parser.error("Use positive threads, >=2 images per graph, and a seed in [0, 2**32)")
    import project_paths as paths
    output = args.output.resolve()
    protected = [(ROOT / "outputs/original").resolve(), Path(paths.prediction_root()).resolve(),
                 Path(paths.image_dir()).resolve(), Path(paths.metadata_path()).resolve().parent,
                 Path(paths.checkpoint_path()).resolve().parent]
    if any(output == p or p in output.parents for p in protected):
        parser.error("Output must be separate from original predictions, data, and checkpoints")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "purpose": "infrastructure smoke test", "host": socket.gethostname(),
              "python": platform.python_version(), "command": sys.argv, "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    started = time.monotonic()
    try:
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        run(args, output, report)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("One-patient smoke passed. Private artifacts:", output, flush=True)


if __name__ == "__main__":
    main()
