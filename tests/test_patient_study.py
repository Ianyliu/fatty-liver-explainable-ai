"""Scientific invariants for the n=1 runner, using synthetic graphs only."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patient_study as study
import summarize_patient_study as summary


class PatientStudyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.images = [f"image{i:02}" for i in range(20)]
        for image in self.images:
            (self.root / f"synthetic_{image}.jpg").touch()
        self.plan = {"patient": "synthetic", "images": self.images,
            "paths": {"images": str(self.root)}, "train_samples": 40,
            "evaluation_samples": 20, "minimum_images": 3, "ridge_alpha": 1.0,
            "adaptive_target_class1": 0.5, "deletion_fractions": [0, .1, .2, .3, .4, .5],
            "streams": {"0": study.stream_seeds(0)}}
        self.calls = []

    def predict(self, selected):
        self.calls.append(tuple(selected))
        # Per-call jitter ensures repeated masks must keep their own response.
        probability = .15 + .55 * ("image00" in selected) + .003 * len(selected) + len(self.calls) * 1e-7
        return {"p_class1": probability, "yhat": int(probability >= .5),
                "logit0": 0., "logit1": float(np.log(probability / (1 - probability))), "edges": 0}

    def execute(self):
        output = self.root / "run"
        output.mkdir()
        return study.experiment(self.plan, 0, output, self.predict), output

    def rows(self, path):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_query_budget_response_alignment_and_novel_fidelity(self):
        report, output = self.execute()
        self.assertEqual(report["total_model_queries"], 147)
        self.assertEqual(len(self.calls), 147)
        ledger = self.rows(output / "queries.csv")
        self.assertEqual(len(ledger), 147)
        keys = {}
        for arm in ("random", "adaptive"):
            training = self.rows(output / f"{arm}_training.csv")
            queried = [row for row in ledger if row["stage"] == arm + "_training"]
            self.assertEqual(training, [{k: row[k] for k in training[0]} for row in queried])
            keys[arm] = {tuple(row[image] for image in self.images) for row in training}
        evaluation = self.rows(output / "evaluation.csv")
        for row in evaluation:
            key = tuple(row[image] for image in self.images)
            for arm in keys:
                self.assertEqual(int(row["in_" + arm + "_training"]), int(key in keys[arm]))
            self.assertEqual(int(row["shared_novel"]), int(all(key not in keys[a] for a in keys)))
        novel = [row for row in evaluation if row["shared_novel"] == "1"]
        self.assertGreater(len(novel), 0)
        for arm in keys:
            mae = np.mean([abs(float(row[arm + "_surrogate"]) - float(row["p_class1"])) for row in novel])
            self.assertAlmostEqual(mae, report["arms"][arm]["shared_novel_evaluation"]["mae"])
            training_mean = np.mean([float(row["p_class1"]) for row in self.rows(output / f"{arm}_training.csv")])
            baseline_mae = np.mean([abs(training_mean - float(row["p_class1"])) for row in novel])
            self.assertAlmostEqual(baseline_mae, report["arms"][arm]["constant_baseline_shared_novel"]["mae"])
        self.assertEqual(sum(report["shared_novel_subset_sizes"].values()), len(novel))

    def test_deletion_rebuilds_nested_subsets_and_shares_control(self):
        report, output = self.execute()
        rows = self.rows(output / "deletion.csv")
        curves = {}
        for arm in ("random", "adaptive"):
            for control in ("descending", "ascending", "random"):
                points = [row for row in rows if row["arm"] == arm and row["control"] == control]
                curves[arm, control] = points
                self.assertEqual([int(p["deleted_count"]) for p in points], [0, 2, 4, 6, 8, 10])
                retained = [{image for image in self.images if row[image] == "1"} for row in points]
                self.assertTrue(all(b < a for a, b in zip(retained, retained[1:])))
                x, y = np.array([float(p["deleted_fraction"]) for p in points]), np.array([float(p["p_class1"]) for p in points])
                self.assertAlmostEqual(np.trapz(y, x), report["deletion"][arm][control]["area_under_curve"])
        self.assertEqual([{k: v for k, v in p.items() if k != "arm"} for p in curves["random", "random"]],
                         [{k: v for k, v in p.items() if k != "arm"} for p in curves["adaptive", "random"]])
        self.assertEqual(report["query_counts"]["shared_random_deletion"], 5)

    def test_no_novel_rows_is_unavailable_not_zero_error(self):
        with patch.object(study, "random_masks", side_effect=lambda count, n, minimum, seed: np.ones((count, n), dtype=int)):
            report, _ = self.execute()
        self.assertEqual(report["shared_novel_evaluation_rows"], 0)
        for arm in report["arms"].values():
            self.assertEqual(arm["shared_novel_evaluation"], {"status": "unavailable_no_rows", "rows": 0})

    def test_metrics_keep_raw_scores_and_class_support(self):
        result = study.metrics(np.array([.1, .9]), np.array([-.2, 1.1]), np.array([0, 1]))
        self.assertEqual(result["outside_probability_range"], 2)
        self.assertAlmostEqual(result["mae"], .25)
        self.assertFalse(result["one_class"])
        with self.assertRaises(ValueError):
            study.metrics(np.array([.1]), np.array([np.nan]), np.array([0]))

    def test_exploratory_rank_similarity_handles_ties_and_constant_vectors(self):
        self.assertAlmostEqual(summary.rank_similarity([0, 0, 1], [0, 0, 2]), 1.)
        self.assertAlmostEqual(summary.rank_similarity([0, 1, 2], [2, 1, 0]), -1.)
        self.assertIsNone(summary.rank_similarity([0, 0, 0], [0, 1, 2]))
        with self.assertRaises(ValueError):
            summary.rank_similarity([0, np.nan], [0, 1])

    def test_ties_counts_and_rng_streams(self):
        self.assertEqual(study.deletion_order(np.array([.2, .2, -.1]), ["b", "a", "c"], True), [1, 0, 2])
        self.assertEqual(study.deletion_order(np.array([.2, .2, -.1]), ["b", "a", "c"], False), [2, 1, 0])
        self.assertEqual(study.deletion_counts(4, [0, .1, .2, .3, .4, .5], 3), [0, 1])
        streams = study.stream_seeds(0)
        self.assertEqual(len(set(streams.values())), 3)
        self.assertEqual(streams, study.stream_seeds(0))
        self.assertNotEqual(streams, study.stream_seeds(1))

    def test_manifest_rejects_changed_inputs(self):
        source = self.root / "input"
        source.write_text("initial")
        plan_file = self.root / "plan.json"
        study.write_json(plan_file, {"files": [{"path": str(source), "sha256": study.sha256(source)}], "versions": {}})
        study.verified_plan(plan_file)
        source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed after planning"):
            study.verified_plan(plan_file)

    def test_existing_output_is_never_overwritten(self):
        plan_file = self.root / "plan.json"
        plan_file.write_text("{}")
        output = self.root / "runs/seed-0"
        output.mkdir(parents=True)
        sentinel = output / "sentinel"
        sentinel.write_text("keep")
        args = argparse.Namespace(plan=plan_file, seed=0, threads=2, output=None, device="cpu")
        with patch.object(study, "verified_plan", return_value={"seeds": [0]}), patch.object(study, "PatientPredictor") as predictor:
            with self.assertRaises(FileExistsError):
                study.run(args)
            predictor.assert_not_called()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_independent_summary_recomputes_evidence_and_rejects_bad_flags(self):
        report, output = self.execute()
        report.update(status="passed", seed=0, patient="synthetic", plan_sha256="synthetic-plan")
        study.write_json(output / "report.json", report)
        checked = summary.validate_run(self.plan, "synthetic-plan", output, 0)
        self.assertEqual(len(checked["fidelity"]), 2)
        self.assertEqual(len(checked["deletion_summary"]), 6)
        rows = self.rows(output / "evaluation.csv")
        rows[0]["shared_novel"] = str(1 - int(rows[0]["shared_novel"]))
        study.write_csv(output / "evaluation.csv", rows)
        with self.assertRaisesRegex(ValueError, "novel-mask flags"):
            summary.validate_run(self.plan, "synthetic-plan", output, 0)

    def test_summary_rejects_failed_runs_and_changed_deletion_masks(self):
        report, output = self.execute()
        report.update(status="failed", seed=0, patient="synthetic", plan_sha256="synthetic-plan")
        study.write_json(output / "report.json", report)
        with self.assertRaisesRegex(ValueError, "unsuccessful"):
            summary.validate_run(self.plan, "synthetic-plan", output, 0)
        report["status"] = "passed"
        study.write_json(output / "report.json", report)
        rows = self.rows(output / "deletion.csv")
        rows[1][self.images[0]] = str(1 - int(rows[1][self.images[0]]))
        study.write_csv(output / "deletion.csv", rows)
        with self.assertRaisesRegex(ValueError, "coefficient ranking"):
            summary.validate_run(self.plan, "synthetic-plan", output, 0)


if __name__ == "__main__":
    unittest.main()
