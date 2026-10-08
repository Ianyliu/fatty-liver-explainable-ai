"""Cohort gates and patient-level aggregation using synthetic evidence."""
import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cohort_study as cohort


class CohortTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = [{"MI_ID": f"synthetic{i}", "image_count": "20"} for i in range(10)]

    def pilot(self):
        for index, row in enumerate(self.rows):
            directory = self.root / f"task-{index}"
            directory.mkdir()
            (directory / "report.json").write_text(json.dumps({"status": "passed", "patient": row["MI_ID"],
                "image_count": 20, "settings": {"samples": 20}, "model_prediction_calls_before_references": 21}))

    def test_pilot_gate_requires_same_complete_cohort(self):
        self.pilot()
        self.assertEqual(len(cohort.pilot_gate(self.rows, self.root)), 10)
        path = self.root / "task-9/report.json"
        report = json.loads(path.read_text()); report["patient"] = "wrong-patient"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "differs"):
            cohort.pilot_gate(self.rows, self.root)

    def test_failed_pilot_cannot_prepare_experiments(self):
        self.pilot()
        path = self.root / "task-9/report.json"
        report = json.loads(path.read_text()); report["status"] = "failed"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "unsuccessful"):
            cohort.pilot_gate(self.rows, self.root)

    def test_missing_pilot_cannot_prepare_experiments(self):
        self.pilot()
        (self.root / "task-9/report.json").unlink()
        with self.assertRaises(FileNotFoundError):
            cohort.pilot_gate(self.rows, self.root)

    def fidelity(self):
        # Deliberately unequal novel row counts: aggregation must weight patients equally.
        return [dict(patient_index=patient, seed=seed, arm=arm, novel_rows=10 if patient == 0 else 200,
                     novel_mae=0.1 + (patient + 1) * 0.1 * (arm == "adaptive"))
                for patient in (0, 1) for seed in (0, 1, 2) for arm in ("random", "adaptive")]

    def test_patient_mean_does_not_weight_masks_as_subjects(self):
        patients, result = cohort.aggregate(self.fidelity(), [0, 1], [0, 1, 2])
        self.assertAlmostEqual(result["mean_patient_paired_mae_difference"], 0.15)
        self.assertEqual(result["patients_with_all_seed_pairs"], 2)
        self.assertAlmostEqual(patients[0]["paired_adaptive_minus_random_mae"], 0.1)

    def test_unavailable_novel_result_keeps_primary_cohort_mean_unavailable(self):
        rows = self.fidelity()
        rows[-1]["novel_mae"] = None
        patients, result = cohort.aggregate(rows, [0, 1], [0, 1, 2])
        self.assertIsNone(result["mean_patient_paired_mae_difference"])
        self.assertEqual(result["missing_primary_patients"], [1])
        self.assertEqual(patients[1]["available_seed_pairs"], 2)

    def test_missing_or_duplicate_seed_arm_rejected(self):
        rows = self.fidelity()
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            cohort.aggregate(rows[:-1], [0, 1], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            cohort.aggregate(rows + [rows[-1]], [0, 1], [0, 1, 2])

    def test_preparation_freezes_exact_thirty_task_mapping(self):
        self.pilot()
        manifest = self.root / "patients.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["MI_ID", "image_count"])
            writer.writeheader(); writer.writerows(self.rows)
        output = self.root / "complete_patient/cohort"
        def prepare_patient(args):
            args.output.mkdir(parents=True)
            (args.output / "plan.json").write_text(json.dumps({"patient": args.patient,
                "images": [f"image{i}" for i in range(20)], "deletion_fractions": [0,.1,.2,.3,.4,.5], "seeds": [0,1,2]}))
        import project_paths
        with patch.object(project_paths, "output_root", return_value=str(self.root)), \
             patch.object(cohort.study, "patient_record", return_value=({"liver_fatty":"1"}, list(range(20)))), \
             patch.object(cohort.study, "prepare", side_effect=prepare_patient):
            cohort.prepare(SimpleNamespace(output=output, manifest=manifest, pilot=self.root))
        plan = cohort.verified_cohort(output / "cohort_plan.json")
        self.assertEqual([(t["patient_index"],t["seed"]) for t in plan["tasks"]],
                         [(p,s) for p in range(10) for s in range(3)])
        self.assertEqual(plan["planned_model_queries"], 67410)
        child = Path(plan["tasks"][0]["plan"])
        child.write_text(child.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            cohort.verified_cohort(output / "cohort_plan.json")

    def test_summary_recomputes_all_thirty_runs_and_rejects_corruption(self):
        images = [f"image{i:02}" for i in range(20)]
        plans, queries = [], 0
        for index in range(10):
            child = self.root / f"patient-{index:02}"
            child.mkdir()
            plan = {"patient": f"synthetic{index}", "images": images,
                "train_samples": 10, "evaluation_samples": 10, "minimum_images": 3,
                "ridge_alpha": 1.0, "adaptive_target_class1": 0.5,
                "paths": {"images": str(child)}, "seeds": [0,1,2],
                "deletion_fractions": [0,.1,.2,.3,.4,.5],
                "streams": {str(s):cohort.study.stream_seeds(s) for s in range(3)}}
            for image in images:
                (child / f"{plan['patient']}_{image}.jpg").touch()
            plan_path = child / "plan.json"
            plan_path.write_text(json.dumps(plan))
            def predict(selected):
                p = .15 + .6 * ("image00" in selected) + .003 * len(selected)
                return {"p_class1":p,"yhat":int(p>=.5),"logit0":0.,
                        "logit1":float(cohort.np.log(p/(1-p))),"edges":0}
            for seed in range(3):
                directory = child / "runs" / f"seed-{seed}"
                directory.mkdir(parents=True)
                with redirect_stdout(io.StringIO()):
                    report = cohort.study.experiment(plan, seed, directory, predict)
                report.update(status="passed",patient=plan["patient"],seed=seed,
                    plan_sha256=cohort.study.sha256(plan_path),gpu_name="synthetic",elapsed_seconds=1.)
                (directory / "report.json").write_text(json.dumps(report))
                queries += report["total_model_queries"]
            plans.append({"patient_index":index,"plan":str(plan_path)})
        frozen = {"patients":plans,"seeds":[0,1,2],"planned_model_queries":queries,"scope":"synthetic"}
        cohort_path = self.root / "cohort_plan.json"
        cohort_path.write_text(json.dumps(frozen))
        with patch.object(cohort,"verified_cohort",return_value=frozen), \
             patch.object(cohort.study,"verified_plan",side_effect=lambda p:json.loads(Path(p).read_text())):
            cohort.summarize(SimpleNamespace(cohort=cohort_path))
            result = json.loads((self.root / "cohort_summary/analysis.json").read_text())
            self.assertEqual(result["total_model_queries"],2310)
            self.assertEqual(result["planned_patients"],10)
            self.assertEqual(len(result["evidence"]),240)
            corrupted = self.root / "patient-09/runs/seed-2/evaluation.csv"
            corrupted.write_text(corrupted.read_text().replace("random_surrogate", "broken_surrogate"))
            with self.assertRaises(KeyError):
                cohort.summarize(SimpleNamespace(cohort=cohort_path))


if __name__ == "__main__":
    unittest.main()
