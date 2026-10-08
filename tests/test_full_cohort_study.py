"""Full-cohort coverage, reuse, immutable evidence and task-budget invariants."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import full_cohort_study as full


class FullCohortTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.patients = [{"patient_index": i, "patient": f"synthetic{i:03}",
                          "plan": f"/synthetic/patient-{i}/plan.json", "reused": i % 13 == 0 and i < 130}
                         for i in range(135)]

    def child(self, count=20):
        return {**copy.deepcopy(full.SETTINGS), "images": list(range(count)),
                "streams": {str(s):full.study.stream_seeds(s) for s in (0,1,2)}}

    def test_tasks_skip_noncontiguous_reused_patients_and_cover_each_new_seed_once(self):
        tasks = full.task_mapping(self.patients)
        self.assertEqual(len(tasks),375)
        self.assertEqual([t["index"] for t in tasks],list(range(375)))
        self.assertEqual({(t["patient_index"],t["seed"]) for t in tasks},
            {(p["patient_index"],s) for p in self.patients if not p["reused"] for s in (0,1,2)})
        self.assertTrue(all(t["plan"]==self.patients[t["patient_index"]]["plan"] for t in tasks))

    def test_variable_image_counts_have_exact_fresh_query_budget(self):
        self.assertEqual(full.queries_per_seed(self.child(20)),2247)
        self.assertEqual(full.queries_per_seed(self.child(35)),2262)
        self.assertEqual(sum(full.queries_per_seed(self.child())*3 for _ in range(125)),842625)

    def test_different_sampler_budget_or_rng_stream_is_rejected(self):
        child = self.child(); full.check_settings(child)
        child["train_samples"]=999
        with self.assertRaisesRegex(ValueError,"settings differ"):
            full.check_settings(child)
        child = self.child(); child["streams"]["0"]["evaluation"]+=1
        with self.assertRaisesRegex(ValueError,"RNG streams"):
            full.check_settings(child)

    def test_eligible_manifest_requires_exact_ids_counts_and_no_duplicates(self):
        path = self.root / "patients.csv"
        def write(rows):
            with path.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=["MI_ID","image_count"])
                writer.writeheader();writer.writerows(rows)
        rows = [{"MI_ID":p["patient"],"image_count":20} for p in self.patients]
        cohort = {p["patient"]:{"images":list(range(20))} for p in self.patients}
        with patch.object(full,"expected_cohort",return_value=cohort):
            write(rows); self.assertEqual(len(full.eligible_rows(path,"unused","unused")),135)
            write(rows[:-1]+[rows[0]])
            with self.assertRaisesRegex(ValueError,"exactly the 135"):
                full.eligible_rows(path,"unused","unused")
            changed=copy.deepcopy(rows); changed[0]["image_count"]=21;write(changed)
            with self.assertRaisesRegex(ValueError,"counts differ"):
                full.eligible_rows(path,"unused","unused")

    def test_changed_old_evidence_and_invalid_dispatch_are_rejected(self):
        evidence = self.root / "old-evidence.csv";evidence.write_text("saved evidence")
        plan = {"patients":self.patients,"tasks":full.task_mapping(self.patients),
                "settings":copy.deepcopy(full.SETTINGS),
                "files":[{"path":str(evidence),"sha256":full.study.sha256(evidence)}]}
        path = self.root / "plan.json";path.write_text(json.dumps(plan))
        self.assertEqual(len(full.verified(path)["tasks"]),375)
        plan["tasks"][-1]["seed"]=0;path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,"task mapping"):
            full.verified(path)
        plan["tasks"]=full.task_mapping(self.patients);path.write_text(json.dumps(plan))
        evidence.write_text("changed evidence")
        with self.assertRaisesRegex(ValueError,"source/input changed"):
            full.verified(path)


if __name__ == "__main__":
    unittest.main()
