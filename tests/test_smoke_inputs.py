"""Input selection checks using synthetic IDs/files, with no private artifacts."""
import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from smoke_patient import patient_record


class PatientInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.paths = SimpleNamespace(
            split_path=lambda: self.root / "split.csv",
            metadata_path=lambda: self.root / "metadata.csv",
            image_dir=lambda: self.root,
        )

    def inputs(self, records, split_ids=None, missing=()):
        with self.paths.metadata_path().open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["MI_ID", "liver_fatty", "IMG_ID_LIST"])
            writer.writeheader()
            for name, label, count in records:
                images = [f"image{i:02}" for i in range(count)]
                writer.writerow({"MI_ID": name, "liver_fatty": label, "IMG_ID_LIST": repr(images)})
                for image in images:
                    if (name, image) not in missing:
                        (self.root / f"{name}_{image}.jpg").touch()
        with self.paths.split_path().open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["MI_ID"])
            writer.writerows([[name] for name in (split_ids if split_ids is not None else [r[0] for r in records])])

    def test_default_selection_respects_cohort_split_completeness_and_stable_order(self):
        self.inputs([
            ("incomplete", 1, 20), ("negative", 0, 20), ("too-small", 1, 19),
            ("outside-split", 1, 20), ("larger", 1, 21), ("beta", 1, 20), ("alpha", 1, 20),
        ], split_ids=["incomplete", "negative", "too-small", "larger", "beta", "alpha"],
            missing=[("incomplete", "image19")])
        row, images = patient_record(None, self.paths)
        self.assertEqual(row["MI_ID"], "alpha")
        self.assertEqual(len(images), 20)

    def test_explicit_patient_cannot_use_partial_images(self):
        self.inputs([("partial", 1, 20)], missing=[("partial", "image00")])
        with self.assertRaisesRegex(ValueError, "all images"):
            patient_record("partial", self.paths)

    def test_duplicate_metadata_cannot_select_an_arbitrary_record(self):
        self.inputs([("duplicate", 1, 20), ("duplicate", 0, 20)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            patient_record("duplicate", self.paths)

    def test_no_complete_patient_fails_before_inference(self):
        self.inputs([("negative", 0, 20)])
        with self.assertRaisesRegex(ValueError, "No complete"):
            patient_record(None, self.paths)


if __name__ == "__main__":
    unittest.main()
