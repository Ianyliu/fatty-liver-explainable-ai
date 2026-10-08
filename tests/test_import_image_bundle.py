"""Synthetic transfer checks: reject conflicts before import and preserve originals."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from import_image_bundle import sha256, stage_and_import, verify


def csv_bytes(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class BundleImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.reports = self.root / "reports"
        self.reports.mkdir()
        self.metadata = self.root / "metadata.csv"
        self.split = self.root / "split.csv"
        ids = [f"image{i:02}" for i in range(20)]
        self.metadata.write_bytes(csv_bytes([dict(MI_ID="synthetic", liver_fatty=1, IMG_ID_LIST=repr(ids))]))
        self.split.write_bytes(csv_bytes([dict(MI_ID="synthetic")]))
        stream = io.BytesIO()
        Image.new("L", (8, 6), 42).save(stream, format="JPEG")
        self.payload = stream.getvalue()
        digest = hashlib.sha256(self.payload).hexdigest()
        manifest, availability = [], []
        self.members = {}
        self.names = [f"synthetic_{image}.jpg" for image in ids]
        for name, image in zip(self.names, ids):
            member = f"bundle/images/{name}"
            source = f"/synthetic/{name}"
            self.members[member] = self.payload
            manifest.append(dict(original_source_path=source, bundle_relative_path=member,
                                 requested_filename=name, sha256=digest, bytes=len(self.payload)))
            availability.append(dict(MI_ID="synthetic", IMG_ID=image, requested_filename=name,
                                     status="found_verified", source_path=source + ";/synthetic-copy/" + name, sha256=digest,
                                     bytes=len(self.payload), format="JPEG", width=8, height=6, mode="L"))
        self.members.update({
            "bundle_manifest.csv": csv_bytes(manifest),
            "image_availability.csv": csv_bytes(availability),
            "patient_coverage.csv": csv_bytes([dict(MI_ID="synthetic", liver_fatty=1,
                required_image_count=20, verified_available_count=20, unresolved_count=0,
                missing_count=0, complete="yes", image_list=" ".join(ids))]),
            "working/metadata_hashes.json": json.dumps(dict(metadata_sha256=sha256(self.metadata),
                split_sha256=dict(test=sha256(self.split)))).encode(),
        })
        self.archive = self.root / "bundle.zip"
        self.write_archive()

    def write_archive(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, data in self.members.items():
                archive.writestr(name, data)

    def verified(self, report):
        return verify(self.archive, self.metadata, self.split, self.images, report,
                      expected_hash=sha256(self.archive))

    def test_import_preserves_identical_file_and_validates_all_outputs(self):
        original = self.images / self.names[0]
        original.write_bytes(self.payload)
        before = original.stat().st_mtime_ns
        report = {}
        entries = self.verified(report)
        stage_and_import(self.archive, entries, self.images, self.reports, report)
        self.assertEqual((report["imported_count"], report["preserved_count"]), (19, 1))
        self.assertEqual(original.stat().st_mtime_ns, before)
        self.assertEqual(report["post_import_verified_images"], 20)
        self.assertTrue(all((self.images / name).read_bytes() == self.payload for name in self.names))

    def test_existing_conflict_prevents_any_import(self):
        original = self.images / self.names[-1]
        original.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.verified({})
        self.assertEqual(list(self.images.iterdir()), [original])
        self.assertEqual(original.read_bytes(), b"different")

    def test_late_destination_conflict_prevents_any_import(self):
        report = {}
        entries = self.verified(report)
        original = self.images / self.names[-1]
        original.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            stage_and_import(self.archive, entries, self.images, self.reports, report)
        self.assertEqual(list(self.images.iterdir()), [original])

    def test_corrupt_payload_rejected(self):
        self.members[f"bundle/images/{self.names[-1]}"] = b"corrupt"
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "size/hash"):
            self.verified({})
        self.assertFalse(list(self.images.iterdir()))

    def test_metadata_mismatch_rejected(self):
        self.metadata.write_bytes(self.metadata.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            self.verified({})

    def test_wrong_cohort_payload_rejected(self):
        del self.members[f"bundle/images/{self.names[-1]}"]
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.verified({})

    def test_unsafe_member_rejected(self):
        self.members["../escape"] = b"unsafe"
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.verified({})

    def test_independent_checksum_mismatch_rejected(self):
        report = {}
        with self.assertRaisesRegex(ValueError, "separately supplied"):
            verify(self.archive, self.metadata, self.split, self.images, report, expected_hash="0" * 64)
        self.assertEqual(report["independent_archive_checksum"], "mismatch")


if __name__ == "__main__":
    unittest.main()
