#!/usr/bin/env python3
"""Verify a source image bundle and optionally import absent cohort images."""
import argparse
import ast
import csv
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def expected_cohort(metadata_path, split_path):
    metadata = csv_rows(Path(metadata_path).read_bytes())
    split = csv_rows(Path(split_path).read_bytes())
    ids = [row["MI_ID"] for row in split]
    id_set = set(ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate test-split patient IDs")
    counts = Counter(row["MI_ID"] for row in metadata)
    if any(counts[patient] != 1 for patient in ids):
        raise ValueError("Test patients lack unique metadata records")
    cohort = {}
    for row in metadata:
        if row["MI_ID"] not in id_set or float(row["liver_fatty"]) <= 0:
            continue
        images = ast.literal_eval(row["IMG_ID_LIST"])
        if not isinstance(images, list) or len(images) != len(set(images)):
            raise ValueError("Malformed or duplicate metadata image IDs")
        if len(images) >= 20:
            cohort[row["MI_ID"]] = {"images": images, "label": float(row["liver_fatty"])}
    if not cohort:
        raise ValueError("No eligible cohort")
    return cohort


def verify(archive, metadata_path, split_path, image_dir, report, expected_hash=None):
    from PIL import Image

    archive = Path(archive)
    actual_hash = sha256(archive)
    report.update(archive=str(archive.resolve()), archive_bytes=archive.stat().st_size,
                  archive_sha256=actual_hash, expected_archive_sha256=expected_hash,
                  independent_archive_checksum="pending" if expected_hash else "not_provided")
    if expected_hash and actual_hash != expected_hash.lower():
        report["independent_archive_checksum"] = "mismatch"
        raise ValueError("Archive SHA-256 differs from the separately supplied checksum")
    if expected_hash:
        report["independent_archive_checksum"] = "matched"
    cohort = expected_cohort(metadata_path, split_path)
    expected = {f"{patient}_{image}.jpg": (patient, image)
                for patient, record in cohort.items() for image in record["images"]}
    if len(expected) != sum(len(record["images"]) for record in cohort.values()):
        raise ValueError("Canonical image filenames collide")
    if any(PurePosixPath(name).name != name or "\\" in name for name in expected):
        raise ValueError("Metadata image names contain path components")
    required_reports = {"bundle_manifest.csv", "image_availability.csv", "patient_coverage.csv",
                        "working/metadata_hashes.json"}
    allowed_reports = required_reports | {"working/aggregate.json", "search_report.md", "search_report.json"}
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member names")
        if sum(item.file_size for item in infos) > 512 * 1024 * 1024:
            raise ValueError("Expanded archive exceeds this transfer's 512-MiB limit")
        for item in infos:
            path = PurePosixPath(item.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in item.filename
                    or path.as_posix() != item.filename.rstrip("/")
                    or stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1):
                raise ValueError("Unsafe, encrypted or noncanonical ZIP member")
        files = {item.filename for item in infos if not item.is_dir()}
        if not required_reports <= files:
            raise ValueError("Required source reports are missing")
        image_members = {name for name in files if name.startswith("bundle/images/")}
        if files - image_members - allowed_reports:
            raise ValueError("Unexpected non-image ZIP payload")
        if image_members != {f"bundle/images/{name}" for name in expected}:
            raise ValueError("Archive images do not exactly match the local eligible cohort")
        source_hashes = json.loads(z.read("working/metadata_hashes.json"))
        metadata_hash = sha256(metadata_path)
        split_hash = sha256(split_path)
        if source_hashes["metadata_sha256"] != metadata_hash or source_hashes["split_sha256"]["test"] != split_hash:
            raise ValueError("Source metadata/test split hashes differ from local inputs")
        report.update(metadata_sha256=metadata_hash, test_split_sha256=split_hash,
                      eligible_patients=len(cohort), required_images=len(expected))
        coverage = csv_rows(z.read("patient_coverage.csv"))
        if len(coverage) != len(cohort) or {row["MI_ID"] for row in coverage} != set(cohort):
            raise ValueError("Source patient coverage does not exactly match the local cohort")
        for row in coverage:
            record = cohort[row["MI_ID"]]
            # Source coverage reports serialize IDs as whitespace-separated text;
            # also accept the metadata's Python-list representation.
            value = row["image_list"].strip()
            images = ast.literal_eval(value) if value.startswith("[") else value.split()
            if not isinstance(images, list) or not all(isinstance(image, str) for image in images):
                raise ValueError("Malformed source coverage image list")
            if (len(images) != len(record["images"]) or set(images) != set(record["images"])
                    or int(row["required_image_count"]) != len(images)
                    or int(row["verified_available_count"]) != len(images)
                    or int(row["missing_count"]) != 0 or int(row["unresolved_count"]) != 0
                    or row["complete"].lower() not in {"true", "1", "yes"}
                    or float(row["liver_fatty"]) != record["label"]):
                raise ValueError("A source patient coverage record is inconsistent")
        manifest = csv_rows(z.read("bundle_manifest.csv"))
        availability = csv_rows(z.read("image_availability.csv"))
        for rows in (manifest, availability):
            if len(rows) != len(expected) or {row["requested_filename"] for row in rows} != set(expected):
                raise ValueError("Image manifest/availability filenames do not exactly match the cohort")
        available = {row["requested_filename"]: row for row in availability}
        entries = []
        existing = 0
        formats = Counter()
        dimensions = Counter()
        for row in manifest:
            name = row["requested_filename"]
            member = f"bundle/images/{name}"
            if row["bundle_relative_path"] != member or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                raise ValueError("Noncanonical manifest path/hash")
            raw = z.read(member)  # Also checks the ZIP member CRC.
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) != int(row["bytes"]) or digest != row["sha256"]:
                raise ValueError("Image payload size/hash mismatch")
            evidence = available[name]
            if (evidence["status"] != "found_verified" or evidence["sha256"] != digest
                    or row["original_source_path"] not in evidence["source_path"].split(";")
                    or int(evidence["bytes"]) != len(raw)
                    or (evidence["MI_ID"], evidence["IMG_ID"]) != expected[name]):
                raise ValueError("Image availability evidence conflicts with the payload")
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                if (image.format != "JPEG" or evidence["format"] != image.format
                        or int(evidence["width"]) != image.width or int(evidence["height"]) != image.height
                        or evidence["mode"] != image.mode):
                    raise ValueError("Image format/dimensions/mode differ from source evidence")
                formats[image.format] += 1
                dimensions[f"{image.width}x{image.height}:{image.mode}"] += 1
            destination = Path(image_dir) / name
            if destination.is_symlink():
                raise ValueError("Existing destination image is a symlink")
            if destination.exists():
                if not destination.is_file() or sha256(destination) != digest:
                    raise ValueError(f"Existing image conflicts with incoming payload; see private source manifest: {name}")
                existing += 1
            entries.append({"name": name, "member": member, "bytes": len(raw), "sha256": digest,
                            "present_before": destination.exists()})
        report.update(verified_images=len(entries), existing_identical_images=existing,
                      missing_images=len(entries) - existing, image_formats=dict(formats),
                      image_dimensions=dict(dimensions), cohort_lists_match=True,
                      payload_hashes_match=True, existing_image_conflicts=0)
    return entries


def stage_and_import(archive, entries, image_dir, report_dir, report):
    """Stage verified bytes, then create absent images exclusively; never overwrite."""
    stage = Path(report_dir) / "staged_bundle"
    if sha256(archive) != report["archive_sha256"]:
        raise ValueError("Archive changed after verification")
    stage.mkdir()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir():
                continue
            destination = stage / item.filename
            if stage.resolve() not in destination.resolve().parents:
                raise ValueError("Unsafe ZIP staging destination")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(z.read(item.filename))
    if sha256(archive) != report["archive_sha256"]:
        raise ValueError("Archive changed after verification")
    # Revalidate all staged bytes and existing targets before the first import.
    for entry in entries:
        if sha256(stage / entry["member"]) != entry["sha256"]:
            raise ValueError("Staged image hash differs from verified payload")
        destination = Path(image_dir) / entry["name"]
        if destination.is_symlink() or (destination.exists() and sha256(destination) != entry["sha256"]):
            raise ValueError("Destination changed/conflicts before import")
    report["imported_images"] = []
    report["preserved_images"] = []
    for entry in entries:
        destination = Path(image_dir) / entry["name"]
        try:
            with destination.open("xb") as handle:
                handle.write((stage / entry["member"]).read_bytes())
            report["imported_images"].append(entry["name"])
        except FileExistsError:
            if destination.is_symlink() or sha256(destination) != entry["sha256"]:
                raise ValueError("Concurrent destination conflict during import")
            report["preserved_images"].append(entry["name"])
    for entry in entries:
        if sha256(Path(image_dir) / entry["name"]) != entry["sha256"]:
            raise ValueError("Post-import image hash mismatch")
    report.update(imported_count=len(report["imported_images"]),
                  preserved_count=len(report["preserved_images"]),
                  post_import_verified_images=len(entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path, help="New ignored private output directory")
    parser.add_argument("--expected-sha256", help="Separately supplied source ZIP checksum, when available")
    parser.add_argument("--execute", action="store_true", help="Import verified absent images after all input checks pass")
    args = parser.parse_args()
    if args.expected_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_sha256):
        parser.error("--expected-sha256 must be 64 hexadecimal characters")
    import project_paths as paths
    output = args.report_dir.resolve()
    private_root = (ROOT / "outputs/reproducibility").resolve()
    if private_root not in output.parents:
        parser.error("Report directory must be inside ignored outputs/reproducibility")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "execute": args.execute, "destination": str(Path(paths.image_dir()).resolve())}
    started = time.monotonic()
    try:
        entries = verify(args.archive, paths.metadata_path(), paths.split_path(), paths.image_dir(), report,
                         expected_hash=args.expected_sha256)
        (output / "verified_image_manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
        if args.execute:
            stage_and_import(args.archive, entries, paths.image_dir(), output, report)
        report["status"] = "imported" if args.execute else "verified_only"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "import_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"imported_images", "preserved_images"}}, indent=2))


if __name__ == "__main__":
    main()
