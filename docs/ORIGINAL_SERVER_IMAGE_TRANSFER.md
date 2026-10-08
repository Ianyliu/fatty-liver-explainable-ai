# Next original-server prompt: package the corrected cohort images

**Completed:** the ZIP and separate checksum have arrived, validation passed, and all 135 eligible patients now have their 3,072 required images. See [image recovery status and the experiment-branch handoff](IMAGE_RECOVERY_STATUS.md). The prompt and pre-transfer counts below are retained as provenance; another source-server search or packaging step is unnecessary for this cohort.

The source report's metadata and test09 split SHA-256 values exactly match the destination's configured files. The destination independently derives 135 eligible patients and 3,072 unique required filenames, of which 306 currently exist locally and 2,766 are missing. The reported source cohort is therefore suitable for transfer reconciliation without another search or the earlier pilot-request ZIP. Patient/image lists and payload hashes must still be verified on arrival.

Use only the corrected investigation at `/home/liuusa_tw/private_image_availability_20261008T0052Z/`. The earlier `...20261008T004944Z` investigation used the wrong eligibility filter and is excluded.

## Copy this prompt to the original-server agent

Please package the corrected image-availability investigation for transfer to Dartmouth. The destination has confirmed that your metadata and test09 split hashes match its actual inputs, and independently obtains the same 135-patient / 3,072-image cohort. The pilot-request ZIP is no longer needed for this transfer. Do not repeat the image search, modify scientific code/environments, run models/fitting, or use the preliminary investigation with the count-field bug.

1. Work only from `/home/liuusa_tw/private_image_availability_20261008T0052Z/`. Recheck every row in its `bundle_manifest.csv` against the corrected `bundle/images/` payload: safe relative path, exact filename, existence, byte size and SHA-256. Require 3,072 unique image records/files, zero missing files, zero hash/size mismatches and no duplicate target names. Check that `patient_coverage.csv` records exactly 135 distinct complete eligible patients and that their metadata image-list union equals the manifest's 3,072 canonical filenames. If a check fails, report the failure and stop packaging; preserve the existing reports/bundle.

2. Create a new private ZIP, refusing to overwrite an existing archive. Include these paths relative to the corrected investigation root:
   - `bundle/images/` with all 3,072 verified original JPEGs;
   - `bundle_manifest.csv`;
   - `image_availability.csv`;
   - `patient_coverage.csv`;
   - `working/metadata_hashes.json`;
   - the final search report Markdown/JSON and `working/aggregate.json`, if present.

   Preserve these relative paths, especially the manifest's `bundle_relative_path` convention. Do not flatten the archive, include the older preliminary investigation, retransmit model/result/environment bundles, or include credentials. Use a name such as `/home/liuusa_tw/private_image_availability_20261008T0052Z_transfer.zip`.

3. Check the finished ZIP's integrity, inventory and extracted payload hashes in a fresh temporary/private staging directory or by reading ZIP members. There must be exactly 3,072 image payload members matching the manifest, with no unsafe paths, symlink members, duplicate ZIP names or unexpected images. Do not include the archive or its own checksum inside itself. Preserve originals.

4. Write a separate `<archive>.sha256` file with the archive SHA-256. Return only the archive's absolute path, exact byte size, SHA-256, aggregate validation results and any necessary extraction notes. Do not perform a remote transfer automatically; I will transfer the archive and checksum privately.

## Destination receiving steps

Transfer the returned ZIP and checksum privately into this existing directory on Polaris:

```text
/dartfs-hpc/rc/home/c/f008hzc/projects/fatty-liver-explainable-ai/outputs/incoming/
```

This directory is ignored by Git. The ZIP can be downloaded from the original server to the user's computer and uploaded to Polaris using the existing remote file-transfer workflow; a direct server-to-server copy is optional and requires the actual source hostname/access configuration. Do not treat a report pasted into chat as the image payload.

On this image-recovery branch, verify the received archive SHA-256 against the separately reported value, inspect safe member paths, and extract to a new ignored staging directory. Compare metadata/split hashes, exact cohort IDs/image lists and every payload size/hash; decode the images. Before import, compare any existing destination image with the incoming version. Preserve identical files; stop and report differing hashes for the same canonical filename. Copy only absent verified images into the configured image directory, retaining the archive, manifests and a private import receipt. Do not overwrite existing images or historical results.

Then rerun the coverage audit and verify **135/135 eligible patients and all 3,072 cohort images**. The full 890-patient test09 strict audit can still report missing images outside this eligible cohort; its nonzero full-test exit status is distinct from the target cohort's completeness. Regenerate the pilot candidate list and report input readiness to the experiment branch. This recovery branch does not launch a pilot or alter the complete-patient branch's code, environment or running jobs.
