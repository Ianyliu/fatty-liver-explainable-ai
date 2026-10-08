# Image recovery complete: pilot inputs ready

**Experiment update:** the ten-patient pilot has passed, and the ten-patient, three-seed P0 comparison is submitted. See [cohort experiment status](COHORT_EXPERIMENTS.md) for job IDs, frozen plans and automatic summary details. The receiving record and original handoff below are retained as provenance.

The corrected original-server bundle has been verified and imported on Polaris. All **135 eligible positive test09 patients have all 3,072 required images**. This supersedes the earlier one-complete-patient coverage gate and 152-image upload request described in `NEXT_RUNS.md` and the original-server search prompts.

The separately uploaded checksum matches the 78,139,734-byte ZIP:

```text
80754e2639cd5b5be263aea992a01ea3b9bdb41208e306a8a3b90feb7956ea10
```

Verification matched source metadata/test-split hashes, exact patient/image lists, every image size and SHA-256, and JPEG decoding. All images are grayscale, 685 × 496 pixels. The importer added **2,766 absent images**, preserved **306 identical existing images**, and verified all 3,072 destination hashes afterward. No existing image was overwritten. Eight synthetic import regressions pass, including rejection of unsafe paths, corrupted payloads, wrong cohorts/checksums/metadata, and existing or newly introduced destination conflicts.

The archive, checksum, source reports, staged payload and import receipts remain in ignored private storage. Patient IDs and filenames are confined to those private artifacts.

## Handoff to the experiment branch

Readiness and provenance are recorded in:

- `outputs/reproducibility/image-transfer/readiness-80754e2639cd/readiness.json`
- `outputs/reproducibility/image-transfer/import-80754e2639cd/import_receipt.json`
- `outputs/reproducibility/image-transfer/post-import-audit/summary.json`

The refreshed **135-patient** candidate list is `outputs/reproducibility/image-transfer/post-import-audit/candidate_patients.csv`. A separate **ten-patient** pilot manifest is frozen at `outputs/reproducibility/image-transfer/readiness-80754e2639cd/pilot_patients.csv`, ordered by image count then patient ID. This operational convenience sample does not establish representative scientific sampling.

The full test split still has 12,174 missing image references **outside the eligible cohort**. Accordingly, the full-test `--strict` audit returns 1; the eligible-cohort coverage gate passes at 135/135. The audit invocation, stdout and exit status are retained beside its summary.

The experiment branch can now proceed with the bounded ten-patient Discovery diagnostic in [NEXT_RUNS.md](NEXT_RUNS.md), using the refreshed manifest:

```bash
sbatch --partition=gpu_preempt --gres=gpu:1 \
  --export=ALL,XAI_DEVICE=cuda:0,XAI_PATIENT_MANIFEST=outputs/reproducibility/image-transfer/readiness-80754e2639cd/pilot_patients.csv \
  slurm/pilot.sbatch
```

Submit from the repository root. The existing template runs ten tasks with at most two concurrently and 20 perturbations per patient. Review successful task reports and the completed n=1 methodology evidence before expanding experiments or launching the full subject/seed arrays. Image availability alone does not validate fidelity or adaptive sampling behavior.

**Preserve n=1 continuity:** newly complete patients can change the default automatically selected patient. Existing frozen study plans retain their patient; future smoke/study preparations that repeat the original patient should pass its explicit private `--patient` value.

At image-import completion, the recovery work had launched no jobs and changed no experiment code, environment, frozen study plans or historical results. Subsequent experiments are recorded separately in [cohort experiment status](COHORT_EXPERIMENTS.md). The import utility is `scripts/import_image_bundle.py`; it verifies by default and requires `--execute` to import absent images. Snapshots of the importer and its tests are retained beside the private readiness receipt. The existing environment and completed Discovery checks remain documented in `NEXT_RUNS.md`.
