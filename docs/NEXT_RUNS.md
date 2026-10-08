# From the Polaris smoke test to Discovery

**Current execution:** image recovery is complete for all 135 eligible patients (3,072 images). Ten-patient GPU pilot `9557618` passed all tasks and independent numerical checks. The ten-patient, three-seed P0 array `9557642` and dependent summary job `9557645` are submitted. See [cohort experiment status](COHORT_EXPERIMENTS.md) for frozen plans, output paths and the next review gate. Environment checks and all 39 regression tests pass.

The complete-patient branch has now **completed** the separate [n=1 study workflow](COMPLETE_PATIENT_STUDY.md): random versus adaptive training with 1,000 draws per arm, 200 shared evaluation draws per seed, signed-coefficient deletion controls and seeds 0, 1 and 2. Validation job `9557483_0` and all three full array tasks `9557485_[0-2]` completed with exit `0:0`. All 6,741 model calls and numerical results passed independent CSV checks; the private report/figures are in `outputs/complete_patient/study-20261007-v1/summary/report.md`. The environment and 23 regression tests pass. Adaptive had higher unseen-mask MAE in all three seeds and missed its 50/50 class target; both strategies' coefficient-ranked deletion curves had lower AUC than seeded random controls. This is a one-patient result; cohort expansion retains the completeness gate below.

As of October 7, 2026, the existing Python 3.9.25 environment passes offline lock/synchronization checks, dependency compatibility, all active imports, and synthetic Ridge and compiled glmnet fits. Both random and repaired adaptive one-patient GPU smoke jobs now pass on Discovery. Run `bash scripts/verify_env.sh` to repeat the environment checks. The project gives ordinary `uv` commands a writable `.uv/cache` default; `UV_CACHE_DIR` still overrides it. `scripts/setup_env.sh` retains its scratch-cache configuration for downloads but now installs Python under shared `.uv/python/`. A fresh installation has not been repeated. The auxiliary `usflc_xai.utils` remains broken by its obsolete `training` import; the inference workflow does not use it.

In Codex's restricted process sandbox, `uv run` was observed to leave its finished Python child as a zombie and hang. A one-line `uv run` and the complete environment verification both exit successfully outside that sandbox on Polaris. Use direct `.venv/bin/python` inside the restricted runner; `verify_env.sh` does this after checking the environment with `uv`. This runner issue is distinct from the dependencies and from normal host execution.

The transferred materials already have useful separate locations: metadata/splits/images in `data/`, graph and DenseNet weights in `checkpoints/`, the supplied source in ignored `usflc_xai/`, and historical runs/bundles in `outputs/original/`. Existing source snapshots and notebook backups remain in ignored storage. No bulk moves or deletions were needed. Paths are configured through `.env`/`project_paths.py`, including the normalized `TORCH_HOME` encoder cache. The external source is still an unversioned upload and must accompany a fresh clone.

## Repeat the bounded smoke test

From the repository root:

```bash
bash scripts/verify_env.sh
.uv/bin/uv run --frozen --offline python scripts/smoke_patient.py \
  --device cpu --samples 20 --seed 42 \
  --output outputs/smoke/my-new-run
```

The smoke harness selects a complete positive test-split patient with at least 20 images, ordered by image count and then patient ID. `--patient` validates an explicit test patient instead. It uses the uploaded `single_data_loader` for each full/subset graph, including the original grayscale/resize/normalize preprocessing and feature-correlation adjacency. It loads the cached DenseNet121 weights and the supplied SETNET_GAT checkpoint, with strict state-dictionary matching and CPU-compatible `map_location`. It requires local weights before model construction.

Twenty random perturbations are divided into 16 training and four holdout rows. The single fixed-alpha Ridge fit targets the GNN's class-1 probability. This is a bounded diagnostic, with no bootstrap, CV tuning, significance estimates or historical coefficient reproduction. The harness caps perturbations at 20; increasing the production sample count requires a separate workflow. Add `--sampling adaptive` to exercise the repaired two-stage sampler with a 0.5 class-1 target. Realized class balance is recorded explicitly rather than guaranteed.

Private `report.json`, `pred_results.csv`, and `ridge_coefficients.csv` record input/source hashes, package versions, explicit seed/device/settings, full-patient logits, graph sizes, masks, probabilities, holdout diagnostics and limitations. Smoke prediction tables include additional diagnostic columns and must not be passed directly to the historical Ridge/Elastic-Net runners. Output directories must be new and separate from reference predictions/data/checkpoints. The JSON status records failed runs too; the presence of a directory alone is not success.

The initial Polaris CPU run completed in 23.2 seconds, including 10.5 seconds for the original prediction plus 20 perturbations. The graph checkpoint matched strictly; the full graph had 20 nodes and 16 directed correlation edges. All 20 masks were distinct, and all predicted class 1, with probabilities approximately 0.837–1.000. Holdout probability MAE was 0.0236 on just four rows. The one-class result makes class agreement uninformative and cannot validate a class-balanced experiment. Detailed artifacts remain under `outputs/smoke/polaris-cpu-20261007/`.

Optionally add `--reference-predictions <patient/pred_results.csv>` to compare up to eight saved masks (first four rows per saved prediction class); repeat this option to check separate runs. Comparisons are diagnostic and do not refit historical explanations or establish which checkpoint/source produced a historical run.

The saved-mask verification also checked eight saved July masks and eight saved November masks, with both saved classes represented in each comparison. All 16 current predictions matched the stored labels. This confirms that the current loader/checkpoint can reproduce these selected predictions, while the 20 random smoke masks still happen to cover only class 1. That run took 26.8 seconds and is recorded separately in `outputs/smoke/polaris-cpu-20261007-verified/report.json`. A final direct-Python invocation exited with status zero and reproduced its masks, predictions and Ridge diagnostics exactly; the final artifacts are in `outputs/smoke/polaris-cpu-20261007-final/`.

## Discovery smoke job

Dartmouth documents Polaris/Andes as interactive development systems and Discovery compute jobs under Slurm; available GPU partitions and access tiers are listed in [Discovery cluster details](https://rc.dartmouth.edu/hpc/discovery-cluster-details/). Select account, partition and GPU resources according to your allocation and the current [Slurm documentation](https://slurm.schedmd.com/sbatch.html). Those site/account choices are not embedded in the templates.

**Submit directly from Polaris:** its Slurm clients reach Discovery's scheduler. SSH from Polaris to Discovery is unnecessary for these jobs. Your locally verified SSH connection can remain as it is. The separate SSH attempt from this environment still lacks a trusted host key, but that is no longer a computation gate.

Use the shared repository on Polaris, without another clone or upload:

```bash
cd /dartfs-hpc/rc/home/c/f008hzc/projects/fatty-liver-explainable-ai
test -x .venv/bin/python
.venv/bin/python --version
mkdir -p logs
# These options worked for the current free account:
sbatch --partition=gpu_preempt --gres=gpu:1 \
  --export=ALL,XAI_DEVICE=cuda:0 slurm/smoke_patient.sbatch
# Optional repaired adaptive smoke:
sbatch --partition=gpu_preempt --gres=gpu:1 \
  --export=ALL,XAI_DEVICE=cuda:0 slurm/smoke_patient.sbatch --sampling adaptive
```

The smoke template defaults to CPU. For a GPU run, request one GPU with the resource options supported by your partition and pass `--export=ALL,XAI_DEVICE=cuda:0` to `sbatch`. Inside the allocation the harness uses visible device 0. `gpu_preempt` can preempt jobs; use a fresh job/output directory for another smoke attempt if interrupted. Account/partition availability can change; the commands above record the configuration actually tested.

The first submitted job, `9557439`, failed with exit code 127 because `.venv/bin/python` pointed to Polaris-local `/scratch`. The identical Python 3.9.25 installation was copied to shared `.uv/python/cpython-3.9.25-linux-x86_64-gnu/`; the Python binary SHA-256 matches, and only the venv interpreter link/home changed. Packages and lockfile stayed unchanged. Job `9557444` then completed the random GPU smoke; job `9557446` completed the adaptive GPU smoke. Both have Slurm `COMPLETED`, exit code `0:0`, on `gv01`. Avoid synchronizing or relinking the environment while jobs use it.

The resource request is two CPUs, one GPU, 8 GB RAM and 15 minutes. Slurm elapsed times were 44 seconds for random sampling and 34 seconds for adaptive sampling; these include startup and are not a benchmark comparison. The random smoke used 21 model calls (one original plus 20 perturbations). Adaptive used 42 (one original, one constructor validation, 20 singleton probes and 20 perturbations). The adaptive realization was 1 negative / 19 positive, with 20 unique masks; it explicitly records `class_balance_reached=false`. CPU/GPU seeded masks and labels agree for both checked runs; maximum probability differences were about 1.4e-6 (random) and 3.5e-6 (adaptive).

Private outputs are `outputs/smoke/slurm-9557444/` and `outputs/smoke/slurm-9557446/`. Logs, scheduler receipts, interpreter relocation evidence, CPU/GPU comparison summaries and exact executed source snapshots are preserved under ignored `logs/` and `outputs/reproducibility/next-run-readiness/`.

## Ten-patient pilot and production gates

The post-import audit finds 890 test patients and **135/135 complete eligible positive patients**, with all 3,072 cohort images verified. Its 12,174 remaining missing test-image references are outside this eligible cohort. The former 152-image pilot request is superseded by the completed [image recovery](IMAGE_RECOVERY_STATUS.md).

Historical private upload lists are retained for provenance and are no longer an input gate:

- `outputs/reproducibility/next-run-readiness/pilot_upload_patients.csv`: ten selected patients and missing-image counts.
- `outputs/reproducibility/next-run-readiness/pilot_upload_images.csv`: exact patient/image IDs and 152 expected filenames; restore those JPEGs into the configured image directory.
- `outputs/reproducibility/next-run-readiness/pilot_upload_summary.json`: counts and selection rule.

The original-server investigation and transfer are complete. The refreshed 135-patient list is `outputs/reproducibility/image-transfer/post-import-audit/candidate_patients.csv`; the frozen ten-patient pilot list is `outputs/reproducibility/image-transfer/readiness-80754e2639cd/pilot_patients.csv`. Selection is by image count and patient ID, so this remains a convenience cohort.

The one-patient methods study is complete and preserved. The ten-patient pipeline pilot now also passes: all ten tasks in array `9557618` completed with exit `0:0`, and all 210 model calls and diagnostic Ridge fits were independently checked. The follow-on P0 array `9557642` uses these same ten patients and seeds 0, 1 and 2. [Cohort experiment status](COHORT_EXPERIMENTS.md) describes its budgets, source/input locks and automatic summary job.

To repeat a coverage audit, choose a new report directory; its candidate CSV contains private patient IDs:

```bash
.venv/bin/python scripts/audit_reproducibility.py --strict \
  --report-dir outputs/reproducibility/pilot-readiness
```

Strict full-test coverage returns nonzero for images outside the eligible cohort. The 135-patient eligible-cohort gate passes. The audit script uses repository-default input paths; supply `--metadata`, `--split`, and `--images` explicitly if your `.env` uses other locations.

The pilot is already complete. To repeat it as a new scheduler attempt:

```bash
sbatch --partition=gpu_preempt --gres=gpu:1 \
  --export=ALL,XAI_DEVICE=cuda:0,XAI_PATIENT_MANIFEST=outputs/reproducibility/image-transfer/readiness-80754e2639cd/pilot_patients.csv \
  slurm/pilot.sbatch
```

The array uses ten patients, caps concurrent tasks at two, and runs the same 20-perturbation diagnostic for each. Every task verifies the ten-patient completeness/cohort gate before loading models. Outputs are separated by array job and task. The former one-row manifest remains deliberately rejected; use the refreshed ten-patient manifest above.

The sampler repairs now have nine focused regressions, alongside four patient-input tests. Image-to-column mapping round-trips; NumPy binary labels normalize to Python integers; negative-quota-first and single-pool cases return the full requested batch; scarce pools reallocate deficits to preserve subset sizes; seeds and append/reset behavior are explicit; invalid bounds/budgets fail before sampling inference. Positive/negative pools now mean predicted disease class 1/0 for either ground-truth label. For positive-ground-truth patients this preserves the original pool meaning; negative-ground-truth behavior intentionally changes. Duplicate masks are permitted and reported, rather than claiming uniqueness. `max_iter` must cover the requested new perturbation count; singleton probes and constructor validation are separate overhead. The all-subject loader now maps checkpoint tensors to its chosen device and normalizes the binary subject label.

These are deliberate changes for future sampling, not a reconstruction of historical sampling. Historical correlation formulas, fitters, uncertainty estimates and saved artifacts remain untouched. The legacy all-subject entry point still has unrelated unsafe defaults and should not replace the bounded smoke/pilot workflow.

The prospective [protocol](P0_EXPERIMENT_PROTOCOL.md) now defines the evaluation budget, overlap handling, shared novel-mask fidelity and deletion interventions used by the [complete-patient runner](COMPLETE_PATIENT_STUDY.md). The active generator already defaults to 1,000 samples; the 10,000 `samples_per_subj` setting belongs to the legacy clustering sampler. Keep perturbation counts separate from the historical runners' 10,000 bootstrap fits. The n=1 runner is separate from a future cohort-scale production entry point.

Before expanding to all 135 patients, review the complete ten-patient P0 evidence, including failed/preempted tasks, weak class support, constant-baseline performance, achieved adaptive balance and deletion controls. Image completeness and the bounded pipeline pilot now pass; the submitted ten-patient P0 comparison is the next evidence gate. Reconcile source/checkpoint/runtime evidence separately if agreement with historical predictions or explanations is required.
