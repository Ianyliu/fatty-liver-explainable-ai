# Ten-patient prospective experiments

Additional CPU design/surrogate/stability analyses and an independent GPU leave-one-image-out baseline now run alongside this study. See [parallel experiment status](PARALLEL_EXPERIMENTS.md) for their separate plans, dependencies and job IDs.

The image coverage gate and ten-patient GPU pipeline pilot have passed. The prospective P0 random-versus-adaptive and deletion/fidelity experiments have been submitted to Discovery. This is a convenience cohort of ten patients selected before comparative inference, ordered by image count then patient ID. All ten have 20 images. It does not establish representative patient-level generalization.

## Execution record

On October 7, 2026:

- Pilot array **9557618**: all ten tasks completed with exit `0:0`; 210 model calls. Independent validation matched input/source hashes, masks, logits/probabilities/labels, Ridge refits, coefficients and holdout MAE.
- P0 array **9557642**: 30 patient/seed tasks submitted, at most two concurrent GPUs. Each patient uses seeds 0, 1 and 2, with 1,000 training perturbations per arm and 200 shared evaluation draws per seed. Sampling fidelity and signed-coefficient deletion/control curves run together. Planned total: **67,410 model calls**.
- CPU summary job **9557645**: depends on successful completion of the entire P0 array (`afterok:9557642`). It independently validates every raw patient/seed table before writing the cohort report.

The environment checks pass, including offline uv lock/sync checks, package compatibility, active imports, and synthetic Ridge/glmnet fits. All **39 regression tests** pass, including an end-to-end synthetic 30-run summary and corruption rejection. Bash syntax checks pass for both new Slurm templates. The existing unused `usflc_xai.utils` import problem remains documented in `NEXT_RUNS.md`.

Submission is distinct from completion; inspect the scheduler and run reports before interpreting results:

```bash
squeue -j 9557642,9557645
sacct -j 9557642,9557645 --format=JobID,State,ExitCode,Elapsed
```

Private pilot evidence is under `outputs/reproducibility/pilot-20261007-v1/`, including `independent_validation.json`, the scheduler receipt, input hashes, source snapshots and regression logs. Pilot inference outputs are under `outputs/pilot/slurm-9557618/`.

## Frozen plans and output

The cohort root is `outputs/complete_patient/cohort-20261007-v1/`:

- `cohort_plan.json`: frozen ten-patient selection, 30-task mapping, settings, budgets and orchestration hashes.
- `patient-00/` through `patient-09/`: separate frozen scientific plans, source snapshots and `runs/seed-{0,1,2}/` evidence.
- `cohort_source_snapshot/`: coordinator, validator and Slurm template snapshots.
- `submission.json`, `summary_submission.json`: scheduler submission receipts.
- `cohort_summary/`: generated only after all 30 runs pass independent validation; includes `report.md`, `analysis.json`, seed/patient fidelity and deletion tables, and runtime/class-balance diagnostics.

Keep frozen scientific sources, metadata, images, weights and the environment stable while jobs run. The coordinator rejects changed plans, sources or pilot reports; the patient runner checks manifested scientific inputs and package versions. Outputs refuse overwrite. Failed or preempted tasks require inspection and a new documented attempt, rather than silently dropping a patient or seed. If any task fails, the dependent summary will not run successfully; inspect its dependency state as well.

The existing [patient-study runner](COMPLETE_PATIENT_STUDY.md) executes each task without a scientific implementation change. Each child remains an n=1 study; the separate coordinator aggregates across ten patients. The original n=1 results, plans, protocol text and historical artifacts remain preserved. References to missing images in those frozen documents describe the earlier state; [image recovery status](IMAGE_RECOVERY_STATUS.md) records its resolution.

## Analysis and next gate

The comparison follows the frozen [P0 protocol](P0_EXPERIMENT_PROTOCOL.md): probability Ridge with alpha 1, shared evaluation draws, common novel-mask primary fidelity, training-mean baselines, and descending/ascending/random deletion controls. Adaptive's achieved class balance, duplicate masks, overlap, class support and inference overhead remain explicit. The earlier one-patient study's unfavorable adaptive fidelity and weak evaluation class support do not prevent measuring the same prespecified comparison across these patients.

For primary cohort fidelity, average paired adaptive-minus-random MAE differences across seeds within each patient, then weight patients equally. An unavailable novel-mask result remains unavailable; the complete-cohort primary mean is withheld if any patient lacks a planned seed pair. The available-patient mean is separately labeled. Deletion AUCs are likewise averaged within patient before reporting cohort means. No perturbation-level population significance or equal-compute claim is made.

Review complete pilot/P0 statuses, failures, class support, baseline performance, achieved adaptive balance and deletion controls before preparing the full 135-patient arrays, bootstrap significance or ablations. The current launch covers ten patients and three seeds.

## Repeat with a new output directory

```bash
.venv/bin/python scripts/cohort_study.py prepare \
  --manifest outputs/reproducibility/image-transfer/readiness-80754e2639cd/pilot_patients.csv \
  --pilot outputs/pilot/slurm-9557618 \
  --output outputs/complete_patient/cohort-NEW

sbatch --parsable --partition=gpu_preempt --gres=gpu:1 \
  --export=ALL,XAI_DEVICE=cuda:0,XAI_COHORT_PLAN=outputs/complete_patient/cohort-NEW/cohort_plan.json \
  slurm/cohort_study.sbatch

# Replace ARRAY_JOB_ID with the returned job number.
sbatch --parsable --partition=standard --dependency=afterok:ARRAY_JOB_ID \
  --export=ALL,XAI_COHORT_PLAN=outputs/complete_patient/cohort-NEW/cohort_plan.json \
  slurm/cohort_summary.sbatch
```

For manual validation after every seed finishes, run `.venv/bin/python scripts/cohort_study.py summarize --cohort <cohort_plan.json>`. The summary directory must not already exist. Per-patient figures remain available through `scripts/summarize_patient_study.py --plan <patient-NN/plan.json>`.
