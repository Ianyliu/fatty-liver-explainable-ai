# Complete-patient methods study

This workflow executes the available complete positive test09 patient with at least 20 images. It is an n=1 methods study; seeds, perturbations and image nodes are repeated measurements of that patient. Missing images do not prevent these runs, but still prevent the ten-patient pilot and the planned 135-patient comparison.

The prospective [protocol](P0_EXPERIMENT_PROTOCOL.md) specifies 1,000 training perturbations for each of random and repaired adaptive sampling, shared evaluation, fixed probability-Ridge fits, signed-coefficient deletion controls and seeds 0, 1 and 2. This workflow does not invoke the historical classifier/bootstrap runners. It retains historical artifacts and their source/provenance uncertainty.

## Prepare and execute

From the repository root, after `bash scripts/verify_env.sh`:

```bash
# Choose a new directory; preparation and runs refuse existing output directories.
.venv/bin/python scripts/patient_study.py prepare \
  --output outputs/complete_patient/my-study \
  --train-samples 1000 --eval-samples 200 --seeds 0 1 2

# Polaris can submit directly to Discovery; no separate SSH connection is needed.
mkdir -p logs
sbatch --parsable --partition=gpu_preempt --gres=gpu:1 \
  --export=ALL,XAI_STUDY_PLAN=outputs/complete_patient/my-study/plan.json,XAI_DEVICE=cuda:0 \
  slurm/patient_study.sbatch
```

The template requests two CPUs, 8 GB RAM and 30 minutes per task, with array indices/seeds 0, 1 and 2 and a concurrency limit of two. Pass cluster account/partition/resource choices supported by your allocation. `gpu_preempt` can interrupt jobs. A failed or preempted task remains a failed study attempt; inspect its status and logs before preparing a new study. Do not overwrite an output or drop an unfavorable seed. The summary requires every planned seed to pass.

For a smaller runner validation, prepare a separate plan with `--train-samples 40 --eval-samples 20 --seeds 0` and submit with `sbatch --array=0` plus the same GPU/export options. To execute on CPU, use `.venv/bin/python scripts/patient_study.py run --plan <plan.json> --seed 0 --device cpu`; repeat for all planned seeds. Prefer Discovery for the full study.

Preparation selects the smallest complete eligible patient, or validates `--patient`. It hashes metadata, split, all patient images, graph/encoder weights, scientific source, protocol, lockfile and Slurm template. It copies scientific source snapshots before comparative inference. Runs reject changed source/inputs or package versions. Inference uses the frozen image/checkpoint/cache paths even if `.env` later changes. Keep those sources and the environment stable while tasks are running. A changed implementation requires a newly prepared study directory.

## Evaluation and query accounting

Each mask removes image nodes; the uploaded loader re-encodes retained images and rebuilds the correlation graph. Models run in evaluation/inference mode. DenseNet weights come from the manifested local cache, and the GAT checkpoint loads strictly. No feature or prediction cache approximates subset inference.

The surrogate target is class-1 softmax probability, using unweighted `Ridge(alpha=1.0)` with an intercept. Both strategies use 1,000 training rows; duplicates remain and consume fresh inference calls. Adaptive sampling additionally validates the full graph and probes all singleton images. A 0.5 predicted-class target is requested and its achieved counts are recorded; it is not guaranteed.

Two hundred evaluation draws use an independent RNG stream and are queried once for both arms after fitting. Independent draws can overlap training. Primary fidelity uses exactly the shared evaluation draws whose masks appeared in neither training arm. Both arms are compared on this common subset; raw all-draw metrics are also retained. Filtering changes the evaluation distribution and can remove large subsets, so every report records overlap, novel row counts, class support and subset-size histograms. Draw multiplicity is retained. Each arm's training-mean probability is a constant baseline; no evaluation response fits that baseline. Raw Ridge scores remain unclipped, including scores outside [0,1].

Deletion removes 0%, 10%, 20%, 30%, 40% and 50% of nodes, using floored counts and retaining at least three images. Descending signed coefficients are the main ranking, with ascending and seeded random controls. Image IDs break ties. The random control is shared across strategies, as is the original zero-deletion prediction. AUC uses actual deleted fractions. A nonmonotonic curve or weak ranking remains part of the result.

For the 20-image patient, a full seed uses exactly 2,247 GNN calls: original 1, random training 1,000, adaptive validation 1, adaptive singletons 20, adaptive training 1,000, shared evaluation 200, shared random deletion 5, and ranked deletion 20. Three seeds use 6,741 calls. This compares equal training rows with unequal sampler overhead; it does not establish computational efficiency.

## Validate and summarize

After all tasks finish:

```bash
.venv/bin/python scripts/summarize_patient_study.py \
  --plan outputs/complete_patient/my-study/plan.json
```

The summary checks all planned statuses and plan hashes, binary masks, query-stage budgets, training/evaluation response alignment, overlap flags, coefficient-based predictions, metric calculations and deletion orders/AUC against the raw CSVs. It rejects inconsistent evidence. Its new `summary/` directory contains a readable report, numerical summary tables, provenance, figure captions and standalone PNG/SVG figures. Summary generation also refuses to overwrite an existing directory; use `--output <study>/summary-v2` for a new export.

Private artifacts below the study directory are:

- `plan.json`, `source_snapshot/`: frozen settings, source and scientific input fingerprints.
- `runs/seed-*/report.json`: status, metrics, model/runtime evidence, sampler diagnostics and timing.
- `runs/seed-*/queries.csv`: every model call with its stage, mask and response, including validation/singletons.
- `runs/seed-*/{random,adaptive}_training.csv` and coefficient CSVs: fitted evidence and image-column mapping.
- `runs/seed-*/evaluation.csv`, `deletion.csv`: shared evaluation and each ranked/control trajectory.
- `summary/report.md`, `fidelity.csv`, `deletion_summary.csv`, `analysis.json`, figures and artifact hashes. An exploratory coefficient heatmap and rank-stability table use the existing fits; their addition after protocol freeze is explicitly labeled in the report.

All patient identifiers, images and numerical output remain in ignored storage. The tracked workflow and tests use synthetic patient/image IDs. Regression tests run with `MPLCONFIGDIR=outputs/.matplotlib .venv/bin/python -m unittest discover -s tests -v`.

## Execution record

The 40-training/20-evaluation validation job `9557483_0` completed successfully on Discovery with exit code `0:0`, followed by independent CSV validation and figure export. Its separate artifacts are in `outputs/complete_patient/validation-20261007-v1/`.

The full input/source-locked plan is `outputs/complete_patient/study-20261007-v1/plan.json`. Array tasks `9557485_0`, `9557485_1` and `9557485_2` all finished `COMPLETED`, exit `0:0`, in 4:17, 4:17 and 3:54 respectively. Seeds 0/2 used an RTX A5000; seed 1 used a V100. Both strategies within each seed shared the same allocation. Slurm's internal numeric job IDs in the reports differ from its displayed array task names. The scheduler completion receipt is saved at the study root.

Independent table validation passed for all seeds, confirming exactly **6,741 GNN calls**. The final private report and standalone figures are in `outputs/complete_patient/study-20261007-v1/summary/report.md`. All four exported PNG figures were visually inspected; SVG versions, captions, numerical tables, source snapshots and hashes accompany them. Environment/dependency/numerical checks and 23 regression tests pass.

| Seed | Shared novel evaluation draws | Random Ridge MAE | Adaptive Ridge MAE | Adaptive training negative / positive |
| --- | --- | --- | --- | --- |
| 0 | 174 | 0.036745 | 0.080511 | 84 / 916 |
| 1 | 176 | 0.037393 | 0.084610 | 84 / 916 |
| 2 | 177 | 0.035150 | 0.090650 | 114 / 886 |

Adaptive had higher shared-novel probability MAE in all three seeds; the mean paired increase was 0.048828. It obtained more negative training predictions than random sampling, but never reached the 50/50 target. Random Ridge improved on its own constant training-mean baseline in only one of three seeds, so its lower MAE does not establish strong surrogate fidelity. The novel evaluation sets contained just 2, 1 and 3 negative GNN labels respectively; class agreement alone is weak evidence here. The common novel filter excluded 26, 24 and 23 evaluation draws that overlapped at least one training arm.

Both strategies' descending-coefficient deletion curves had lower AUC than the seeded random control in all three seeds. At 50% deletion, the original probability of approximately 0.990 fell to 0.470–0.515 for random-trained rankings and 0.496–0.654 for adaptive-trained rankings; random-control probabilities remained 0.976–0.993. These curves evaluate the GNN intervention directly and do not contradict weak probability-surrogate fidelity. Exploratory coefficient rankings were stable within each strategy: Spearman correlations across seed pairs were 0.962–0.979 for random and 0.904–0.955 for adaptive, with the same top-five image set within each strategy across all three seeds.

These are descriptive outcomes for one available patient under the frozen prospective protocol. They do not establish patient-level generalization, clinical image importance, historical explanation reproduction or an equal-compute comparison. The completed n=1 workflow requires no additional images; the ten-patient and 135-patient branches still do.
