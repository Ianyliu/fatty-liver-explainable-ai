# Full-cohort secondary experiments

Submitted October 8, 2026 at 16:49 Eastern, following explicit authorization to extend every completed ten-patient analysis to all eligible patients. This is an exploratory expansion after inspection of both pilot and primary results. No completed primary inference or pilot secondary task is resubmitted.

## Coverage and fixed methods

| Analysis | Reused pilot | New work | Complete evidence |
| --- | ---: | ---: | ---: |
| CPU surrogate and design analysis | 30 patient–seed records | 375 records: one passed locally, 374 Slurm tasks | 405 records, 135 patients × seeds 0/1/2 |
| Leave-one-image-out (LOO) | 10 patients, 210 queries | 125 patients, 2,997 queries | 135 patients, 3,207 queries |
| Independent Ridge/selected-Elastic-Net refits | 120 previously validated | 1,500 | 1,620 |
| Image-influence profiles | Existing pilot preserved | 135 patients × two sampling arms | 270 aligned four-panel profiles |

The CPU work revalidates original masks/responses and recomputes fixed-alpha Ridge, training-only five-fold Elastic Net selection, Pearson correlations, design conditioning, image inclusion and stage-II pool composition. It adds no GNN queries. Elastic Net preserves the pilot's 25 penalties, four mixing ratios, shuffled folds matched to each seed, tolerance and iteration limit. Independent selected-model refits check coefficients, held-out scores and recorded CV minima without tuning on evaluation responses.

LOO preserves the existing predictor, checkpoint, graph reconstruction, preprocessing and seed 0, using one full-set prediction and every single-image removal for each new patient. LOO is one baseline per patient, not three independent replications. Saved rankings provide three-seed stability, sign agreement, top-five overlap/frequency and ranking-versus-LOO association. Constant or otherwise undefined vectors remain unavailable.

## Execution and provenance

Execution root: `outputs/parallel_experiments/full-135-20261008-v1/`.

Frozen plan SHA-256: `a1b91d2c53318b369b4b58898ccb89faf5a968d63988501f13c552f024d2f880`.

Frozen execution commit: `e53b319f9dfa30cc5bbe216cfc33f99f6d20d7a8`.

The plan binds 135 child plans, full primary validation, pilot secondary evidence/review, exact settings/runtime versions, explicit identity-based reuse/task mappings, reporting/scheduler sources and the reviewed Figure 1–5 exports. Preparation verified 3,087 distinct frozen child inputs, including all 3,072 images. Source snapshots are retained. The plan rejects changed source/input hashes or settings. Existing inference code, environment, checkpoints and scientific plans are unchanged.

`submissions.json` contains the actual commands, resource requests, log paths, dependencies, job IDs, timestamps and plan/source commit. Local CPU task 000 passed, including four independent refits, before submission; it is excluded from the submitted CPU array.

| Stage | Job ID | Resources | Dependency |
| --- | --- | --- | --- |
| CPU analysis, tasks 1–374, concurrency 8 | 9563646 | standard, 2 CPUs, 4 GB, 20 min/task | None |
| GPU LOO, tasks 0–124, concurrency 2 | 9563647 | gpu_preempt, 1 GPU, 2 CPUs, 8 GB, 20 min/task | None |
| Complete-cohort validation and aggregates | 9563655 | standard, 2 CPUs, 8 GB, 1 hour | afterok:9563646:9563647 |
| Tables, figures and patient profiles | 9563656 | standard, 2 CPUs, 8 GB, 1 hour | afterok:9563655 |
| Existing manuscript integration | 9563657 | standard, 2 CPUs, 8 GB, 30 min | afterok:9563656 |

At initial post-submission inspection both arrays had started; the first two GPU LOO records and eleven CPU records (including the local preflight) passed. This is a startup check, not completed-cohort validation. Inspect live scheduler/evidence status rather than treating this snapshot as final.

```bash
squeue -j 9563646,9563647,9563655,9563656,9563657
sacct -j 9563646,9563647,9563655,9563656,9563657 \
  --format=JobID,State,ExitCode,Elapsed
```

Scheduler completion alone is insufficient. The summary requires all 405 CPU records, all 135 LOO baselines, exact seed/image identity, input/output hashes, 1,500 new refits and all 3,207 LOO calls. It cross-checks Ridge MAE against the already validated primary study and full-set LOO predictions against primary seed 0. No missing patient, seed or undefined metric can silently create a complete-cohort mean. All three seeds/pairs must be defined before patient averaging; summaries weight patients equally. Available-case diagnostics report their counts separately.

Any failed/preempted task blocks the dependent jobs. Inspect its preserved report/log and document the problem before a new attempt. Do not blindly rerun arrays or change the frozen methods. The primary study remains validated separately: array 9558368 and summary 9558371 completed; historical manuscript job 9560474 failed its source guard and was superseded by the existing local full manuscript package.

## Generated outputs and deterministic commands

After validation, `summary/analysis.json` and ten CSVs preserve fidelity, rankings, conditioning, inclusion, pool composition, seed stability/top-five frequencies, LOO and agreement. `report/` contains generated numerical tables, patient-level aggregates, evidence ledger/captions, three quantitative figure sets (PDF/SVG/300-dpi PNG), a contact sheet and 270 patient profiles with corresponding CSVs. Clinical thumbnails remain restricted to private review; no publication permission is inferred.

The report requires the existing `.venv` with its frozen NumPy, SciPy, scikit-learn, pandas, Matplotlib, Pillow and threadpoolctl dependencies. No R packages or new environment installation are needed.

The following are the same commands used by the dependent jobs. Run only if the matching job has not already produced its output; their output directories refuse overwrite.

```bash
.venv/bin/python scripts/full_secondary_study.py summarize \
  --plan outputs/parallel_experiments/full-135-20261008-v1/plan.json
.venv/bin/python scripts/report_full_secondary.py \
  --plan outputs/parallel_experiments/full-135-20261008-v1/plan.json
.venv/bin/python scripts/integrate_full_secondary.py \
  --plan outputs/parallel_experiments/full-135-20261008-v1/plan.json \
  --manuscript outputs/proceedings_2026/full-submission-prep-20261008-v1
```

## Preserve the user's manuscript edits

The target remains **`outputs/proceedings_2026/full-submission-prep-20261008-v1`**, as explicitly requested. No replacement manuscript directory is created. Separate experiment outputs do not require moving the user's edits.

Integration snapshots the current user-edited `latex/` directory and prior PDFs into `history/full-secondary-TIMESTAMP/` within that same package. It stages a build, changes only result/scope statements, generated macros/tables and new quantitative exports, and records `minimal_text_changes.diff`. The original README workflow and original result-chart composite (Figures 1 and 2) must remain byte-identical. Other user-written paragraphs, abstract wording, main source and bibliography are preserved. A concurrent textual edit or ambiguous replacement stops installation and retains the staged update for manual reconciliation.

The full-cohort secondary diagnostics replace the pilot-only secondary table/heatmap with explicit full-population captions and counts. Supplemental Elastic Net fidelity and LOO figures are added. The pilot's original evidence, example, interpretation and pre-integration package remain preserved. Abstract changes are limited to the generated analysis-population sentence; current user-edited abstract prose is not replaced.

The updater compiles the user's staged source, checks citations, glyphs, overflows, PDF size/security/fonts and private patient IDs, installs the resulting PDFs into the same package, refreshes the evidence ledger/contact sheet, and renders pages. `full_secondary_integration.json` records provenance and before/after source hashes. Final visual page review remains a separate gate; automated QA does not claim that human review happened. If an in-progress textual edit fails compile/anchor checks, the existing manuscript stays intact.

## Verification and limits

Seven focused tests pass for identity-based reuse, relative/absolute plan paths, changed-evidence rejection, complete LOO coverage/signs, missing/duplicate seeds, independent coefficient refits and protected text replacements. Python compilation, all five Slurm syntax checks and uniqueness checks against the actual edited manuscript passed before freezing. Both pilot-based influence-profile previews were rendered; the random-arm preview was visually inspected and its thumbnail-axis defect corrected before submission.

Completed primary sampling, Ridge fidelity and deletion experiments are not rerun. Matched-query budgets, new sampling ratios, new-method multi-node deletion, repeated random deletion orders, bootstrap inference, physician evaluation and clustering analyses were not completed in the pilot and are outside this submission. No public release, paper submission or Git push is authorized. Author declarations, image/secondary-use permissions, final manuscript/figure review and explicit release approval remain required.
