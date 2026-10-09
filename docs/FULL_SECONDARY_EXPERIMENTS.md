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


## Recovery update — October 8 evening (current submission)

All original experiments completed: 374 scheduled CPU tasks plus the local CPU task, and all 125 GPU LOO tasks. Scheduler accounting confirms **499/499 allocations completed with exit 0:0**; artifact reports contain all **405 CPU records and 135 LOO baselines**, including the pilot. No missing CPU experiment requires resubmission.

Validation job **9563655 failed**, rather than passing scientific validation. Its strict cross-study check found 38 full-set LOO probabilities outside the original 1e-6 absolute/relative tolerance when compared with primary seed 0. All 38 used different GPU models between executions. All 135 full-set predicted classes match, but one patient's full-graph edge count also differs. Maximum absolute probability difference is 0.00880664587020874. This is a substantive reconciliation issue, not a reason to weaken the tolerance. Hardware dependence is an observed association at this stage; same-model retries must demonstrate reconciliation.

Blocked figure/manuscript jobs **9563656/9563657** were cancelled. The first exact-node recovery (`recovery-20261008-v1`, preflight 9565576 and dependent 9565602–9565613) was entirely queued behind occupied nodes and was cancelled **before any inference**. Its receipts, scheduler snapshot, original mismatch audit and source snapshot are retained. It is superseded, with zero recovery queries consumed.

The active replacement is **`outputs/parallel_experiments/full-135-20261008-v1/recovery-20261008-v2/`**. Every retry must use the same GPU model as that patient's primary seed-0 execution, on an explicitly eligible node. RTX A5000/A5500 use typed GPU requests; V100 excludes the PCIe-only node; H200 requests exclude all non-H200 nodes. The runner checks the actual CUDA GPU name and eligible hostname, then verifies full-set class, probability (unchanged tolerance) **and edge count before querying any image removals**. Any mismatch stops the task and preserves its report. Original inference/model/checkpoint/source hashes remain unchanged.

The four arrays contain exactly 38 unique retry patients: 12 A5000, 7 V100-SXM2, 7 A5500 and 12 H200 NVL. Two dependency lanes and per-array concurrency one limit active recovery to at most two GPUs. All 405 CPU analyses and the other 97 LOO baselines are reused. One of the 38 retries is a pilot patient; the selected full evaluation therefore retains nine original pilot LOO baselines. Original pilot/full artifacts are never overwritten.

| Active stage | Job | Work/dependency |
| --- | --- | --- |
| RTX A5000 recovery | 9565638 | 12 targeted patients |
| V100-SXM2 recovery | 9565639 | 7 targeted patients |
| RTX A5500 recovery | 9565640 | 7 targeted patients; afterok:9565639 |
| H200 NVL recovery | 9565641 | 12 targeted patients; afterok:9565638 |
| Complete-cohort validation | 9565642 | afterok of all four retry arrays |
| Figures, numerical tables and 270 profiles | 9565643 | afterok:9565642 |
| In-place manuscript integration | 9565644 | afterok:9565643 |

The separate recovery plan binds all original completed record reports, the original frozen plan and separately versioned recovery sources. It supplies a strictly checked replacement mapping to the unchanged original summary/report implementations; no probability/graph validation, independent refit, raw-artifact or complete-cohort check is bypassed. Three focused recovery tests pass, including an edge-only mismatch and rejection of same-model probability disagreement before any removal calls. Python compilation, scheduler syntax and current manuscript anchor checks pass.

Recovery adds **956** model calls if all tasks complete. Selected analysis still contains **3,207** full-set/removal calls; total executed LOO calls become **4,163**, including **210** historical pilot calls and **3,953** new calls across original/recovery executions. Selected and executed counts are reported separately. All unused first-pass LOO artifacts remain preserved. Original seed runs used heterogeneous GPUs, so stability is labeled as potentially including hardware numerical variability.

The existing user-edited manuscript target is unchanged: `outputs/proceedings_2026/full-submission-prep-20261008-v1/`. The recovery integration adapts to the latest rewritten Data paragraph, preserves unrelated text and original Figures 1/2, backs up the package, and records minimal result/scope changes plus accurate recovery query accounting. The original failed validation left no summary directory or partially installed manuscript update.

```bash
squeue -j 9565638,9565639,9565640,9565641,9565642,9565643,9565644
sacct -j 9565638,9565639,9565640,9565641,9565642,9565643,9565644 \
  --format=JobID,State,ExitCode,Elapsed
```

Compatible H200 resources remain occupied at submission, so that stage may wait for availability. Do not promise a completion time from scheduler acceptance alone. A queued retry or scheduler completion is not full-cohort validation; `recovery-20261008-v2/summary/analysis.json` must pass before integrating any new results. Source changes to the superseded recovery are documented; original inference and the original secondary plan remain hash-valid. Public release remains prohibited without author/user approval.

Active recovery plan SHA-256: `60339c7d24e7d87e9a1b57bd179ae966ea4a8ef2f2cc021efb5ab5ce8612c55d`. Execution commit: `cf28920ae42b03ef28365ba8b0c5db2b79d266b5`. Exact commands/resources are in the active `submissions.json`.

## Completed recovery and manuscript repair — October 8, 21:20 Eastern

This completion record supersedes the queued-status snapshot above. All 38 GPU-model-matched retries completed, and validation **9565642 completed with exit 0:0**. The selected evidence passes the unchanged probability and graph checks and includes **405 CPU patient–seed records, 135 patients, three seeds per patient and 135 LOO baselines**. Undefined descriptive metrics remain explicitly counted; they are not replaced with zero or silently treated as complete-cohort estimates.

Report **9565643 failed during final contact-sheet assembly** because the recovery manifest refers to the original figure manifest indirectly. The reporter looked for Figure 1 directly in the recovery manifest. Its 270 patient profiles and numerical exports had already been generated. Blocked manuscript job **9565644 was cancelled**. The replacement finalizer resolves and hash-checks all 15 original Figure 1–5 exports through the original frozen plan, checks every existing profile CSV against validated coefficients, and verifies the PDF/SVG/PNG profile exports before reusing them. It performs no GNN inference.

Replacement report **9565855 completed (0:0, 1 minute 28 seconds)**. Its first dependent manuscript attempt, **9565856**, stopped during staged PDF QA before installation: the current user-edited source referenced Figure 2 but lacked its include block, and one model-description paragraph exceeded the line width. The narrow layout adapter restores the original Figure 2 block and permits local paragraph wrapping without changing its words. Exact staged integration then produced a 17-page PDF with resolved references and no missing characters, overfull boxes or oversized floats. Replacement manuscript **9565877 completed (0:0, 55 seconds)**.

The existing `full-submission-prep-20261008-v1` package was updated in place. Its pre-install user source and PDFs are retained in `history/full-secondary-20261009T012031Z/`, together with `minimal_text_changes.diff`. Original Figures 1 and 2 remain byte-identical; nine unrelated source/BibTeX files remain unchanged. All 55 installed artifact hashes match the build manifest, and the review, candidate and LaTeX PDFs are identical. The report contact sheet and rendered manuscript pages 3, 6, 14, 15, 16 and 17 were visually inspected; this is not a claim of complete page-by-page author approval. The full-cohort secondary table, heatmap, Elastic Net fidelity and LOO figures are installed. All 270 private patient influence profiles remain available in the secondary report.

Three focused report-finalizer tests pass, including nested manifest resolution, changed/missing figure rejection and preservation of undefined profile values. Repair sources and submission scripts have separate atomic commits. Exact submission commands, resources, source/configuration hashes and logs are recorded in `report_repair_submissions.json`, `manuscript_repair_submission.json` and `report/assembly_receipt.json` under the active recovery directory. No inference was repeated for this report/manuscript repair. Final Slurm inspection found no remaining queued/running jobs for this user.

The PDFs remain private author-review products. Author declarations, clinical-image/workflow publication permissions and final user/Prof. Yen review remain required. The user's abbreviated workflow caption and remaining historical-caption cross-references should be reconciled during editorial review; this scheduler repair does not silently rewrite those passages. No public submission, release or push occurred.
