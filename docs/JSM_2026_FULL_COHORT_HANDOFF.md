# Final full-cohort author-review handoff — October 8, 2026

The primary manuscript is complete for final author review. The clean candidate is private and is not approved for submission. No inference was launched and no frozen study inputs were changed. This package supersedes earlier review archives for prose, tables and QA; the latest original-README Figures 1 and 2 are preserved byte for byte.

## Completed-run reconciliation

Fresh scheduler receipt: all 375 expansion tasks in 9558368 completed with exit code 0:0; summary 9558371 completed 0:0. The 30 reused pilot runs reconcile to 135 unique patients, seeds 0/1/2 and 405 complete records. The full summary and fresh raw-artifact validation both passed. Primary unavailable patient pairs: 0. The report job 9560474 failed 1:0 because publication sources changed during its earlier build; the new local latest-source build replaces that report, not any scientific run. See `scheduler_status.json`, `superseded_report_failure.json`, `validation_reconciliation.json` and `review_validation.json`.

## Validated findings

All cohort summaries average three seeds within patients first. The full cohort includes the inspected convenience pilot and is exploratory, not an independent replication. This table is generated from `outputs/proceedings_2026/full-submission-prep-20261008-v1/latex/generated/metrics.json`; ledger entries `claims:primary`, `claims:pilot`, `queries:stages` and `comparison:full_vs_pilot` define its sources and calculations.

| Finding | Full cohort | Pilot |
| --- | ---: | ---: |
| Patients / seeds | 135 / 3 | 10 / 3 |
| Random Ridge MAE | 0.066109 | 0.067116 |
| Adaptive Ridge MAE | 0.119002 | 0.111150 |
| Adaptive minus random MAE | +0.052894 | +0.044034 |
| Random / adaptive / tied patients | 129 / 2 / 4 | 9 / 1 / 0 |
| Class-1 proportions, random / adaptive | 91.48% / 78.68% | 87.21% / 77.04% |
| Exact 500/500 target successes | 1 / 405 | 0 / 30 |
| Duplicate rows, random / adaptive | 8.40% / 12.46% | 10.23% / 15.38% |
| Descending minus random AUC, random-trained | -0.153123 | -0.162640 |
| Descending minus random AUC, adaptive-trained | -0.148524 | -0.161970 |

Adaptive proportions are closer to 0.5 in 126/135 patients, but random sampling has lower MAE in 129/135. Class coverage and surrogate fidelity are different goals. Both predicted classes occur in 356/405 random and 377/405 adaptive training sets. Only 1/405 adaptive runs attain the exact target.

Pool deficits affect 122,575/175,726 biased draws (69.75%); 15 runs use one-pool fallback. Unique masks average 916.0 and 875.4 per 1,000 draws. No centered design is rank deficient. Own-training-mean constant MAE is 0.092272 and 0.190432; Ridge beats those respective constants in 43/135 and 103/135 patients.

Primary shared-novel support is 170–194 draws per patient–seed; 101/405 sets contain one predicted class, but probability MAE remains defined. Raw deletion AUC uses actual deletion fractions; descending is below random control in 135/135 patients in each arm. This concerns model behavior, not clinical or causal importance. Primary inference totals 911,151, including 9,621 adaptive singleton/constructor calls beyond the equal training-row budgets. Pilot LOO adds 210 separate calls and is not included in this primary budget.

## Manuscript and figure integration

The accepted title and original abstract are preserved. The revised abstract retains the GNN/LIME/adaptive/conditional/marginal narrative and uses the full-cohort finding. Results and Discussion now interpret coverage, fidelity, constants, capacity constraints, duplicate masks, deletion and query cost. Potential mechanisms are labeled as interpretations, not experimentally established causes. Positive-label selection, one checkpoint, post-pilot expansion, unequal cost and absent clinical validation remain explicit.

Figure 1 is the original workflow; Figure 2 is the exact original three-bar-chart composite. Both export PDFs and the contact sheet match the latest prior full package byte for byte. Their native pixels were independently checked against the recovered source PNGs. Historical classifier/CV/interval/significance and physician labels are qualified; they are not current quantitative evidence. Current image-influence estimates remain in Appendix Figure 7. Figures 3–5 use all 135 patients; Figure 6 and Table 5 are explicitly ten-patient secondary analyses. Elastic Net, Pearson repeatability and LOO are not full-cohort experiments. No significance tests or new uncertainty estimates were added.

## Verification and reproducibility

Fresh build source: e40e200 (publication files unchanged by subsequent QA/documentation commits). Ten reporting tests passed. Independent QA verified 32 publication source hashes and 47 artifact hashes, patient-level numerical summaries, stage query counts and original-image pixels. Six verified bibliography records retain original-source comments, resolve in the manuscript and have no duplicated keys/DOIs. All 17 PDF pages were rendered and inspected individually. Letter size, ASA margins, single-column 10-point embedded Times New Roman, nonsecured PDF and no overfull boxes were verified. The official requirements are recorded in `submission_requirements.json`.

Original raster labels and clinical thumbnails still require publication/privacy review; text screening alone cannot clear them. Original fine print remains small, with source-resolution artwork retained. Minor bibliography/appendix float pagination is documented in QA and was not treated as a scientific blocker.

Regenerate in the repository with the existing environment (no sync/install):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/build_proceedings.py --phase full --output outputs/proceedings_2026/NEW_DIRECTORY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/verify_proceedings_package.py outputs/proceedings_2026/NEW_DIRECTORY
```

The source package contains `outputs/proceedings_2026/full-submission-prep-20261008-v1/latex/` (manuscript, references, all generated tables/macros and figure exports), `source_snapshot/`, original-source comparisons, evidence ledger/hashes and the author checklist. Restricted raw images, checkpoints and raw run ledgers are not bundled. The evidence ledger contains private source paths: do not publish this archive wholesale.

## Remaining author and release decisions

Confirm author order, current affiliations and correspondence; contribution assignments; work-specific funding; both competing-interest declarations; secondary-use/consent authorization; data/code/checkpoint access statements; AI disclosure; artwork/image rights and original embedded image-label privacy; actual eligible presentation/proof-of-progress compliance. Proposed wording and exact missing information are in `author_statement_proposals.md`. Confirm final manuscript and figures with Ian and Prof. Yen; then insert agreed statements, rebuild and perform final layout QA before explicit release approval. No missing declaration has been converted to “none.”

The scientific full-cohort package is ready for final author review before October 9. Public submission remains blocked by those author/eligibility/image confirmations and final approval, not by incomplete primary experiments. No ten-patient fallback is required. Optional matched-query studies, expanded Elastic Net/Pearson/LOO, sensitivity analyses, bootstrap inference and physician validation remain deferred.

Package: `outputs/proceedings_2026/full-submission-prep-20261008-v1/`.
Archive: `outputs/proceedings_2026/jsm2026-full-submission-prep-20261008.zip` (private; checksum sidecar).
