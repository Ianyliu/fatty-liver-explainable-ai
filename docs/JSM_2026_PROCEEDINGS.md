# JSM 2026 proceedings implementation and review handoff

Updated October 8, 2026. **Private review only; no submission or public release is authorized.** Ian Liu must review the final manuscript and figures, and Tso-Jung Yen must have an opportunity to approve authorship and content. Author order is provisionally Ian Liu, Tso-Jung Yen.

## Critical path and status

The ten-patient exploratory manuscript is built and is the fallback if the expansion cannot be validated in time. The internal target is October 9 at noon Eastern. Remaining critical work is complete-run validation, interpretation of the expanded results, author/declaration confirmation, required typography, and final PDF review. Optional visual refinements and new experiments must not delay a scientifically complete package.

| Item | Status and evidence |
| --- | --- |
| Environment and migration | Existing uv environment, CPU/GPU smoke tests, image-transfer hashes and eligible-cohort coverage validated before the studies; no environment or inference changes made for publication. |
| Ten-patient comparison | 30 completed patient–seed runs; independently validated raw tables, fits and deletion areas. |
| Additional ten-patient analyses | Training-only Elastic Net CV, marginal Pearson rankings, seed stability and LOO completed; independent completion review refitted 60 Ridge and 60 Elastic Net models. |
| Additional manuscript calculations | CPU-only centered-design diagnostics, exact pool composition reconstruction, and saved-ranking/LOO agreement; no new GNN calls. These are descriptive diagnostics, not a claim of a completed sensitivity study. |
| Full eligible-cohort expansion | 135 patients, three seeds, 405 total runs including 30 reused pilot runs; 375 new GPU tasks in array **9558368**. Complete validation is pending. |
| Full validation | CPU summary **9558371**, dependent on successful completion of array 9558368. |
| Pilot report toolchain | CPU job **9560471** completed with exit 0:0; XeLaTeX/BibTeX, all figure/table exports and automated PDF checks passed on a compute node. |
| Full report generation | CPU job **9560474**, dependent on successful completion of summary 9558371; no new inference or publication. |
| Manuscript and presentation | Complete pilot draft, five generated numerical tables, five multi-panel figures, contact sheet, six verified references, evidence ledger, hashes and internal declaration placeholders. |
| Release | Blocked pending final author review, declarations and Times New Roman or a documented ASA font exception. No paper was submitted or published. |

Scheduler completion is distinct from scientific validation. A full build requires all 405 preserved runs, all planned patients, matching plan/source/evidence hashes, and a passed summary. It rejects absent or incomplete validation before creating an output directory. It never substitutes partial aggregates or pilot estimates for full-cohort results.

## Review package and source locations

The completed private pilot package is `outputs/proceedings_2026/pilot-review-20261008-final/`. The preceding CPU-toolchain validation package remains preserved at `outputs/proceedings_2026/pilot-review-cpu-20261008-v1/`:

- `manuscript_review.pdf`: nonsecured, letter-size review PDF.
- `latex/`: copied LaTeX source, bibliography, generated numerical macros/CSV/TeX tables and figures.
- `latex/figures/contact_sheet.{pdf,svg,png}`: all five figures together; individual figures have PDF/SVG and 300-dpi PNG exports at 5.5-inch manuscript width.
- `supplementary/`: complete historical workflow PDF/SVG/PNG, original source PNG, recovered editable `.drawio` source and provenance.
- `page_review/`: rendered manuscript pages for inspection. `review_validation.json` separately records the agent's visual review, release-gate checks and hashes for exported figures/tables; author approval is still pending.
- `evidence_ledger.json`, `build_manifest.json`, `figure_captions.json`: definitions, validated sources, artifact hashes, commands and package versions.
- `reference_verification.json`: canonical publication URLs, verified fields and access limitations.
- `analysis_status.json`, `release_confirmations.json`, `pdf_qa.json`, `git_commits.txt`: completed/pending/deferred work, author checklist, automated checks and commits.

Full report job 9560474 will use a NEW directory, `outputs/proceedings_2026/full-review-cpu-20261008-v1/`. This is an expected path, not evidence that a full review PDF exists. Private launch configurations, configuration hashes, resources, submission commands/job IDs and execution receipts are in `outputs/proceedings_2026/jobs-20261008-v1/`. Logs are `logs/xai-jsm-review-{JOB_ID}.out`.

Tracked manuscript sources are in `manuscript/proceedings_2026/`; plotting, numerical export and read-only validation are `scripts/proceedings_*.py`. The build entry point is `scripts/build_proceedings.py`. Scheduler execution uses `scripts/run_proceedings_review.py` and `slurm/proceedings_review.sbatch`. Publication source is selected and hashed at report-job execution, checked unchanged during that build, and copied into the package; inference plans remain frozen throughout.

The evidence ledger contains restricted source paths and must remain private. Neither a whole review package nor its raw input directories are a public code/data release. The final PDF contains aggregate results and ordinal patient indices, with automated extracted-text checks for private identifiers. The original workflow's illustrative images remain subject to author publication permission.

## Scientific boundaries

This paper evaluates the perturbation distribution's effect on probability-surrogate fidelity, sampling feasibility and explanation repeatability. The poster narrative is retained with explicit historical/current distinctions. The expansion followed inspection of the convenience pilot, includes those patients, and is exploratory rather than preregistered or an independent replication.

Both arms have 1,000 training draws, seeds 0/1/2 and 200 shared evaluation draws per patient–seed. Current Ridge uses alpha 1, an unpenalized intercept and unscaled binary columns; Elastic Net selection is shuffled five-fold training-only CV on the ten-patient cohort. Historical hard-label classifiers, ten-fold tuning, bootstrap intervals and physician validation are not current completed analyses. The GNN implementation uses a correlation threshold greater than 0.95; the source paper describes 0.995, and this discrepancy is stated explicitly.

Primary MAE is on shared evaluation masks unseen in either training arm, retaining multiplicity. Each arm's constant baseline is its own training-mean probability. All three seed results are averaged within patients first; patients receive equal weight. Missing primary seed metrics make that patient's primary metric unavailable, and any missing patient makes the complete-cohort primary mean unavailable. Available-patient summaries are explicitly labeled. SD and other dispersion are descriptive, with no significance tests or confidence intervals.

Observed disease labels define positive-patient eligibility; singleton model predictions define image pools. Intended 85/15 mixtures are distinguished from achieved proportions, capacity clamping and reallocation, and target failures. Duplicates consume fresh calls; matched training-row budgets do not imply matched query costs. Deletion is a model-behavior intervention, using actual deletion fractions for raw trapezoidal AUC and signed coefficient rankings. It does not establish clinical or causal importance.

## Figures, tables and bibliography

Figure 1 is adapted from the author's original workflow. The permanent GitHub asset was retrieved, preserved and hashed; its PNG contains an embedded editable draw.io diagram, which was recovered without altering the diagram. The full historical source accompanies the explicit current-method vector adaptation.

Figures 2–4 use the primary build population for sampling feasibility, fidelity and deletion; Figure 3D and Figure 5 remain ten-patient only. Method/arm encodings use consistent navy, vermilion, teal, purple/slate and gray, with shapes and line styles as additional distinctions. Captions define populations, metrics, controls, units and direction. Illustrative deletion trajectories select a patient deterministically near the median paired deletion contrast and do not pool unequal grids.

Five numbered tables cover design/eligibility, paired fidelity and availability, sampling/design diagnostics, deletion controls, and ten-patient additional analyses. Values and manuscript macros are generated from validated evidence. The ledger records every table and primary/pilot numerical claim set; all source artifacts are hashed. No result is manually transcribed into manuscript prose.

The six included references are Yen et al. (2024), Ribeiro et al. (2016), Ying et al. (2019), Hoerl and Kennard (1970), Zou and Hastie (2005), and Lundberg and Lee (2017). Every BibTeX entry has an immediately preceding original-publication URL/DOI comment, and all citations resolve. Efron (1979) is omitted: current bootstrap inference is not performed, and original full-text bibliographic verification was incomplete. This omission does not block the paper.

## Deterministic regeneration and checks

Use the validated environment directly; do not sync/relink it while inference jobs run. Output directories must be NEW and below ignored `outputs/proceedings_2026/`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/build_proceedings.py \
  --phase pilot --output outputs/proceedings_2026/pilot-review-NEW

# Only after the full raw-evidence summary passes:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/build_proceedings.py \
  --phase full --output outputs/proceedings_2026/full-review-NEW

MPLCONFIGDIR=outputs/proceedings_2026/.matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -v

bash -n slurm/proceedings_review.sbatch
squeue -j 9558368,9558371,9560474
```

The current suite has **58 passing tests**. Reporting tests verify seed-before-patient averaging, missing-seed rejection, unavailable metrics, constant ranking handling, the full-run gate, bibliography resolution and blocked submission without author confirmation. Existing inference/sampling/transfer tests also pass. System tools are XeLaTeX, BibTeX, Fontconfig, Poppler (`pdftotext`, `pdfinfo`, `pdffonts`, `pdftoppm`) and the LaTeX packages listed in the package README. Python/Matplotlib is used consistently; R dependencies are unnecessary. No network retrieval or inference occurs during report builds.

Automated PDF QA confirms resolved citations, nonsecured letter-size output, no missing glyphs, no overfull text boxes and no detected patient identifiers in extracted text. This does not replace review of the rendered figures, image content, numerical interpretation or author declarations. The current environment substitutes Nimbus Roman for required Times New Roman; review PDFs are explicitly marked as such. Install a legitimately available Times New Roman font in an approved location or obtain a documented ASA exception before submission-ready output. Do not silently rename the substituted font.

`--submission-pdf` requires actual recorded author confirmations, removal of internal placeholders and review markings, final typography and explicit release status. It only builds a PDF and never submits or publishes. Missing declarations must not become statements of “none.”

## After full validation finishes

1. Check summary 9558371 and build 9560474 exit codes and their JSON evidence receipts. Reconcile 135 patients, 405 runs and the frozen query budget, without merging incompatible attempts.
2. Read the generated full tables and patient-level distributions. Update the abstract, Results and Discussion to explain the full findings while retaining separate pilot results and the post-pilot expansion disclosure. Do not infer an adaptive benefit from the pilot or from selected diagnostics.
3. Inspect all full figures and manuscript pages at the 5.5-inch printed figure width, including actual-grid deletion trajectories, defined/unavailable counts, contact sheet and grayscale distinctions. Record QA against the PDF's SHA-256.
4. Obtain author confirmation of order, affiliations, correspondence, contributions, funding, interests from both authors, secondary-use authorization, approved data access, releasable code, original-workflow permission and AI disclosure. Prof. Yen's 2024 affiliation is historical evidence, not current confirmation.
5. Resolve typography, rebuild a new immutable package, and repeat numerical/citation/PDF checks on that exact PDF. Deliver it to Ian for final review and provide Prof. Yen an approval opportunity. Public release remains blocked until those conditions are satisfied.

If validation misses the review window, use the clearly labeled ten-patient manuscript and disclose the pending expansion rather than reporting invented or partial full-cohort results. No new inference is needed to finish that paper.

## Deferred safely

Matched total-query-budget comparisons, pool-ratio/subset-size sensitivity, repeated random-deletion orders, deletion with additional rankings, expanded Elastic Net/Pearson/LOO/stability, historical bootstrap inference, physician validation and clustering ablations are not completed full-cohort experiments. They are optional future work and must not delay this proceedings manuscript.
