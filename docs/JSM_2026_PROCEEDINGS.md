# JSM 2026 proceedings: revised author-review package

Updated October 8, 2026. Private review only. Ian Liu and Tso-Jung Yen remain the provisional authors in that order. No submission, public release, or author approval has occurred.

## Editorial and visual revision

The revised manuscript retains the accepted title and the original research sequence: grouped-ultrasound GNN prediction, LIME-inspired image-subset perturbations, proposed two-stage Adaptive Class-Balanced Sampling, complementary conditional and marginal influence, and evaluation of those explanations. Fidelity and feasibility findings qualify the framework without replacing its motivation. The original accepted abstract is preserved verbatim in `manuscript/proceedings_2026/accepted_abstract.txt`, with the official program URL and source hash in the adjacent JSON. `abstract_revision_notes.md` explains necessary changes to classifier/CV/bootstrap claims, population, and conclusions.

The clean author-review and candidate PDFs have no internal-review banner or engineering status paragraph. Their content is identical until confirmed author information is supplied. Funding, competing interests, current secondary-use authorization and author affiliations are not invented. They must be supplied in `author_declarations.json` and incorporated before final approval; `author_confirmation_checklist.md` records the outstanding items separately. A candidate PDF is not evidence of submission eligibility or permission to release it.

Figures are generated with Python/Matplotlib at a 5.5-inch manuscript width, with vector PDF/SVG and 300-dpi PNG exports:

1. **Original README workflow:** restored unchanged as main Figure 1, preserving the author's layout, artwork, colors, model architecture and arrows. The PDF/SVG embeds the original raster at native resolution. The caption qualifies historical classifier/ten-fold-CV/physician-evaluation labels against current probability regression, fixed-alpha Ridge, five-fold Elastic Net and proposed physician assessment. The PNG and editable draw.io remain preserved.
2. **Original README result composite:** the three original PNGs, stacked in README order: A, marginal Pearson correlation; B, conditional Elastic Net; C, conditional Ridge. Only panel headings are added outside the artwork. Original values, thumbnails, labels, ordering, colors, error bars and faded bars are unchanged. The caption explicitly distinguishes these historical illustrations from current validated probability-regression evidence. SHA-256 checks enforce the original sources. This follows Ian's latest October 8 instruction to use the original charts rather than recreate them.
3. **Sampling:** paired achieved prediction proportions and patient-level distributions of pool reallocation and duplicate masks. Target attainment is annotated.
4. **Fidelity:** paired patient MAE, adaptive-minus-random differences and exact arm-specific training-mean constant comparisons.
5. **Deletion:** control trajectories on one patient's actual deletion grid and paired patient-level descending-minus-random raw AUC. Normalized-AUC correlation does not occupy a main panel.
6. **Supplementary pilot diagnostics:** compact method/metric similarity matrix with defined counts and Elastic Net-versus-Ridge differences. These are ten-patient analyses only.
7. **Supplementary current image-influence example:** current validated three-seed Pearson/Elastic Net/Ridge estimates for twenty images, drawn in the README thumbnail-on-bar style. All panels share image order; anonymous labels I01–I20 replace source IDs. No unvalidated error bars or faded significance coding are used. This figure is separate from the original main Figure 2.

The patient example and historical artwork contain clinical images. Ian authorized their use for private review on October 8. Publication eligibility remains unconfirmed. The unchanged original bar charts retain their image labels; extracted-text checks do not inspect labels embedded in raster images. Do not publish the PDF, package or image derivatives until eligibility is resolved. Visual inspection and extracted-text checks supplement, rather than establish, eligibility.

## Current results and full-cohort dependency

The ten-patient pilot has 30 validated patient–seed runs. Independent completion review refitted 60 Ridge and 60 Elastic Net models and checked the selected training-only five-fold CV settings. Pearson rankings, seed stability and LOO are complete for these ten patients. Publication calculations reconstruct realized pool mixtures, centered-design conditioning and ranking/LOO agreement from preserved evidence; no new GNN inference is performed.

The pilot patient-mean shared-novel MAE is 0.067116 for random and 0.111150 for adaptive sampling, paired difference +0.044034; random has lower error in nine of ten patients. Adaptive achieves the requested 500/500 prediction balance in zero of 30 runs. These unfavorable results remain explicit. They are descriptive findings, with no bootstrap intervals, significance tests or physician-validation claims.

The full study targets 135 eligible positive-label test09 patients with complete image availability, 3,072 images and three seeds per patient. It includes the pilot's 30 runs plus 375 new tasks, not an independent replication. Expansion followed inspection of the pilot; the analysis is exploratory.

| Job | Purpose | Checked October 8, after full-cohort validation |
| --- | --- | --- |
| 9558368 | 375 new GPU patient–seed tasks, concurrency two | All 375 completed, exit 0:0 |
| 9558371 | Complete raw-evidence validation summary | Completed, exit 0:0; summary status passed |
| 9560474 | Earlier full private manuscript/figures build | Failed, exit 1:0; publication-source integrity guard detected the subsequently requested Figure 2 revision |

Raw scheduler rows: `outputs/proceedings_2026/jobs-20261008-v1/readme_originals_sacct.txt`. The completed summary validates all 405 runs and all 135 patient comparisons, with no missing primary patients. Full-cohort patient-mean shared-novel MAE is 0.066109 for random and 0.119002 for adaptive sampling; paired difference +0.052894. Random has lower MAE for 129 patients, adaptive for two, with four ties. Adaptive attained the requested balance in one of 405 runs. The main evaluation used 911,151 GNN queries, including 67,410 reused pilot queries. The expansion agrees with the pilot's unfavorable adaptive-fidelity direction and remains exploratory. These values are cross-checked against the preserved patient CSV in `outputs/proceedings_2026/full_validated_numerical_crosscheck.json`.

The full build rejects absent/partial summaries and requires all 405 preserved runs and source/configuration consistency. Job 9560474 began before the latest editorial instruction, and its source-change failure does not invalidate the experiment or summary. Its output is superseded. The CPU-only local full build uses the committed original-figure layout at `full-readme-originals-20261008-v2`; it repeats artifact validation and publication calculations, with no new GNN queries. Frozen inference inputs, checkpoints and running-job configurations are unchanged. No new experiments were launched for this revision.

## Private deliverables

The latest full-cohort package is built at `outputs/proceedings_2026/full-readme-originals-20261008-v2/`. Consult its manifest and QA records to establish completion. Its predecessor, `full-readme-originals-20261008-v1`, has the requested original charts but predates the final native-resolution workflow export and removal of the image-clearance reminder from the caption. Clearance remains on the separate author checklist. The preceding pilot package, `pilot-original-figures-20261008-v1`, predates the request for an exact original-chart composite. Earlier packages remain preserved; their QA applies only to those PDFs. The original review remains at `pilot-review-20261008-final/` for comparison.

The final full build passed on October 8. Its 16-page clean PDF has embedded Times New Roman, letter dimensions, resolved citations, no encryption and no overflowing text. All 30 publication-source hashes and 42 artifact hashes match; the evidence ledger has 20 entries. Exported PDF image streams preserve the original workflow and all three bar charts at native resolution, with identical visible RGB pixels and transparency. Eight reporting checks pass. Rendered-page and contact-sheet review found no clipped or overlapping content. Final pages 3 and 6 were inspected after the last export changes; the other fourteen page renders are pixel-identical to the preceding reviewed build. These checks do not establish clinical-image publication eligibility or author approval.

- `manuscript_review.pdf` and `manuscript_candidate.pdf`: clean private scientific copies, pending declarations and approval.
- `latex/`: manuscript source, bibliography, generated macros, CSV/TeX tables and all figure exports.
- `latex/figures/contact_sheet.pdf`: revised figures together.
- `comparisons/workflow_original_vs_revised.pdf`: original and restored main workflow.
- `comparisons/contact_sheet_before_after.pdf`: previous and revised figure presentation.
- `abstract_original_vs_revised.md`: exact accepted abstract, populated revised abstract and substantive change notes.
- `author_confirmation_checklist.md`: remaining author facts, image permissions and approval requirements.
- `evidence_ledger.json`, `build_manifest.json`, `pdf_qa.json`, `review_validation.json`: scientific definitions, validated source hashes, artifact hashes, automated PDF checks and separate rendered-page review.
- `reference_verification.json`, `git_commits.txt`: verified bibliography and source history.
- `supplementary/`: original workflow, editable source, unchanged README result PNGs and their provenance.

The earlier scheduler output `full-review-cpu-20261008-v1/` is superseded and must not be delivered as the latest manuscript. Its configuration/submission/execution receipts remain under `outputs/proceedings_2026/jobs-20261008-v1/`; its log is `logs/xai-jsm-review-9560474.out`.

The evidence ledger and source paths are restricted. Do not publish the whole review archive or raw data/checkpoints. Public code availability must be distinguished from access to private inputs.

## Scientific and reproducibility boundaries

Patient-level summaries are primary: average all three seeds within patient, then weight patients equally. Missing seed metrics make a patient metric unavailable; any missing patient makes the complete-cohort primary mean unavailable, with available-patient summaries labeled separately. Dispersion is descriptive.

Each arm has 1,000 training draws and 200 shared independent evaluation draws per seed. Primary MAE uses masks seen in neither training arm, retaining multiplicity. Each constant is its arm's training-mean probability on exactly the same evaluation rows. Ridge is alpha-1 probability regression, unscaled indicators and an unpenalized intercept. Elastic Net is training-only shuffled five-fold CV in ten patients. Historical hard-label classifiers, ten-fold tuning, bootstrap intervals and physician evaluation are not current completed analyses.

Observed positive disease labels define cohort eligibility; singleton model predictions define pools. Intended 85/15 mixtures differ from achieved pool proportions and achieved subset predictions. Capacity clamping, reallocation, failed targets, duplicates and adaptive overhead are reported. Equal row budgets are not equal query budgets. Raw deletion AUC integrates actual patient-specific fractions; deletion evaluates model behavior, not clinical or causal image importance. The supplied model's feature-correlation threshold (>0.95) differs from the source paper's 0.995; the analysis concerns the supplied checkpoint and does not reproduce conformal-risk validation.

Five numerical tables cover design, fidelity/baselines/availability, adaptive diagnostics, deletion controls, and pilot supplementary analyses. Numerical macros and tables are generated from validated artifacts and entered in the evidence ledger. Six verified references remain unchanged; every BibTeX entry retains its immediately preceding original-source URL comment. No unverified citations were added.

## Build and checks

Use the existing uv environment directly. Do not sync or relink it during inference. Every build requires a NEW output directory.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/build_proceedings.py \
  --phase pilot --output outputs/proceedings_2026/pilot-review-NEW

# Only after complete full validation:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/build_proceedings.py \
  --phase full --output outputs/proceedings_2026/full-review-NEW

MPLCONFIGDIR=outputs/proceedings_2026/.matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests

squeue -j 9558368,9558371,9560474
```

Dependencies: pinned existing NumPy/Pandas/SciPy/scikit-learn/Matplotlib/Pillow; XeLaTeX, BibTeX, fontspec, geometry, natbib, booktabs, tabularx, graphicx, Fontconfig and Poppler. R is unnecessary for the chosen implementation. Local Times New Roman regular/bold/italic/bold-italic files are verified by family and SHA-256 against `font_source.json`; the original package/license remain in ignored `outputs/proceedings_2026/fonts/`. Extracted font files are not redistributed. The build records the actual embedded family and never silently calls a substitute Times New Roman.

Figure 2 requires the three private cached originals listed with source URLs and SHA-256 in `manuscript/proceedings_2026/assets/readme_result_references.json`. They are included in the private package under `supplementary/readme_examples/`; restore them to the manifest's ignored cache paths when reproducing from a fresh checkout. Restricted clinical PNGs are not newly added to Git. Original Figure 1 and its editable draw.io source remain tracked.

Reporting tests cover patient weighting, missingness, constant rankings, full-run completeness, bibliography resolution and author release gates. All 59 tests pass; the final reporting-only rerun passes all eight checks. A new test verifies that checked approval boxes cannot replace missing declaration text. Automated PDF checks cover citations, glyphs, encryption, page size, identifiers and font embedding; manual visual review remains separate.

## Completion and release gate

Full-cohort validation is complete, so a full-cohort author-review package before October 9 noon Eastern is feasible. Final readiness depends on PDF review and author declarations rather than further inference. The full manuscript uses the expanded results in the main analysis, retains the separate preliminary pilot comparison and labels additional explanation analyses as ten-patient only. Review the exact revised PDF and original-figure captions before approval. The ASA submission page states October 9 without a cutoff: https://ww2.amstat.org/meetings/jsm/2026/submissions.cfm.

The ASA page also requires an eligible oral presentation (including a poster presentation); confirm actual presentation and any applicable proof-of-progress requirement. Acceptance alone does not establish these facts.

Ian and Prof. Yen must supply or confirm author order, current affiliations/correspondence, contribution assignments, work-specific funding, interests from both authors, secondary-use/consent coverage, approved data/code statements and AI disclosure. Clinical-image and original-workflow publication permission remain outstanding. Add those verified statements, rebuild, and review that exact PDF before any release.

`--submission-pdf` requires actual confirmation flags, complete declaration text, final typography and explicit final-review approval. It only builds a PDF; it never submits. The clean candidate does not bypass this gate. No ten-patient fallback will be submitted automatically.

Matched total-query budgets, sampling-ratio/subset-size sensitivity, repeated random deletion controls, expanded Elastic Net/Pearson/stability/LOO, bootstrap inference, physician validation and clustering ablations are deferred. They must not delay a scientifically complete proceedings package.
