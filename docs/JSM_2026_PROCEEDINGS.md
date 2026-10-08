# JSM 2026 proceedings: revised author-review package

Updated October 8, 2026. Private review only. Ian Liu and Tso-Jung Yen remain the provisional authors in that order. No submission, public release, or author approval has occurred.

## Editorial and visual revision

The revised manuscript retains the accepted title and the original research sequence: grouped-ultrasound GNN prediction, LIME-inspired image-subset perturbations, proposed two-stage Adaptive Class-Balanced Sampling, complementary conditional and marginal influence, and evaluation of those explanations. Fidelity and feasibility findings qualify the framework without replacing its motivation. The original accepted abstract is preserved verbatim in `manuscript/proceedings_2026/accepted_abstract.txt`, with the official program URL and source hash in the adjacent JSON. `abstract_revision_notes.md` explains necessary changes to classifier/CV/bootstrap claims, population, and conclusions.

The clean author-review and candidate PDFs have no internal-review banner or engineering status paragraph. Their content is identical until confirmed author information is supplied. Funding, competing interests, current secondary-use authorization and author affiliations are not invented. They must be supplied in `author_declarations.json` and incorporated before final approval; `author_confirmation_checklist.md` records the outstanding items separately. A candidate PDF is not evidence of submission eligibility or permission to release it.

Figures are generated with Python/Matplotlib at a 5.5-inch manuscript width, with vector PDF/SVG and 300-dpi PNG exports:

1. **Original README workflow:** restored unchanged as main Figure 1, preserving the author's layout, artwork, colors, model architecture and arrows. The caption qualifies historical classifier/ten-fold-CV/physician-evaluation labels against current probability regression, fixed-alpha Ridge, five-fold Elastic Net and proposed physician assessment. The PNG and editable draw.io remain preserved.
2. **README-style image-influence results:** signed vertical cyan/coral bars with corresponding grayscale ultrasound thumbnails beside the bar ends, matching the visual language of the three original example plots. Current validated three-seed Pearson/Elastic Net/Ridge estimates replace historical numbers. All three panels share twenty images sorted by Pearson mean; I01–I20 are anonymous display labels. No unvalidated error bars or faded significance coding are shown. The three unmodified README result images are preserved privately as historical source references with URL/SHA-256 provenance.
3. **Sampling:** paired achieved prediction proportions and patient-level distributions of pool reallocation and duplicate masks. Target attainment is annotated.
4. **Fidelity:** paired patient MAE, adaptive-minus-random differences and exact arm-specific training-mean constant comparisons.
5. **Deletion:** control trajectories on one patient's actual deletion grid and paired patient-level descending-minus-random raw AUC. Normalized-AUC correlation does not occupy a main panel.
6. **Supplementary pilot diagnostics:** compact method/metric similarity matrix with defined counts and Elastic Net-versus-Ridge differences. These are ten-patient analyses only.

The patient example and historical artwork contain clinical images. Ian authorized their use for private review on October 8. Publication eligibility remains unconfirmed, including for the adapted workflow. Do not publish the PDF, package or image derivatives until that issue is resolved. Visual inspection and extracted-text checks supplement, rather than establish, eligibility.

## Current results and full-cohort dependency

The ten-patient pilot has 30 validated patient–seed runs. Independent completion review refitted 60 Ridge and 60 Elastic Net models and checked the selected training-only five-fold CV settings. Pearson rankings, seed stability and LOO are complete for these ten patients. Publication calculations reconstruct realized pool mixtures, centered-design conditioning and ranking/LOO agreement from preserved evidence; no new GNN inference is performed.

The pilot patient-mean shared-novel MAE is 0.067116 for random and 0.111150 for adaptive sampling, paired difference +0.044034; random has lower error in nine of ten patients. Adaptive achieves the requested 500/500 prediction balance in zero of 30 runs. These unfavorable results remain explicit. They are descriptive findings, with no bootstrap intervals, significance tests or physician-validation claims.

The full study targets 135 eligible positive-label test09 patients with complete image availability, 3,072 images and three seeds per patient. It includes the pilot's 30 runs plus 375 new tasks, not an independent replication. Expansion followed inspection of the pilot; the analysis is exploratory.

| Job | Purpose | Live status at 12:30 p.m. Eastern, October 8 |
| --- | --- | --- |
| 9558368 | 375 new GPU patient–seed tasks, concurrency two | 339 completed, two running, 34 pending |
| 9558371 | Complete raw-evidence validation summary | Pending successful array completion |
| 9560474 | Full private manuscript/figures build | Pending successful validation summary |

Snapshot and raw scheduler rows: `outputs/proceedings_2026/jobs-20261008-v1/revision_final_status.json`. Recent 20 completed tasks averaged about 560 seconds. With two concurrent tasks, approximately three hours of inference remained at the final snapshot, subject to preemption and queue availability. A full-cohort review before October 9 noon Eastern appears feasible if completion and validation succeed; it is not guaranteed. The ASA submission page states October 9, without specifying a cutoff: https://ww2.amstat.org/meetings/jsm/2026/submissions.cfm.

No full-cohort result is included before complete validation. The full build rejects absent/partial summaries and requires all 405 preserved runs and source/configuration consistency. The pending report job selects and hashes publication sources at execution, checks that they stay unchanged during its build, and preserves a source snapshot. Frozen inference inputs, checkpoints and running-job configurations are unchanged. No new experiments were launched for this revision.

## Private deliverables

The revised pilot package is `outputs/proceedings_2026/pilot-original-figures-20261008-v1/`. Earlier revision packages remain preserved. The original review package remains at `outputs/proceedings_2026/pilot-review-20261008-final/` for comparison. The new build restores the original workflow and README plot style; consult its PDF/build/visual-QA records. Earlier revision QA remains specific to those earlier PDFs.

- `manuscript_review.pdf` and `manuscript_candidate.pdf`: clean private scientific copies, pending declarations and approval.
- `latex/`: manuscript source, bibliography, generated macros, CSV/TeX tables and all figure exports.
- `latex/figures/contact_sheet.pdf`: revised figures together.
- `comparisons/workflow_original_vs_revised.pdf`: original and restored main workflow.
- `comparisons/contact_sheet_before_after.pdf`: previous and revised figure presentation.
- `abstract_original_vs_revised.md`: exact accepted abstract, populated revised abstract and substantive change notes.
- `author_confirmation_checklist.md`: remaining author facts, image permissions and approval requirements.
- `evidence_ledger.json`, `build_manifest.json`, `pdf_qa.json`, `review_validation.json`: scientific definitions, validated source hashes, artifact hashes, automated PDF checks and separate rendered-page review.
- `reference_verification.json`, `git_commits.txt`: verified bibliography and source history.
- `supplementary/`: historical original workflow and editable source.

Expected full output: `outputs/proceedings_2026/full-review-cpu-20261008-v1/`. This path alone is not evidence that a full manuscript exists. Report configuration/submission/execution receipts remain under `outputs/proceedings_2026/jobs-20261008-v1/`; logs are `logs/xai-jsm-review-{JOB_ID}.out`.

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

Reporting tests cover patient weighting, missingness, constant rankings, full-run completeness, bibliography resolution and author release gates. All 59 tests pass; the final reporting-only rerun passes all eight checks. A new test verifies that checked approval boxes cannot replace missing declaration text. Automated PDF checks cover citations, glyphs, encryption, page size, identifiers and font embedding; manual visual review remains separate.

## Completion and release gate

Once summary 9558371 passes, inspect the full build receipt, generated numerical tables and 135 patient-level distributions. Interpret full-cohort results in Results and Discussion, including any agreement or disagreement with the pilot, before final author review. Check every full figure and rendered page, its captions, defined counts and actual deletion grids. Preserve the ten-patient supplementary population and post-pilot expansion disclosure.

The ASA page also requires an eligible oral presentation (including a poster presentation); confirm actual presentation and any applicable proof-of-progress requirement. Acceptance alone does not establish these facts.

Ian and Prof. Yen must supply or confirm author order, current affiliations/correspondence, contribution assignments, work-specific funding, interests from both authors, secondary-use/consent coverage, approved data/code statements and AI disclosure. Clinical-image and original-workflow publication permission remain outstanding. Add those verified statements, rebuild, and review that exact PDF before any release.

`--submission-pdf` requires actual confirmation flags, complete declaration text, final typography and explicit final-review approval. It only builds a PDF; it never submits. The clean candidate does not bypass this gate. No ten-patient fallback will be submitted automatically.

Matched total-query budgets, sampling-ratio/subset-size sensitivity, repeated random deletion controls, expanded Elastic Net/Pearson/stability/LOO, bootstrap inference, physician validation and clustering ablations are deferred. They must not delay a scientifically complete proceedings package.
