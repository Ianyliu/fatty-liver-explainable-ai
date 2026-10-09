# JSM 2026 reviewed manuscript sources

These text sources preserve the supplied revised manuscript and describe the explanation method in place. Its editable private package is `outputs/proceedings_2026/revised_20261008/JSM2026_revised_package/`. The prior private output directory is backed up at `outputs/private_backups/proceedings_2026_before_revision_20261008.tar.gz`. Existing historical receipts describe their original artifacts and remain unchanged.

Build this revision directly. Do not use `build_proceedings.py`, older finalizers, or historical exporters to regenerate its prose or figures. The source directory here intentionally contains no generated tables, restricted clinical figures, full numeric evidence, PDF or ZIP. All seven referenced figure PDFs remain in the ignored editable package. Figures 1 and 2 preserve the reviewed bytes; no replacement influence bars or duplicate influence figure are allowed.

From the repository root, run:

```sh
.venv/bin/python scripts/build_revised_proceedings.py \
  --package outputs/proceedings_2026/revised_20261008/JSM2026_revised_package \
  --font-dir outputs/proceedings_2026/fonts/times-new-roman \
  --latexmk /absolute/path/to/latexmk
```

The wrapper verifies licensed local Times New Roman hashes/family, regenerates rounded summaries from original full-precision evidence, runs `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`, checks critical TeX diagnostics, letter size and embedding, and saves the new PDF and build checks. It consumes the package's revised text, never an older exporter. Omit `--latexmk` when the program is on PATH. This server's locally downloaded latexmk is `outputs/revision_audit_20261008/tools/latexmk` (version 4.87).

For evidence checks without inference:

```sh
.venv/bin/python scripts/verify_revised_proceedings.py \
  --package outputs/proceedings_2026/revised_20261008/JSM2026_revised_package \
  --cohort-plan outputs/complete_patient/full-135-20261007-v2/full_cohort_plan.json
```

After editing text, copy only those reviewed `.tex`/`.bib` files into the package before rebuilding. Do not copy these historical assets or regenerate workflow/image panels. The summary generator takes `--package` when invoked from the repository; its editable-package copy needs no argument. No fonts are redistributed. A rebuild changes the PDF hash and requires renewed visual inspection; build checks alone do not certify visual QA or submission eligibility.

See [source_confirmations.md](source_confirmations.md) for verified facts and exact outstanding author/source questions. Affiliations and the requested AI disclosure are preserved. Missing funding, competing-interest, contribution or ethics statements are not declarations of “none.” “Supplied test split” remains because historical checkpoint training/tuning linkage is unresolved despite zero overlap with the supplied training/validation lists.

Internal review only. Do not push, submit, publish or publicly release the manuscript/package until Ian Liu reviews it and Prof. Tso-Jung Yen has an opportunity to approve authorship and content. Clinical-image and workflow publication permissions remain unconfirmed. Publication/submission authorization remains pending; the manuscript body contains no review banner or unresolved checklist.
