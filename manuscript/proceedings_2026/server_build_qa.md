# Server build and formatting QA — October 9, 2026

The current private PDF is `outputs/proceedings_2026/revised_20261008/JSM2026_revised_package/JSM2026_server_revised_manuscript.pdf`.

SHA-256: `e3cca428be4869612cf30da30975eb4d9de685c3800f596036af9ed7548fe9a0`.

The bibliography is fully justified. The abstract, affiliations, figure/table captions, table text and table notes now use 10-point Times New Roman. Table notes form justified paragraphs. Body prose retains 10-point Times New Roman; titles/headings, mathematical notation and preserved figure artwork retain their conventional typography. The summary generator enforces these table sizes on future builds.

Passed: 17 nonsecured US-letter pages; one-inch top/bottom and 1.5-inch side margins; single column and spacing; centered figures/tables; one sentence space. All 17 pages' body word bounds were checked, excluding footer page numbers, with 0.1 PDF-point tolerance. Times New Roman regular/bold/italic and all other PDF fonts are embedded. No unresolved references, missing glyphs, overfull boxes or oversized floats. One underfull alignment in the repeatability table was visually inspected and remains readable.

Every final page was rendered and inspected, including enlarged tables and notes, justified references, equations, workflow labels, original clinical panels and appendix plots. Exactly the seven allowed figures remain; all seven PDFs are byte-identical to the preceding build, including Figures 1 and 2. All 41 numerical macros, five table display records and original evidence hashes are unchanged. No manuscript prose, patient selection, frozen inference settings or original full-precision evidence changed. No new inference or significance testing ran. Prior checks covering 3,448 source hashes and 405 saved runs remain valid for these unchanged scientific inputs.

ASA requirements: https://ww2.amstat.org/meetings/jsm/2026/submissions.cfm#proceedings . Full current private records: `server_revision_qa.json`, `server_build_checks.json`, `server_evidence/formatting_verification.json`, `server_evidence/formatting_margin_checks.json` and `revision_evidence_ledger.json`. The preceding build, editable ZIP, sources and QA were backed up at `outputs/private_backups/before_formatting_update_20261009T040438Z`.

Source/author clearance remains pending: historical checkpoint provenance, exact illustrative-panel linkage, author declarations, secondary-use/consent and public-use permissions. The requested abstract's 10-fold CV and 85% allocation/informativeness claims remain flagged externally against the saved five-fold CV and rounding/pool-reallocation rules. These scientific issues are separate from this formatting QA; no review banner or unresolved checklist was added to the manuscript.

Private review only. No push, submission or publication is authorized. No patient artifacts, clinical figures, font files, full numerical evidence or PDF/ZIP were added to Git. Unrelated local changes were preserved.
