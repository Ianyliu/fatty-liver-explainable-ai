# Server revision QA — October 8, 2026

Private PDF: `outputs/proceedings_2026/revised_20261008/JSM2026_revised_package/JSM2026_server_revised_manuscript.pdf`.

SHA-256: `87787246581441daba250f137471a2d73354fccbda81f814d74bd697c876f70c`.

Passed: 17 nonsecured letter pages; Times New Roman body fonts and all other PDF fonts embedded; requested geometry and body word bounds checked (footer page numbers excluded). Text height was reduced four TeX points to keep descenders inside the bottom margin. No unresolved references, missing glyphs, overfull boxes or oversized floats. One underfull table alignment was visually checked and is readable.

Every final page was rendered and inspected after the margin adjustment: affiliations, workflow labels, original image panels, equations, tables, references and appendix plots. Figures 1 and 2 match the incoming reviewed bytes; exactly the seven allowed figures are referenced. No duplicate influence plot returned.

Read-only checks matched all 3,448 original ledger source hashes and recomputed MAE/actual-fraction deletion AUC across 405 saved runs, with seed-then-patient aggregation. Display tables/macros match the supplied revision. Sampling allocation, covariance, Ridge/Elastic Net loss scaling and all 11 bibliography entries were checked. No inference settings, patient selection, source precision or historical receipts changed. No new inference or significance tests ran.

Full private records: `server_revision_qa.json`, `server_scientific_checks.json`, `revision_evidence_ledger.json`, and `source_confirmations.md`. Historical training/tuning provenance, exact illustrative-panel linkage and author/institutional declarations/permissions remain pending. ASA typography rules were rechecked at https://ww2.amstat.org/meetings/jsm/2026/submissions.cfm .

This is a local private-review build, with publication/submission approval pending. No clinical figures, patient artifacts, font files, full numeric evidence or built PDF/ZIP were added to Git. Unrelated local changes were preserved.
