# Abstract wording update

The latest user-supplied abstract is installed in `abstract.tex` and the private editable package. The displayed text follows that request exactly; LaTeX escapes the percent sign and retains the evidence-backed MAE macros, displaying 0.066 and 0.119.

Two scientific discrepancies remain for author review outside the manuscript:

- The requested “10-fold cross-validated Elastic Net” conflicts with the frozen five-fold training-only CV and the appendix. No inference or tuning settings were changed.
- Stage II requests an 85% deficient-class pool allocation, with integer rounding and pool-shortage reallocation. The wording “drawing 85% ... to ensure an informative design matrix” overstates the achieved fraction and guarantee. The source implementation is `sampling_marginal_relation_pipeline.py:289,319–322`.

See the private `server_evidence/abstract_author_revision.json` for the exact request and backup location. These statements were retained as explicitly requested wording; they are not newly verified scientific claims. Publication/submission remains pending author review and source permissions.
