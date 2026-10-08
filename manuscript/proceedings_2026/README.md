# JSM 2026 proceedings review package

This is an exploratory evaluation of image-subset explanations for a fixed fatty-liver GNN. The manuscript follows the accepted poster's narrative while documenting the implemented probability-regression methods. The ten-patient pilot and the later 135-patient expansion must remain distinguishable.

Build instructions and the evidence ledger are generated with each private package. Generated results, figures, manuscripts and patient-level data belong below ignored `outputs/proceedings_2026/`, never in Git. Source files here and the publication exporter are versioned. No inference source or frozen configuration is changed for this manuscript.

## Release gate

**Internal review only. Do not submit, push, publish or publicly release the manuscript or generated package until Ian Liu has reviewed it and Tso-Jung Yen has had an opportunity to approve authorship and manuscript content.**

The release checklist records author order, affiliations, correspondence, contributions, funding, competing interests, secondary-use authorization, code/data restrictions, AI-assisted preparation, final figures and typography. Missing information is not a declaration of “none.”

The review draft may use Nimbus Roman where Times New Roman is unavailable. This substitution must be resolved or explicitly accepted by ASA before marking a PDF submission-ready.

## Analysis boundaries

- Primary response: GNN class-1 probability; fixed-alpha Ridge regression.
- Elastic Net: training-only shuffled five-fold CV on the ten-patient cohort.
- Primary fidelity: shared evaluation masks unseen in either training arm, with draw multiplicity retained.
- Seeds are repeated measurements; average within patients before cohort summaries.
- Deletion interventions measure model behavior, not clinical or causal importance.
- No new bootstrap inference, significance testing, physician evaluation or inference experiments are included.
- Full mode must reject absent or incomplete full-cohort validation, never substitute partial results.

The official proceedings deadline is October 9, 2026. Internal review target: October 9 at noon Eastern, subject to completed-run validation and author confirmations.
