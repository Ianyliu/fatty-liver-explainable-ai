# Accepted abstract versus revised proceedings abstract

The exact accepted text is preserved in `accepted_abstract.txt`, with the official JSM URL and SHA-256 in `accepted_abstract.json`. The revised source is `abstract.tex`; each built package includes the populated revised text and a comparison with the accepted version.

The title and the original sequence of ideas are retained: GNN diagnosis and interpretability → LIME applied to image subsets → two-stage adaptive sampling → conditional and marginal explanations → study application and visual explanation.

| Accepted statement | Revision and scientific reason |
| --- | --- |
| “computer-aided diagnosis” | Specifies multiple ultrasound images to make the grouped prediction setting explicit. |
| “images subsets (subgraphs)” | Grammatical correction to “image subsets (subgraphs)”; contribution unchanged. |
| Stage II draws 85% from the minority class pool “to ensure an informative design matrix” | Retains the intended 85/15 strategy, identifies singleton-predicted pools and the underrepresented subset class, and removes the unsupported guarantee. |
| Ridge/Elastic Net classifiers with 10-fold CV | Updates only the inaccurate method: fixed-alpha Ridge probability regression and training-only cross-validated Elastic Net; five-fold settings are detailed in Methods. |
| Bootstrap standard errors and confidence intervals | Removed because current completed experiments do not validate bootstrap inference. No alternative uncertainty estimates are invented. |
| Efficacy on 135 patients and intuitive explanations for clinicians | Uses the latest fully validated build population and a restrained principal finding. Clinical benefit is not established; visual explanations support examination of model behavior. |

The revised abstract avoids implementation repairs, configuration discrepancies and lists of unperformed analyses. The unfavorable fidelity and balance findings remain explicit in Results and Discussion, with a concise qualification in the abstract. Full and pilot populations are never interchanged silently.
