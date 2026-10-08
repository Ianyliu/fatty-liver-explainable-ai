# Parallel diagnostics and baselines

These additional exploratory analyses use the same frozen ten-patient cohort as [the ongoing P0 study](COHORT_EXPERIMENTS.md). They run alongside it using separate scripts, plans and output directories. Existing P0 scientific sources, protocol, environment and results remain unchanged.

## Submitted work

| Work | Execution | Dependency |
| --- | --- | --- |
| Design diagnostics, Stage-II audit, Elastic Net and marginal-correlation rankings | CPU array **9557701**, up to four tasks concurrently; task 0 already passed locally | Each corresponding P0 patient/seed task succeeds |
| Leave-one-image-out baseline | GPU array **9557699**, one task at a time | Independent of P0 sampling and fits |
| Stability and combined report | CPU job **9557715** | Both additional arrays succeed |

The plan is `outputs/parallel_experiments/ten-patient-20261007-v1/plan.json`. Submission receipts and source snapshots are retained beside it. The GPU baseline makes **210 additional GNN calls** across ten patients. All other added analyses use saved tables and require no GNN inference.

The CPU array uses `aftercorr:9557642`, matching each array index to its P0 task. Tasks 1–7 had already finished upstream when this dependency was created and remained pending; their passed reports were checked and those seven dependencies explicitly cleared. Task 0 was separately validated locally before bulk submission. This adjustment is recorded in `dependency_releases.json`; later tasks retain their per-task success dependency. The final summary still requires every one of the 30 CPU analyses and ten LOO tasks to pass.

## Analysis settings

**Design audit:** achieved predicted-class balance, duplicate masks, subset-size distributions, per-image inclusion/exclusion, and rank/condition number of the column-centered binary design. Singular designs have an explicit flag and unavailable finite condition number. Stage-II membership and requested/realized pool counts are reconstructed from the singleton query ledger and sequential class quotas; every biased row must match the sampler's documented deficit-reallocation rule. The first real-data task passed this check and recorded reallocation in all 500 biased draws, a diagnostic for that patient/seed rather than a cohort conclusion.

**Elastic Net:** class-1 probability target; unscaled binary inclusion features with intercept. Five shuffled training-only CV folds use the existing training seed, 25 alpha values from 0.0001 through 1, and l1 ratios 0.1, 0.5, 0.9 and 1. Settings minimize CV mean squared error. Evaluation responses never select settings or fit coefficients. Convergence warnings fail the task. Ridge and Elastic Net are evaluated on the same existing all-draw and shared-novel sets, with training-mean baselines. Raw surrogate scores remain unclipped.

**Marginal correlations:** Pearson correlation of each inclusion column with the training GNN probability provides a separate ranking. Constant columns/responses produce unavailable correlations. This is a ranking comparison, not a fitted multivariable probability surrogate.

**Stability:** across the three existing seeds, report Spearman correlation, top-five Jaccard similarity, coefficient/correlation sign agreement, and per-image top-five frequency. Image IDs break ranking ties, with a 1e-10 zero tolerance. Constant or undefined ranking vectors are marked unavailable rather than producing artificial perfect top-five agreement. Final output covers all ten patients; any available `stability_preview/` is explicitly partial and includes only patients with all three CPU tasks passed.

**LOO:** query the full graph and each graph omitting exactly one image through the unchanged real predictor. Record both class-1 probability change and original predicted-class probability change; these have opposite signs when the original prediction is class 0. Each subset rebuilds its graph. This measures single-node removal; ranked multi-node deletion for Elastic Net/correlation, repeated random-deletion controls, sampling-ratio sensitivity, matched-query-budget comparisons and a five-seed/30-patient stability study remain separate experiments.

## Validation and outputs

All **46 regression tests** pass. Added tests cover pool-deficit reconstruction, singular designs, constant rankings, known ranking stability, negative-class LOO interpretation, complete summary coverage and evidence corruption. A leakage test changes evaluation responses and confirms that fitted rankings and selected Elastic Net settings remain identical while evaluation metrics change.

Under the new private experiment root:

- `cpu/task-NN/`: diagnostics, image inclusion, Stage-II pool counts, rankings, fitted Elastic Net settings, evaluation scores and fidelity.
- `loo/task-NN/`: original prediction and every omitted-image prediction/probability change.
- `summary/`: generated only after all planned work passes; seed/patient fidelity, design diagnostics, stability, top-five frequency and LOO tables, plus provenance and limitations.
- `regression-tests.log`, `submissions.json`, `source_snapshot/`: validation and execution evidence.

The summary hashes all input/output evidence, checks identities and LOO budgets/deltas, then averages paired surrogate MAE differences across seeds within patient before weighting patients equally. An unavailable patient result keeps the complete-cohort mean unavailable. All identifiers and numerical evidence stay under ignored storage.

```bash
squeue -j 9557642,9557645,9557699,9557701,9557715
sacct -j 9557699,9557701,9557715 --format=JobID,State,ExitCode,Elapsed
```

The final summary can also be run manually after all tasks succeed with `.venv/bin/python scripts/parallel_experiments.py summarize --plan outputs/parallel_experiments/ten-patient-20261007-v1/plan.json`; its output directory must be new.
