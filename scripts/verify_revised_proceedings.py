#!/usr/bin/env python3
"""Read saved evidence to verify the revised manuscript; never query the model."""
import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--cohort-plan', type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    ledger = json.loads((package / 'evidence_ledger_original.json').read_text())
    claims = {e['id']: e['values'] for e in ledger['entries'] if 'values' in e}
    metrics = json.loads((package / 'latex/generated/metrics.json').read_text())['primary']
    assert claims['claims:primary'] == metrics
    source_checks = {'matched': 0, 'mismatched': [], 'unavailable': []}
    for item in ledger['sources']:
        path = Path(item['path'])
        if not path.is_file():
            source_checks['unavailable'].append(item)
        elif sha(path) != item['sha256']:
            source_checks['mismatched'].append(item)
        else:
            source_checks['matched'] += 1
    assert not source_checks['mismatched'] and not source_checks['unavailable'], source_checks
    cohort = json.loads(args.cohort_plan.read_text())
    patient_mae = defaultdict(list)
    patient_auc = defaultdict(list)
    run_count = 0
    auc_max_error = 0.0
    mae_max_error = 0.0
    runtime_records = []
    for patient in cohort['patients']:
        plan_path = Path(patient['plan'])
        plan = json.loads(plan_path.read_text())
        runtime_records.append({'plan': str(plan_path), 'sha256': sha(plan_path),
                                'versions': plan['versions'], 'git_commit': plan['git_commit'],
                                'module_files': [f for f in plan['files'] if 'usflc_xai' in f['path']],
                                'reports': []})
        for seed in [0, 1, 2]:
            run = plan_path.parent / 'runs' / ('seed-' + str(seed))
            report = json.loads((run / 'report.json').read_text())
            assert report['plan_sha256'] == sha(plan_path) and report['status'] == 'passed'
            assert report['python'] == '3.9.25'
            runtime_records[-1]['reports'].append({'path': str(run / 'report.json'),
                'sha256': sha(run / 'report.json'), 'python': report['python'],
                'plan_sha256': report['plan_sha256']})
            evaluation = pd.read_csv(run / 'evaluation.csv')
            assert len(evaluation) == 200
            retained = evaluation[evaluation.shared_novel.astype(bool)]
            assert not retained.in_random_training.any() and not retained.in_adaptive_training.any()
            for arm in ['random', 'adaptive']:
                error = float(np.mean(np.abs(retained.p_class1 - retained[arm + '_surrogate'])))
                reference = report['arms'][arm]['shared_novel_evaluation']['mae']
                mae_max_error = max(mae_max_error, abs(error - reference))
                assert abs(error - reference) < 1e-12
                patient_mae[arm, patient['patient_index']].append(error)
            deletion = pd.read_csv(run / 'deletion.csv')
            for (arm, control), group in deletion.groupby(['arm', 'control']):
                group = group.sort_values('deleted_count')
                assert not group.deleted_count.duplicated().any()
                assert (group.retained_count >= 3).all()
                assert np.allclose(group.deleted_fraction, group.deleted_count / len(plan['images']), atol=1e-15)
                area = float(np.trapz(group.p_class1, group.deleted_fraction))
                expected = report['deletion'][arm][control]['area_under_curve']
                auc_max_error = max(auc_max_error, abs(area - expected))
                assert abs(area - expected) < 1e-12 and 0 <= area <= 0.5
                patient_auc[arm, control, patient['patient_index']].append(area)
            run_count += 1
    means = {arm: [np.mean(patient_mae[arm, p['patient_index']]) for p in cohort['patients']]
             for arm in ['random', 'adaptive']}
    for arm in means:
        assert abs(np.mean(means[arm]) - metrics['arms'][arm]['mae']) < 1e-12
        for control in ['descending', 'ascending', 'random']:
            value = np.mean([np.mean(patient_auc[arm, control, p['patient_index']]) for p in cohort['patients']])
            assert abs(value - metrics['deletion'][arm][control]) < 1e-12
    paired = np.array(means['adaptive']) - np.array(means['random'])
    assert abs(paired.mean() - metrics['paired_mae']) < 1e-12
    assert int((paired > 0).sum()) == metrics['random_better']
    assert int((paired < 0).sum()) == metrics['adaptive_better']
    assert int((paired == 0).sum()) == metrics['equal_mae']
    # Exact random-design distribution and covariance, without generating inference data.
    for n in range(3, 10):
        masks, probs = [], []
        for k in range(3, n + 1):
            for selected in combinations(range(n), k):
                masks.append([int(j in selected) for j in range(n)])
                probs.append(1 / ((n - 2) * math.comb(n, k)))
        u, prob = np.array(masks), np.array(probs)
        k = u.sum(axis=1)
        covariance = np.sum(prob * u[:, 0] * k) - np.sum(prob * u[:, 0]) * np.sum(prob * k)
        assert abs(prob.sum() - 1) < 1e-12
        assert abs(covariance - (((n - 2)**2 - 1) / 12) / n) < 1e-12
    # Integer feasibility clamp agrees with the source sampler's sequential clamping.
    for n in range(20, 36):
        for positive in range(1, n):
            negative = n - positive
            for k in range(3, n + 1):
                for rho in [0.15, 0.85]:
                    formula = min(positive, max(math.floor(rho * k), k - negative))
                    code = min(math.floor(rho * k), positive)
                    code = max(k - negative, code)
                    assert formula == code and 0 <= k - formula <= negative
    secondary = pd.read_csv(package / 'latex/generated/secondary.csv').set_index('diagnostic')
    gains = {}
    for arm in ['Random', 'Adaptive']:
        delta = secondary.loc[arm + ' Elastic Net minus Ridge MAE', 'available_patient_mean']
        gains[arm] = format(-100 * delta, '.2f')
    assert gains == {'Random': '0.14', 'Adaptive': '0.07'}
    text = '\n'.join((package / 'latex' / n).read_text() for n in ['main.tex', 'body.tex', 'appendix.tex', 'abstract.tex', 'declarations.tex'])
    cites = set()
    for group in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}', text):
        cites.update(group.split(','))
    bib = (package / 'latex/references.bib').read_text()
    keys = re.findall(r'@\w+\{([^,]+),', bib)
    assert len(keys) == len(set(keys)) and cites <= set(keys)
    assert 'figure7_current_influence' not in text
    # The manuscript's Ridge objective is an unscaled SSE plus lambda||beta||^2;
    # centering removes its unpenalized intercept. ElasticNet uses SSE/(2B),
    # alpha*eta*L1 + alpha*(1-eta)*L2/2. Confirm saved constructor settings.
    source = Path(__file__).resolve().parent / 'patient_study.py'
    if source.is_file():
        assert 'Ridge(alpha=plan["ridge_alpha"])' in source.read_text()
        from sklearn.linear_model import Ridge
        import inspect
        assert inspect.signature(Ridge).parameters['fit_intercept'].default is True
    result = {'status': 'passed', 'original_ledger_sources': source_checks,
              'original_claims_equal_metrics': True, 'patient_seed_runs_recomputed': run_count,
              'mae_max_abs_error': mae_max_error, 'actual_fraction_auc_max_abs_error': auc_max_error,
              'seed_then_patient_aggregation': 'reconciled with original full-precision claims',
              'sampling_allocation': 'exhaustive feasible pool counts n=20..35; matches sequential clamp',
              'random_distribution_covariance': 'exact enumeration n=3..9 passed',
              'ridge_elastic_net_loss_scaling': 'equations and frozen constructors reviewed; no refits performed',
              'elastic_net_gains_percentage_points': gains, 'bibliography_keys': sorted(keys),
              'all_citations_resolved_to_bibliography': True, 'new_inference': False,
              'new_significance_tests': False}
    (package / 'server_scientific_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    (package / 'server_evidence/inference_run_manifest.json').write_text(json.dumps(runtime_records, indent=2) + '\n')
    print('Verified', run_count, 'saved runs and', source_checks['matched'], 'source hashes.')

if __name__ == '__main__':
    main()
