#!/usr/bin/env python3
"""Independently check a private full-cohort package; never run inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    if not np.isclose(actual, expected, rtol=0, atol=1e-13):
        raise ValueError(f'Numerical mismatch: {actual} != {expected}')


def verify(package):
    manifest = json.loads((package/'build_manifest.json').read_text())
    for entry in manifest['sources'] + manifest['artifacts']:
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Changed source or artifact: '+entry['path'])
    assert manifest['phase']=='full' and manifest['validated_runs']==405
    assert manifest['analysis_patients']==135 and manifest['new_inference_queries']==0
    receipt=json.loads((package/'validation_reconciliation.json').read_text())
    assert receipt['all_patient_seed_sets_complete'] and receipt['primary_patients_unavailable']==0
    plan=json.loads(Path(receipt['full_plan']['path']).read_text())
    summary=Path(receipt['validated_summary']['path']).parent
    f=pd.read_csv(summary/'fidelity_by_seed.csv')
    assert len(f)==810 and not f.duplicated(['patient_index','seed','arm']).any()
    assert f.patient_index.nunique()==135 and set(f.seed)=={0,1,2}
    counts=f.groupby(['patient_index','arm']).seed.agg(list)
    assert counts.apply(lambda values: sorted(values)==[0,1,2]).all()
    m=json.loads((package/'latex/generated/metrics.json').read_text())['primary']
    means=f.groupby(['patient_index','arm']).mean(numeric_only=True)
    paired=means.novel_mae.unstack('arm')
    delta=paired.adaptive-paired.random
    close(delta.mean(),m['paired_mae'])
    assert int((delta>0).sum())==m['random_better']
    assert int((delta<0).sum())==m['adaptive_better']
    assert int((delta==0).sum())==m['equal_mae']
    proportions=(means.training_positive_rows/1000).unstack('arm')
    distances=(proportions-.5).abs()
    assert int((distances.adaptive<distances.random).sum())==m['balance_comparison']['adaptive_closer_patients']
    deletion=pd.read_csv(summary/'deletion_by_patient.csv')
    for arm in ('random','adaptive'):
        rows=means.xs(arm,level='arm')
        raw=f[f.arm==arm]
        close(rows.novel_mae.mean(),m['arms'][arm]['mae'])
        close(rows.constant_baseline_novel_mae.mean(),m['arms'][arm]['constant_mae'])
        close(rows.training_positive_rows.mean()/1000,m['arms'][arm]['positive_fraction'])
        close(rows.unique_training_masks.mean(),m['arms'][arm]['unique_masks_mean'])
        assert int(((raw.training_positive_rows>0)&(raw.training_negative_rows>0)).sum())==m['arms'][arm]['both_prediction_classes_runs']
        for control in ('descending','ascending','random'):
            close(deletion.loc[deletion.arm==arm,control+'_mean_auc'].mean(),m['deletion'][arm][control])
    query_counts=Counter()
    for item in plan['patients']:
        for seed in (0,1,2):
            path=Path(item['plan']).parent/'runs'/f'seed-{seed}'/'report.json'
            query_counts.update(json.loads(path.read_text())['query_counts'])
    assert dict(query_counts)==m['query_counts'] and sum(query_counts.values())==911151

    # PDF image streams reverse rows. Compare native pixels, not screenshots.
    check=package/'figure_source_check';check.mkdir(exist_ok=True)
    figures=package/'latex/figures'
    for filename,prefix in [('figure1_workflow.pdf','workflow'),('figure2_influence.pdf','panel')]:
        subprocess.run(['pdfimages','-png',str(figures/filename),str(check/prefix)],check=True)
    originals=json.loads((ROOT/'manuscript/proceedings_2026/assets/readme_result_references.json').read_text())['references']
    original_files=[ROOT/'manuscript/proceedings_2026/assets/original_workflow.png']+[ROOT/r['path'] for r in originals]
    embedded=[check/'workflow-000.png']+[check/f'panel-{2*i:03}.png' for i in range(3)]
    for original,pdf_image in zip(original_files,embedded):
        a=np.asarray(Image.open(original).convert('RGBA'))
        b=np.flipud(np.asarray(Image.open(pdf_image).convert('RGB')))
        assert a.shape[:2]==b.shape[:2]
        mask=pdf_image.with_name(pdf_image.stem[:-3]+f'{int(pdf_image.stem[-3:])+1:03}.png')
        if mask.exists():
            assert np.array_equal(a[:,:,3],np.flipud(np.asarray(Image.open(mask))))
        else:
            assert (a[:,:,3]==255).all()
        assert np.array_equal(a[:,:,:3][a[:,:,3]>0],b[a[:,:,3]>0])
    ledger=json.loads((package/'evidence_ledger.json').read_text())
    historical=next(e for e in ledger['entries'] if e['id']=='figure:InfluenceCaption')
    assert historical['population']=='historical README illustrations' and 'not revalidated' in historical['validation']
    qa=json.loads((package/'pdf_qa.json').read_text())
    assert qa['letter_page'] and qa['times_new_roman_embedded'] and not qa['font_substitution']
    assert qa['nonsecured_pdf'] and qa['citations_resolved'] and not qa['overfull_boxes']
    result={'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'sources_verified':len(manifest['sources']),'artifacts_verified':len(manifest['artifacts']),
        'patients':135,'patient_seed_records':405,'missing_primary_patients':0,
        'primary_metrics_independently_reconciled':True,'query_stages_reconciled':dict(query_counts),
        'original_workflow_and_three_bar_charts_native_pixels_match':True,
        'historical_uncertainty_not_presented_as_current_evidence':True,
        'pdf_technical_checks':'passed; visual inspection recorded separately',
        'verifier_sha256':digest(__file__),'publication_allowed':False}
    (package/'review_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path)
    verify(parser.parse_args().package.resolve())
