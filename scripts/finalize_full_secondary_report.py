#!/usr/bin/env python3
"""Resume final report assembly, preserving completed inference and 270 profiles.

Recovery plans bind the original plan indirectly. Resolve original figure
exports through that plan rather than assuming they appear in the retry
manifest. All frozen scientific/source checks remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import full_secondary_recovery as recovery
import full_secondary_study as secondary
import patient_study as study
import report_full_secondary as plotting
from proceedings_style import configure, save

STEMS=['figure1_workflow','figure2_influence','figure3_sampling','figure4_fidelity','figure5_deletion']


def primary_sources(plan):
    original=json.loads(Path(plan['recovery_base_plan']).read_text())
    folder=Path(plan['primary_figure_package'])/'latex/figures'
    entries={Path(row['path']).name:row for row in original['files'] if Path(row['path']).parent==folder}
    expected={stem+'.'+suffix for stem in STEMS for suffix in ('pdf','svg','png')}
    secondary.require(set(entries)==expected,'Original plan must bind all 15 primary figure exports')
    secondary.check_hashes(entries.values())
    return [entries[name] for name in sorted(expected)]


def check_profile_csv(path,expected):
    actual=pd.read_csv(path)
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_names=False,check_exact=False,atol=1e-12,rtol=1e-9)


def validate_export(stem):
    for suffix in ('pdf','svg','png'):
        secondary.require(stem.with_suffix('.'+suffix).stat().st_size>100,'Empty figure: '+str(stem))
    with Image.open(stem.with_suffix('.png')) as preview:preview.verify()
    ET.parse(stem.with_suffix('.svg'))
    info=subprocess.check_output(['pdfinfo',str(stem.with_suffix('.pdf'))],text=True)
    secondary.require(any(line.split()==['Pages:','1'] for line in info.splitlines()),'Invalid single-page figure PDF: '+str(stem))


def validate_profiles(plan,summary,folder):
    rankings=pd.read_csv(summary/'rankings.csv');loo=pd.read_csv(summary/'leave_one_out.csv');profiles=[]
    expected_names={f"patient-{item['patient_index']:03}-{arm}" for item in plan['patients'] for arm in secondary.extra.ARMS}
    for suffix in ('csv','pdf','svg','png'):
        secondary.require({p.stem for p in folder.glob('*.'+suffix)}==expected_names,'Incomplete or unexpected profile exports: '+suffix)
    for item in plan['patients']:
        index=item['patient_index'];child=json.loads(Path(item['plan']).read_text());images=child['images']
        for arm in secondary.extra.ARMS:
            group=rankings[(rankings.patient_index==index)&(rankings.arm==arm)]
            means=plotting.patient_means(group,['value'],['image_id','method']).pivot(index='image_id',columns='method',values='value').loc[images]
            means['LOO']=loo[loo.patient_index==index].set_index('omitted_image').loc[images,'delta_class1'].to_numpy()
            means['image_label']=[f'I{i+1:02}' for i in range(len(images))]
            expected=means.sort_values('marginal_correlation',ascending=False,kind='stable',na_position='last').reset_index(drop=True)
            name=f'patient-{index:03}-{arm}'
            check_profile_csv(folder/(name+'.csv'),expected);validate_export(folder/name)
            profiles.append({'patient_index':index,'arm':arm,'figures':name,'images':len(images),
                             'publication_permission':'unconfirmed; private review only'})
    return profiles


def finalize(args):
    plan=recovery.verified(args.plan);root=args.plan.resolve().parent;summary=root/'summary';output=root/'report'
    analysis=json.loads((summary/'analysis.json').read_text())
    secondary.require(analysis['status']=='passed' and analysis['validated_cpu_tasks']==405
        and analysis['validated_loo_patients']==135 and analysis['plan_sha256']==study.sha256(args.plan),
        'Complete scientific validation must pass before resuming report assembly')
    secondary.require(not (output/'report_manifest.json').exists(),'Completed report already exists; inspect it rather than repeat assembly')
    secondary.check_hashes(analysis['evidence']+analysis['summary_files'])
    entries=primary_sources(plan)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');backup=root/('report-assembly-before-'+stamp)
    backup.mkdir(exist_ok=False)
    for name in ('tables','figures'):
        shutil.copytree(output/name,backup/name)
    before=[secondary.hashed(p) for p in sorted((output/'patient_profiles').glob('*')) if p.is_file()]
    profiles=validate_profiles(plan,summary,output/'patient_profiles')
    tables=output/'tables';figures=output/'figures'
    f,s,l,loo,rows=plotting.cohort_tables(summary,tables)
    captions=plotting.summary_figures(f,s,l,loo,figures)
    primary=output/'preserved_primary_figures';primary.mkdir(exist_ok=True)
    for entry in entries:
        destination=primary/Path(entry['path']).name
        if destination.exists():secondary.require(study.sha256(destination)==entry['sha256'],'Existing preserved figure differs')
        else:shutil.copy2(entry['path'],destination)
    secondary.check_hashes([{'path':str(primary/Path(row['path']).name),'sha256':row['sha256']} for row in entries])
    configure();fig,axes=plt.subplots(3,2,figsize=(11,16));fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.96,hspace=.15,wspace=.10)
    previews=[primary/'figure1_workflow.png',primary/'figure2_influence.png',figures/'secondary_fidelity.png',figures/'secondary_diagnostics.png',figures/'loo_behavior.png',primary/'figure4_fidelity.png']
    for ax,path in zip(axes.flat,previews):
        ax.imshow(plt.imread(path));ax.axis('off');ax.set_title(path.stem.replace('_',' '),fontsize=10)
    save(fig,output,'contact_sheet')
    for name in ('secondary_fidelity','secondary_diagnostics','loo_behavior'):validate_export(figures/name)
    validate_export(output/'contact_sheet')
    secondary.check_hashes(before)
    sources=[secondary.hashed(args.plan),secondary.hashed(Path(plan['recovery_base_plan'])),secondary.hashed(summary/'analysis.json'),secondary.hashed(Path(__file__)),*[secondary.hashed(p) for p in sorted(summary.glob('*.csv'))]]
    ledger={'created_at':datetime.now(timezone.utc).isoformat(),'population':'135 eligible positive patients; exploratory post-pilot secondary expansion',
        'analysis_unit':'patient; average all three seeds or three seed pairs before cohort summaries','sources':sources,
        'claims':rows,'figure_captions':captions,'patient_profiles':profiles,'recovery':analysis['recovery'],
        'uncertainty':'Descriptive SD only; no confidence intervals, significance tests or clinical validation.',
        'publication_allowed':False}
    study.write_json(output/'evidence_ledger.json',ledger);study.write_json(output/'figure_captions.json',captions)
    (output/'README.md').write_text('# Full-cohort secondary results — private review\n\nAll 405 CPU records and 135 selected LOO baselines passed complete validation. All 270 patient profiles were checked against validated coefficients and LOO values, and their PDF/SVG/PNG exports were verified without regeneration. Original Figures 1–5 were resolved from the original frozen plan and preserved byte-for-byte.\n\nThirty CPU records and nine selected LOO baselines are reused from the pilot. The other pilot LOO baseline is among 38 same-primary-model recovery runs. Selected analysis contains 3,207 calls; total executed LOO queries are 4,163 including 956 recovery calls. Original artifacts remain preserved.\n\nSee tables/, figures/, patient_profiles/, contact_sheet.pdf, evidence_ledger.json and assembly_receipt.json. Undefined metrics retain explicit availability counts; no bootstrap, significance or clinical validation is implied. Clinical thumbnails remain private pending publication permission. The dependent build updates the existing user-edited manuscript in place with a backup and conflict checks. No public release is authorized.\n')
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True);shutil.copy2(__file__,snapshot/Path(__file__).name)
    receipt={'status':'passed','resumed_failed_job':'9565643','root_cause':'Primary figure entries are in the original frozen plan, indirectly referenced by the recovery plan.',
        'validated_patients':135,'patient_profiles':270,'profiles_preserved_unchanged':True,
        'primary_exports_verified':15,'original_figures_unchanged':[row for row in entries if Path(row['path']).stem in STEMS[:2]],
        'original_partial_tables_figures_backup':str(backup),'source':secondary.hashed(Path(__file__)),
        'new_gnn_queries':0,'publication_allowed':False}
    study.write_json(output/'assembly_receipt.json',receipt)
    study.write_json(output/'report_manifest.json',{'status':'passed','plan_sha256':study.sha256(args.plan),'validated_patients':135,
        'patient_profiles':270,'new_gnn_queries_in_report':0,'publication_allowed':False,'recovery':analysis['recovery'],
        'resumed_report_assembly':True,'files':[secondary.hashed(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='report_manifest.json']})
    print('Completed report assembly; 270 existing profiles preserved:',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):finalize(args)
