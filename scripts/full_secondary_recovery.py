#!/usr/bin/env python3
"""Recover only LOO baselines that fail the unchanged primary-comparison gate.

The original secondary coordinator and inference sources stay byte-identical.
This adapter supplies a separately hashed retry mapping to the original summary
and report entry points; every scientific validation in those functions remains.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import full_secondary_study as base
import patient_study as study

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_VERIFIED=base.verified
GPU_POOLS={
    'NVIDIA RTX A5000':[f'amp{i:02}' for i in range(1,7)],
    'NVIDIA RTX A5500':[f'centurion{i:02}' for i in range(1,10)],
    'Tesla V100-SXM2-32GB':[f'p{i:02}' for i in range(1,5)],
    'NVIDIA H200 NVL':['h200a','h200b']}

SOURCES=['scripts/full_secondary_recovery.py','scripts/integrate_full_secondary_recovery.py',
         'slurm/full_secondary_recovery_loo.sbatch','slurm/full_secondary_recovery_stage.sbatch']


def primary_report(record):
    return json.loads((Path(record['plan']).parent/'runs/seed-0/report.json').read_text())


def match_original(actual,expected):
    return (actual['yhat']==expected['yhat'] and actual['edges']==expected['edges']
            and np.isclose(actual['p_class1'],expected['p_class1'],atol=1e-6,rtol=1e-6))


def mapping(original,root):
    records=[];tasks=[]
    for item in original['loo_records']:
        report=base.check_record(item,'loo');primary=primary_report(item)
        if match_original(report['original_prediction'],primary['original_prediction']):
            records.append(item.copy());continue
        index=len(tasks)
        replacement={**item,'task':index,'reused':False,'source_patient_index':item['patient_index'],
            'source_plan':str(root/'plan.json'),'directory':str(root/'loo'/f'task-{index:03}'),
            'original_directory':item['directory'],'primary_host':primary['host'].split('.')[0],
            'required_gpu_name':primary['gpu_name'],'allowed_hosts':GPU_POOLS[primary['gpu_name']]}
        records.append(replacement);tasks.append(replacement)
    return records,tasks


def verified(path):
    path=Path(path).resolve();plan=json.loads(path.read_text())
    original=ORIGINAL_VERIFIED(Path(plan['recovery_base_plan']))
    base.check_hashes(plan['files'])
    # Identity/source mappings, completed CPU outputs and fixed method settings
    # come from the original frozen plan; only the audited LOO retries differ.
    for key in ('patients','cpu_records','elastic_net','stability','versions','primary_figure_package','full_plan','full_summary','pilot_plan','pilot_review'):
        base.require(plan[key]==original[key],'Recovery changed original settings/mapping: '+key)
    expected_records,expected_tasks=mapping(original,path.parent)
    base.require(plan['loo_records']==expected_records and plan['loo_tasks']==expected_tasks,
                 'Recovery retry selection or same-model GPU requirement changed')
    base.require(len(expected_tasks)==38 and len(expected_records)==135,'Recovery coverage changed')
    return plan


def prepare(args):
    original=ORIGINAL_VERIFIED(args.base_plan);root=args.output.resolve()
    base.require(root.parent==args.base_plan.resolve().parent and root.name.startswith('recovery-'),
                 'Recovery must remain beside the original artifacts')
    base.require(not (root/'plan.json').exists(),'Recovery plan already exists')
    root.mkdir(exist_ok=True)
    reports=[]
    for command in ('cpu','loo'):
        for record in original[command+'_records']:
            base.check_record(record,command);reports.append(Path(record['directory'])/'report.json')
    records,tasks=mapping(original,root)
    base.require(len(tasks)==38,'Unexpected audited retry count')
    queries=sum(1+len(json.loads(Path(t['plan']).read_text())['images']) for t in tasks)
    base.require(queries==956,'Unexpected recovery query budget')
    plan={**original,'root':str(root),'recovery_base_plan':str(args.base_plan.resolve()),
        'recovery_created_at':datetime.now(timezone.utc).isoformat(),'cpu_tasks':[],
        'loo_records':records,'loo_tasks':tasks,'recovery_model_queries':queries,
        'original_loo_queries':3207,'total_executed_loo_queries':3207+queries,
        'recovery_policy':'Rerun every full-set probability OR edge-count mismatch on its primary seed-0 GPU model on an eligible node; unchanged 1e-6 absolute/relative tolerance, checkpoint and inference.',
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'files':[base.hashed(p) for p in [args.base_plan.resolve(),*reports,*[ROOT/name for name in SOURCES]]]}
    for name in SOURCES:
        destination=root/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
    study.write_json(root/'plan.json',plan)
    print(json.dumps({'plan':str(root/'plan.json'),'retries':len(tasks),'additional_queries':queries,
        'groups':dict(Counter(t['required_gpu_name'] for t in tasks))},indent=2))


def run(args):
    plan=verified(args.plan);base.require(0<=args.task<len(plan['loo_tasks']),'Retry outside frozen mapping')
    item=plan['loo_tasks'][args.task];child=study.verified_plan(Path(item['plan']))
    primary=primary_report(item);output=Path(item['directory']);output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','command':'loo','task':args.task,'patient_index':item['patient_index'],
        'seed':0,'plan_sha256':study.sha256(args.plan),'slurm_job_id':os.getenv('SLURM_JOB_ID'),
        'host':socket.gethostname(),'device':'cuda:0','input_hashes':[base.hashed(Path(item['plan']).parent/'runs/seed-0/report.json')],
        'original_directory':item['original_directory'],'primary_host':item['primary_host'],
        'allowed_hosts':item['allowed_hosts'],
        'required_gpu_name':item['required_gpu_name']}
    study.write_json(output/'report.json',report);started=time.monotonic();calls=0
    try:
        base.require(socket.gethostname().split('.')[0] in item['allowed_hosts'],'Retry outside eligible same-model node pool')
        predictor=study.PatientPredictor(child,'cuda:0',2,0)
        report['gpu_name']=predictor.torch.cuda.get_device_name(predictor.device)
        base.require(report['gpu_name']==item['required_gpu_name'],'Retry not on original GPU model')
        def predict(selected):
            nonlocal calls
            result=predictor(selected);calls+=1
            if calls==1:
                base.require(selected==child['images'],'First retry query must be full image set')
                report['full_set_comparison']={'primary':primary['original_prediction'],'retry':result}
                base.require(match_original(result,primary['original_prediction']),
                             'Same-model full-set probability/graph mismatch; keep gate closed')
            return result
        report.update(base.extra.loo_analysis(child,output,predict))
        base.validate_loo(child,report,base.read_rows(output/'leave_one_out.csv'))
        base.require(calls==1+len(child['images']),'Retry query ledger count mismatch')
        report['status']='passed';report['output_hashes']=[base.hashed(p) for p in sorted(output.iterdir()) if p.name!='report.json']
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        report['actual_model_queries']=calls;report['elapsed_seconds']=time.monotonic()-started
        study.write_json(output/'report.json',report)
    print('Passed same-model LOO recovery:',args.task,'patient ordinal',item['patient_index'])


def summarize(args):
    plan=verified(args.plan)
    # Explicit adapter for the recovery mapping. This retains every raw-artifact,
    # independent-refit, probability, identity and complete-cohort check in the
    # unchanged original summarize() implementation.
    base.verified=verified
    base.summarize(SimpleNamespace(plan=args.plan))
    output=args.plan.resolve().parent/'summary/analysis.json';analysis=json.loads(output.read_text())
    analysis['recovery']={'policy':plan['recovery_policy'],'repeated_patients':38,
        'original_executed_loo_queries':3207,'additional_recovery_queries':956,
        'total_executed_loo_queries':4163,'selected_analysis_queries':3207,
        'original_loo_artifacts_preserved':True}
    # These original-coordinator counts describe the first pass; state selected
    # records separately because one of the ten pilot LOO baselines was repeated.
    analysis['recovery']['selected_pilot_loo_patients']=sum(r['reused'] for r in plan['loo_records'])
    analysis['recovery']['selected_recovery_patients']=len(plan['loo_tasks'])
    analysis['limitations'].append('GPU-dependent full-set differences required 38 same-primary-model LOO reruns. Original seed runs used heterogeneous GPUs, so seed stability can include hardware numerical variability.')
    study.write_json(output,analysis)


def report(args):
    import sys
    import report_full_secondary as reporting
    base.verified=verified;sys.argv=[sys.argv[0],'--plan',str(args.plan)]
    reporting.main()
    root=args.plan.resolve().parent/'report';readme=root/'README.md';s=readme.read_text()
    s=s.replace('Thirty CPU records and ten LOO baselines are reused from the pilot.',
        'Thirty CPU records and nine selected LOO baselines are reused from the pilot. The other pilot LOO baseline is among 38 same-primary-model recovery runs. Selected analysis uses 3,207 LOO calls; 4,163 were executed in total, including 956 recovery calls. Original artifacts are retained.')
    readme.write_text(s)
    ledger=json.loads((root/'evidence_ledger.json').read_text())
    ledger['recovery']=json.loads((args.plan.resolve().parent/'summary/analysis.json').read_text())['recovery']
    study.write_json(root/'evidence_ledger.json',ledger)
    manifest=json.loads((root/'report_manifest.json').read_text())
    manifest['files']=[base.hashed(Path(r['path'])) for r in manifest['files']]
    manifest['files'].append(base.hashed(readme));manifest['recovery']=ledger['recovery']
    study.write_json(root/'report_manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--base-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for command in ('loo','summarize','report'):
        p=commands.add_parser(command);p.add_argument('--plan',type=Path,required=True)
        if command=='loo':p.add_argument('--task',type=int,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):globals()[args.command if args.command!='loo' else 'run'](args)


if __name__=='__main__':main()
