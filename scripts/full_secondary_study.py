#!/usr/bin/env python3
"""Expand completed pilot secondary analyses without altering frozen inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge

import full_cohort_study as full
import parallel_experiments as extra
import patient_study as study
from summarize_patient_study import column, matrix, read_rows

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['scripts/full_secondary_study.py','scripts/report_full_secondary.py',
         'scripts/integrate_full_secondary.py',
         'scripts/parallel_experiments.py','scripts/summarize_patient_study.py',
         'scripts/full_cohort_study.py','scripts/cohort_study.py','scripts/proceedings_style.py',
         'slurm/full_secondary_cpu.sbatch','slurm/full_secondary_loo.sbatch',
         'slurm/full_secondary_summary.sbatch','slurm/full_secondary_report.sbatch',
         'slurm/full_secondary_integrate.sbatch']


def require(condition, message):
    if not condition: raise ValueError(message)


def hashed(path):
    return {'path':str(Path(path).resolve()),'sha256':study.sha256(path)}


def check_hashes(entries):
    for row in entries:
        require(study.sha256(row['path'])==row['sha256'],'Evidence/source changed: '+row['path'])


def mappings(patients, pilot, pilot_path, root):
    old={p['patient']:p for p in pilot['patients']}
    cpu_records=[];loo_records=[];cpu_tasks=[];loo_tasks=[]
    for item in patients:
        previous=old.get(item['patient']);index=item['patient_index']
        for seed in (0,1,2):
            task=3*previous['patient_index']+seed if previous else len(cpu_tasks)
            record={'patient_index':index,'seed':seed,'plan':item['plan'],'task':task,
                'reused':previous is not None,'source_patient_index':previous['patient_index'] if previous else index,
                'source_plan':str(pilot_path.resolve()) if previous else str(root/'plan.json'),
                'directory':str(pilot_path.parent/'cpu'/f'task-{task:02}') if previous else str(root/'cpu'/f'task-{task:03}')}
            cpu_records.append(record)
            if not previous:cpu_tasks.append(record)
        task=previous['patient_index'] if previous else len(loo_tasks)
        record={'patient_index':index,'seed':0,'plan':item['plan'],'task':task,
            'reused':previous is not None,'source_patient_index':previous['patient_index'] if previous else index,
            'source_plan':str(pilot_path.resolve()) if previous else str(root/'plan.json'),
            'directory':str(pilot_path.parent/'loo'/f'task-{task:02}') if previous else str(root/'loo'/f'task-{task:03}')}
        loo_records.append(record)
        if not previous:loo_tasks.append(record)
    return cpu_records,loo_records,cpu_tasks,loo_tasks


def check_mapping(plan):
    patients=plan['patients']
    require(len(patients)==135 and [p['patient_index'] for p in patients]==list(range(135))
            and len({p['patient'] for p in patients})==135,'Invalid full patient mapping')
    expected=mappings(patients,json.loads(Path(plan['pilot_plan']).read_text()),
                      Path(plan['pilot_plan']),Path(plan['root']))
    for key,value in zip(('cpu_records','loo_records','cpu_tasks','loo_tasks'),expected):
        require(plan[key]==value,'Changed task mapping: '+key)
    require(len(plan['cpu_tasks'])==375 and len(plan['loo_tasks'])==125,'Unexpected expansion coverage')
    require(len(plan['cpu_records'])==405 and len(plan['loo_records'])==135,'Incomplete secondary coverage')


def verified(path):
    plan=json.loads(Path(path).read_text());check_hashes(plan['files']);check_mapping(plan)
    for name,version in plan['versions'].items():
        require(importlib.metadata.version(name)==version,'Changed runtime version: '+name)
    pilot=json.loads(Path(plan['pilot_plan']).read_text())
    require(plan['elastic_net']==pilot['elastic_net'] and plan['stability']==pilot['stability'],
            'Secondary settings differ from completed pilot')
    return plan


def check_record(record, command, verify_hashes=True):
    directory=Path(record['directory']);report=json.loads((directory/'report.json').read_text())
    require(report['status']=='passed' and report['command']==command and report['task']==record['task']
            and report['patient_index']==record['source_patient_index'] and report['seed']==record['seed']
            and report['plan_sha256']==study.sha256(record['source_plan']),
            'Failed or misidentified '+command+' record: '+str(directory))
    require(bool(report['output_hashes']),'Missing output evidence manifest')
    if verify_hashes:check_hashes(report.get('input_hashes',[])+report['output_hashes'])
    return report


def prepare(args):
    output=args.output.resolve()
    require((ROOT/'outputs/parallel_experiments').resolve() in output.parents and not output.exists(),
            'Use a new private parallel-experiment directory')
    primary=full.verified(args.full_plan);pilot=extra.verified(args.pilot_plan)
    summary=args.full_plan.parent/'summary/analysis.json';analysis=json.loads(summary.read_text())
    require(analysis['status']=='passed' and analysis['validated_patient_seed_runs']==405
            and analysis['plan_sha256']==study.sha256(args.full_plan),'Full P0 gate has not passed')
    review=json.loads(args.pilot_review.read_text())
    require(review['status']=='validated' and review['patients']==10 and review['independent_refits']==120,
            'Pilot independent-refit review is incomplete')
    require(review['report_source_sha256']==study.sha256(ROOT/'scripts/report_ten_patient_completion.py'),
            'Pilot review source changed')
    check_hashes(review['evidence'])
    # Verify shared model/data files once, instead of rereading the same weights for every patient.
    inputs={}
    images=0
    for item in primary['patients']:
        child=json.loads(Path(item['plan']).read_text());full.check_settings(child)
        require(child['patient']==item['patient'] and child['ground_truth']==1,'Changed child identity/eligibility')
        images+=len(child['images'])
        for entry in child['files']:
            require(entry['path'] not in inputs or inputs[entry['path']]['sha256']==entry['sha256'],
                    'Conflicting frozen child input hashes')
            inputs[entry['path']]=entry
    check_hashes(inputs.values());require(images==3072,'Changed image coverage')
    records=mappings(primary['patients'],pilot,args.pilot_plan,output)
    total_loo=sum(1+len(json.loads(Path(p['plan']).read_text())['images']) for p in primary['patients'])
    plan={'schema':1,'prepared_at':datetime.now(timezone.utc).isoformat(),'root':str(output),
        'scope':'All 135 eligible positive test09 patients; secondary expansion after pilot and primary results were inspected',
        'full_plan':str(args.full_plan.resolve()),'full_summary':str(summary.resolve()),
        'pilot_plan':str(args.pilot_plan.resolve()),'pilot_review':str(args.pilot_review.resolve()),
        'patients':primary['patients'],'elastic_net':pilot['elastic_net'],'stability':pilot['stability'],
        'loo':{**pilot['loo'],'planned_model_queries':total_loo,'reused_model_queries':210,'new_model_queries':total_loo-210},
        **dict(zip(('cpu_records','loo_records','cpu_tasks','loo_tasks'),records)),
        'new_independent_refits':1500,'reused_independent_refits':120,
        'versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','pandas','matplotlib','Pillow')},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'primary_figure_package':str(args.primary_package.resolve()),'publication_allowed':False}
    files=[args.full_plan,summary,args.pilot_plan,args.pilot_review,
           *[Path(p['plan']) for p in primary['patients']],*[ROOT/name for name in SOURCES]]
    for command,items in [('cpu',records[0]),('loo',records[1])]:
        for record in items:
            if record['reused']:
                check_record(record,command);files.append(Path(record['directory'])/'report.json')
    for suffix in ('pdf','svg','png'):
        found=sorted((args.primary_package/'latex/figures').glob('figure[1-5]_*.'+suffix))
        require(len(found)==5,'Missing validated primary figure exports: '+suffix);files.extend(found)
    plan['files']=[hashed(p) for p in dict.fromkeys(files)]
    check_mapping(plan)
    output.mkdir(parents=True)
    for name in SOURCES:
        target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    study.write_json(output/'plan.json',plan)
    print(json.dumps({'plan':str(output/'plan.json'),'new_cpu_tasks':375,'new_loo_patients':125,
                      'new_gnn_queries':total_loo-210,'total_loo_queries':total_loo,'frozen_inputs_verified_once':len(inputs)},indent=2))


def independent_cpu_check(child, directory, output, seed, settings):
    """Match pilot review: refit Ridge and selected EN; verify saved CV minimum."""
    images=child['images'];evaluation=read_rows(directory/'evaluation.csv')
    ex=matrix(evaluation,images);rankings=read_rows(output/'rankings.csv')
    for arm in extra.ARMS:
        training=read_rows(directory/f'{arm}_training.csv');x=matrix(training,images);y=column(training,'p_class1')
        ridge=Ridge(alpha=child['ridge_alpha']).fit(x,y)
        ridge_rows=[r for r in rankings if r['arm']==arm and r['method']=='ridge']
        require([r['image_id'] for r in ridge_rows]==images,'Changed Ridge column order')
        require(np.allclose(ridge.coef_,column(ridge_rows,'value'),atol=1e-9,rtol=1e-7),'Ridge refit mismatch')
        require(np.allclose(ridge.predict(ex),column(evaluation,arm+'_surrogate'),atol=1e-9,rtol=1e-7),'Ridge score mismatch')
        fitted=json.loads((output/f'{arm}_elastic_net.json').read_text());mse=np.asarray(fitted['cv_mean_mse'])
        ratio=settings['l1_ratios'].index(fitted['l1_ratio']);grid=np.asarray(fitted['alpha_grid'])
        grid=grid[ratio] if grid.ndim==2 else grid;alpha=int(np.argmin(abs(grid-fitted['alpha'])))
        require(np.isclose(mse[ratio,alpha],mse.min(),atol=1e-12,rtol=1e-9),'Selected EN settings do not minimize recorded training CV MSE')
        require(fitted['evaluation_rows_used_for_fit']==0,'Evaluation leakage in fit record')
        model=ElasticNet(alpha=fitted['alpha'],l1_ratio=fitted['l1_ratio'],tol=settings['tol'],max_iter=settings['max_iter'],selection='cyclic')
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y)
        coefficients=[r for r in rankings if r['arm']==arm and r['method']=='elastic_net']
        require([r['image_id'] for r in coefficients]==images,'Changed Elastic Net column order')
        require(np.allclose(model.coef_,column(coefficients,'value'),atol=1e-9,rtol=1e-7),'Elastic Net refit mismatch')
        scores=read_rows(output/f'{arm}_evaluation.csv')
        require(np.allclose(model.predict(ex),column(scores,'elastic_net_score'),atol=1e-9,rtol=1e-7),'Elastic Net evaluation refit mismatch')
        correlations=[r for r in rankings if r['arm']==arm and r['method']=='marginal_correlation']
        require([r['image_id'] for r in correlations]==images,'Changed Pearson column order')
        expected=extra.marginal_correlations(x,y)
        actual=[float(r['value']) if r['value'] else None for r in correlations]
        require(all((a is None and b is None) or (a is not None and b is not None and np.isclose(a,b,atol=1e-12))
                    for a,b in zip(actual,expected)),'Pearson reconstruction mismatch')
    return 4


def run(args):
    plan=verified(args.plan);items=plan[args.command+'_tasks']
    require(0<=args.task<len(items),'Task outside immutable mapping');item=items[args.task]
    child=study.verified_plan(Path(item['plan']));directory=Path(item['plan']).parent/'runs'/f"seed-{item['seed']}"
    output=Path(item['directory']);output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','command':args.command,'task':args.task,'patient_index':item['patient_index'],
        'seed':item['seed'],'plan_sha256':study.sha256(args.plan),'slurm_job_id':os.getenv('SLURM_JOB_ID'),
        'host':socket.gethostname(),'input_hashes':[]};started=time.monotonic()
    study.write_json(output/'report.json',report)
    try:
        if args.command=='cpu':
            report['input_hashes']=[hashed(p) for p in [*sorted(directory.glob('*.csv')),directory/'report.json']]
            report.update(extra.cpu_analysis(plan,item,child,directory,output))
            report['independent_refits']=independent_cpu_check(child,directory,output,item['seed'],plan['elastic_net'])
        else:
            predictor=study.PatientPredictor(child,args.device,args.threads,plan['loo']['seed'])
            report.update(extra.loo_analysis(child,output,predictor));report['device']=args.device
            report['gpu_name']=predictor.torch.cuda.get_device_name(predictor.device) if predictor.device.type=='cuda' else None
        report['status']='passed';report['output_hashes']=[hashed(p) for p in sorted(output.iterdir()) if p.name!='report.json']
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;study.write_json(output/'report.json',report)
    print('Passed secondary task:',args.command,args.task,'patient ordinal',item['patient_index'])


def validate_loo(child, report, rows):
    require(len(rows)==len(child['images']) and {r['omitted_image'] for r in rows}==set(child['images']),
            'Incomplete or duplicate LOO image coverage')
    require(report['model_queries']==1+len(rows),'LOO query budget differs')
    original=report['original_prediction']
    require(np.isfinite(original['p_class1']) and 0<=original['p_class1']<=1 and original['yhat'] in (0,1),
            'Invalid original LOO probability/class')
    for row in rows:
        p=float(row['p_class1']);delta=original['p_class1']-p
        require(np.isfinite(p) and 0<=p<=1 and int(row['retained_count'])==len(rows)-1,'Invalid LOO response')
        require(np.allclose([float(row['delta_class1']),float(row['delta_original_class'])],
                           [delta,delta if original['yhat'] else -delta],atol=1e-12,rtol=1e-9),
                'LOO sign or probability change mismatch')


def correlation(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(spearmanr(a,b).correlation)


def complete_mean(values, count):
    return float(np.mean(values)) if len(values)==count and all(v is not None and np.isfinite(v) for v in values) else None


def summarize(args):
    plan=verified(args.plan);root=args.plan.resolve().parent;output=root/'summary'
    require(not output.exists(),'Summary output already exists; preserve it')
    evidence=[];fidelity=[];rankings=[];design=[];inclusion=[];stages=[];loo=[];loo_queries=0;refits=0
    for command in ('cpu','loo'):
        for record in plan[command+'_records']:
            report=check_record(record,command);directory=Path(record['directory'])
            evidence.extend([hashed(directory/'report.json'),*report.get('input_hashes',[]),*report['output_hashes']])
            index=record['patient_index'];seed=record['seed'];identity={'patient_index':index,'seed':seed}
            child=json.loads(Path(record['plan']).read_text())
            if command=='cpu':
                require(report['model_queries']==0,'CPU analysis consumed model queries')
                require(record['reused'] or report.get('independent_refits')==4,'New task independent refits missing')
                refits+=0 if record['reused'] else report['independent_refits']
                rows=read_rows(directory/'fidelity.csv')
                require(len(rows)==4 and {(r['arm'],r['method']) for r in rows}=={(a,m) for a in extra.ARMS for m in ('ridge','elastic_net')},'Incomplete method/arm fidelity pairs')
                fidelity.extend({**identity,**r} for r in rows)
                own_rows=read_rows(directory/'rankings.csv')
                require(len(own_rows)==6*len(child['images']),'Wrong ranking row count')
                for a in extra.ARMS:
                    for m in extra.METHODS:
                        group=[r['image_id'] for r in own_rows if r['arm']==a and r['method']==m]
                        require(len(group)==len(child['images']) and set(group)==set(child['images']),
                                'Duplicate or missing ranking image columns')
                for row in own_rows:
                    rankings.append({**identity,**row,'value':float(row['value']) if row['value'] else None})
                own=rankings[-6*len(child['images']):]
                require(len(own)==6*len(child['images']) and all(len([r for r in own if r['arm']==a and r['method']==m])==len(child['images']) for a in extra.ARMS for m in extra.METHODS), 'Ranking coverage differs')
                inclusion.extend({**identity,**r} for r in read_rows(directory/'image_inclusion.csv'))
                design.extend({**identity,'arm':a,**{k:v for k,v in d.items() if k!='subset_size_counts'}} for a,d in report['design'].items())
                stages.append({**identity,'balance_reached':report['class_balance_reached'],
                    'biased_rows':report['stage_two']['stages']['biased'],'deficit_rows':report['stage_two']['biased_rows_with_pool_deficit'],
                    'one_pool_fallback':report['stage_two']['one_pool_fallback']})
            else:
                rows=read_rows(directory/'leave_one_out.csv');validate_loo(child,report,rows)
                loo_queries+=report['model_queries'];loo.extend({'patient_index':index,**r} for r in rows)
                primary=json.loads((Path(record['plan']).parent/'runs/seed-0/report.json').read_text())['original_prediction']
                require(report['original_prediction']['yhat']==primary['yhat'] and np.isclose(report['original_prediction']['p_class1'],primary['p_class1'],atol=1e-6,rtol=1e-6),'LOO and P0 full-set predictions differ')
    require(loo_queries==plan['loo']['planned_model_queries']==3207,'Aggregate LOO budget differs')
    require(refits==plan['new_independent_refits']==1500,'Aggregate new refit count differs')
    stability=[];frequency=[];agreement=[];patient_fidelity=[]
    for index in range(135):
        r=[x for x in rankings if x['patient_index']==index]
        pairs,counts=extra.stability_tables(r,index,[0,1,2],plan['stability']['sign_zero_tolerance'])
        stability.extend(pairs);frequency.extend(counts)
        effects={row['omitted_image']:float(row['delta_class1']) for row in loo if row['patient_index']==index}
        for arm in extra.ARMS:
            values={}
            for method in ('ridge','elastic_net'):
                records=[x for x in fidelity if x['patient_index']==index and x['arm']==arm and x['method']==method]
                require(len(records)==3 and {x['seed'] for x in records}=={0,1,2},'Incomplete patient fidelity seeds')
                values[method]=complete_mean([float(x['novel_mae']) if x['novel_mae'] else None for x in records],3)
            patient_fidelity.append({'patient_index':index,'arm':arm,'ridge_mean_novel_mae':values['ridge'],
                'elastic_net_mean_novel_mae':values['elastic_net'],
                'elastic_net_minus_ridge_mae':values['elastic_net']-values['ridge'] if all(v is not None for v in values.values()) else None})
            for method in extra.METHODS:
                for seed in (0,1,2):
                    rows=[x for x in r if x['arm']==arm and x['method']==method and x['seed']==seed]
                    require({x['image_id'] for x in rows}==set(effects),'LOO/ranking image alignment differs')
                    agreement.append({'patient_index':index,'arm':arm,'method':method,'seed':seed,
                        'spearman':correlation([x['value'] for x in rows],[effects[x['image_id']] for x in rows])})
    primary=pd.read_csv(Path(plan['full_summary']).parent/'fidelity_by_seed.csv')
    for row in fidelity:
        if row['method']=='ridge':
            match=primary[(primary.patient_index==row['patient_index'])&(primary.seed==row['seed'])&(primary.arm==row['arm'])]
            require(len(match)==1 and np.isclose(float(row['novel_mae']),match.novel_mae.iloc[0],atol=1e-13), 'Secondary Ridge comparison differs from primary study')
    require(len(fidelity)==1620 and len(stability)==2430 and len(agreement)==2430 and len(loo)==3072,'Incomplete aggregate coverage')
    output.mkdir()
    for name,rows in [('fidelity_by_seed',fidelity),('fidelity_by_patient',patient_fidelity),('rankings',rankings),
                      ('design_diagnostics',design),('image_inclusion',inclusion),('stage_two_review',stages),
                      ('stability',stability),('top_five_frequency',frequency),('leave_one_out',loo),('loo_agreement_by_seed',agreement)]:
        study.write_csv(output/(name+'.csv'),rows)
    analysis={'status':'passed','scope':plan['scope'],'plan_sha256':study.sha256(args.plan),'patients':135,
        'validated_cpu_tasks':405,'new_cpu_tasks':375,'reused_cpu_tasks':30,
        'validated_loo_patients':135,'new_loo_patients':125,'reused_loo_patients':10,
        'loo_model_queries':loo_queries,'new_loo_model_queries':2997,'new_independent_refits':refits,
        'previously_validated_pilot_refits':120,
        'mean_patient_elastic_net_minus_ridge_mae':{a:complete_mean([r['elastic_net_minus_ridge_mae'] for r in patient_fidelity if r['arm']==a],135) for a in extra.ARMS},
        'evidence':list({r['path']:r for r in evidence}.values()),'publication_allowed':False,
        'limitations':['Exploratory secondary expansion after inspection of pilot and full primary findings.',
            'Patients, not seeds or image subsets, are the primary analysis unit.',
            'LOO is one full-set removal baseline per patient at seed 0; no three-seed LOO replication is implied.',
            'Undefined constant-vector metrics remain unavailable; no significance tests or confidence intervals.',
            'No new sampling ratios, query matching, bootstrap or physician evaluation.']}
    analysis['summary_files']=[hashed(p) for p in sorted(output.glob('*.csv'))]
    study.write_json(output/'analysis.json',analysis)
    print('Validated full secondary cohort:',output)


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare')
    for name in ('full-plan','pilot-plan','pilot-review','primary-package','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('cpu','loo'):
        p=commands.add_parser(name);p.add_argument('--plan',type=Path,required=True);p.add_argument('--task',type=int,required=True)
        p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu');p.add_argument('--threads',type=int,default=2)
    p=commands.add_parser('summarize');p.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='summarize':summarize(args)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):run(args)


if __name__=='__main__':main()
