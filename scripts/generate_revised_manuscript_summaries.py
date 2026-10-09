#!/usr/bin/env python3
"""Regenerate display tables/macros from the supplied validated aggregate evidence.

Source values retain their original precision. This script changes presentation,
not inference, patient inclusion, or the definition of a metric.
"""
from pathlib import Path
import csv
import hashlib
import json
import re

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
ROOT = parser.parse_args().package.resolve()
GEN = ROOT / 'latex/generated'
ledger = json.loads((ROOT / 'evidence_ledger_original.json').read_text())
entries = {e['id']: e for e in ledger['entries']}
primary = json.loads((GEN / 'metrics.json').read_text())['primary']
with (GEN / 'secondary.csv').open() as f:
    secondary = {r['diagnostic']: r for r in csv.DictReader(f)}
records = []

def fmt(v, digits=3):
    if isinstance(v, str):
        if v.strip() in ('', '—', 'N/A'): return '--'
        try: v = float(v)
        except ValueError: return v.replace('−', '$-$').replace('–', '--')
    return f'{v:.{digits}f}'

def display(v, digits=3):
    return fmt(v, digits).replace('-', '$-$') if isinstance(v, (float, int)) else fmt(v, digits)

macros = {}
def macro(name, value, source, digits=3, integer=False, signed=False):
    result = str(int(value)) if integer else f'{value:+.{digits}f}' if signed else fmt(value,digits)
    macros[name] = result
    records.append({'id':'macro:'+name,'unrounded_value':value,'display':result,'source':source})

for name,key in [('PrimaryN','patients'),('PrimaryRuns','runs'),('RandomFavored','random_better'),
                 ('AdaptiveFavored','adaptive_better'),('MAETies','equal_mae'),('TargetSuccesses','target_successes')]:
    macro(name,primary[key],'claims:primary/'+key,integer=True)
macro('BalanceCloser',primary['balance_comparison']['adaptive_closer_patients'],'claims:primary/balance_comparison',integer=True)
macro('PairedMAE',primary['paired_mae'],'claims:primary/paired_mae',signed=True)
for name,key in [('PairedMedian','median'),('PairedQOne','q1'),('PairedQThree','q3')]:
    macro(name,primary['paired_mae_dispersion'][key],'claims:primary/paired_mae_dispersion/'+key,signed=name=='PairedMedian')
macro('DeficitPercent',100*primary['pool_deficit_rows']/primary['biased_rows'],'claims:primary/pool_deficit_rows / biased_rows',digits=2)
for arm, label in [('random','Random'),('adaptive','Adaptive')]:
    a=primary['arms'][arm];d=primary['deletion'][arm]
    for suffix,key in [('MAE','mae'),('ConstantMAE','constant_mae')]: macro(label+suffix,a[key],'claims:primary/arms/'+arm+'/'+key)
    macro(label+'PositivePercent',100*a['positive_fraction'],'claims:primary/arms/'+arm+'/positive_fraction',digits=2)
    for suffix,key in [('BothClasses','both_prediction_classes_runs'),('BeatsConstant','better_than_constant_patients')]:
        macro(label+suffix,a[key],'claims:primary/arms/'+arm+'/'+key,integer=True)
    for suffix,key in [('DeletionDescending','descending'),('DeletionAscending','ascending'),('DeletionRandom','random'),('DeletionDelta','descending_minus_random')]:
        macro(label+suffix,d[key],'claims:primary/deletion/'+arm+'/'+key)
    for suffix,diagnostic in [('EnetMAE',label+' Elastic Net MAE'),('RidgeStability',label+' Ridge seed spearman'),('RidgeLOO',label+' Ridge versus LOO Spearman')]:
        macro(label+suffix,float(secondary[diagnostic]['available_patient_mean']),'table:secondary/'+diagnostic)
    diagnostic=label+' Elastic Net minus Ridge MAE'
    macro('FullSecondary'+label+'ENDelta',float(secondary[diagnostic]['available_patient_mean']),'table:secondary/'+diagnostic)
    macro(label+'EnetGainPP',-100*float(secondary[diagnostic]['available_patient_mean']),'table:secondary/'+diagnostic+'; probability error expressed in percentage points',digits=2)
macro('LOOAbsoluteMean',float(secondary['LOO mean absolute class-1 probability change']['available_patient_mean']),'table:secondary/LOO mean absolute class-1 probability change')
captions = {
 'SamplingCaption':'Sampling behavior across 135 patients. A: paired three-seed patient means of the class-1 proportion; the dashed line marks the adaptive target. B: patient-level reallocation rates for the 130 patients with Stage II draws. C: duplicate-mask rates. Diamonds denote cohort means.',
 'FidelityCaption':'Held-out probability fidelity across 135 patients. A: paired patient-mean Ridge MAEs. B: adaptive-minus-random MAE; positive values favor random sampling. C: each training-mean constant MAE minus its corresponding Ridge MAE; positive values favor Ridge. Diamonds denote cohort means.',
 'DeletionCaption':'Deletion of images ranked by signed Ridge coefficients. A--B: probability trajectories for one examination, selected as nearest the median random-trained descending-minus-random AUC contrast, using the same example in both panels. C: paired patient-mean descending-minus-random AUC contrasts across 135 patients. Negative values indicate lower probability trajectories under descending deletion; diamonds denote cohort means.'
}
(GEN/'results.tex').write_text(''.join('\\newcommand{\\'+n+'}{'+v.replace('−','-')+'}\n' for n,v in {**macros,**captions}.items()))

def table(path, caption, label, headers, rows, spec=None, foot=''):
    spec = spec or '@{}>{\\raggedright\\arraybackslash}X'+'r'*(len(headers)-1)+'@{}'
    text='\\begin{table}[!htbp]\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n'
    text+='\\begin{tabularx}{\\linewidth}{'+spec+'}\\toprule\n'+' & '.join(headers)+r' \\\midrule'+'\n'
    text+='\n'.join(' & '.join(row)+' \\\\' for row in rows)
    text+='\n\\bottomrule\\end{tabularx}\n'
    if foot: text+='\\par\\vspace{0.3em}{\\footnotesize '+foot+'}\n'
    text+='\\end{table}\n'
    (GEN/path).write_text(text)
    records.append({'id':'table:'+label,'headers':headers,'display_rows':rows,'source':'claims:primary and table:secondary; original validated ledger'})

rows=[]
for arm in ('random','adaptive'):
    a=primary['arms'][arm]
    rows.append([arm.title()+' Ridge',fmt(a['mae']),fmt(a['mae_dispersion']['sd']),fmt(a['constant_mae'])])
rows.append(['Adaptive $-$ random', '+'+fmt(primary['paired_mae']),fmt(primary['paired_mae_dispersion']['sd']),'--'])
table('fidelity.tex','Held-out probability fidelity. Means and standard deviations summarize three-seed patient means; lower MAE is better.','tab:fidelity',
      ['Training design','Mean MAE','SD','Constant MAE'],rows,
      foot='All 135 patients had complete paired errors. Random/adaptive/tied: 129/2/4. Retained evaluation draws: 170--194 per patient and seed; 101 of 405 sets contained one predicted class. Each constant is its own design\'s training-mean probability.')

sr=[]
for key,desc in [('positive_fraction','Achieved class-1 proportion'),('both_prediction_classes_runs','Designs with both classes'),('unique_masks_mean','Unique masks per 1,000 draws'),('duplicate_fraction','Duplicate rows (\\%)')]:
    vals=[]
    for arm in ('random','adaptive'):
        v=primary['arms'][arm][key]
        vals.append(str(v)+' / 405' if key=='both_prediction_classes_runs' else fmt(v*100,2) if key=='duplicate_fraction' else fmt(v,2) if key=='unique_masks_mean' else fmt(v))
    sr.append([desc,*vals])
sr.extend([['Target class-1 proportion','--','0.500'],['Runs attaining target','--','1 / 405'],
           ['Stage II draws reallocated','--',fmt(100*primary['pool_deficit_rows']/primary['biased_rows'],2)+'\\%'],
           ['Single-pool fallback runs','--',str(primary['one_pool_runs'])+' / 405']])
table('sampling.tex','Sampling coverage and feasibility. Proportions and mask counts are averaged within patient, then across patients.','tab:sampling',
      ['Diagnostic','Random','Adaptive'],sr,foot='Reallocation: 122,575 of 175,726 Stage II draws. Requested pool compositions and achieved prediction classes are distinct quantities.')

dr=[]
for key,desc in [('descending','Descending deletion'),('ascending','Ascending deletion'),('random','Random deletion'),
                 ('descending_minus_random','Descending $-$ random'),('descending_minus_ascending','Descending $-$ ascending')]:
    dr.append([desc,*[display(primary['deletion'][arm][key]) for arm in ('random','adaptive')]])
dr.append(['Patients below random control','135 / 135','135 / 135'])
table('deletion.tex','Mean raw deletion AUC by training design. AUC integrates probability over actual deleted fractions; its maximum is at most 0.5.','tab:deletion',
      ['Deletion order or contrast','Random','Adaptive'],dr)

st=[]
for arm in ('Random','Adaptive'):
    for method in ('Ridge','Elastic Net','Pearson'):
        names=[arm+' '+method+' seed '+x for x in ('spearman','top_five_jaccard','sign_agreement')]+[arm+' '+method+' versus LOO Spearman']
        vals=[]
        for name in names:
            r=secondary[name];vals.append(fmt(float(r['available_patient_mean']))+' ('+r['defined']+')')
        st.append([arm,method,*vals])
table('stability.tex','Repeatability and agreement with leave-one-image-out effects. Cells give the mean and defined patient count in parentheses, out of 135.','tab:stability',
      ['Design','Method','Seed $r_s$','Top-five','Sign','LOO $r_s$'],st,
      spec='@{}llrrrr@{}',foot='$r_s$: Spearman correlation. Top-five: Jaccard similarity. Sign: sign agreement. Three seed-pair values are averaged within patient; LOO correlations are averaged over the three fitted rankings.')

diag=[]
for key,name in [('singular_designs','Rank-deficient designs'),('condition_median','Median condition number')]:
    vals=[str(primary['arms'][a][key])+' / 405' if key=='singular_designs' else fmt(primary['arms'][a][key],2) for a in ('random','adaptive')]
    diag.append([name,*vals])
diag.append(['Out-of-range held-out scores',format(primary['arms']['random']['out_of_range_novel_scores'],','),format(primary['arms']['adaptive']['out_of_range_novel_scores'],',')])
table('design_diagnostics.tex','Additional binary-design and score diagnostics. Condition numbers refer to centered inclusion designs.','tab:design-diagnostics',
      ['Diagnostic','Random','Adaptive'],diag)
runtime = json.loads((ROOT/'server_evidence/inference_runtime.json').read_text())
assert runtime['verification']['plan_hash_mismatches'] == []
assert runtime['verification']['python_run_counts'] == {'3.9.25': 405}
(GEN/'software_versions.tex').write_text(
    'Saved inference plans and completed run reports recorded Python '+runtime['python']+
    ', PyTorch '+runtime['torch']+', torchvision '+runtime['torchvision']+
    ' and torch-geometric '+runtime['torch-geometric']+'. '
    'The local usflc\\_xai module had no package version; its source-file hashes were preserved in the inference plans. '
    'Recorded numerical and reporting versions were NumPy 1.22.4, pandas 1.4.2, SciPy 1.7.3, scikit-learn 1.0.2, Matplotlib 3.8.0 and Pillow 9.0.1.\n')
records.append({'id':'claim:inference_runtime','values':{k:runtime[k] for k in ('python','torch','torchvision','torch-geometric','usflc_xai_version','usflc_xai_commit')},'source':'server_evidence/inference_runtime.json; frozen plans and 405 run reports'})
records.append({'id':'claim:architecture','source':'server_evidence/server_source_audit.json; usflc_xai/models.py:50-105,240-245; datasets.py:465-480; scripts/patient_study.py:180-202','validation':'constructor, checkpoint state dimensions and frozen source hashes inspected; no inference'})
records.append({'id':'claim:illustrative_panel_provenance','source':'server_evidence/figure2_source_inventory.json; original follow-up report','validation':'candidate tables located; exact fit-to-panel linkage unresolved; original PDF bytes preserved'})
for record in records:
    if record['id'].startswith('table:'):
        record['precision_policy']='3 decimal places for response/rank metrics; 2 for percentages and design descriptives'
manifest={'source_ledger_sha256':hashlib.sha256((ROOT/'evidence_ledger_original.json').read_bytes()).hexdigest(),
          'source_metrics_sha256':hashlib.sha256((GEN/'metrics.json').read_bytes()).hexdigest(),
          'source_secondary_sha256':hashlib.sha256((GEN/'secondary.csv').read_bytes()).hexdigest(),
          'revision_source_hashes':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'latex/body.tex',ROOT/'latex/appendix.tex',ROOT/'server_evidence/inference_runtime.json',ROOT/'server_evidence/server_source_audit.json']},
          'entries':records,'new_inference':False}
(ROOT/'revision_evidence_ledger.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Generated',len(macros),'numerical macros and 5 tables from preserved evidence.')
