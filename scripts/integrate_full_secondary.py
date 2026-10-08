#!/usr/bin/env python3
"""Update the user's existing manuscript in place after complete secondary validation.

Compile a staged copy, check for concurrent textual edits, then install only
minimal scope/result changes. Originals and prior PDFs are retained in history.
"""
import argparse
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import full_secondary_study as secondary
import patient_study as study


def replace_once(text,old,new,label):
    if text.count(old)!=1:raise ValueError('Text conflict; preserve user edits and review manually: '+label)
    return text.replace(old,new,1)


def run(command,cwd):
    result=subprocess.run(command,cwd=cwd,text=True,capture_output=True,timeout=180)
    if result.returncode:raise RuntimeError('Build failed: '+' '.join(command)+'\n'+result.stdout[-2500:]+result.stderr[-1000:])
    return result.stdout


def integrate(args):
    plan=secondary.verified(args.plan);root=args.plan.resolve().parent;report=root/'report';summary=root/'summary'
    a=json.loads((summary/'analysis.json').read_text());r=json.loads((report/'report_manifest.json').read_text())
    secondary.require(a['status']=='passed' and a['validated_cpu_tasks']==405 and a['validated_loo_patients']==135
        and r['status']=='passed' and r['validated_patients']==135 and r['patient_profiles']==270,'Secondary report gate failed')
    secondary.check_hashes(a['summary_files']+r['files'])
    target=args.manuscript.resolve()
    secondary.require(target==Path(plan['primary_figure_package']).resolve(),'Unexpected manuscript target')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');history=target/'history'/('full-secondary-'+stamp)
    history.mkdir(parents=True,exist_ok=False);stage=history/'compiled_update'
    source=target/'latex';shutil.copytree(source,stage)
    watched={str(p.relative_to(source)):study.sha256(p) for p in source.rglob('*') if p.suffix in ('.tex','.bib')}
    for name,digest in watched.items():
        secondary.require(study.sha256(stage/name)==digest,'Text changed during snapshot: '+name)
    unchanged={name:study.sha256(source/'figures'/name) for name in ('figure1_workflow.pdf','figure2_influence.pdf')}
    for path in target.iterdir():
        if path.is_file():shutil.copy2(path,history/path.name)
    shutil.copytree(source,history/'latex_before')
    body=(stage/'body.tex').read_text();appendix=(stage/'appendix.tex').read_text();results=(stage/'generated/results.tex').read_text()
    # The user has already rewritten this Methods paragraph; only change an
    # obsolete scope label if it is still present, preserving their new wording.
    old_en='The ten-patient analysis also uses Elastic Net probability regression'
    if old_en in body:
        body=replace_once(body,old_en,'The expanded analysis also uses Elastic Net probability regression','Elastic Net scope')
    else:
        secondary.require('Elastic Net' in body and 'Training-only shuffled five-fold CV' in body,
                          'Elastic Net Methods changed; preserve user edits and review scope manually')
    body=replace_once(body,'Ridge fidelity and deletion are evaluated in the main study; Elastic Net, Pearson ranking diagnostics, repeatability and leave-one-image-out comparisons remain ten-patient analyses.',
        'Ridge fidelity and deletion are evaluated in the main study. Elastic Net, Pearson ranking diagnostics, repeatability and leave-one-image-out comparisons were subsequently extended to all 135 patients after inspection of the pilot and primary findings.','population scope')
    body=replace_once(body,'Ten-patient repeatability is measured','Repeatability is measured','repeatability scope')
    body=replace_once(body,'The pilot-only LOO comparisons cannot establish that this relationship generalizes to all 135 patients.',
        'The full-cohort LOO comparisons remain descriptive and do not establish clinical validity or a causal mechanism.','LOO limitation')
    body=replace_once(body,'Supplementary Table~\\ref{tab:secondary} and Figure~\\ref{fig:diagnostics} report repeatability and LOO comparisons with their availability counts.',
        'The corresponding expanded analyses are reported in supplementary Table~\\ref{tab:secondary} and Figure~\\ref{fig:diagnostics}; the ten-patient outputs remain preserved separately.','pilot/full table reference')
    body=body.replace('deletion and pilot repeatability analyses','deletion and repeatability analyses')
    rows=json.loads((report/'evidence_ledger.json').read_text())['claims']
    lookup={row['diagnostic']:row for row in rows}
    fmt=lambda v:'NA' if v is None else f'{v:+.6f}'
    macro_values={
        'FullSecondaryRandomENDelta':fmt(lookup['Random Elastic Net minus Ridge MAE']['complete_cohort_mean']),
        'FullSecondaryAdaptiveENDelta':fmt(lookup['Adaptive Elastic Net minus Ridge MAE']['complete_cohort_mean'])}
    extra=('Full-cohort secondary analyses subsequently completed 405 patient--seed fits and 135 LOO baselines, including the reused pilot. '
        'Mean patient Elastic Net-minus-Ridge MAE is \\FullSecondaryRandomENDelta{} under random sampling and \\FullSecondaryAdaptiveENDelta{} under adaptive sampling '
        '(Table~\\ref{tab:secondary}; Figure~\\ref{fig:full-secondary-fidelity}). '
        'Stability and ranking-versus-LOO agreement retain explicit defined counts (Figure~\\ref{fig:diagnostics}). '
        'The expansion adds 1,500 independent Ridge/selected-Elastic-Net refits to the 120 pilot checks; recorded CV minima and held-out scores are checked without using evaluation responses for selection. '
        'LOO comprises 3,207 calls, including 210 reused pilot calls and 2,997 new calls, and measures model behavior rather than clinical importance.\n\n')
    body=replace_once(body,r'\subsection{Preliminary and additional ten-patient analyses}',extra+r'\subsection{Preliminary and additional ten-patient analyses}','new full secondary results placement')
    old=('Table~\\ref{tab:secondary} and Figure~\\ref{fig:diagnostics} concern the ten-patient convenience cohort, regardless of the primary analysis population. '
         'They do not imply expanded Elastic Net, Pearson stability or LOO experiments.')
    appendix=replace_once(appendix,old,'Table~\\ref{tab:secondary} and Figure~\\ref{fig:diagnostics} report the completed 135-patient secondary expansion, including the ten preliminary patients. The expansion followed inspection of the pilot and primary findings.','appendix population')
    appendix=replace_once(appendix,r'\section{Additional diagnostics: ten patients only}',r'\section{Additional diagnostics: expanded cohort}','appendix heading')
    appendix=appendix.replace('expanded secondary analyses, bootstrap inference','bootstrap inference')
    body=body.replace('and expand the supplementary explanation methods','and evaluate the supplementary explanation methods in additional populations')
    newcaption=json.loads((report/'figure_captions.json').read_text())
    for key,value in macro_values.items():results+='\\newcommand{\\'+key+'}{'+value+'}\n'
    lines=results.splitlines()
    for k,line in enumerate(lines):
        if line.startswith(r'\newcommand{\DiagnosticsCaption}'):
            lines[k]=r'\newcommand{\DiagnosticsCaption}{'+newcaption['secondary_diagnostics']+'}'
    results='\n'.join(lines)+'\n'
    results=replace_once(results,'; supplementary explanation methods are examined in ten patients.',
        '; secondary explanation diagnostics are subsequently evaluated in the same 135 patients.','abstract method scope')
    design=(stage/'generated/design.tex').read_text()
    design=replace_once(design,'Elastic Net (pilot only)','Elastic Net (135 patients)','design table method scope')
    table=(report/'tables/secondary_summary.tex').read_text().replace(r'\begin{table}[!htbp]\centering\small',r'\begin{table}[!htbp]\centering\small'+'\n'+r'\label{tab:secondary}')
    # Place labels after captions for correct cross-reference numbering.
    table=table.replace(r'\label{tab:secondary}'+'\n','')
    table=table.replace(r'\begin{tabularx}',r'\label{tab:secondary}'+'\n'+r'\begin{tabularx}',1)
    for name,caption,label in [('secondary_fidelity','secondary_fidelity','full-secondary-fidelity'),('loo_behavior','loo_behavior','full-secondary-loo')]:
        appendix+='\n'+r'\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{figures/'+name+r'.pdf}\caption{'+newcaption[caption]+r'}\label{fig:'+label+r'}\end{figure}'+'\n'
    edits={'body.tex':body,'appendix.tex':appendix,'generated/results.tex':results,'generated/design.tex':design,'generated/secondary.tex':table}
    for name,text in edits.items():(stage/name).write_text(text)
    diffs=''.join(''.join(difflib.unified_diff((source/name).read_text().splitlines(True),text.splitlines(True),fromfile=name+' before',tofile=name+' after')) for name,text in edits.items())
    (history/'minimal_text_changes.diff').write_text(diffs)
    for suffix in ('pdf','svg','png'):
        shutil.copy2(report/'figures'/('secondary_diagnostics.'+suffix),stage/'figures'/('figure6_diagnostics.'+suffix))
        for name in ('secondary_fidelity','loo_behavior'):shutil.copy2(report/'figures'/(name+'.'+suffix),stage/'figures'/(name+'.'+suffix))
    shutil.copy2(report/'tables/secondary_summary.csv',stage/'generated/secondary.csv')
    metrics=json.loads((stage/'generated/metrics.json').read_text());metrics['full_secondary']={'plan_sha256':study.sha256(args.plan),'claims':rows}
    (stage/'generated/metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    import matplotlib.pyplot as plt
    from proceedings_style import configure,save
    configure();fig,axes=plt.subplots(4,2,figsize=(11,21));fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.97,hspace=.15,wspace=.10)
    names=['figure1_workflow','figure2_influence','figure3_sampling','figure4_fidelity','figure5_deletion','figure6_diagnostics','secondary_fidelity','loo_behavior']
    for ax,name in zip(axes.flat,names):ax.imshow(plt.imread(stage/'figures'/(name+'.png')));ax.axis('off');ax.set_title(name.replace('_',' '),fontsize=10)
    save(fig,stage/'figures','contact_sheet')
    commands=[['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['bibtex','main'],['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex']]
    for k,cmd in enumerate(commands):(history/f'build-{k}.log').write_text(run(cmd,stage))
    info=run(['pdfinfo','main.pdf'],stage);fonts=run(['pdffonts','main.pdf'],stage);text=run(['pdftotext','main.pdf','-'],stage)
    log=(stage/'main.log').read_text(errors='replace')
    secondary.require(not re.search(r'Citation .* undefined|There were undefined references|Missing character:|Overfull \\hbox|Float too large',log),'PDF references/glyph/overflow QA failed')
    secondary.require(re.search(r'Encrypted:\s+no\b',info) and re.search(r'Page size:\s+612 x 792 pts',info),'PDF page/security QA failed')
    secondary.require(re.search(r'TimesNewRoman|TimesNewRomanPS',fonts),'Times New Roman is not embedded')
    for item in plan['patients']:
        secondary.require(not re.search(r'(?<!\w)'+re.escape(str(item['patient']))+r'(?!\w)',text),'Private patient ID in PDF text')
    for name,digest in watched.items():secondary.require(study.sha256(source/name)==digest,'Concurrent textual edit detected; staged update retained, existing manuscript untouched: '+name)
    for name,digest in unchanged.items():secondary.require(study.sha256(stage/'figures'/name)==digest,'Original Figure 1/2 changed')
    # Install only changed assets and the five narrowly edited text files.
    paths=[*edits,'generated/secondary.csv','generated/metrics.json','main.pdf']
    paths += ['figures/'+name+'.'+suffix for name in ['figure6_diagnostics','secondary_fidelity','loo_behavior','contact_sheet'] for suffix in ('pdf','svg','png')]
    for name in paths:shutil.copy2(stage/name,source/name)
    for name in ('manuscript_review.pdf','manuscript_candidate.pdf'):shutil.copy2(stage/'main.pdf',target/name)
    (target/'manuscript_text.txt').write_text(text);(target/'pdfinfo.txt').write_text(info);(target/'pdffonts.txt').write_text(fonts)
    ledger=json.loads((target/'evidence_ledger.json').read_text())
    for entry in ledger['entries']:
        if entry['id'] in ('table:secondary','figure:DiagnosticsCaption'):
            entry['id']+=':pilot_before_expansion';entry['superseded_in_manuscript']=True
    ledger['entries'].extend([{'id':'table:secondary','population':'full secondary cohort','values':rows,'validation':'passed'},
        {'id':'figure:DiagnosticsCaption','population':'full secondary cohort','definition':newcaption['secondary_diagnostics'],'validation':'passed'},
        {'id':'claims:full_secondary','population':'full secondary cohort','values':{**macro_values,'cpu_records':405,'loo_patients':135,'new_refits':1500,'pilot_refits':120,'loo_queries':3207,'new_loo_queries':2997,'reused_loo_queries':210},'scope':a['scope'],'validation':'passed'}])
    ledger['sources'].extend([secondary.hashed(args.plan),secondary.hashed(summary/'analysis.json'),secondary.hashed(report/'evidence_ledger.json')])
    study.write_json(target/'evidence_ledger.json',ledger)
    receipt={'status':'passed','integrated_at':datetime.now(timezone.utc).isoformat(),'manuscript_directory':str(target),
        'plan_sha256':study.sha256(args.plan),'patients':135,'cpu_records':405,'loo_patients':135,'history':str(history),
        'original_figures_unchanged':unchanged,'minimal_text_changes':str(history/'minimal_text_changes.diff'),
        'user_source_files_before':watched,'source_files_after':[secondary.hashed(source/name) for name in edits],
        'visual_review':'pending final rendered-page inspection; automated PDF checks passed',
        'publication_allowed':False,'new_inference_queries':0}
    study.write_json(target/'full_secondary_integration.json',receipt)
    for name in ('full_secondary_study.py','report_full_secondary.py','integrate_full_secondary.py'):shutil.copy2(secondary.ROOT/'scripts'/name,target/'source_snapshot/scripts'/name)
    manifest=json.loads((target/'build_manifest.json').read_text());manifest.setdefault('integrations',[]).append(receipt)
    artifact_paths=[Path(e['path']) for e in manifest['artifacts']]
    artifact_paths.extend(source/'figures'/(name+'.'+suffix) for name in ('secondary_fidelity','loo_behavior') for suffix in ('pdf','svg','png'))
    artifact_paths.append(target/'full_secondary_integration.json')
    manifest['artifacts']=[secondary.hashed(p) for p in dict.fromkeys(artifact_paths)]
    manifest['manuscript_source_mode']='User-edited latex directory preserved, with five focused scope/result updates; see integration receipt and diff.'
    manifest['latex_sources']=[secondary.hashed(p) for p in source.rglob('*') if p.suffix in ('.tex','.bib')]
    study.write_json(target/'build_manifest.json',manifest)
    qa=json.loads((target/'pdf_qa.json').read_text());qa.update(visual_review='Pending: full-secondary integrated PDF needs final human/assistant page inspection.',submission_ready=False)
    study.write_json(target/'pdf_qa.json',qa)
    status_path=target/'analysis_status.json'
    if status_path.exists():
        status=json.loads(status_path.read_text())
        status['completed'].append('Full-cohort secondary expansion: 405 CPU records, 135 LOO baselines and 270 patient influence profiles validated; original Figures 1 and 2 preserved.')
        status['deferred']=[item for item in status['deferred'] if item!='full-cohort Elastic Net, Pearson, stability and leave-one-image-out']
        status['full_secondary_integration']='full_secondary_integration.json'
        study.write_json(status_path,status)
    readme=target/'README.md'
    if readme.exists():
        with readme.open('a') as handle:
            handle.write('\n\nFull secondary expansion integrated in place. See full_secondary_integration.json for provenance and history/minimal_text_changes.diff for the focused changes. All 405 CPU records and 135 LOO baselines passed; final visual review and author/publication approvals remain pending. Full patient profiles are in '+str(report/'patient_profiles')+'.\n')
    # Old handoff hashes/QA refer to the retained pre-integration package, not this update.
    for name in ('handoff_manifest.json','review_validation.json'):
        if (target/name).exists():
            value=json.loads((target/name).read_text());value['status']='superseded_by_full_secondary_integration'
            value['current_validation_receipt']='full_secondary_integration.json';study.write_json(target/name,value)
    preview=target/'rendered_pages/full_secondary';preview.mkdir(parents=True,exist_ok=False)
    subprocess.run(['pdftoppm','-scale-to','1300','-png',str(target/'manuscript_review.pdf'),str(preview/'page')],check=True)
    print('Updated existing user manuscript:',target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--manuscript',required=True,type=Path)
    integrate(parser.parse_args())
