#!/usr/bin/env python3
"""Private full-cohort secondary tables and publication figures, after validation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import full_secondary_study as secondary
import patient_study as study
from proceedings_style import COLORS, LABELS, MARKERS, configure, save


def patient_means(frame, columns, keys):
    rows=[]
    for group,values in frame.groupby(keys):
        if not isinstance(group,tuple):group=(group,)
        row=dict(zip(keys,group))
        if len(values)!=3:raise ValueError('Incomplete three-seed/pair patient metric')
        if 'seed' in values and set(values.seed)!={0,1,2}:raise ValueError('Duplicate/missing patient seed')
        for col in columns:
            a=pd.to_numeric(values[col],errors='raise').to_numpy(dtype=float)
            row[col]=float(a.mean()) if np.isfinite(a).all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def stats(values,n=135):
    v=np.asarray(values,dtype=float);valid=v[np.isfinite(v)]
    return {'planned':n,'defined':len(valid),'undefined':n-len(valid),
        'complete_cohort_mean':float(valid.mean()) if len(v)==n and len(valid)==n else None,
        'available_patient_mean':float(valid.mean()) if len(valid) else None,
        'available_patient_sd':float(valid.std(ddof=1)) if len(valid)>1 else None,
        'available_patient_median':float(np.median(valid)) if len(valid) else None}


def cohort_tables(summary,output):
    fidelity=pd.read_csv(summary/'fidelity_by_seed.csv')
    f=patient_means(fidelity,['novel_mae','constant_baseline_novel_mae'],['patient_index','arm','method'])
    stability=pd.read_csv(summary/'stability.csv')
    # Invalid vectors must not turn top-five/sign tie values into apparent stability.
    stability.loc[~stability.defined,['spearman','top_five_jaccard','sign_agreement']]=np.nan
    s=patient_means(stability,['spearman','top_five_jaccard','sign_agreement'],['patient_index','arm','method'])
    l=patient_means(pd.read_csv(summary/'loo_agreement_by_seed.csv'),['spearman'],['patient_index','arm','method'])
    loo=pd.read_csv(summary/'leave_one_out.csv')
    effects=loo.assign(absolute_change=loo.delta_class1.abs()).groupby('patient_index').absolute_change.mean()
    rows=[]
    def add(label,values):rows.append({'diagnostic':label,**stats(values)})
    for arm in secondary.extra.ARMS:
        for method in ('ridge','elastic_net'):
            g=f[(f.arm==arm)&(f.method==method)]
            add(LABELS[arm]+' '+LABELS[method]+' MAE',g.novel_mae)
            add(LABELS[arm]+' '+LABELS[method]+' own constant MAE',g.constant_baseline_novel_mae)
        paired=f[f.arm==arm].pivot(index='patient_index',columns='method',values='novel_mae')
        add(LABELS[arm]+' Elastic Net minus Ridge MAE',paired.elastic_net-paired.ridge)
        for method in secondary.extra.METHODS:
            g=s[(s.arm==arm)&(s.method==method)];a=l[(l.arm==arm)&(l.method==method)]
            for metric in ('spearman','top_five_jaccard','sign_agreement'):
                add(LABELS[arm]+' '+LABELS[method]+' seed '+metric,g[metric])
            add(LABELS[arm]+' '+LABELS[method]+' versus LOO Spearman',a.spearman)
    add('LOO mean absolute class-1 probability change',effects)
    frame=pd.DataFrame(rows);frame.to_csv(output/'secondary_summary.csv',index=False)
    f.to_csv(output/'fidelity_by_patient.csv',index=False);s.to_csv(output/'stability_by_patient.csv',index=False)
    l.to_csv(output/'loo_agreement_by_patient.csv',index=False)
    fmt=lambda v:'NA' if v is None or pd.isna(v) else f'{v:.4f}'
    lines=[r'\begin{table}[!htbp]\centering\small',
        r'\caption{Full-cohort secondary analyses. Seeds or seed pairs are averaged within patients first. MAE means require all 135 patients; other means use the explicitly stated defined patient counts. Undefined metrics are not assigned zero.}',
        r'\begin{tabularx}{\linewidth}{@{}Xrr@{}}\toprule',
        r'Diagnostic & Mean & Defined / 135 \\\midrule']
    for row in rows:
        label=row['diagnostic'].replace('_',r'\_')
        value=row['complete_cohort_mean'] if 'MAE' in row['diagnostic'] else row['available_patient_mean']
        lines.append(label+' & '+fmt(value)+' & '+str(row['defined'])+r' \\')
    lines += [r'\bottomrule\end{tabularx}\end{table}']
    (output/'secondary_summary.tex').write_text('\n'.join(lines)+'\n')
    return f,s,l,loo,rows


def summary_figures(f,s,l,loo,output):
    configure();captions={}
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.8));fig.subplots_adjust(left=.08,right=.98,bottom=.20,top=.83,wspace=.32)
    for k,(arm,ax) in enumerate(zip(secondary.extra.ARMS,axes)):
        g=f[f.arm==arm].pivot(index='patient_index',columns='method',values='novel_mae')
        for _,r in g.iterrows():
            if np.isfinite(r).all():ax.plot([0,1],[r.ridge,r.elastic_net],c='#B4BDC7',lw=.5,alpha=.5)
        for j,method in enumerate(('ridge','elastic_net')):
            ax.scatter(np.full(len(g),j),g[method],s=12,c=COLORS[method],marker=MARKERS[method],alpha=.6)
            valid=g[method].dropna()
            if len(valid):ax.scatter([j],[valid.mean()],s=50,marker='D',c=COLORS[method],edgecolor='white',zorder=5)
        ax.set(xticks=[0,1],xticklabels=['Ridge','Elastic Net'],ylabel='Shared-novel probability MAE (lower is better)',ylim=(0,None))
        ax.set_title(chr(65+k)+'  '+LABELS[arm]+' training',loc='left',fontweight='bold')
    fig.suptitle('135 patients · paired probability-surrogate fidelity',fontsize=11,fontweight='bold')
    fig.text(.08,.035,'Points: patient means across three seeds. Diamonds: available-patient means. No confidence intervals.',fontsize=8)
    save(fig,output,'secondary_fidelity')
    captions['secondary_fidelity']='Paired Ridge/Elastic Net MAE on identical shared-novel masks, by training arm. Training-only five-fold CV selects Elastic Net settings. Patient means require all three seeds; lower MAE is better. Undefined values are omitted from marks and counted in tables. Diamonds summarize defined patient means, without inferential intervals.'
    fig,ax=plt.subplots(figsize=(8.5,4.7));fig.subplots_adjust(left=.28,right=.87,bottom=.19,top=.85)
    mat=[];counts=[];labels=[]
    for method in secondary.extra.METHODS:
        for arm in secondary.extra.ARMS:
            g=s[(s.arm==arm)&(s.method==method)];a=l[(l.arm==arm)&(l.method==method)]
            values=[g.spearman,g.top_five_jaccard,g.sign_agreement,a.spearman]
            mat.append([v.mean() for v in values]);counts.append([v.notna().sum() for v in values]);labels.append(LABELS[method]+' · '+LABELS[arm])
    mat=np.asarray(mat);cmap=LinearSegmentedColormap.from_list('association',['#B35836','#FFFFFF','#2166AC']);cmap.set_bad('#E7EAEE')
    im=ax.imshow(np.ma.masked_invalid(mat),vmin=-1,vmax=1,cmap=cmap,aspect='auto')
    ax.set(xticks=range(4),xticklabels=['Seed-pair\nSpearman','Top-five\nJaccard','Sign\nagreement','Versus LOO\nSpearman'],yticks=range(6),yticklabels=labels)
    for i in range(6):
        for j in range(4):
            v=mat[i,j];ax.text(j,i,(f'{v:.3f}' if np.isfinite(v) else 'NA')+f'\n{counts[i][j]}/135',ha='center',va='center',fontsize=9,color='white' if np.isfinite(v) and abs(v)>.72 else '#222222')
    ax.tick_params(length=0);fig.colorbar(im,ax=ax,fraction=.04,pad=.04,ticks=[-1,0,1])
    fig.suptitle('135-patient explanation diagnostics',fontsize=11,fontweight='bold')
    fig.text(.28,.06,'Cells: available-patient means and defined counts.\nShade represents similarity, not statistical significance.',fontsize=8)
    save(fig,output,'secondary_diagnostics')
    captions['secondary_diagnostics']='All 135 patients are assessed; cells state the available-patient mean and defined count. Seed stability requires all three seed pairs; LOO agreement requires all three training-seed correlations. Constant/nonfinite vectors remain unavailable. Jaccard/sign range 0–1 and Spearman −1–1; shading is descriptive, not significance.'
    fig,axes=plt.subplots(1,2,figsize=(8.5,4));fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.84,wspace=.36)
    effect=loo.assign(absolute=loo.delta_class1.abs()).groupby('patient_index').absolute.mean().sort_values()
    axes[0].plot(np.arange(1,136),effect,marker='o',ms=2,lw=.8,c=COLORS['ridge'])
    axes[0].set(xlabel='Patient rank by mean absolute LOO change',ylabel='Mean absolute class-1 probability change',ylim=(0,None),title='A  Single-image removals')
    labels=[]
    for j,(method,arm) in enumerate((m,a) for m in secondary.extra.METHODS for a in secondary.extra.ARMS):
        vals=l[(l.method==method)&(l.arm==arm)].spearman.dropna()
        axes[1].scatter(vals,np.full(len(vals),j)+np.linspace(-.12,.12,len(vals)),s=10,c=COLORS[arm],marker=MARKERS[arm],alpha=.5)
        labels.append(f'{LABELS[method]} {arm[0].upper()} ({len(vals)}/135)')
    axes[1].set(yticks=range(6),yticklabels=labels,xlabel='Ranking versus LOO Spearman',xlim=(-1.05,1.05),ylim=(5.5,-.5),title='B  Agreement across patients')
    axes[1].axvline(0,c='#777777',ls='--',lw=.7)
    fig.suptitle('Full-cohort LOO model-behavior evaluation',fontsize=11,fontweight='bold')
    fig.text(.09,.035,'LOO queries each full image set and every single-image removal once. R/A: random/adaptive training.',fontsize=8)
    save(fig,output,'loo_behavior')
    captions['loo_behavior']='A: one patient estimate averages absolute class-1 probability changes over its images; the horizontal axis orders those 135 estimates. B: patient means of three seed-level ranking-versus-LOO Spearman correlations, with defined counts. LOO uses one fixed-seed full-set/removal baseline per patient; it is not repeated-seed or clinical validation.'
    return captions


def influence_profiles(plan,summary,output):
    configure();rankings=pd.read_csv(summary/'rankings.csv');loo=pd.read_csv(summary/'leave_one_out.csv')
    output.mkdir();exports=[]
    for item in plan['patients']:
        child=json.loads(Path(item['plan']).read_text());images=child['images'];index=item['patient_index'];n=len(images)
        for arm in secondary.extra.ARMS:
            g=rankings[(rankings.patient_index==index)&(rankings.arm==arm)]
            means=patient_means(g,['value'],['image_id','method']).pivot(index='image_id',columns='method',values='value').loc[images]
            means['LOO']=loo[loo.patient_index==index].set_index('omitted_image').loc[images,'delta_class1'].to_numpy()
            means['image_label']=[f'I{i+1:02}' for i in range(n)]
            order=means.sort_values('marginal_correlation',ascending=False,kind='stable',na_position='last')
            fig,axes=plt.subplots(4,1,figsize=(8.5,10));fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.91,hspace=.65)
            for k,(method,title) in enumerate([('marginal_correlation','Pearson correlation'),('elastic_net','Elastic Net'),('ridge','Ridge'),('LOO','Leave-one-image-out')]):
                ax=axes[k];values=order[method].to_numpy();valid=np.isfinite(values);x=np.arange(n)
                ax.bar(x[valid],values[valid],width=.78,color=np.where(values[valid]>=0,'#2AC6F2','#FA9B90'),edgecolor='#555555',linewidth=.35)
                ax.axhline(0,c='#555555',lw=.6);ax.set(xticks=x,xticklabels=order.image_label,ylabel='Correlation' if k==0 else 'Probability change' if k==3 else 'Probability coefficient')
                ax.tick_params(axis='x',labelsize=6.5,labelrotation=55);ax.set_title(chr(65+k)+'  '+title,loc='left',fontsize=10,fontweight='bold')
                for position in x[~valid]:ax.text(position,0,'NA',rotation=90,fontsize=5,ha='center',va='bottom')
                if k==0:
                    span=max(float(np.ptp(values[valid])) if valid.any() else 0,.05)
                    for j,image in enumerate(order.index):
                        if not valid[j]:continue
                        path=Path(child['paths']['images'])/f"{child['patient']}_{image}.jpg"
                        center=values[j]+(.10*span if values[j]>=0 else -.10*span)
                        with Image.open(path) as thumbnail:
                            ax.imshow(thumbnail,cmap='gray',vmin=0,vmax=255,extent=(j-.36,j+.36,center-.07*span,center+.07*span),aspect='auto',zorder=4)
                    if valid.any():ax.set_ylim(min(0,values[valid].min())-.22*span,max(0,values[valid].max())+.22*span)
            fig.suptitle(f'Patient ordinal {index+1:03} · {LABELS[arm]} sampling · {n} images',fontsize=12,fontweight='bold')
            fig.text(.10,.945,'Pearson, Elastic Net and Ridge: means across three seeds. LOO: one seed-0 removal baseline.',fontsize=8)
            fig.text(.10,.025,'Private clinical-image review. I labels follow the frozen image order; panels share Pearson-sorted order.\nCyan: positive; coral: negative association. NA: undefined. No confidence intervals or significance coding.',fontsize=8)
            name=f'patient-{index:03}-{arm}';save(fig,output,name)
            order.drop(columns=[]).reset_index(drop=True).to_csv(output/(name+'.csv'),index=False)
            exports.append({'patient_index':index,'arm':arm,'figures':name,'images':n,'publication_permission':'unconfirmed; private review only'})
    return exports


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True,type=Path);args=parser.parse_args()
    plan=secondary.verified(args.plan);root=args.plan.resolve().parent;summary=root/'summary'
    analysis=json.loads((summary/'analysis.json').read_text())
    secondary.require(analysis['status']=='passed' and analysis['validated_cpu_tasks']==405 and analysis['validated_loo_patients']==135
                      and analysis['plan_sha256']==study.sha256(args.plan),'Full secondary validation gate failed')
    secondary.check_hashes(analysis['evidence']+analysis['summary_files'])
    output=root/'report';output.mkdir(exist_ok=False);tables=output/'tables';tables.mkdir();figures=output/'figures';figures.mkdir()
    f,s,l,loo,rows=cohort_tables(summary,tables);captions=summary_figures(f,s,l,loo,figures)
    profiles=influence_profiles(plan,summary,output/'patient_profiles')
    primary=output/'preserved_primary_figures';primary.mkdir()
    for entry in plan['files']:
        path=Path(entry['path'])
        if path.parent==Path(plan['primary_figure_package'])/'latex/figures':shutil.copy2(path,primary/path.name)
    fig,axes=plt.subplots(3,2,figsize=(11,16));fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.96,hspace=.15,wspace=.10)
    previews=[primary/'figure1_workflow.png',primary/'figure2_influence.png',figures/'secondary_fidelity.png',figures/'secondary_diagnostics.png',figures/'loo_behavior.png',primary/'figure4_fidelity.png']
    for ax,path in zip(axes.flat,previews):
        ax.imshow(plt.imread(path));ax.axis('off');ax.set_title(path.stem.replace('_',' '),fontsize=10)
    save(fig,output,'contact_sheet')
    ledger={'created_at':datetime.now(timezone.utc).isoformat(),'population':'135 eligible positive patients; exploratory post-pilot secondary expansion',
        'analysis_unit':'patient; average all three seeds or three seed pairs before cohort summaries',
        'sources':[secondary.hashed(args.plan),secondary.hashed(summary/'analysis.json'),*[secondary.hashed(p) for p in sorted(summary.glob('*.csv'))]],
        'claims':rows,'figure_captions':captions,'patient_profiles':profiles,
        'uncertainty':'Descriptive SD only; no confidence intervals, significance tests or clinical validation.',
        'publication_allowed':False}
    study.write_json(output/'evidence_ledger.json',ledger)
    study.write_json(output/'figure_captions.json',captions)
    study.write_json(output/'report_manifest.json',{'status':'passed','plan_sha256':study.sha256(args.plan),'validated_patients':135,
        'patient_profiles':len(profiles),'new_gnn_queries_in_report':0,'publication_allowed':False,
        'files':[secondary.hashed(p) for p in sorted(output.rglob('*')) if p.is_file()]})
    (output/'README.md').write_text('# Full-cohort secondary results — private review\n\nAll 405 CPU analyses and 135 LOO baselines passed. Thirty CPU records and ten LOO baselines are reused from the pilot. Inspect tables/secondary_summary.csv and .tex, figures/, all 270 patient_profiles/, preserved_primary_figures/ and contact_sheet.pdf. Each patient has both random/adaptive influence profiles with aligned Pearson, Elastic Net, Ridge and LOO panels.\n\nUndefined values remain unavailable and counts are explicit. Complete-cohort means require all 135 patients. No bootstrap or significance claims are made. The cohort expansion followed inspection of pilot and primary findings. LOO is a single baseline per patient, not three independent repeats.\n\nThe dependent integration job updates only result-related files and minimal scope statements in the existing user-edited full-submission-prep-20261008-v1 package, retaining a backup and refusing conflicting text replacements. Original workflow/result figures remain preserved. Clinical images and evidence paths are restricted; publication permission and author approval remain required.\n')
    print('Full secondary figures and tables:',output)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=int(os.environ.get('OMP_NUM_THREADS','2'))):main()
