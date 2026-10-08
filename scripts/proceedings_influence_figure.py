"""Actual patient-level marginal/conditional explanations from validated outputs.

No inference or fitting. Clinical images are included for private author review;
publication permission must be confirmed before any release.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from proceedings_data import PILOT, PARALLEL, hash_entry, require
from proceedings_style import COLORS, configure, save

POSITIVE='#2166AC'
NEGATIVE='#B35836'


def example_data():
    plan=json.loads(PILOT.read_text())
    for item in plan['patients']:
        index=item['patient_index'];frames=[];sources=[]
        for seed in (0,1,2):
            path=PARALLEL.parent/'cpu'/f'task-{3*index+seed:02}'/'rankings.csv'
            frame=pd.read_csv(path);frame['seed']=seed;frames.append(frame);sources.append(hash_entry(path))
        frame=pd.concat(frames,ignore_index=True)
        if not all(np.isfinite(g.value).all() and np.ptp(g.value)>0
                   for _,g in frame.groupby(['arm','method','seed'])):
            continue
        child_path=Path(item['plan']);child=json.loads(child_path.read_text());images=child['images']
        values=frame.query('arm == "random"').groupby(['image_id','method']).value.mean().unstack('method').loc[images]
        require(len(frame)==len(images)*3*3*2,'Illustrative ranking coverage differs')
        require(values.notna().all().all(),'Missing illustrative image coefficient')
        paths=[Path(child['paths']['images'])/f"{child['patient']}_{image}.jpg" for image in images]
        sources.extend([hash_entry(child_path),*[hash_entry(path) for path in paths]])
        values=values.reset_index(drop=True);values.insert(0,'image_label',[f'I{j+1:02}' for j in range(len(images))])
        return {'index':index,'values':values,'images':paths,'sources':sources,
                'selection':'First patient in frozen pilot ordering with nonconstant rankings for all three methods, arms and seeds; no fidelity/deletion outcome selection.',
                'publication_eligibility':'Unconfirmed; user authorized private review use on 2026-10-08.'}
    raise ValueError('No eligible illustrative patient with defined explanation vectors')


def influence_figure(pilot,output):
    configure();example=example_data();values=example['values'];n=len(values)
    require(n==20,'The validated pilot illustration expects twenty images')
    fig=plt.figure(figsize=(5.5,7.2))
    fig.text(.025,.975,'A',fontsize=11,weight='bold',va='top')
    fig.text(.085,.975,'One patient, twenty ultrasound images',fontsize=10,weight='bold',va='top')
    fig.text(.085,.935,'Validated ten-patient cohort · random sampling · means over three seeds',fontsize=8)
    for j,path in enumerate(example['images']):
        row,col=divmod(j,5)
        ax=fig.add_axes([.085+col*.177,.825-row*.095,.151,.070])
        ax.imshow(Image.open(path),cmap='gray');ax.axis('off')
        ax.text(.5,-.07,values.image_label.iloc[j],ha='center',va='top',transform=ax.transAxes,fontsize=8)
    methods=[('marginal_correlation','B','Pearson','Correlation'),
             ('ridge','C','Ridge','Probability\ncoefficient'),
             ('elastic_net','D','Elastic Net','Probability\ncoefficient')]
    for k,(method,letter,title,xlabel) in enumerate(methods):
        ax=fig.add_axes([.13+k*.285,.12,.235,.35])
        g=values[method].to_numpy()
        color=COLORS['pearson' if method=='marginal_correlation' else method]
        ax.barh(np.arange(n),g,height=.68,color=np.where(g>=0,POSITIVE,NEGATIVE),zorder=3)
        ax.axvline(0,c='#343D46',lw=.8,zorder=4)
        ax.set(yticks=np.arange(n),yticklabels=values.image_label if k==0 else [],ylim=(n-.5,-.5),xlabel=xlabel)
        ax.set_title(title,loc='left',fontsize=9.5,color=color,pad=14,weight='bold')
        ax.text(-.20,1.065,letter,transform=ax.transAxes,fontsize=11,weight='bold')
        bound=max(np.max(np.abs(g))*1.12,1e-4);ax.set_xlim(-bound,bound)
        ax.tick_params(axis='y',length=0,labelsize=8)
        ax.spines['left'].set_visible(False);ax.grid(axis='y',color='#EDF0F2',lw=.4,zorder=0)
        ax.tick_params(axis='x',labelsize=7.7)
        ax.locator_params(axis='x',nbins=3)
    fig.text(.13,.035,'Positive association',color=POSITIVE,fontsize=8.3,weight='bold')
    fig.text(.47,.035,'Negative association',color=NEGATIVE,fontsize=8.3,weight='bold')
    fig.text(.13,.012,'Image ordering is shared across panels; coefficient and correlation scales differ.',fontsize=7.8)
    save(fig,Path(output),'figure2_influence')
    values.to_csv(Path(output)/'image_influence_example.csv',index=False)
    caption=(f'Patient-level explanation example from the validated ten-patient cohort (pilot ordinal {example["index"]+1}). '
        'A, twenty cropped ultrasound views, labeled I01–I20 in the frozen image order. '
        'B–D, Pearson inclusion–probability correlations and Ridge/Elastic Net probability-regression coefficients '
        'from random sampling, averaged over seeds 0, 1 and 2; image ordering is identical across all panels. '
        'Blue and terracotta denote positive and negative associations with class-1 probability. '
        'Correlation and coefficient axes have different units and scales; magnitudes are not directly interchangeable. '
        'The illustration is the first pilot patient with nonconstant vectors for every method, arm and seed, '
        'rather than a patient selected for fidelity or deletion success. No significance coding or confidence intervals are shown. '
        'These are perturbation-distribution-dependent model-output associations, not clinical or causal image importance.')
    return caption,example
