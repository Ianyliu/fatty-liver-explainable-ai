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
    # Preserve the README's visual language: signed vertical bars with ultrasound
    # thumbnails next to the bar ends. No historical uncertainty is transferred.
    display=values.sort_values('marginal_correlation',ascending=False,kind='stable')
    order=display.index.to_numpy();positions=np.arange(n)
    fig=plt.figure(figsize=(5.5,7.1))
    fig.text(.13,.975,'Image-level marginal and conditional influence',fontsize=10,weight='bold')
    fig.text(.13,.947,'One validated pilot patient · random sampling · three-seed means',fontsize=8)
    methods=[('marginal_correlation','A','Pearson correlation','Correlation'),
             ('elastic_net','B','Elastic Net','Probability coefficient'),
             ('ridge','C','Ridge','Probability coefficient')]
    for k,(method,letter,title,ylabel) in enumerate(methods):
        ax=fig.add_axes([.13,.70-k*.30,.84,.18])
        g=display[method].to_numpy();span=max(float(np.ptp(g)),float(np.max(np.abs(g))),1e-4)
        ax.bar(positions,g,width=.79,color=np.where(g>=0,'#2AC6F2','#FA9B90'),
               edgecolor=np.where(g>=0,'#168BAA','#BC645B'),linewidth=.4,zorder=2)
        ax.axhline(0,c='#4E5964',lw=.65,zorder=3)
        for position,index,value in zip(positions,order,g):
            direction=1 if value>=0 else -1
            center=value+direction*.105*span
            ax.imshow(Image.open(example['images'][index]),
                extent=(position-.365,position+.365,center-.075*span,center+.075*span),
                aspect='auto',zorder=4)
        ax.set(xlim=(-.65,n-.35),ylim=(min(0,float(g.min()))-.25*span,max(0,float(g.max()))+.25*span),
               xticks=positions,xticklabels=display.image_label,ylabel=ylabel)
        ax.tick_params(axis='x',labelsize=7,labelrotation=55,length=2,pad=1)
        ax.tick_params(axis='y',labelsize=7.5)
        ax.locator_params(axis='y',nbins=4)
        color=COLORS['pearson' if method=='marginal_correlation' else method]
        ax.set_title(title,loc='left',fontsize=9.5,color=color,weight='bold',pad=10)
        ax.text(-.10,1.16,letter,transform=ax.transAxes,fontsize=11,weight='bold')
    fig.text(.13,.025,'Cyan: positive association     Coral: negative association',fontsize=8)
    fig.text(.13,.005,'Same image order in all panels; no significance fading or uncertainty intervals.',fontsize=7.5)
    save(fig,Path(output),'figure2_influence')
    values.to_csv(Path(output)/'image_influence_example.csv',index=False)
    caption=(f'Image-influence plots in the original README style, using current validated outputs (pilot ordinal {example["index"]+1}). '
        'A, marginal Pearson inclusion–probability correlations; B–C, conditional Elastic Net and Ridge probability-regression coefficients. '
        'Bars show means over seeds 0, 1 and 2 under random sampling, with the corresponding cropped ultrasound view beside each bar end. '
        'All panels share the same twenty images and order, sorted by mean Pearson correlation. I01–I20 are display labels assigned in the frozen image order. '
        'Cyan and coral denote positive and negative associations. Correlation and coefficient axes have different units and scales. '
        'The example is the first pilot patient with nonconstant vectors for every method, arm and seed, not selected for evaluation success. '
        'Historical error bars and faded significance coding are omitted because current uncertainty estimates have not been validated. '
        'These are perturbation-distribution-dependent model-output associations, not clinical or causal importance.')
    return caption,example
