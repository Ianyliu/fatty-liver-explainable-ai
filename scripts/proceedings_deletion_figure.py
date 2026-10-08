"""Deletion plots that respect actual fractions and matched patient grids."""
import numpy as np
import matplotlib.pyplot as plt
from proceedings_data import ARMS, patient_average
from proceedings_style import COLORS, MARKERS, LABELS, configure, panel, save


def deletion_figure(data,output):
    configure();fig,axes=plt.subplots(2,2,figsize=(5.5,5.0))
    fig.subplots_adjust(left=.12,right=.97,bottom=.12,top=.91,wspace=.48,hspace=.70)
    d=patient_average(data['deletion'],['area_under_curve','normalized_area'],('patient_index','arm','control'))
    raw={a:d[d.arm==a].pivot(index='patient_index',columns='control',values='area_under_curve') for a in ARMS}
    diffs={a:raw[a].descending-raw[a].random for a in ARMS}
    # Deterministic descriptive illustration: nearest median random-arm paired AUC.
    illustration=int((diffs['random']-diffs['random'].median()).abs().sort_index().idxmin())
    controls={'descending':('#222222','-','o','Descending'),
              'ascending':('#888888',':','s','Ascending'),
              'random':('#555555','--','^','Seeded random')}
    for ax,arm,letter in ((axes[0,0],'random','A'),(axes[1,1],'adaptive','D')):
        panel(ax,letter,LABELS[arm]+'-trained')
        g=data['trajectories'];g=g[(g.patient_index==illustration)&(g.arm==arm)]
        for control,(color,style,marker,label) in controls.items():
            curve=g[g.control==control].groupby('deleted_fraction').p_class1.mean()
            ax.plot(curve.index,curve,ls=style,marker=marker,c=color,label=label,markersize=3)
        ax.set(xlabel='Actual fraction deleted',ylabel='Class-1 probability',ylim=(0,1.04),xlim=(0,.51))
        if letter=='A':ax.legend(frameon=False,fontsize=7,loc='lower left',handlelength=1.7)
    ax=axes[0,1];panel(ax,'B','Raw AUC differences')
    for arm in ARMS:
        x=ARMS.index(arm);values=diffs[arm]
        ax.scatter(np.full(len(values),x)+np.linspace(-.08,.08,len(values)),values,c=COLORS[arm],marker=MARKERS[arm],s=13,alpha=.75)
        ax.plot([x-.18,x+.18],[values.mean()]*2,c='#222222',lw=1.5)
    ax.axhline(0,c='#555555',ls='--',lw=.8)
    ax.set(xticks=[0,1],xticklabels=['Random','Adaptive'],ylabel='Descending − random AUC',xlim=(-.4,1.4))
    ax=axes[1,0];panel(ax,'C','Span-normalized AUC')
    ax.axhline(0,c='#BBBBBB',lw=.65);ax.axvline(0,c='#BBBBBB',lw=.65)
    for arm in ARMS:
        norm=d[d.arm==arm].pivot(index='patient_index',columns='control',values='normalized_area')
        ax.scatter(diffs[arm],norm.descending-norm.random,c=COLORS[arm],marker=MARKERS[arm],s=13,alpha=.7)
    ax.set(xlabel='Raw AUC difference',ylabel='Normalized AUC difference')
    fig.suptitle(f"{data['n']} patients · three seeds · rebuilt graphs for every deletion",fontsize=9,y=.995)
    save(fig,output,'figure4_deletion')
    return (f'Primary population: {data["n"]} patients, with seed means computed within patient. '
        f'A and D, the same illustrative patient (ordinal {illustration+1}), selected deterministically as nearest the median random-trained descending-minus-random AUC. '
        'Curves average three seeds on that patient’s actual fraction grid; no unequal patient grids are pooled. '
        'B, patient descending-minus-random raw trapezoidal AUC; negative values indicate lower class-1 trajectories for descending deletion. '
        'C, raw versus span-normalized paired AUC, showing patient-specific integration ranges. '
        'Colors/markers identify sampling arms in B–C; line styles identify controls in A–D. One seeded random order per patient–seed is shared between arms. '
        'Points show patient variability, bars show means, and no confidence bands are implied. These are model-behavior interventions, not clinical or causal importance.')
