"""Paired achieved balance and patient-level sampling distributions."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from proceedings_data import ARMS,patient_average
from proceedings_style import COLORS,LABELS,MARKERS,configure,panel,save


def ecdf(ax,values,color,label,style='-'):
    x=np.sort(np.asarray(values.dropna(),float))
    if len(x):ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',c=color,label=label,ls=style,lw=1.7)


def sampling_figure(data,output):
    configure();fig=plt.figure(figsize=(5.5,4.8))
    a=fig.add_axes([.14,.55,.80,.30]);b=fig.add_axes([.14,.13,.34,.27]);c=fig.add_axes([.63,.13,.33,.27])
    d=patient_average(data['design'],['positive_fraction','duplicate_rows']);p=d.pivot(index='patient_index',columns='arm',values='positive_fraction')
    panel(a,'A','Achieved subset-class predictions')
    for _,row in p.iterrows():a.plot([0,1],[row.random,row.adaptive],c='#AEB9C4',alpha=.35,lw=.6,zorder=1)
    for arm in ARMS:
        j=ARMS.index(arm);v=p[arm]
        a.scatter(np.full(len(v),j),v,s=14,c=COLORS[arm],marker=MARKERS[arm],alpha=.65,zorder=3)
        a.scatter([j],[v.mean()],s=50,c=COLORS[arm],marker='D',edgecolor='white',linewidth=.9,zorder=5)
    a.axhline(.5,c='#59636E',ls='--',lw=.9)
    a.set(xticks=[0,1],xticklabels=['Random','Adaptive'],ylabel='Class-1 proportion',ylim=(0,1.06),xlim=(-.30,1.30))
    a.yaxis.set_major_formatter(PercentFormatter(1))
    success=int(data['stages'].balance_reached.sum());runs=len(data['stages'])
    fig.text(.14,.94,f"{data['n']} patients · three seeds per patient",fontsize=9.3,weight='bold')
    fig.text(.14,.90,f'Adaptive 50/50 target attained in {success}/{runs} runs.',fontsize=8)
    panel(b,'B','Pool-deficit reallocation')
    s=data['stages'];rates=s.assign(rate=s.deficit_rows/s.biased_rows.replace(0,np.nan)).groupby('patient_index').rate.mean()
    ecdf(b,rates,COLORS['adaptive'],'Adaptive')
    b.set(xlabel='Biased draws reallocated',ylabel='Fraction of patients',xlim=(0,1.02),ylim=(0,1.04))
    b.xaxis.set_major_formatter(PercentFormatter(1));b.yaxis.set_major_formatter(PercentFormatter(1))
    b.text(.04,.96,f'n = {rates.notna().sum()}',transform=b.transAxes,va='top',fontsize=8)
    panel(c,'C','Repeated training masks')
    for arm in ARMS:ecdf(c,d[d.arm==arm].duplicate_rows/1000,COLORS[arm],LABELS[arm],'-' if arm=='random' else '--')
    c.set(xlabel='Duplicate draws',ylabel='Fraction of patients',xlim=(0,max(.25,d.duplicate_rows.max()/1000*1.08)),ylim=(0,1.04))
    c.xaxis.set_major_formatter(PercentFormatter(1));c.yaxis.set_major_formatter(PercentFormatter(1))
    c.legend(frameon=False,fontsize=7.8,loc='lower right')
    save(fig,output,'figure3_sampling')
    return (f'Sampling comparison for {data["n"]} patients and {runs} patient–seed runs. '
        'A, paired achieved class-1 proportions: each point is a three-seed patient mean, connecting lines pair patients, '
        'and diamonds are equal-patient means. The dashed 50% line is the adaptive objective, not a requirement for random sampling; '
        f'exact 500/500 target attainment is {success}/{runs} adaptive runs. '
        'B–C, empirical cumulative distributions of patient-level reallocation and duplicate-mask rates. '
        f'B includes {rates.notna().sum()} patients with at least one biased stage; per-seed reallocated/biased proportions are averaged over those seeds within patient. '
        'C averages duplicate rows beyond unique masks divided by 1,000 over all three seeds within patient. '
        'These distributions describe patient variation, not uncertainty intervals. Intended singleton-pool composition and achieved subset classes are different quantities.')
