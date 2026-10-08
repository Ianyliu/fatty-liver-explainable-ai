"""Paired probability-fidelity diagnostics, with a labeled pilot-only panel."""
import numpy as np
import matplotlib.pyplot as plt
from proceedings_data import ARMS, patient_average
from proceedings_style import COLORS, MARKERS, LABELS, configure, panel, save


def fidelity_figure(data,pilot,output):
    configure();fig,axes=plt.subplots(2,2,figsize=(5.5,5.0))
    fig.subplots_adjust(left=.12,right=.97,bottom=.12,top=.91,wspace=.48,hspace=.68)
    p=data['patients'];p=p[p.primary_available]
    ax=axes[0,0];panel(ax,'A','Paired patient fidelity')
    for row in p.itertuples():
        ax.plot([0,1],[row.random_mean_novel_mae,row.adaptive_mean_novel_mae],c='#999999',alpha=.38,lw=.6)
    for arm in ARMS:
        ax.scatter(np.full(len(p),ARMS.index(arm)),p[arm+'_mean_novel_mae'],c=COLORS[arm],marker=MARKERS[arm],s=13,alpha=.8)
    high=max(.01,p[['random_mean_novel_mae','adaptive_mean_novel_mae']].max().max()*1.12)
    ax.set(xticks=[0,1],xticklabels=['Random','Adaptive'],ylabel='Novel-mask MAE',ylim=(0,high),xlim=(-.25,1.25))
    ax=axes[0,1];panel(ax,'B','Paired MAE difference')
    values=np.sort(p.paired_adaptive_minus_random_mae)
    ax.axvline(0,c='#555555',ls='--',lw=.8)
    ax.scatter(values,np.arange(1,len(values)+1),s=13,c='#333333')
    ax.set(xlabel='Adaptive − random MAE',ylabel='Ordered patient',ylim=(0,len(values)+1))
    ax.text(.02,.98,'<0 favors adaptive',va='top',fontsize=7,transform=ax.transAxes)
    ax=axes[1,0];panel(ax,'C','Own training-mean baseline')
    f=patient_average(data['fidelity'],['novel_mae','constant_baseline_novel_mae'])
    high=max(.01,f[['novel_mae','constant_baseline_novel_mae']].max().max()*1.08)
    ax.plot([0,high],[0,high],c=COLORS['reference'],ls='--',lw=.8)
    for arm in ARMS:
        g=f[f.arm==arm]
        ax.scatter(g.constant_baseline_novel_mae,g.novel_mae,s=13,c=COLORS[arm],marker=MARKERS[arm],alpha=.75,label=LABELS[arm])
    ax.set(xlabel='Constant-baseline MAE',ylabel='Ridge MAE',xlim=(0,high),ylim=(0,high))
    ax.text(.04,.96,'Below diagonal: Ridge better',va='top',fontsize=7,transform=ax.transAxes)
    ax=axes[1,1];panel(ax,'D','Elastic Net: pilot only (n=10)')
    e=patient_average(pilot['enet'],['novel_mae'],('patient_index','arm','method'))
    for arm in ARMS:
        g=e[e.arm==arm].pivot(index='patient_index',columns='method',values='novel_mae')
        delta=g.elastic_net-g.ridge
        ax.scatter(np.full(len(g),ARMS.index(arm))+np.linspace(-.07,.07,len(g)),delta,s=15,c=COLORS[arm],marker=MARKERS[arm])
        ax.plot([ARMS.index(arm)-.18,ARMS.index(arm)+.18],[delta.mean()]*2,c='#222222',lw=1.5)
    ax.axhline(0,c=COLORS['reference'],ls='--',lw=.8)
    ax.set(xticks=[0,1],xticklabels=['Random','Adaptive'],ylabel='Elastic Net − Ridge MAE',xlim=(-.45,1.45))
    fig.suptitle(f"Primary population: {data['n']} patients · {len(p)} with all primary seed pairs",fontsize=9,y=.995)
    save(fig,output,'figure3_fidelity')
    return (f'A–C, primary population ({data["n"]} planned patients; {len(p)} with all primary seed pairs). '
        'A, lines join patient means across three seeds. B, patient adaptive-minus-random MAE; zero denotes equal error and negative values favor adaptive. '
        'C, each arm’s fixed training-mean constant versus its Ridge error on identical shared-novel evaluation rows; below the identity line favors Ridge. '
        'D, ten-patient pilot only: paired Elastic Net-minus-Ridge MAE, with training-only five-fold selection. '
        'Points represent patients, horizontal bars their means. Lower MAE is better; no inferential intervals are shown.')
