"""Ten-patient-only rank stability and single-image deletion diagnostics."""
import numpy as np
import matplotlib.pyplot as plt
from proceedings_tables import stability_patient
from proceedings_style import COLORS, LABELS, configure, panel, save

METHODS=('ridge','elastic_net','marginal_correlation')


def diagnostic_panel(ax,frame,metric,letter,title,ylabel):
    panel(ax,letter,title)
    labels=[]
    for j,method in enumerate(METHODS):
        ns=[]
        for arm,offset,marker in (('random',-.17,'o'),('adaptive',.17,'^')):
            values=frame[(frame.method==method)&(frame.arm==arm)][metric].dropna()
            color=COLORS['pearson' if method=='marginal_correlation' else method]
            ax.scatter(np.full(len(values),j+offset)+np.linspace(-.035,.035,len(values)),values,
                       s=14,c=color,marker=marker,alpha=.8)
            if len(values):ax.plot([j+offset-.09,j+offset+.09],[values.mean()]*2,c='#222222',lw=1.2)
            ns.append(len(values))
        labels.append(LABELS[method]+f'\nR:{ns[0]} A:{ns[1]}')
    ax.set(xticks=range(3),xticklabels=labels,ylabel=ylabel,xlim=(-.5,2.5))


def diagnostics_figure(pilot,output):
    configure();fig,axes=plt.subplots(2,2,figsize=(5.5,5.3))
    fig.subplots_adjust(left=.12,right=.98,bottom=.13,top=.91,wspace=.48,hspace=.76)
    s=stability_patient(pilot['stability'])
    diagnostic_panel(axes[0,0],s,'spearman','A','Seed ranking stability','Mean seed-pair Spearman')
    axes[0,0].set_ylim(-1.05,1.05)
    diagnostic_panel(axes[0,1],s,'top_five_jaccard','B','Top-five agreement','Mean seed-pair Jaccard')
    axes[0,1].set_ylim(-.03,1.05)
    diagnostic_panel(axes[1,0],pilot['loo_agreement'],'spearman','C','LOO rank agreement','Mean ranking–LOO Spearman')
    axes[1,0].set_ylim(-1.05,1.05)
    ax=axes[1,1];panel(ax,'D','LOO change magnitude')
    means=pilot['loo'].assign(abs_delta=pilot['loo'].delta_class1.abs()).groupby('patient_index').abs_delta.mean()
    ax.scatter(means.index+1,means,s=18,c=COLORS['reference'])
    ax.set(xlabel='Pilot patient index',ylabel='Mean absolute LOO change',ylim=(0,max(.001,means.max()*1.15)),xlim=(.5,10.5))
    fig.text(.12,.025,'Markers: ○ Random    △ Adaptive    Bars: patient mean    R/A: defined patients',fontsize=7.5)
    fig.suptitle('Supplementary diagnostics · ten patients only · three seeds',fontsize=9,y=.995)
    save(fig,output,'figure5_diagnostics')
    return ('Ten-patient convenience pilot only. A–B, Spearman rank correlation and top-five Jaccard averaged across all three seed pairs within patient. '
        'C, rank agreement with full-minus-omitted-image class-1 probability changes, averaged over three training seeds within patient; this descriptive CPU calculation uses saved LOO inference. '
        'D, patient mean absolute class-1 change across 20 single-image deletions. '
        'Method colors distinguish Ridge, Elastic Net and Pearson; circles/triangles distinguish random/adaptive training. '
        'Available counts are printed under each method. Constant/undefined vectors are unavailable; no arbitrary tie-broken list is treated as stability. '
        'Points represent patients and bars the mean of defined patient values. No bootstrap, significance tests, clinical validation or 135-patient secondary analysis is implied.')
