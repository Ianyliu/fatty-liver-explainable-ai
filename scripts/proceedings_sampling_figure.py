"""Four-panel sampling feasibility figure from validated saved predictions."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from proceedings_data import ARMS, patient_average
from proceedings_style import COLORS, LABELS, MARKERS, configure, panel, save


def sampling_figure(data, output):
    configure()
    fig, axes=plt.subplots(2,2,figsize=(5.5,4.8))
    fig.subplots_adjust(left=.12,right=.97,bottom=.12,top=.91,wspace=.48,hspace=.65)
    n=data['n']; d=patient_average(data['design'],['positive_fraction','duplicate_rows'])
    ax=axes[0,0];panel(ax,'A','Achieved predictions')
    for arm,offset in (('random',-.1),('adaptive',.1)):
        g=d[d.arm==arm]
        ax.scatter(g.patient_index+1+offset,g.positive_fraction,s=13,c=COLORS[arm],marker=MARKERS[arm],label=LABELS[arm],alpha=.8)
    ax.axhline(.5,color=COLORS['reference'],ls='--',lw=.8)
    ax.set(xlabel='Patient index',ylabel='Class-1 predictions (%)',ylim=(0,1.04),xlim=(.5,n+.5))
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,loc='lower right',handletextpad=.3)
    ax=axes[0,1];panel(ax,'B','Requested target attainment')
    success=int(data['stages'].balance_reached.sum()); total=len(data['stages'])
    ax.barh([1,0],[success,total-success],color=['white',COLORS['adaptive']],edgecolor='#333333',height=.55,hatch=['///',''])
    ax.set(yticks=[1,0],yticklabels=['50/50 attained','Not attained'],xlabel='Patient–seed runs',xlim=(0,max(total,1)*1.18))
    ax.text(success+total*.035,1,str(success),va='center',fontsize=8)
    ax.text(total-success+total*.035,0,str(total-success),va='center',fontsize=8)
    ax=axes[1,0];panel(ax,'C','Pool-deficit reallocation')
    s=data['stages']; by=s.groupby('patient_index')[['deficit_rows','biased_rows']].sum()
    fractions=by.deficit_rows/by.biased_rows.replace(0,np.nan)
    valid=fractions.dropna()
    ax.scatter(valid.index+1,valid,s=14,color=COLORS['adaptive'],marker='^')
    ax.set(xlabel='Patient index',ylabel='Biased draws reallocated (%)',ylim=(-.02,1.04),xlim=(.5,n+.5))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.text(.04,.08,f'{len(valid)}/{n} patients with biased draws',transform=ax.transAxes,fontsize=7.5)
    ax=axes[1,1];panel(ax,'D','Repeated training masks')
    for arm,offset in (('random',-.12),('adaptive',.12)):
        g=d[d.arm==arm]
        ax.scatter(np.full(len(g),ARMS.index(arm))+np.linspace(-.06,.06,len(g)),g.duplicate_rows/1000,
                   s=14,c=COLORS[arm],marker=MARKERS[arm],alpha=.75)
        ax.plot([ARMS.index(arm)-.18,ARMS.index(arm)+.18],[g.duplicate_rows.mean()/1000]*2,c='#222222',lw=1.5)
    ax.set(xticks=[0,1],xticklabels=['Random','Adaptive'],ylabel='Duplicate rows (%)',ylim=(0,max(.2,d.duplicate_rows.max()/1000*1.15)))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle(f"{'Full cohort' if n==135 else 'Exploratory pilot'}: {n} patients · three seeds per patient",fontsize=9,y=.995)
    save(fig,output,'figure2_sampling')
    return (f'Primary population: {n} patients and {total} patient–seed runs. '
        'A, achieved training class-1 proportions averaged across seeds within patient; dashed line is the adaptive target, not a random-arm requirement. '
        'B, adaptive runs attaining exactly 500 predictions of each class versus failing the target. '
        'C, reallocated biased draws divided by all biased draws within patient; patients without biased draws are unavailable. '
        'D, repeated rows beyond unique masks divided by 1,000, averaged within patient; points show patients and bars their mean. '
        'The intended 85/15 singleton-pool mixture is distinct from achieved subset classes; realized pool composition and conditioning are in the sampling table. No confidence intervals are shown.')
