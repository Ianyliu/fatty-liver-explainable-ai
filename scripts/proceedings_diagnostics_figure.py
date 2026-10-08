"""Compact ten-patient similarity matrix and auxiliary surrogate comparison."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from proceedings_data import patient_average
from proceedings_tables import stability_patient
from proceedings_style import COLORS,LABELS,MARKERS,configure,panel,save

METHODS=('ridge','elastic_net','marginal_correlation')


def diagnostics_figure(pilot,output):
    configure();fig=plt.figure(figsize=(5.5,5.1));a=fig.add_axes([.30,.48,.60,.36]);b=fig.add_axes([.18,.13,.72,.19])
    s=stability_patient(pilot['stability']);loo=pilot['loo_agreement'];matrix=[];counts=[];labels=[]
    for method in METHODS:
        for arm in ('random','adaptive'):
            g=s[(s.method==method)&(s.arm==arm)];l=loo[(loo.method==method)&(loo.arm==arm)]
            vals=[g.spearman,g.top_five_jaccard,g.sign_agreement,l.spearman]
            matrix.append([v.mean() for v in vals]);counts.append([v.notna().sum() for v in vals])
            labels.append(LABELS[method]+' · '+LABELS[arm])
    matrix=np.array(matrix);counts=np.array(counts)
    cmap=LinearSegmentedColormap.from_list('association',['#B35836','#FFFFFF','#2166AC']);cmap.set_bad('#E7EAEE')
    im=a.imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=-1,vmax=1,aspect='auto')
    a.set(xticks=range(4),xticklabels=['Seed\nSpearman','Top-five\nJaccard','Sign\nagreement','LOO\nSpearman'],yticks=range(6),yticklabels=labels)
    a.tick_params(length=0,labelsize=8)
    for spine in a.spines.values():spine.set_visible(False)
    for r in range(6):
        color=COLORS['pearson' if METHODS[r//2]=='marginal_correlation' else METHODS[r//2]]
        a.get_yticklabels()[r].set_color(color)
        for c in range(4):
            value=matrix[r,c];text=f'{value:.3f}\n{counts[r,c]}/10' if np.isfinite(value) else 'NA\n0/10'
            a.text(c,r,text,ha='center',va='center',fontsize=8,color='white' if np.isfinite(value) and abs(value)>.72 else '#20262D')
    a.set_xticks(np.arange(-.5,4,1),minor=True);a.set_yticks(np.arange(-.5,6,1),minor=True)
    a.grid(which='minor',color='white',lw=2);a.tick_params(which='minor',length=0)
    fig.text(.035,.875,'A',fontsize=11,weight='bold');fig.text(.14,.875,'Complementary explanation diagnostics',fontsize=9.5,weight='bold')
    cax=fig.add_axes([.93,.48,.022,.36]);cb=fig.colorbar(im,cax=cax);cb.set_ticks([-1,0,1]);cb.ax.tick_params(labelsize=7.5);cb.outline.set_visible(False)
    panel(b,'B','Elastic Net versus Ridge fidelity')
    e=patient_average(pilot['enet'],['novel_mae'],('patient_index','arm','method'))
    for arm in ('random','adaptive'):
        g=e[e.arm==arm].pivot(index='patient_index',columns='method',values='novel_mae');delta=g.elastic_net-g.ridge;j=('random','adaptive').index(arm)
        b.scatter(delta,np.full(len(delta),j)+np.linspace(-.13,.13,len(delta)),s=20,c=COLORS[arm],marker=MARKERS[arm],alpha=.75)
        b.scatter([delta.mean()],[j],s=48,c=COLORS[arm],marker='D',edgecolor='white',linewidth=.9,zorder=5)
    b.axvline(0,c='#4F5A65',ls='--',lw=.9)
    b.set(yticks=[0,1],yticklabels=['Random','Adaptive'],xlabel='Elastic Net MAE − Ridge MAE',ylim=(1.45,-.45))
    b.spines['left'].set_visible(False);b.tick_params(axis='y',length=0)
    fig.text(.14,.96,'Additional analyses · ten-patient cohort only',fontsize=9.5,weight='bold')
    fig.text(.14,.925,'Cells: descriptive mean and defined patients. Shade: −1 to 1, not significance.',fontsize=7.8)
    fig.text(.18,.025,'B: three-seed patient means; diamonds show means.\nNegative favors Elastic Net.',fontsize=7.6)
    save(fig,output,'figure6_diagnostics')
    return ('Ten-patient supplementary analyses only. A, mean seed-pair Spearman, top-five Jaccard and sign agreement, '
        'and mean ranking-versus-LOO Spearman; each cell states the mean and number of defined patients out of ten. '
        'Seed comparisons first average all three pairs within patient; LOO comparisons average three training-seed correlations within patient. '
        'Constant/undefined vectors are unavailable and excluded from explicitly labeled available-patient means. '
        'Spearman ranges from −1 to 1; Jaccard and sign agreement range from 0 to 1. Shade represents descriptive similarity, not statistical significance. '
        'B, paired patient Elastic Net-minus-Ridge shared-novel MAE from training-only five-fold selection, with mean diamonds and no confidence intervals. '
        'No full-cohort Elastic Net, stability or LOO claim is implied.')
