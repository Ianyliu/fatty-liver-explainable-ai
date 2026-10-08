"""Primary paired fidelity, paired changes and exact constant-baseline comparisons."""
import numpy as np
import matplotlib.pyplot as plt
from proceedings_data import ARMS,patient_average
from proceedings_style import COLORS,MARKERS,LABELS,configure,panel,save


def fidelity_figure(data,pilot,output):
    configure();fig=plt.figure(figsize=(5.5,4.9))
    a=fig.add_axes([.14,.55,.34,.31]);b=fig.add_axes([.64,.55,.32,.31]);c=fig.add_axes([.17,.15,.77,.22])
    p=data['patients'];p=p[p.primary_available]
    panel(a,'A','Paired patient fidelity')
    for row in p.itertuples():a.plot([0,1],[row.random_mean_novel_mae,row.adaptive_mean_novel_mae],c='#AFBBC7',lw=.6,alpha=.38)
    for arm in ARMS:
        v=p[arm+'_mean_novel_mae'];j=ARMS.index(arm)
        a.scatter(np.full(len(v),j),v,s=15,c=COLORS[arm],marker=MARKERS[arm],alpha=.7)
        a.scatter([j],[v.mean()],s=47,c=COLORS[arm],marker='D',edgecolor='white',linewidth=.8,zorder=5)
    high=max(.01,p[['random_mean_novel_mae','adaptive_mean_novel_mae']].max().max()*1.12)
    a.set(xticks=[0,1],xticklabels=['Random','Adaptive'],ylabel='Shared-novel MAE',ylim=(0,high),xlim=(-.28,1.28))
    panel(b,'B','Paired MAE changes')
    values=p.paired_adaptive_minus_random_mae.to_numpy()
    if len(values):
        bins=np.histogram_bin_edges(values,bins=min(15,max(5,int(np.sqrt(len(values))))))
        counts,edges=np.histogram(values,bins=bins)
        b.bar(edges[:-1],counts,width=np.diff(edges)*.94,align='edge',color='#577DA8',ec='white',lw=.4)
        b.axvline(values.mean(),c='#20262D',lw=1.2,ls=':')
        b.text(.96,.95,f'Mean {values.mean():+.4f}',transform=b.transAxes,ha='right',va='top',fontsize=8)
    b.axvline(0,c='#4E5964',ls='--',lw=.9)
    b.set(xlabel='Adaptive − random MAE',ylabel='Patients',ylim=(0,max(1,b.get_ylim()[1])*1.08))
    fig.text(.64,.455,'Negative favors adaptive;\npositive favors random',fontsize=7.8)
    panel(c,'C','Ridge versus its own training-mean constant')
    f=patient_average(data['fidelity'],['novel_mae','constant_baseline_novel_mae'])
    counts=[]
    for arm in ARMS:
        g=f[f.arm==arm];v=(g.constant_baseline_novel_mae-g.novel_mae).dropna();j=ARMS.index(arm)
        c.scatter(v,np.full(len(v),j)+np.linspace(-.12,.12,len(v)),s=15,c=COLORS[arm],marker=MARKERS[arm],alpha=.65,zorder=3)
        if len(v):c.scatter([v.mean()],[j],s=46,c=COLORS[arm],marker='D',edgecolor='white',linewidth=.8,zorder=5)
        counts.append(f'{LABELS[arm]}: {int((v>0).sum())}/{len(v)} patients favor Ridge')
    c.axvline(0,c='#555F69',ls='--',lw=.9)
    c.set(yticks=[0,1],yticklabels=['Random','Adaptive'],xlabel='Constant MAE − Ridge MAE',ylim=(1.48,-.48))
    c.spines['left'].set_visible(False);c.tick_params(axis='y',length=0)
    fig.text(.17,.025,'; '.join(counts).replace(' patients favor Ridge','')+' favor Ridge.\nPositive values favor Ridge over its constant baseline.',fontsize=7.7)
    fig.text(.14,.95,f"{data['n']} planned patients · {len(p)} with all primary seed pairs",fontsize=9.3,weight='bold')
    fig.text(.14,.91,'Patient means over three seeds; diamonds show means.',fontsize=8)
    save(fig,output,'figure4_fidelity')
    return (f'Probability fidelity in the primary population ({data["n"]} planned patients; {len(p)} with complete paired seed metrics). '
        'A, three-seed patient means connected across random and adaptive training; lower shared-novel MAE is better. '
        'B, the distribution of patient paired adaptive-minus-random MAE, with zero indicating equal error and a dotted line indicating the mean. '
        'C, each arm’s own training-mean-probability constant baseline minus Ridge MAE on exactly the same shared-novel rows. '
        'Positive values favor Ridge over its baseline. Points represent available patient means and diamonds their means; counts are stated. '
        'No confidence intervals or significance tests are shown. Elastic Net comparisons are restricted to the ten-patient supplementary analysis.')
