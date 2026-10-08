"""Control deletion trajectories and paired patient heterogeneity on actual grids."""
import numpy as np
import matplotlib.pyplot as plt
from proceedings_data import ARMS,patient_average
from proceedings_style import COLORS,MARKERS,LABELS,configure,panel,save


def deletion_figure(data,output):
    configure();fig=plt.figure(figsize=(5.5,4.7))
    a=fig.add_axes([.14,.54,.34,.31]);b=fig.add_axes([.64,.54,.32,.31]);c=fig.add_axes([.17,.13,.77,.24])
    d=patient_average(data['deletion'],['area_under_curve','normalized_area'],('patient_index','arm','control'))
    raw={arm:d[d.arm==arm].pivot(index='patient_index',columns='control',values='area_under_curve') for arm in ARMS}
    diffs={arm:raw[arm].descending-raw[arm].random for arm in ARMS}
    illustration=int((diffs['random']-diffs['random'].median()).abs().sort_index().idxmin())
    controls={'descending':('#008B8B','-','o','Descending'),
              'ascending':('#9B6B52','--','s','Ascending'),
              'random':('#49535E',':','^','Random order')}
    for ax,arm,letter in ((a,'random','A'),(b,'adaptive','B')):
        panel(ax,letter,LABELS[arm]+'-trained Ridge')
        g=data['trajectories'];g=g[(g.patient_index==illustration)&(g.arm==arm)]
        for control,(color,style,marker,label) in controls.items():
            curve=g[g.control==control].groupby('deleted_fraction').p_class1.mean()
            ax.plot(curve.index,curve,ls=style,marker=marker,c=color,label=label,markersize=3.4,lw=1.6)
        ax.set(xlabel='Actual fraction deleted',ylabel='Class-1 probability',ylim=(0,1.04),xlim=(0,.51))
    handles,labels=a.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.955),ncol=3,frameon=False,fontsize=8,handlelength=2)
    panel(c,'C','Paired deletion effects across patients')
    for patient in diffs['random'].index:c.plot([0,1],[diffs[arm].loc[patient] for arm in ARMS],c='#AEB9C4',alpha=.35,lw=.6)
    for arm in ARMS:
        j=ARMS.index(arm);v=diffs[arm]
        c.scatter(np.full(len(v),j),v,c=COLORS[arm],marker=MARKERS[arm],s=16,alpha=.65,zorder=3)
        c.scatter([j],[v.mean()],c=COLORS[arm],s=48,marker='D',edgecolor='white',linewidth=.8,zorder=5)
    c.axhline(0,c='#4E5964',ls='--',lw=.9)
    c.set(xticks=[0,1],xticklabels=['Random-trained','Adaptive-trained'],ylabel='Descending − random AUC',xlim=(-.3,1.3))
    fig.text(.14,.975,f"Deletion controls · {data['n']} patients · three seeds",fontsize=9.3,weight='bold')
    fig.text(.17,.025,'Negative: lower class-1 trajectory under descending deletion.',fontsize=7.9)
    save(fig,output,'figure5_deletion')
    return (f'Deletion-based model behavior for {data["n"]} patients. '
        f'A–B, the same illustrative patient (ordinal {illustration+1}), selected as nearest the median random-trained descending-minus-random raw AUC. '
        'Descending and ascending signed Ridge rankings are compared with one seeded random order per seed shared across arms. '
        'Curves average three seeds on that patient’s actual deletion-fraction grid; unequal patient grids are not pooled. '
        'C, paired patient mean differences in raw trapezoidal AUC between descending and random deletion, with equal-patient means shown by diamonds. '
        'A negative difference indicates a lower class-1 trajectory, not clinical or causal importance. '
        'Points and connecting lines show patient heterogeneity; no inferential uncertainty bands are implied. '
        'Ascending contrasts and normalized-AUC diagnostics remain in the numerical evidence package.')
