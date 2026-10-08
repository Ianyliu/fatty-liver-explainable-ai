"""Embed the author's original README workflow without changing its artwork.

Historical labels are qualified in the manuscript caption and Methods. This
export preserves the original pixels, colors, architecture, positions and arrows.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from proceedings_style import configure, save

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'manuscript/proceedings_2026/assets'


def workflow(output):
    configure();original=plt.imread(ASSETS/'original_workflow.png')
    height,width=original.shape[:2]
    fig=plt.figure(figsize=(5.5,5.5*height/width))
    ax=fig.add_axes([0,0,1,1]);ax.imshow(original,interpolation='none');ax.axis('off')
    save(fig,Path(output),'figure1_workflow')


def workflow_comparison(output,revised):
    configure();fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    fig.subplots_adjust(left=.02,right=.98,bottom=.12,top=.90,wspace=.05)
    for ax,path,title in zip(axes,[ASSETS/'original_workflow.png',Path(revised)/'figure1_workflow.png'],
                             ['Original README workflow','Restored main Figure 1']):
        ax.imshow(plt.imread(path));ax.axis('off');ax.set_title(title,fontsize=12,pad=12)
    fig.text(.03,.035,'Original artwork restored unchanged. The caption identifies historical labels and the current validated settings.',fontsize=9)
    save(fig,Path(output),'workflow_original_vs_revised')
