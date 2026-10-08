"""Refined visual adaptation of the recovered draw.io workflow.

Ultrasound artwork is decoded from the original editable source, not substituted
with unrelated imagery. Clinical publication eligibility remains an author gate.
"""
import base64
import io
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
from proceedings_style import COLORS, configure, save

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'manuscript/proceedings_2026/assets'


def original_assets():
    cells={c.get('id'):c for c in ET.parse(ASSETS/'original_workflow.drawio').getroot().iter('mxCell')}
    def decode(identifier):
        style=cells[identifier].get('style')
        value=style.split('image=',1)[1].split(';',1)[0]
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(value.split(',',1)[1]))).convert('RGB'))
    return [decode(i) for i in ('SnyPj6xBETERhtKptUh--12','SnyPj6xBETERhtKptUh--13',
        'SnyPj6xBETERhtKptUh--15','CVcjwiK0NmAejh44XMRx-105')]


def workflow(output):
    configure();images=original_assets()
    fig,ax=plt.subplots(figsize=(5.5,7.2));fig.subplots_adjust(left=.015,right=.985,top=.99,bottom=.01)
    ax.set(xlim=(0,10),ylim=(0,13.5));ax.axis('off')
    def label(x,y,text,size=8.3,ha='center',weight=None,color='#20262D',va='center'):
        ax.text(x,y,text,fontsize=size,ha=ha,va=va,weight=weight,color=color,linespacing=1.25)
    def band(x,y,w,h,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05,rounding_size=.15',fc=color,ec='none',zorder=0))
    def arrow(x,y,xx,yy,color='#303840',style='-|>',lw=1):
        ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle=style,mutation_scale=9,color=color,lw=lw))
    def image(x,y,w=.68,h=.50,index=0,edge=None):
        ax.imshow(images[index%len(images)],extent=(x,x+w,y,y+h),aspect='auto',zorder=3)
        if edge:ax.add_patch(Rectangle((x,y),w,h,fill=False,ec=edge,lw=1.5,zorder=4))
    def heading(y,letter,text):
        label(.08,y,letter,11.5,ha='left',weight='bold');label(.48,y,text,9.5,ha='left',weight='bold')
    def stack(x,y,index,edge):
        for d in (2,1,0):image(x+d*.11,y+d*.08,index=index,edge=edge)
    heading(13.2,'A','Patient image sets and predicted image pools')
    label(1.25,12.75,'Patient image set',8.3,weight='bold')
    for j in range(3):image(.2+j*.75,12.05,index=j)
    label(1.3,11.8,'$I_1$     $I_2$     …     $I_n$',8)
    arrow(2.5,12.30,3.02,12.30)
    label(4.18,12.65,'Single-image inference',8.0,weight='bold')
    for j in range(3):
        image(3.08+j*.72,12.02,w=.57,h=.43,index=j)
        arrow(3.37+j*.72,11.98,3.37+j*.72,11.64)
    band(3.02,11.02,2.4,.65,'#FAF0BE');label(4.22,11.34,'Grouped-image\nGNN',7.9)
    arrow(5.44,11.37,5.85,11.37)
    band(5.95,11.02,1.8,1.65,'#F6DAD3');band(7.98,11.02,1.8,1.65,'#DDEBD6')
    label(6.85,12.33,'Predicted\nclass 0',8,weight='bold');label(8.88,12.33,'Predicted\nclass 1',8,weight='bold')
    stack(6.38,11.48,1,'#BA6A58');stack(8.39,11.48,0,'#5D8050')
    ax.plot([5.68,5.68,8.88],[11.37,10.98,10.98],c='#58616A',lw=.8)
    arrow(8.88,10.98,8.88,11.12,lw=.8)
    label(5,10.72,'Image pools reflect model predictions, not image-level disease labels.',7.5)
    heading(10.22,'B','Two-stage Adaptive Class-Balanced Sampling')
    band(.15,8.02,4.65,1.85,'#E6EFFA');band(5.12,8.02,4.65,1.85,'#F7E5F2')
    label(2.47,9.58,'Stage I · random subsets',8.8,weight='bold',color=COLORS['random'])
    label(2.47,9.18,'Random until one class quota is met',7.9)
    label(7.45,9.58,'Stage II · class-biased subsets',8.8,weight='bold',color='#804568')
    label(7.45,9.18,'Target 85/15; capacities constrain it',7.8)
    for r in range(2):
        for j in range(5-r):
            image(.53+j*.79,8.12+r*.52,w=.59,h=.43,index=j+r)
            image(5.49+j*.79,8.12+r*.52,w=.59,h=.43,index=0 if j<4 else 1,edge='#5D8050' if j<4 else '#BA6A58')
    arrow(4.84,8.90,5.08,8.90)
    ax.plot([1.25,.035,.035,2.47],[11.63,11.63,9.98,9.98],c='#58616A',lw=.8)
    arrow(2.47,9.98,2.47,9.84,lw=.8)
    ax.plot([9.79,9.96,9.96],[11.37,11.37,9.0],c='#58616A',lw=.8)
    arrow(9.96,9.0,9.77,9.0,lw=.8)
    ax.plot([2.47,2.47,7.45,7.45],[7.98,7.91,7.91,7.98],c='#58616A',lw=.8)
    ax.plot([2.47,.035,.035],[7.91,7.91,5.53],c='#58616A',lw=.8)
    arrow(.035,5.53,.17,5.53,lw=.8)
    label(5,7.66,'1,000 total training draws · requested 50/50 subset predictions · attainment evaluated',7.8)
    label(5,7.31,'Reference arm: random subsets throughout, using the same training-row budget',7.8,color=COLORS['random'])
    heading(6.88,'C','Binary inclusion design and grouped-image GNN')
    # Matrix is schematic; only its binary structure is asserted.
    pattern=np.array([[1,0,1,1,0],[0,1,1,0,1],[1,1,0,1,0],[0,1,0,1,1]])
    label(1.02,6.34,'$U$: binary\ninclusion matrix',8.2,weight='bold')
    for r in range(4):
        for c in range(5):
            x=.18+c*.34;y=5.04+(3-r)*.25
            ax.add_patch(Rectangle((x,y),.31,.22,fc='#DBE6F3' if pattern[r,c] else '#F1F3F5',ec='white',lw=.5))
            label(x+.155,y+.11,str(pattern[r,c]),7.5)
    label(1.02,4.73,'1: included\n0: omitted',7.5)
    arrow(1.99,5.52,2.34,5.52)
    # Actual SETNET_GAT: shared CNN -> correlation graph -> GAT/MLP -> pooled logits.
    band(2.42,4.91,7.35,1.22,'#FAF2C8')
    for j in range(3):
        image(2.60,5.01+j*.29,w=.39,h=.25,index=j)
        arrow(3.02,5.14+j*.29,3.31,5.53)
    for j in range(3):
        ax.add_patch(Polygon([(3.38+j*.09,5.18+j*.06),(4.0+j*.09,5.18+j*.06),
                             (4.12+j*.09,5.82+j*.06),(3.50+j*.09,5.82+j*.06)],fc='#E9D474',ec='#8F7A2A',lw=.7,zorder=3-j))
    label(3.84,5.55,'CNN',8,weight='bold');label(3.85,4.67,'DenseNet121',7.8)
    arrow(4.37,5.55,4.65,5.55)
    nodes=[(4.83,5.26),(5.23,5.27),(4.90,5.84),(5.33,5.72)]
    for a,b in [(0,1),(0,2),(1,2),(1,3),(2,3)]:
        ax.plot([nodes[a][0],nodes[b][0]],[nodes[a][1],nodes[b][1]],color='#69745B',lw=.9)
    for x,y in nodes:ax.add_patch(Circle((x,y),.085,fc='#788A58',ec='white',lw=.6,zorder=4))
    label(5.12,4.67,'Correlation\ngraph',7.8)
    arrow(5.49,5.55,5.80,5.55)
    for j in range(3):
        for r in range(3):ax.add_patch(Circle((6.02+j*.24,5.20+r*.28),.06,fc=COLORS['ridge'],ec='none'))
    for j in range(2):
        for r in range(3):
            for rr in range(3):ax.plot([6.02+j*.24,6.26+j*.24],[5.2+r*.28,5.2+rr*.28],c='#8BB6AD',lw=.3,zorder=0)
    label(6.28,4.67,'GAT + MLP',7.8)
    arrow(6.68,5.55,6.98,5.55)
    for r in range(3):ax.add_patch(Rectangle((7.06,5.18+r*.25),.70,.17,fc='#D5C269',ec='none'))
    label(7.42,4.67,'Linear +\nmean pool',7.8)
    arrow(7.87,5.55,8.13,5.55)
    ax.barh([5.33,5.70],[.20,.80],left=8.20,height=.16,color=['#8E9DAE',COLORS['random']])
    label(9.28,5.53,'$q(u)$',10);label(8.66,4.67,'Softmax\nprobability',7.8)
    label(5,4.27,'Every sampled image subset receives graph reconstruction and model inference.',7.8)
    heading(3.78,'D','Conditional and marginal image influence')
    ax.plot([9.72,9.96,9.96,1.65],[5.53,5.53,3.45,3.45],c='#58616A',lw=.8)
    ax.text(5,3.45,'(U, q)',fontsize=7.6,ha='center',va='center',bbox=dict(fc='white',ec='none',pad=2))
    methods=[(.2,'Ridge',COLORS['ridge'],'Fixed α = 1'),(3.52,'Pearson',COLORS['pearson'],'Marginal correlation'),(6.85,'Elastic Net',COLORS['elastic_net'],'Training-only\nfive-fold CV')]
    for x,name,color,detail in methods:
        arrow(x+1.45,3.42,x+1.45,3.13,color=color)
        label(x+1.45,2.88,name,9.4,weight='bold',color=color)
        label(x+1.45,2.51,detail,7.9)
        label(x+1.45,2.18,'Conditional' if name!='Pearson' else 'Marginal',7.6)
    label(5,1.99,'Image-influence profiles',9,weight='bold')
    for j,h in enumerate([.40,.28,.20,.12,-.14,-.26]):
        x=2.66+j*.72
        ax.add_patch(Rectangle((x,1.08 if h>0 else 1.08+h),.46,abs(h),fc='#3C5488' if h>0 else '#C66351',ec='none'))
        image(x,1.11+h if h>0 else .63+h,w=.46,h=.28,index=j)
    ax.plot([2.5,7.55],[1.08,1.08],c='#4D555E',lw=.7)
    label(5,.27,'Schematic artwork from the original workflow; evaluated patient explanations appear in Figure 2.',7.3)
    save(fig,Path(output),'figure1_workflow')


def workflow_comparison(output,revised):
    configure();fig,axes=plt.subplots(1,2,figsize=(12,8.2),gridspec_kw={'width_ratios':[1,1]})
    fig.subplots_adjust(left=.025,right=.975,bottom=.08,top=.93,wspace=.08)
    for ax,path,title in zip(axes,[ASSETS/'original_workflow.png',Path(revised)/'figure1_workflow.png'],
                             ['Original accepted workflow','Revised main Figure 1']):
        ax.imshow(plt.imread(path));ax.axis('off');ax.set_title(title,fontsize=12,pad=16)
    fig.text(.04,.025,'Original retained without alteration. Revised artwork preserves ultrasound sets, predicted pools, two sampling stages, design, GNN and explanation branches.',fontsize=9)
    save(fig,Path(output),'workflow_original_vs_revised')
